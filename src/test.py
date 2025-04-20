# src/test.py

import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import Dataset, DataLoader, RandomSampler

from src.config import load_config
import src.utils as U
from src.models import DrugClassifier

warnings.filterwarnings("ignore")


class CraveDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.tensor(s["bpm"], dtype=torch.float32),
            torch.tensor(s["steps"], dtype=torch.float32),
            s["id"],
            torch.tensor(s["label"], dtype=torch.float32),
        )


def make_samples(df_user, label_col, win_size):
    """
    Create sliding windows of size win_size with stride=1.
    Label=1 if any hour in window has a positive event.
    """
    out = []
    df_user = df_user.sort_values("datetime").reset_index(drop=True)
    n = len(df_user)
    for i in range(0, n - win_size + 1):
        chunk = df_user.iloc[i : i + win_size]
        out.append({
            "id":    chunk["id"].iat[0],
            "bpm":   chunk.get("bpm_scaled", chunk["bpm"]).values,
            "steps": chunk.get("steps_scaled", chunk["steps"]).values,
            "label": int(chunk[label_col].max() == 1)
        })
    return out


def main():
    cfg = load_config()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # 1) Load & merge sensor + label data
    df_sensor = U.load_sensor_data(cfg.data.biosignal_dir)
    df_label  = U.load_label_data(cfg.data.label_root)
    df_hour   = U.pivot_label_data(df_label)
    df        = U.merge_sensors_with_labels(df_sensor, df_hour)
    df        = df.groupby("id").apply(U.scale_per_user).reset_index(drop=True)

    # 2) Identify label columns
    lbl_cols = [c for c in df.columns if c.endswith("_label")]
    if not lbl_cols:
        raise SystemExit("No label columns found – aborting")

    # 3) Prepare output directory
    res_root = os.path.join(cfg.results_root, "test")
    os.makedirs(res_root, exist_ok=True)
    all_results = []

    # 4) Per-user loops
    for uid in sorted(df["id"].unique()):
        user_dir = os.path.join(res_root, f"user_{uid}")
        os.makedirs(user_dir, exist_ok=True)
        df_u = df[df["id"] == uid]
        if len(df_u) < 2 * cfg.window.size:
            print(f"[User {uid}] Insufficient data – skipping")
            continue

        for lbl in lbl_cols:
            samples = make_samples(df_u, lbl, cfg.window.size)
            if len(samples) < 15:
                continue

            y = np.array([s["label"] for s in samples])
            # require at least 2 positives and 2 negatives overall
            if y.sum() < 2 or (len(y) - y.sum()) < 2:
                continue

            # train/val/test split (70/15/15)
            idx = np.arange(len(samples))
            idx_tv, idx_test = train_test_split(
                idx, test_size=0.15, stratify=y, random_state=cfg.seed
            )
            val_frac = 0.15 / 0.85
            idx_train, idx_val = train_test_split(
                idx_tv, test_size=val_frac, stratify=y[idx_tv], random_state=cfg.seed
            )

            def mk_loader(indices, shuffle):
                ds = CraveDataset([samples[i] for i in indices])
                return DataLoader(
                    ds,
                    batch_size=cfg.test.batch_size,
                    sampler=RandomSampler(ds) if shuffle else None
                )

            train_loader = mk_loader(idx_train, True)
            val_loader   = mk_loader(idx_val,   False)
            test_loader  = mk_loader(idx_test,  False)

            # 5) Build model & load personalized SSL weights
            model = DrugClassifier(window_size=cfg.window.size).to(device)
            ckpt  = os.path.join(cfg.results_root, "train", f"user_{uid}", "personalized_ssl.pt")
            if os.path.isfile(ckpt):
                ssl_sd   = torch.load(ckpt, map_location="cpu")
                model_sd = model.state_dict()
                matched  = {
                    k: v for k, v in ssl_sd.items()
                    if k in model_sd and v.shape == model_sd[k].shape
                }
                model_sd.update(matched)
                model.load_state_dict(model_sd)
                print(f"[U{uid}] Transferred {len(matched)}/{len(model_sd)} layers")
            else:
                print(f"[U{uid}] No SSL checkpoint – random init")

            # 6) Unfreeze classifier head + LSTM layers
            for name, param in model.named_parameters():
                if "classifier" in name or "lstm" in name:
                    param.requires_grad = True
                else:
                    param.requires_grad = False

            # 7) Weighted loss to counter class imbalance in training set
            train_labels = np.array([samples[i]["label"] for i in idx_train])
            pos = max(train_labels.sum(), 1)
            neg = max(len(train_labels) - pos, 1)
            pos_weight = torch.tensor([neg / pos], device=device)
            criterion = torch.nn.BCEWithLogitsLoss(pos_weight=pos_weight)

            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=cfg.test.lr
            )
            scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
                optimizer,
                mode='min',
                patience=cfg.test.scheduler.patience,
                factor=cfg.test.scheduler.factor,
                min_lr=cfg.test.scheduler.min_lr,
                verbose=True
            )

            # 8) Train/validate loop
            best_val, no_imp = float("inf"), 0
            lbl_dir = os.path.join(user_dir, lbl)
            os.makedirs(lbl_dir, exist_ok=True)
            best_ck = os.path.join(lbl_dir, "best.pt")

            for ep in range(1, cfg.test.num_epochs + 1):
                if no_imp >= cfg.test.patience:
                    break

                # — TRAIN —
                model.train()
                tr_loss = 0.0
                for bpm, steps, _id, tgt in train_loader:
                    bpm, steps, tgt = bpm.to(device), steps.to(device), tgt.to(device)
                    optimizer.zero_grad()
                    logits = model(bpm, steps)
                    loss   = criterion(logits, tgt)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    tr_loss += loss.item()
                tr_loss /= len(train_loader)

                # — VALIDATE —
                model.eval()
                val_loss, preds, gts = 0.0, [], []
                with torch.no_grad():
                    for bpm, steps, _id, tgt in val_loader:
                        bpm, steps, tgt = bpm.to(device), steps.to(device), tgt.to(device)
                        logits = model(bpm, steps)
                        val_loss += criterion(logits, tgt).item()
                        preds.extend(torch.sigmoid(logits).cpu().tolist())
                        gts.extend(tgt.cpu().tolist())
                val_loss /= len(val_loader)
                acc_val = ((np.array(preds) >= 0.5) == np.array(gts)).mean() * 100
                print(f"[U{uid}|{lbl}] ep{ep:02d} tr{tr_loss:.4f} vl{val_loss:.4f} acc{acc_val:5.1f}%")

                # step scheduler on validation loss
                scheduler.step(val_loss)

                if val_loss < best_val:
                    best_val, no_imp = val_loss, 0
                    torch.save(model.state_dict(), best_ck)
                else:
                    no_imp += 1

            if not os.path.isfile(best_ck):
                continue
            model.load_state_dict(torch.load(best_ck, map_location=device))

            # 9) Select best threshold on validation set
            pv, yv = [], []
            model.eval()
            with torch.no_grad():
                for bpm, steps, _id, tgt in val_loader:
                    logits = model(bpm.to(device), steps.to(device))
                    pv.extend(torch.sigmoid(logits).cpu().tolist())
                    yv.extend(tgt.cpu().tolist())
            pv, yv = np.array(pv), np.array(yv)
            best_thr = max(np.linspace(0, 1, 101),
                           key=lambda t: ((pv >= t) == yv).mean())

            # 10) Final test evaluation
            pt, yt = [], []
            with torch.no_grad():
                for bpm, steps, _id, tgt in test_loader:
                    logits = model(bpm.to(device), steps.to(device))
                    pt.extend(torch.sigmoid(logits).cpu().tolist())
                    yt.extend(tgt.cpu().tolist())
            pt, yt = np.array(pt), np.array(yt)
            auc  = np.nan if len(np.unique(yt)) == 1 else roc_auc_score(yt, pt)
            pred = (pt >= best_thr).astype(int)
            acc  = (pred == yt).mean() * 100
            tn, fp, fn, tp = confusion_matrix(yt, pred, labels=[0, 1]).ravel()

            # Compute sensitivity and specificity
            sensitivity = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            specificity = tn / (tn + fp) if (tn + fp) > 0 else np.nan

            print(f"[U{uid}|{lbl}] TEST thr={best_thr:.2f} acc={acc:.2f}% auc={auc:.3f} "
                  f"sens={sensitivity:.3f} spec={specificity:.3f}")

            # Plot & save confusion matrix
            plt.figure(figsize=(3, 3))
            plt.imshow([[tn, fp], [fn, tp]], cmap="Blues")
            for i, v in enumerate([tn, fp, fn, tp]):
                plt.text(i % 2, i // 2, str(v),
                         ha="center", va="center",
                         color="white" if v > max(tn, fp, fn, tp) / 2 else "black")
            plt.title(f"U{uid} {lbl}\nth={best_thr:.2f} acc={acc:.1f}%")
            plt.tight_layout()
            plt.savefig(os.path.join(lbl_dir, f"cm_{best_thr:.2f}.png"))
            plt.close()

            all_results.append({
                "user_id":   uid,
                "label_col": lbl,
                "pos":       int(yt.sum()),
                "neg":       int(len(yt) - yt.sum()),
                "thr":       best_thr,
                "auc":       auc,
                "acc":       acc,
                "tn":        tn,
                "fp":        fp,
                "fn":        fn,
                "tp":        tp,
                "sensitivity": sensitivity,
                "specificity": specificity
            })

    # 11) Save summary CSV
    if all_results:
        pd.DataFrame(all_results).to_csv(
            os.path.join(res_root, "classification_summary.csv"),
            index=False
        )


if __name__ == "__main__":
    main()
