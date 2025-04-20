# src/test.py

import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import Dataset, DataLoader, RandomSampler
from sklearn.model_selection import train_test_split

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


def make_samples(df_user, label_col, win_size, *, stride):
    """
    Sliding / non‑overlapping window builder.

    Parameters
    ----------
    stride : int
        • stride == win_size → non‑overlapping  
        • stride  < win_size → overlapping
    """
    out, df_user = [], df_user.sort_values("datetime").reset_index(drop=True)
    n = len(df_user)
    for i in range(0, n - win_size + 1, stride):
        chunk = df_user.iloc[i:i + win_size]
        out.append({
            "id":    chunk["id"].iat[0],
            "bpm":   chunk.get("bpm_scaled",   chunk["bpm"]).values,
            "steps": chunk.get("steps_scaled", chunk["steps"]).values,
            "label": int(chunk[label_col].max() == 1),
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

        # --- split raw records into chronological train_val vs test ---
        n = len(df_u)
        test_frac = 0.15
        cut = int((1 - test_frac) * n)
        df_train_val = df_u.iloc[:cut]
        df_test      = df_u.iloc[cut:]

        for lbl in lbl_cols:
            # 4a) create windows only after splitting
            samples_tv   = make_samples(df_train_val, lbl, cfg.window.size, stride=1)
            samples_test = make_samples(df_test, lbl, cfg.window.size, stride=cfg.window.size)

            # catch empty test-window case
            if len(samples_test) == 0:
                print(f"[U{uid}|{lbl}] WARNING: test slice too short for window → skipping")
                continue

            # require enough windows in train_val to stratify
            if len(samples_tv) < 15:
                continue
            y_tv = np.array([s["label"] for s in samples_tv])
            if y_tv.sum() < 2 or (len(y_tv) - y_tv.sum()) < 2:
                continue

            # 5) split train_val windows into train / val
            idx = np.arange(len(samples_tv))
            idx_tv, idx_val = train_test_split(
                idx,
                test_size=test_frac / (1 - test_frac),
                stratify=y_tv,
                random_state=cfg.seed
            )

            idx_train, idx_val = train_test_split(
                np.arange(len(samples_tv)),
                test_size=0.176,          
                stratify=y_tv,
                random_state=cfg.seed,
            )

            def mk_loader(samples, indices, shuffle):
                ds = CraveDataset([samples[i] for i in indices])
                return DataLoader(
                    ds,
                    batch_size=cfg.test.batch_size,
                    sampler=RandomSampler(ds) if shuffle else None
                )

            train_loader = mk_loader(samples_tv, idx_train, True)
            val_loader   = mk_loader(samples_tv, idx_val,   False)
            test_loader  = DataLoader(
                CraveDataset(samples_test),
                batch_size=cfg.test.batch_size,
                shuffle=False
            )

            # 6) Build model & load SSL weights
            model = DrugClassifier(window_size=cfg.window.size).to(device)
            ckpt  = os.path.join(cfg.results_root, "train", f"user_{uid}", "personalized_ssl.pt")
            if os.path.isfile(ckpt):
                ssl_sd   = torch.load(ckpt, map_location="cpu")
                model_sd = model.state_dict()
                matched  = {k: v for k, v in ssl_sd.items()
                            if k in model_sd and v.shape == model_sd[k].shape}
                model_sd.update(matched)
                model.load_state_dict(model_sd)
                print(f"[U{uid}] Transferred {len(matched)}/{len(model_sd)} layers")
            else:
                print(f"[U{uid}] No SSL checkpoint – random init")

            # 7) Freeze all but classifier & LSTM
            for name, param in model.named_parameters():
                param.requires_grad = ("classifier" in name or "rnn" in name)

            # 8) Compute class‐weight from train only
            train_labels = np.array([samples_tv[i]["label"] for i in idx_train])
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

            # 9) Train/val loop
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

                # — VAL —
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

                scheduler.step(val_loss)
                if val_loss < best_val:
                    best_val, no_imp = val_loss, 0
                    torch.save(model.state_dict(), best_ck)
                else:
                    no_imp += 1

            if not os.path.isfile(best_ck):
                continue
            model.load_state_dict(torch.load(best_ck, map_location=device))

            # 10) Select threshold by maximizing Youden's J (sens + spec – 1) on val
            pv, yv = [], []
            model.eval()
            with torch.no_grad():
                for bpm, steps, _id, tgt in val_loader:
                    logits = model(bpm.to(device), steps.to(device))
                    pv.extend(torch.sigmoid(logits).cpu().tolist())
                    yv.extend(tgt.cpu().tolist())
            pv, yv = np.array(pv), np.array(yv)

            best_j, best_thr = -1, 0.5
            for t in np.linspace(0, 1, 101):
                preds_t = (pv >= t).astype(int)
                tn, fp, fn, tp = confusion_matrix(yv, preds_t, labels=[0, 1]).ravel()
                sens = tp / (tp + fn) if (tp + fn) > 0 else 0
                spec = tn / (tn + fp) if (tn + fp) > 0 else 0
                j = sens + spec - 1
                if j > best_j:
                    best_j, best_thr = j, t

            # 11) Final test evaluation
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
            sens = tp / (tp + fn) if (tp + fn) > 0 else np.nan
            spec = tn / (tn + fp) if (tn + fp) > 0 else np.nan

            print(f"[U{uid}|{lbl}] TEST thr={best_thr:.2f} acc={acc:.2f}% auc={auc:.3f} "
                  f"sens={sens:.3f} spec={spec:.3f}")

            # save confusion matrix
            plt.figure(figsize=(3, 3))
            plt.imshow([[tn, fp], [fn, tp]], cmap="Blues")
            for i, v in enumerate([tn, fp, fn, tp]):
                plt.text(i % 2, i // 2, str(v),
                         ha="center", va="center",
                         color="white" if v > max(tn, fp, fn, tp)/2 else "black")
            plt.title(f"U{uid} {lbl}\nth={best_thr:.2f} acc={acc:.1f}%")
            plt.tight_layout()
            plt.savefig(os.path.join(lbl_dir, f"cm_{best_thr:.2f}.png"))
            plt.close()

            # record results on test set windows only
            all_results.append({
                "user_id":     uid,
                "label_col":   lbl,
                "n_test":      len(yt),           # total test windows
                "pos":         int(yt.sum()),     # positives in test
                "neg":         int(len(yt) - yt.sum()),  # negatives in test
                "thr":         best_thr,
                "auc":         auc,
                "acc":         acc,
                "tn":          tn,
                "fp":          fp,
                "fn":          fn,
                "tp":          tp,
                "sensitivity": sens,
                "specificity": spec
            })

    # 12) Save summary CSV
    if all_results:
        pd.DataFrame(all_results).to_csv(
            os.path.join(res_root, "classification_summary.csv"),
            index=False
        )


if __name__ == "__main__":
    main()
