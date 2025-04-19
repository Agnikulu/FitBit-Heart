# --------------------------------------------------------------------------- #
# 0 . Imports
# --------------------------------------------------------------------------- #
import os, glob, random, warnings
import numpy  as np
import pandas as pd
import torch, torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader, RandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

# Local helper modules
from models import DrugClassifier, partially_unfreeze_backbone
import utils as U   # per‑user scaling utilities etc.

# --------------------------------------------------------------------------- #
# 1 . Hyper‑parameters & constants
# --------------------------------------------------------------------------- #
WINDOW_SIZE         = 6          # hours in one classification window
BATCH_SIZE          = 32
LEARNING_RATE       = 1e-3
NUM_EPOCHS          = 20
PATIENCE            = 5          # early‑stopping patience
UNFREEZE_RATIO      = 0.30       # fraction of backbone layers to keep trainable
RESULTS_ROOT        = "results"  # base results directory

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
rng    = np.random.default_rng(42)
torch.manual_seed(42); random.seed(42)

# Mapping fruit codes → correct drug names (lower‑case, no spaces/slashes)
FRUIT_TO_DRUG = {
    "melon":      "methamphetamine",
    "almond":     "alcohol",
    "carrot":     "cannabis",
    "orange":     "opioid",
    "coconut":    "cocaine",
    "strawberry": "sedative_benzodiazepine",
    "nectarine":  "nicotine",
}

# --------------------------------------------------------------------------- #
# 2 . Data‑loading helpers
# --------------------------------------------------------------------------- #
def load_sensor_data() -> pd.DataFrame:
    """
    Load minute‑level sensor CSVs and aggregate to hourly rows.
    Output columns: id, datetime, date, hour, bpm, steps
    """
    data_dir  = "data/Personalized AI Data/Biosignal"
    csv_files = glob.glob(os.path.join(data_dir, "*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSVs found in {data_dir!r}")

    dfs = []
    for fp in csv_files:
        df = pd.read_csv(fp, parse_dates=["time"])
        if "participant_id" not in df.columns:           # derive from filename
            base = os.path.splitext(os.path.basename(fp))[0]
            pid  = int(base[1:] if base.lower().startswith("p") else base)
            df["participant_id"] = pid
        dfs.append(df)

    df_raw = pd.concat(dfs, ignore_index=True)
    df_raw.loc[df_raw["data_type"] == "hr", "data_type"] = "heart_rate"
    df_raw = df_raw[df_raw["data_type"].isin(["heart_rate", "steps"])].copy()
    df_raw.drop_duplicates(["participant_id", "time", "data_type"], inplace=True)

    df_piv = df_raw.pivot(index=["participant_id", "time"],
                          columns="data_type", values="value").reset_index()
    df_piv.rename(columns={"participant_id": "id", "heart_rate": "bpm"}, inplace=True)
    df_piv["bpm"]   = pd.to_numeric(df_piv["bpm"],   errors="coerce")
    df_piv["steps"] = pd.to_numeric(df_piv["steps"], errors="coerce")
    df_piv.dropna(subset=["bpm", "steps"], inplace=True)

    df_piv["datetime"] = pd.to_datetime(df_piv["time"])
    df_piv["date"]     = df_piv["datetime"].dt.date
    df_piv["hour"]     = df_piv["datetime"].dt.hour

    df_sensor = df_piv[["id", "datetime", "date", "hour", "bpm", "steps"]].copy()
    df_sensor.sort_values(["id", "datetime"], inplace=True)
    df_sensor.reset_index(drop=True, inplace=True)

    # aggregate to hourly
    df_sensor["date"] = pd.to_datetime(df_sensor["date"])
    def _to_hour(dt, hr): return pd.to_datetime(dt) + pd.to_timedelta(hr, unit="H")
    df_hour = (
        df_sensor
        .groupby(["id", "date", "hour"], as_index=False)
        .agg({"bpm": "mean", "steps": "sum"})
    )
    df_hour["datetime"] = df_hour.apply(lambda r: _to_hour(r["date"], r["hour"]), axis=1)
    df_hour.sort_values(["id", "datetime"], inplace=True)
    df_hour.reset_index(drop=True, inplace=True)
    return df_hour


def load_label_data() -> pd.DataFrame:
    """
    Read all label spreadsheets under data/Personalized AI Data/Label/ID*/ and
    return a tall dataframe with columns:
        id, datetime, drug_name (mapped), label_str ("use" / "crave")
    """
    label_root = "data/Personalized AI Data/Label"
    rows = []

    for id_dir in glob.glob(os.path.join(label_root, "ID*")):
        try:
            pid = int(os.path.basename(id_dir).replace("ID", ""))
        except ValueError:
            continue

        for fp in glob.glob(os.path.join(id_dir, "*.*")):
            if fp.endswith(".csv"):
                try: df = pd.read_csv(fp)
                except Exception: continue
            elif fp.endswith(".xlsx"):
                try: df = pd.read_excel(fp)
                except Exception: continue
            else:
                continue
            if df.empty: continue

            time_col = next((c for c in
                             ["hawaii_use_time", "hawaii_createdat_time", "created_at"]
                             if c in df.columns), None)
            if time_col is None: continue

            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df.dropna(subset=[time_col], inplace=True)

            for _, r in df.iterrows():
                fruit_raw = str(r.get("substance_fruit_label", "")).lower().strip()
                drug_name = FRUIT_TO_DRUG.get(fruit_raw, fruit_raw)  # map → drug
                lab       = str(r.get("crave_use_none_label", "")).lower().strip()
                if lab not in {"use", "crave"}: continue
                rows.append(dict(id=pid,
                                 datetime=r[time_col],
                                 drug_name=drug_name,
                                 label_str=lab))
    return (pd.DataFrame(rows)
              .sort_values(["id", "datetime"])
              .reset_index(drop=True))


def pivot_label_data(df_long: pd.DataFrame) -> pd.DataFrame:
    """
    Convert the tall `df_long` from `load_label_data()` into a wide hourly frame
    with one binary column per (drug, label_type):

        methamphetamine_use_label, opioid_crave_label, ...

    Returns a dataframe with id, date, hour, datetime, and the binary cols.
    """
    if df_long.empty: return df_long.copy()

    drugs      = sorted(df_long["drug_name"].unique())
    label_typs = ["use", "crave"]

    for d in drugs:
        for l in label_typs:
            df_long[f"{d}_{l}_label"] = (
                (df_long["drug_name"] == d) & (df_long["label_str"] == l)
            ).astype(int)

    df_long["date"] = df_long["datetime"].dt.date
    df_long["hour"] = df_long["datetime"].dt.hour
    df_long["date"] = pd.to_datetime(df_long["date"])

    agg = {f"{d}_{l}_label": "max" for d in drugs for l in label_typs}

    df_hour = (
        df_long
        .groupby(["id", "date", "hour"], as_index=False)
        .agg(agg)
    )
    df_hour["datetime"] = df_hour.apply(
        lambda r: pd.to_datetime(r["date"]) + pd.to_timedelta(r["hour"], unit="H"),
        axis=1
    )
    return df_hour


def merge_sensors_with_labels(df_sensor_hour: pd.DataFrame,
                              df_label_hour:  pd.DataFrame) -> pd.DataFrame:
    """
    Outer‑join sensor rows with label columns; fill missing labels with 0.
    """
    if df_label_hour.empty:
        return df_sensor_hour.copy()

    df_sensor_hour["datetime"] = df_sensor_hour["datetime"].dt.tz_localize(None)
    df_label_hour["datetime"]  = df_label_hour["datetime"].dt.tz_localize(None)

    df = pd.merge(df_sensor_hour, df_label_hour,
                  on=["id", "datetime"], how="left")
    lbl_cols = [c for c in df_label_hour.columns if c.endswith("_label")]
    for c in lbl_cols: df[c] = df[c].fillna(0).astype(int)
    return df


# --------------------------------------------------------------------------- #
# 3 . Dataset helpers
# --------------------------------------------------------------------------- #
class CravingDataset(Dataset):
    """
    Each item yields:
        bpm_in    – torch.FloatTensor [WINDOW_SIZE]
        steps_in  – torch.FloatTensor [WINDOW_SIZE]
        uid       – int
        label     – torch.float32 (0/1)
    """
    def __init__(self, samples): self.samples = samples
    def __len__(self):           return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.tensor(s["bpm_input"],   dtype=torch.float32),
            torch.tensor(s["steps_input"], dtype=torch.float32),
            s["id"],
            torch.tensor(s["label"],       dtype=torch.float32),
        )


def create_window_samples(df_user: pd.DataFrame, label_col: str):
    """
    Slice single‑user dataframe into **non‑overlapping** windows of size WINDOW_SIZE.
    The window label = 1 iff ANY row in the window has label_col == 1.
    """
    out, n = [], len(df_user := df_user.sort_values("datetime").reset_index(drop=True))
    for i in range(0, n, WINDOW_SIZE):
        chunk = df_user.iloc[i:i+WINDOW_SIZE]
        if len(chunk) < WINDOW_SIZE: break
        lbl  = int(chunk[label_col].max() == 1)

        bpm   = chunk.get("bpm_scaled",   chunk["bpm"]).values
        steps = chunk.get("steps_scaled", chunk["steps"]).values

        out.append(dict(
            id             = chunk["id"].iat[0],
            bpm_input      = bpm,
            steps_input    = steps,
            label          = lbl,
            datetime_start = chunk["datetime"].iat[0],
            datetime_end   = chunk["datetime"].iat[-1],
        ))
    return out


# --------------------------------------------------------------------------- #
# 4 . Model initialisation per user
# --------------------------------------------------------------------------- #
def init_classifier_for_user(user_id: int) -> DrugClassifier:
    """
    Build DrugClassifier, transfer SSL weights (if present), and freeze layers.
    """
    model = DrugClassifier(window_size=WINDOW_SIZE).to(device)

    ckpt = os.path.join(RESULTS_ROOT, "train",
                        f"user_{user_id}", "personalized_ssl.pt")
    if os.path.isfile(ckpt):
        ssl_state   = torch.load(ckpt, map_location="cpu")
        model_state = model.state_dict()
        matched = {k: v for k, v in ssl_state.items()
                   if k in model_state and v.shape == model_state[k].shape}
        model_state.update(matched)
        model.load_state_dict(model_state)
        print(f"[User {user_id}] transferred {len(matched)}/{len(model_state)} layers "
              f"from SSL checkpoint")
    else:
        print(f"[User {user_id}] SSL checkpoint not found – training from scratch")

    partially_unfreeze_backbone(model, unfreeze_ratio=UNFREEZE_RATIO)
    return model


# --------------------------------------------------------------------------- #
# 5 . Main procedure
# --------------------------------------------------------------------------- #
def main() -> None:
    warnings.filterwarnings("ignore")

    # 5.1 Load & merge data -------------------------------------------------- #
    df_sensor_hour = load_sensor_data()
    df_label_long  = load_label_data()
    df_label_hour  = pivot_label_data(df_label_long)
    df_merged      = merge_sensors_with_labels(df_sensor_hour, df_label_hour)

    # 5.2 Per‑user scaling --------------------------------------------------- #
    df_merged = df_merged.groupby("id").apply(U.scale_per_user).reset_index(drop=True)

    label_cols = sorted([c for c in df_merged.columns if c.endswith("_label")])
    if not label_cols:
        print("No label columns found – abort"); return

    global_results = []

    for uid in sorted(df_merged["id"].unique()):
        user_root = os.path.join(RESULTS_ROOT, "test", f"user_{uid}")
        os.makedirs(user_root, exist_ok=True)

        df_user = df_merged[df_merged["id"] == uid].copy()
        if len(df_user) < 2 * WINDOW_SIZE:
            print(f"[User {uid}] not enough rows – skip"); continue

        for lbl in label_cols:
            samples = create_window_samples(df_user, lbl)
            if len(samples) < 15:
                print(f"[U{uid}|{lbl}] <15 windows – skip"); continue

            y   = np.array([s["label"] for s in samples])
            pos = int(y.sum()); neg = int(len(y) - pos)
            if pos < 2 or neg < 2:
                print(f"[U{uid}|{lbl}] minority class ≤1 – skip"); continue

            idx            = np.arange(len(samples))
            idx_trainval, idx_test = train_test_split(
                idx, test_size=0.15, stratify=y, random_state=42)
            val_frac       = 0.15 / 0.85
            idx_train, idx_val = train_test_split(
                idx_trainval, test_size=val_frac,
                stratify=y[idx_trainval], random_state=42)

            if not (len(idx_train) and len(idx_val) and len(idx_test)):
                print(f"[U{uid}|{lbl}] empty split – skip"); continue

            # DataLoaders --------------------------------------------------- #
            def make_loader(indices, shuffle=True):
                ds = CravingDataset([samples[i] for i in indices])
                return DataLoader(
                    ds, batch_size=BATCH_SIZE,
                    sampler=RandomSampler(ds) if shuffle else None)
            train_loader = make_loader(idx_train)
            val_loader   = make_loader(idx_val)
            test_loader  = DataLoader(
                CravingDataset([samples[i] for i in idx_test]),
                batch_size=BATCH_SIZE)

            # Model & optim ------------------------------------------------- #
            model     = init_classifier_for_user(uid)
            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(
                filter(lambda p: p.requires_grad, model.parameters()),
                lr=LEARNING_RATE)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

            best_val_loss, no_improve = float("inf"), 0
            lbl_dir  = os.path.join(user_root, lbl)
            os.makedirs(lbl_dir, exist_ok=True)
            best_ckpt = os.path.join(lbl_dir, "best_model.pth")

            # Training ------------------------------------------------------ #
            for ep in range(1, NUM_EPOCHS + 1):
                if no_improve >= PATIENCE: break

                # ---- train one epoch -------------------------------------- #
                model.train(); tr_loss = 0.0
                for bpm, steps, *_ , tgt in train_loader:
                    bpm, steps, tgt = bpm.to(device), steps.to(device), tgt.to(device)
                    optimizer.zero_grad()
                    loss = criterion(model(bpm, steps), tgt)
                    loss.backward(); nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step(); tr_loss += loss.item()
                tr_loss /= len(train_loader); scheduler.step()

                # ---- validate -------------------------------------------- #
                model.eval(); val_loss, preds, gold = 0.0, [], []
                with torch.no_grad():
                    for bpm, steps, *_ , tgt in val_loader:
                        logits = model(bpm.to(device), steps.to(device))
                        val_loss += criterion(logits, tgt.to(device)).item()
                        probs = torch.sigmoid(logits).cpu().numpy()
                        preds.extend(probs); gold.extend(tgt.numpy())
                val_loss /= len(val_loader)
                acc_val = ((np.array(preds) >= 0.5) == np.array(gold)).mean() * 100

                print(f"[U{uid}|{lbl}] ep {ep:02d}/{NUM_EPOCHS} "
                      f"train {tr_loss:.4f} | val {val_loss:.4f} | acc {acc_val:5.2f}%")

                if val_loss < best_val_loss:
                    best_val_loss, no_improve = val_loss, 0
                    torch.save(model.state_dict(), best_ckpt)
                else:
                    no_improve += 1

            if not os.path.isfile(best_ckpt):
                print(f"[U{uid}|{lbl}] no improvement – skip"); continue
            model.load_state_dict(torch.load(best_ckpt, map_location=device))
            model.eval()

            # Select threshold on val set ----------------------------------- #
            probs_val, y_val = [], []
            with torch.no_grad():
                for bpm, steps, *_ , tgt in val_loader:
                    probs_val.extend(torch.sigmoid(
                        model(bpm.to(device), steps.to(device))).cpu().numpy())
                    y_val.extend(tgt.numpy())
            probs_val, y_val = np.array(probs_val), np.array(y_val)
            best_thr = max(np.linspace(0, 1, 101),
                           key=lambda t: ((probs_val >= t) == y_val).mean())

            # Test evaluation ---------------------------------------------- #
            probs_test, y_test = [], []
            with torch.no_grad():
                for bpm, steps, *_ , tgt in test_loader:
                    probs_test.extend(torch.sigmoid(
                        model(bpm.to(device), steps.to(device))).cpu().numpy())
                    y_test.extend(tgt.numpy())
            probs_test, y_test = np.array(probs_test), np.array(y_test)

            auc  = (np.nan if len(np.unique(y_test)) == 1
                    else roc_auc_score(y_test, probs_test))
            pred = (probs_test >= best_thr).astype(int)
            acc  = (pred == y_test).mean() * 100
            tn, fp, fn, tp = confusion_matrix(y_test, pred, labels=[0, 1]).ravel()

            print(f"[U{uid}|{lbl}] TEST thr={best_thr:.2f} "
                  f"acc={acc:.2f}% auc={auc:.3f} "
                  f"TP={tp} FP={fp} TN={tn} FN={fn}")

            # Confusion‑matrix plot ---------------------------------------- #
            cm_png = os.path.join(lbl_dir, f"cm_{best_thr:.2f}.png")
            plt.figure(figsize=(3, 3))
            plt.imshow([[tn, fp], [fn, tp]], cmap="Blues")
            for i, v in enumerate([tn, fp, fn, tp]):
                plt.text(i % 2, i // 2, v,
                         ha="center", va="center",
                         color="white" if v > max(tn, fp, fn, tp) / 2 else "black")
            plt.title(f"U{uid} {lbl}\nth={best_thr:.2f} acc={acc:.1f}%")
            plt.xlabel("Predicted"); plt.ylabel("True")
            plt.tight_layout(); plt.savefig(cm_png); plt.close()

            global_results.append(dict(
                user_id=uid, label_col=lbl,
                pos_count=pos, neg_count=neg,
                best_thr=best_thr, auc=auc, test_acc=acc,
                tn=tn, fp=fp, fn=fn, tp=tp
            ))

    if global_results:
        res_df  = pd.DataFrame(global_results)
        out_csv = os.path.join(RESULTS_ROOT, "test", "classification_summary.csv")
        res_df.to_csv(out_csv, index=False)
        print(f"\nSaved summary → {out_csv}")
    else:
        print("No models were trained – nothing saved")


# --------------------------------------------------------------------------- #
if __name__ == "__main__":
    main()
