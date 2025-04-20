"""Shared utilities for data loading, scaling, window slicing, batching,
plotting, and auto‑updating README tables."""
from __future__ import annotations

import os, re, glob, json, numpy as np, pandas as pd, matplotlib.pyplot as plt, torch
from pathlib import Path
from torch.utils.data import Dataset, Sampler
from sklearn.preprocessing import StandardScaler

plt.switch_backend("Agg")  # always save to file, never pop GUI

# ------------------------------------------------------------------
# Scaling helpers (unchanged)
# ------------------------------------------------------------------
user_scalers: dict[int, dict[str, float]] = {}

def scale_per_user(g: pd.DataFrame, col_bpm="bpm", col_steps="steps") -> pd.DataFrame:
    uid = g.name
    sb, ss = StandardScaler(), StandardScaler()
    g["bpm_scaled"]   = sb.fit_transform(g[[col_bpm]]).ravel()
    g["steps_scaled"] = ss.fit_transform(g[[col_steps]]).ravel()
    user_scalers[uid] = dict(bpm_mean=sb.mean_[0], bpm_scale=sb.scale_[0],
                             steps_mean=ss.mean_[0], steps_scale=ss.scale_[0])
    return g

def inverse_transform(uid: int, bpm_s: np.ndarray, steps_s: np.ndarray):
    m = user_scalers[uid]
    return bpm_s * m["bpm_scale"] + m["bpm_mean"], steps_s * m["steps_scale"] + m["steps_mean"]

# ------------------------------------------------------------------
# I/O helpers (unchanged from original)
# ------------------------------------------------------------------
FRUIT_TO_DRUG = {"melon":"methamphetamine","almond":"alcohol","carrot":"cannabis","orange":"opioid","coconut":"cocaine","strawberry":"sedative_benzodiazepine","nectarine":"nicotine"}

def load_sensor_data(biosignal_dir: str) -> pd.DataFrame:
    csvs = glob.glob(os.path.join(biosignal_dir, "*.csv"))
    if not csvs:
        raise FileNotFoundError(f"No CSVs in {biosignal_dir}")
    dfs = []
    for fp in csvs:
        df = pd.read_csv(fp, parse_dates=["time"])
        if "participant_id" not in df.columns:
            base = os.path.splitext(os.path.basename(fp))[0]
            pid  = int(base[1:] if base.lower().startswith("p") else base)
            df["participant_id"] = pid
        dfs.append(df)
    df = pd.concat(dfs, ignore_index=True)
    df.loc[df["data_type"]=="hr", "data_type"] = "heart_rate"
    df = df[df["data_type"].isin(["heart_rate", "steps"])].copy()
    df.drop_duplicates(["participant_id", "time", "data_type"], inplace=True)
    df = df.pivot(index=["participant_id", "time"], columns="data_type", values="value").reset_index()
    df.rename(columns={"participant_id":"id", "heart_rate":"bpm"}, inplace=True)
    df[["bpm","steps"]] = df[["bpm","steps"]].apply(pd.to_numeric, errors="coerce")
    df.dropna(subset=["bpm","steps"], inplace=True)
    df["datetime"] = pd.to_datetime(df["time"])
    df["date"]     = df["datetime"].dt.date
    df["hour"]     = df["datetime"].dt.hour
    df = df[["id","datetime","date","hour","bpm","steps"]].copy()
    df.sort_values(["id","datetime"], inplace=True)

    # minute→hour aggregation
    df["date"] = pd.to_datetime(df["date"])
    to_hour = lambda d,h: pd.to_datetime(d) + pd.to_timedelta(h, unit="H")
    dfh = (df.groupby(["id","date","hour"], as_index=False)
             .agg({"bpm":"mean","steps":"sum"}))
    dfh["datetime"] = dfh.apply(lambda r: to_hour(r["date"], r["hour"]), axis=1)
    return dfh.sort_values(["id","datetime"]).reset_index(drop=True)

def load_label_data(label_root: str) -> pd.DataFrame:
    rows = []
    for id_dir in glob.glob(os.path.join(label_root, "ID*")):
        try: pid = int(os.path.basename(id_dir).replace("ID", ""))
        except ValueError: continue
        for fp in glob.glob(os.path.join(id_dir, "*.*")):
            if fp.endswith(".csv"):
                try: df = pd.read_csv(fp)
                except Exception: continue
            elif fp.endswith(".xlsx"):
                try: df = pd.read_excel(fp)
                except Exception: continue
            else: continue
            if df.empty: continue
            time_col = next((c for c in ["hawaii_use_time","hawaii_createdat_time","created_at"] if c in df.columns), None)
            if time_col is None: continue
            df[time_col] = pd.to_datetime(df[time_col], errors="coerce")
            df.dropna(subset=[time_col], inplace=True)
            for _, r in df.iterrows():
                fruit = str(r.get("substance_fruit_label", "")).lower().strip()
                drug  = FRUIT_TO_DRUG.get(fruit, fruit)
                lab   = str(r.get("crave_use_none_label", "")).lower().strip()
                if lab not in {"use", "crave"}: continue
                rows.append(dict(id=pid, datetime=r[time_col], drug_name=drug, label_str=lab))
    return (pd.DataFrame(rows)
              .sort_values(["id","datetime"])
              .reset_index(drop=True))

def pivot_label_data(df_long: pd.DataFrame) -> pd.DataFrame:
    if df_long.empty: return df_long.copy()
    for d in df_long["drug_name"].unique():
        for l in ["use","crave"]:
            df_long[f"{d}_{l}_label"] = ((df_long["drug_name"]==d) & (df_long["label_str"]==l)).astype(int)
    df_long["date"] = df_long["datetime"].dt.date
    df_long["hour"] = df_long["datetime"].dt.hour
    df_long["date"] = pd.to_datetime(df_long["date"])
    agg = {c: "max" for c in df_long.columns if c.endswith("_label")}
    dfh = (df_long.groupby(["id","date","hour"], as_index=False).agg(agg))
    dfh["datetime"] = dfh.apply(lambda r: pd.to_datetime(r["date"]) + pd.to_timedelta(r["hour"], unit="H"), axis=1)
    return dfh

def merge_sensors_with_labels(df_sensor: pd.DataFrame, df_label: pd.DataFrame) -> pd.DataFrame:
    if df_label.empty: return df_sensor.copy()
    df_sensor["datetime"] = df_sensor["datetime"].dt.tz_localize(None)
    df_label["datetime"]  = df_label["datetime"].dt.tz_localize(None)
    df = pd.merge(df_sensor, df_label, on=["id","datetime"], how="left")
    for c in df_label.columns:
        if c.endswith("_label"):
            df[c] = df[c].fillna(0).astype(int)
    return df

# ------------------------------------------------------------------
# Forecasting sample builder (unchanged)
# ------------------------------------------------------------------

def create_forecasting_samples(df: pd.DataFrame, *, col_bpm="bpm_scaled", col_steps="steps_scaled", window_size=6, input_windows=2, predict_windows=1):
    samples = []
    for uid, g in df.groupby("id"):
        g = g.sort_values("datetime").reset_index(drop=True)
        wins = [g.iloc[i:i+window_size] for i in range(0, len(g), window_size) if len(g.iloc[i:i+window_size])==window_size]
        for i in range(len(wins)-input_windows-predict_windows+1):
            hist=wins[i:i+input_windows]; fut=wins[i+input_windows:i+input_windows+predict_windows]
            bpm_hist   = np.stack([w[col_bpm].values   for w in hist])
            steps_hist = np.stack([w[col_steps].values for w in hist])
            bpm_t      = np.stack([w[col_bpm].values   for w in fut])
            steps_t    = np.stack([w[col_steps].values for w in fut])
            bpm_o      = np.stack([w["bpm"].values     for w in fut])
            steps_o    = np.stack([w["steps"].values   for w in fut])
            samples.append(dict(user_id=uid, bpm_input=bpm_hist, steps_input=steps_hist,
                                current_bpm_scaled=bpm_t, current_steps_scaled=steps_t,
                                bpm_target=bpm_t, steps_target=steps_t,
                                bpm_target_original=bpm_o, steps_target_original=steps_o,
                                datetime=fut[0]["datetime"].values[0]))
    return samples

# ------------------------------------------------------------------
# Dataset & sampler (unchanged)
# ------------------------------------------------------------------
class ForecastDataset(Dataset):
    def __init__(self, data): self.data=data
    def __len__(self): return len(self.data)
    def __getitem__(self,i): 
        d=self.data[i]
        t=lambda x: torch.tensor(x,dtype=torch.float32)
        return (t(d["bpm_input"]), t(d["steps_input"]), t(d["current_bpm_scaled"]), t(d["current_steps_scaled"]),
                t(d["bpm_target"]), t(d["steps_target"]), d["user_id"],
                torch.tensor(d["bpm_target_original"]), torch.tensor(d["steps_target_original"]), d["datetime"])

def forecasting_collate_fn(batch):
    s=lambda xs: torch.stack(xs)
    b0,b1,b2,b3,b4,b5,uid,b6,b7,dt=zip(*batch)
    return (s(b0),s(b1),s(b2),s(b3),s(b4),s(b5),list(uid),list(b6),list(b7),list(dt))

class PerUserBatchSampler(Sampler):
    def __init__(self, user_dict, batch_size=128):
        self.user_dict=user_dict; self.uids=list(user_dict.keys()); self.batch_size=batch_size
    def __iter__(self):
        np.random.shuffle(self.uids)
        for uid in self.uids:
            idx=self.user_dict[uid]; np.random.shuffle(idx)
            for i in range(0,len(idx),self.batch_size): yield idx[i:i+self.batch_size]
    def __len__(self):
        return sum((len(v)+self.batch_size-1)//self.batch_size for v in self.user_dict.values())

# ------------------------------------------------------------------
# Plot helpers (unchanged)
# ------------------------------------------------------------------

def plot_prediction_user(uid:int, samples:list[dict], tag:str, pred_win:int, win:int, out_dir:str):
    os.makedirs(out_dir, exist_ok=True)
    for i,s in enumerate(samples):
        bp_p=s["bpm_pred"].flatten(); bp_t=s["bpm_true"].flatten(); st_p=s["steps_pred"].flatten(); st_t=s["steps_true"].flatten(); n=len(bp_p)
        fig,axs=plt.subplots(2,2,figsize=(15,10)); fig.suptitle(f"User {uid} sample {i+1} {s['datetime']}")
        axs[0,0].plot(bp_t,label="True BPM"); axs[0,0].plot(bp_p,label="Pred BPM"); axs[0,0].legend()
        axs[0,1].plot(st_t,label="True Steps"); axs[0,1].plot(st_p,label="Pred Steps"); axs[0,1].legend()
        axs[1,0].bar(range(n),np.abs(bp_t-bp_p)); axs[1,0].set_title("BPM abs err")
        axs[1,1].bar(range(n),np.abs(st_t-st_p)); axs[1,1].set_title("Steps abs err")
        plt.tight_layout(rect=[0,0.04,1,0.96])
        plt.savefig(os.path.join(out_dir,f"{tag}_u{uid}_s{i+1}.png")); plt.close()

# ------------------------------------------------------------------
# README sync helper (now supports custom template pre‑/post text)
# ------------------------------------------------------------------

def _format_df(df: pd.DataFrame) -> pd.DataFrame:
    fmt=df.copy()
    for c in fmt.columns:
        if fmt[c].dtype==float: fmt[c]=fmt[c].round(3)
    return fmt

def sync_readme_table(csv_path:str, *, markdown_path:str="README.md", tag:str|None=None, static_head:str|None=None) -> None:
    """Insert CSV→Markdown table into README. If *static_head* given it
    is inserted above the auto‑table (used for comparison section).
    """
    tag=tag or Path(csv_path).stem.upper()
    if not os.path.isfile(csv_path):
        print(f"[sync_readme_table] CSV not found → {csv_path}"); return
    df=_format_df(pd.read_csv(csv_path)); tbl=df.to_markdown(index=False)
    start,end=f"<!-- START {tag} TABLE -->",f"<!-- END {tag} TABLE -->"
    block=f"{start}\n\n"+(static_head+"\n\n" if static_head else "")+f"{tbl}\n\n{end}"
    if not os.path.isfile(markdown_path):
        with open(markdown_path,"w") as f: f.write(block+"\n"); return
    with open(markdown_path,"r") as f: txt=f.read()
    txt=re.sub(f"{start}[\s\S]*?{end}", block, txt) if re.search(f"{start}[\s\S]*?{end}",txt) else txt+"\n"+block
    with open(markdown_path,"w") as f: f.write(txt)