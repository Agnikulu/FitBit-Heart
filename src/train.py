# src/train.py – personalized fine‑tuning (full, unabridged, with README sync)
from __future__ import annotations
import os, glob, random, warnings
import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import DataLoader

from src.config import load_config
import src.utils as U
from src.models import PersonalizedForecastingModel

warnings.filterwarnings("ignore")

def main():
    # 1) Config & seeds
    cfg = load_config()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # 2) Load & preprocess biosignal CSVs
    df_list = []
    for fn in os.listdir(cfg.data.biosignal_dir):
        if not fn.endswith(".csv"):
            continue
        path = os.path.join(cfg.data.biosignal_dir, fn)
        df = pd.read_csv(path, parse_dates=["time"])
        if "participant_id" not in df.columns:
            pid = int(os.path.splitext(fn)[0].lstrip("pP"))
            df["participant_id"] = pid
        df_list.append(df)
    if not df_list:
        raise FileNotFoundError(f"No CSVs in {cfg.data.biosignal_dir!r}")

    df_raw = pd.concat(df_list, ignore_index=True)
    df_raw.loc[df_raw["data_type"]=="hr", "data_type"] = "heart_rate"
    df_raw = df_raw[df_raw["data_type"].isin(["heart_rate", "steps"])].copy()
    df_raw.drop_duplicates(["participant_id", "time", "data_type"], inplace=True)

    # pivot and type conversion
    df_piv = df_raw.pivot(index=["participant_id","time"], columns="data_type", values="value").reset_index()
    df_piv.rename(columns={"participant_id":"id","heart_rate":"bpm"}, inplace=True)
    df_piv[["bpm","steps"]] = df_piv[["bpm","steps"]].apply(pd.to_numeric, errors="coerce")
    df_piv.dropna(subset=["bpm","steps"], inplace=True)

    # datetime features
    df_piv["datetime"] = pd.to_datetime(df_piv["time"])
    df_piv["date"]     = df_piv["datetime"].dt.date
    df_piv["hour"]     = df_piv["datetime"].dt.hour

    # hourly aggregation
    df_sensor = df_piv[["id","datetime","date","hour","bpm","steps"]].copy()
    df_sensor.sort_values(["id","datetime"], inplace=True)
    df_sensor["date"] = pd.to_datetime(df_sensor["date"])
    dfh = (
        df_sensor
        .groupby(["id","date","hour"], as_index=False)
        .agg({"bpm":"mean","steps":"sum"})
    )
    dfh["datetime"] = dfh.apply(
        lambda r: pd.to_datetime(r["date"]) + pd.to_timedelta(r["hour"], unit="H"),
        axis=1
    )
    dfh.sort_values(["id","datetime"], inplace=True)
    dfh.reset_index(drop=True, inplace=True)

    # 3) Per‑user scaling
    dfh = dfh.groupby("id").apply(U.scale_per_user).reset_index(drop=True)

    # 4) Create windowed samples
    samples = U.create_forecasting_samples(
        dfh,
        window_size     = cfg.window.size,
        input_windows   = cfg.window.input_windows,
        predict_windows = cfg.window.predict_windows,
    )
    print(f"Created {len(samples)} samples from {dfh['id'].nunique()} users.")

    # 5) Check SSL backbone
    if not os.path.isfile(cfg.ssl_ckpt):
        raise FileNotFoundError(f"SSL checkpoint not found → {cfg.ssl_ckpt}")

    sched_cfg = cfg.train.scheduler
    train_root = os.path.join(cfg.results_root, "train")
    os.makedirs(train_root, exist_ok=True)
    global_stats = []

    # 6) Per‑user fine‑tuning
    for uid in sorted(dfh["id"].unique()):
        user_dir = os.path.join(train_root, f"user_{uid}")
        os.makedirs(user_dir, exist_ok=True)
        usamp = [s for s in samples if s["user_id"] == uid]
        if len(usamp) < 10:
            print(f"[User {uid}] skip (<10 samples)")
            continue

        cut = int(0.8 * len(usamp))
        tr_loader = DataLoader(
            U.ForecastDataset(usamp[:cut]),
            batch_size=cfg.train.batch_size,
            shuffle=True,
            collate_fn=U.forecasting_collate_fn,
        )
        vl_loader = DataLoader(
            U.ForecastDataset(usamp[cut:]),
            batch_size=cfg.train.batch_size,
            shuffle=False,
            collate_fn=U.forecasting_collate_fn,
        )

        # weighted loss coefficients
        udf = dfh[dfh["id"]==uid]
        α = 1.0 / max(udf["bpm"].abs().mean(), 1e-6)
        β = 1.0 / max(udf["steps"].abs().mean(), 1e-6)
        α, β = α/(α+β), β/(α+β)

        # model + load SSL weights
        model = PersonalizedForecastingModel(
            window_size     = cfg.window.size,
            predict_windows = cfg.window.predict_windows,
            cfg_model       = dict(cfg.model),
        ).to(device)
        sd = torch.load(cfg.ssl_ckpt, map_location="cpu")
        ms = model.state_dict()
        ms.update({k:v for k,v in sd.items() if k in ms and v.shape==ms[k].shape})
        model.load_state_dict(ms)

        # freezing strategy
        for name, param in model.named_parameters():
            if any(x in name for x in ["attn","ffn","fusion","agg_current","pos_emb"]):
                param.requires_grad = True
            else:
                param.requires_grad = False

        criterion = torch.nn.SmoothL1Loss()
        optimizer = torch.optim.Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=cfg.train.lr
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            patience=sched_cfg.patience,
            factor=sched_cfg.factor,
            min_lr=sched_cfg.min_lr,
            verbose=True
        )

        best_val, no_imp = float("inf"), 0
        stats, val_errs = [], []

        for ep in range(1, cfg.train.num_epochs+1):
            if no_imp >= cfg.train.patience:
                break

            # — TRAIN —
            model.train()
            tr_loss = 0.0
            for bp,st,cbp,cst,bpt,stt,*_ in tr_loader:
                bp,st,cbp,cst,bpt,stt = [x.to(device) for x in (bp,st,cbp,cst,bpt,stt)]
                optimizer.zero_grad()
                p_bp,p_st = model(bp,st,cbp,cst)
                loss = α*criterion(p_bp,bpt) + β*criterion(p_st,stt)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(),1.0)
                optimizer.step()
                tr_loss += loss.item()
            tr_loss /= len(tr_loader)

            # — VALIDATE —
            model.eval()
            vl_loss=eb=es=0.0
            n_samples=0
            with torch.no_grad():
                for bp,st,cbp,cst,bpt,stt,uids,_,_,dts in vl_loader:
                    bp,st,cbp,cst,bpt,stt = [x.to(device) for x in (bp,st,cbp,cst,bpt,stt)]
                    p_bp,p_st = model(bp,st,cbp,cst)
                    vl_loss += α*criterion(p_bp,bpt).item() + β*criterion(p_st,stt).item()
                    pb, sb = p_bp.cpu().numpy(), p_st.cpu().numpy()
                    tb, ts = bpt.cpu().numpy(), stt.cpu().numpy()
                    P, W = cfg.window.predict_windows, cfg.window.size
                    for i, uid2 in enumerate(uids):
                        pr_bp = pb[i].reshape(-1); pr_st = sb[i].reshape(-1)
                        gt_bp = tb[i].reshape(-1); gt_st = ts[i].reshape(-1)
                        pu, su = U.inverse_transform(uid2, pr_bp, pr_st)
                        gu, tu = U.inverse_transform(uid2, gt_bp, gt_st)
                        pu, su, gu, tu = (x.reshape(P,W) for x in (pu,su,gu,tu))
                        e1 = np.mean(np.abs(pu-gu)); e2 = np.mean(np.abs(su-tu))
                        eb += e1; es += e2; n_samples += 1
                        val_errs.append({
                            "user_id":uid2, "epoch":ep,
                            "bpm_error":e1, "steps_error":e2,
                            "bpm_pred":pu,  "steps_pred":su,
                            "bpm_true":gu,  "steps_true":tu,
                            "datetime":dts[i]
                        })
            vl_loss /= len(vl_loader)
            sb1, ss1 = eb/max(n_samples,1), es/max(n_samples,1)
            stats.append({"epoch":ep, "train":tr_loss, "val":vl_loss, "bpm":sb1, "steps":ss1})
            print(f"[U{uid}] ep{ep:02d} tr{tr_loss:.4f} vl{vl_loss:.4f} bpm{sb1:.2f} steps{ss1:.2f}")

            scheduler.step(vl_loss)
            if vl_loss < best_val:
                best_val, no_imp = vl_loss, 0
                torch.save(model.state_dict(), os.path.join(user_dir, "personalized_ssl.pt"))
                print(" ↳ saved BEST")
            else:
                no_imp += 1

        # Save per-user CSVs
        pd.DataFrame(stats).to_csv(os.path.join(user_dir,"stats.csv"), index=False)
        pd.DataFrame(val_errs).to_csv(os.path.join(user_dir,"validation_errors.csv"), index=False)

        # Plots
        df_s = pd.DataFrame(stats)
        plt.figure(figsize=(10,6))
        plt.plot(df_s.epoch, df_s.train, label="Train Loss")
        plt.plot(df_s.epoch, df_s.val,   label="Val Loss")
        plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.legend()
        plt.savefig(os.path.join(user_dir,"loss.png")); plt.close()
        for col,title,fname in [("bpm","BPM MAE","bpm_error.png"),("steps","Steps MAE","steps_error.png")]:
            plt.figure(figsize=(10,6))
            plt.plot(df_s.epoch, df_s[col])
            plt.xlabel("Epoch"); plt.ylabel("MAE"); plt.title(f"User {uid} {title}")
            plt.savefig(os.path.join(user_dir,fname)); plt.close()

        # Best/worst analysis
        if val_errs:
            df_v = pd.DataFrame(val_errs)
            df_v["total_error"] = df_v.bpm_error + df_v.steps_error
            last   = df_v[df_v.epoch==df_s.epoch.max()]
            best5  = last.nsmallest(5, "total_error")
            worst5 = last.nlargest(5,  "total_error")
            ana_dir = os.path.join(user_dir,"analysis"); os.makedirs(ana_dir, exist_ok=True)
            for _,r in best5.iterrows():
                U.plot_prediction_user(uid, [r.to_dict()], "best_finetune", cfg.window.predict_windows, cfg.window.size, ana_dir)
            for _,r in worst5.iterrows():
                U.plot_prediction_user(uid, [r.to_dict()], "worst_finetune", cfg.window.predict_windows, cfg.window.size, ana_dir)

        # Append global summary
        if stats:
            f = stats[-1]
            global_stats.append({
                "user_id":          uid,
                "final_bpm_error":  f["bpm"],
                "final_steps_error":f["steps"],
            })

    # Save global summary CSV
    summary_path = os.path.join(train_root, "personalized_finetune_summary.csv")
    if global_stats:
        pd.DataFrame(global_stats).to_csv(summary_path, index=False)
        print(f"Saved summary ➞ {summary_path}")
        # 7) Sync README table
        if cfg.autoupdate_readme:
            U.sync_readme_table(summary_path, tag="FINETUNE")

if __name__ == "__main__":
    main()
