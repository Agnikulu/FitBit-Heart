# src/pretrain.py

import os
import random
import warnings

import numpy as np
import pandas as pd
import torch
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from src.config import load_config
import src.utils as U
from src.models import SSLForecastingModel

warnings.filterwarnings("ignore")


def main():
    cfg = load_config()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)

    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # 1) Load & preprocess LifeSnaps data
    df = pd.read_csv(cfg.data.lifesnaps_csv, parse_dates=["date"])
    df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="H")
    df = (
        df[["id", "datetime", "bpm", "steps"]]
        .dropna()
        .sort_values(["id", "datetime"])
    )
    df = df.groupby("id").apply(U.scale_per_user).reset_index(drop=True)

    # 2) Create forecasting samples & DataLoaders
    samples = U.create_forecasting_samples(
        df,
        window_size     = cfg.window.size,
        input_windows   = cfg.window.input_windows,
        predict_windows = cfg.window.predict_windows
    )
    train_ids, val_ids = train_test_split(
        df["id"].unique(),
        test_size=0.2,
        random_state=cfg.seed
    )

    def idx_map(lst):
        d = {}
        for i, s in enumerate(lst):
            d.setdefault(s["user_id"], []).append(i)
        return d

    train_data = [s for s in samples if s["user_id"] in train_ids]
    val_data   = [s for s in samples if s["user_id"]   in val_ids]

    train_loader = DataLoader(
        U.ForecastDataset(train_data),
        batch_sampler=U.PerUserBatchSampler(idx_map(train_data), cfg.pretrain.batch_size),
        collate_fn=U.forecasting_collate_fn
    )
    val_loader = DataLoader(
        U.ForecastDataset(val_data),
        batch_sampler=U.PerUserBatchSampler(idx_map(val_data), cfg.pretrain.batch_size),
        collate_fn=U.forecasting_collate_fn
    )

    # 3) Weighted loss coefficients
    avg_bpm   = df[df["id"].isin(train_ids)]["bpm"].abs().mean()
    avg_steps = df[df["id"].isin(train_ids)]["steps"].abs().mean()
    alpha, beta = 1/avg_bpm, 1/avg_steps
    alpha, beta = alpha/(alpha+beta), beta/(alpha+beta)

    # 4) Model + optimizer + ReduceLROnPlateau scheduler
    model = SSLForecastingModel(
        window_size     = cfg.window.size,
        predict_windows = cfg.window.predict_windows,
        attn_heads      = cfg.model.attn_heads,
        dropout         = cfg.model.dropout
    ).to(device)

    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=cfg.pretrain.lr,
        weight_decay=cfg.pretrain.weight_decay
    )

    sched_cfg = cfg.pretrain.scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode='min',
        patience=sched_cfg.patience,
        factor=sched_cfg.factor,
        min_lr=sched_cfg.min_lr,
        verbose=True
    )

    # 5) Training loop
    best_val, no_imp = float("inf"), 0
    stats, val_errors = [], []

    P = cfg.window.predict_windows
    W = cfg.window.size

    for ep in range(1, cfg.pretrain.num_epochs + 1):
        if no_imp >= cfg.pretrain.patience:
            break

        # — train —
        model.train()
        tr_loss = 0.0
        for bp_in, st_in, cbp, cst, bp_t, st_t, *_ in train_loader:
            bp_in, st_in, cbp, cst, bp_t, st_t = [
                x.to(device) for x in (bp_in, st_in, cbp, cst, bp_t, st_t)
            ]
            optimizer.zero_grad()
            p_bp, p_st = model(bp_in, st_in, cbp, cst)
            loss = alpha * criterion(p_bp, bp_t) + beta * criterion(p_st, st_t)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            tr_loss += loss.item()
        tr_loss /= len(train_loader)

        # — validate —
        model.eval()
        vl_loss = tot_bp = tot_st = 0.0

        with torch.no_grad():
            for bp_in, st_in, cbp, cst, bp_t, st_t, uids, _, _, dtimes in val_loader:
                bp_in, st_in, cbp, cst, bp_t, st_t = [
                    x.to(device) for x in (bp_in, st_in, cbp, cst, bp_t, st_t)
                ]
                p_bp, p_st = model(bp_in, st_in, cbp, cst)

                vl_loss += alpha * criterion(p_bp, bp_t).item() \
                         +  beta * criterion(p_st, st_t).item()

                p_bp_np, p_st_np = p_bp.cpu().numpy(), p_st.cpu().numpy()
                bp_t_np, st_t_np = bp_t.cpu().numpy(), st_t.cpu().numpy()

                # record per‐sample errors + original‐scale values
                for i, uid in enumerate(uids):
                    pr_bp = p_bp_np[i].reshape(P * W)
                    pr_st = p_st_np[i].reshape(P * W)
                    gt_bp = bp_t_np[i].reshape(P * W)
                    gt_st = st_t_np[i].reshape(P * W)

                    # unscale
                    pu_bp, pu_st = U.inverse_transform(uid, pr_bp, pr_st)
                    gu_bp, gu_st = U.inverse_transform(uid, gt_bp, gt_st)

                    # reshape [P,W]
                    pu_bp = pu_bp.reshape(P, W)
                    pu_st = pu_st.reshape(P, W)
                    gu_bp = gu_bp.reshape(P, W)
                    gu_st = gu_st.reshape(P, W)

                    e_bp = np.mean(np.abs(pu_bp - gu_bp))
                    e_st = np.mean(np.abs(pu_st - gu_st))
                    tot_bp += e_bp; tot_st += e_st

                    val_errors.append({
                        "user_id":     uid,
                        "epoch":       ep,
                        "bpm_error":   e_bp,
                        "steps_error": e_st,
                        "bpm_pred":    pu_bp,
                        "steps_pred":  pu_st,
                        "bpm_true":    gu_bp,
                        "steps_true":  gu_st,
                        "datetime":    dtimes[i]
                    })

        vl_loss /= len(val_loader)
        bpm_mae = tot_bp / len(val_data)
        stp_mae = tot_st / len(val_data)

        stats.append({
            "epoch": ep,
            "train": tr_loss,
            "val":   vl_loss,
            "bpm":   bpm_mae,
            "steps": stp_mae
        })
        print(f"ep{ep:03d} tr{tr_loss:.4f} vl{vl_loss:.4f} "
              f"{bpm_mae:.2f}bpm {stp_mae:.2f}steps")

        # step scheduler & checkpoint
        scheduler.step(vl_loss)
        if vl_loss < best_val:
            best_val, no_imp = vl_loss, 0
            os.makedirs(os.path.dirname(cfg.ssl_ckpt), exist_ok=True)
            torch.save(model.state_dict(), cfg.ssl_ckpt)
            print("  ↳ saved backbone")
        else:
            no_imp += 1

    # 6) Save stats + plots
    res_dir = os.path.join(cfg.results_root, "pretrain")
    os.makedirs(res_dir, exist_ok=True)
    df_stats = pd.DataFrame(stats)
    df_stats.to_csv(os.path.join(res_dir, "stats_pretrain.csv"), index=False)

    plt.figure(figsize=(10,6))
    plt.plot(df_stats.epoch, df_stats.train, label="Train")
    plt.plot(df_stats.epoch, df_stats.val,   label="Val")
    plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("SSL Pretrain Loss")
    plt.legend(); plt.tight_layout()
    plt.savefig(os.path.join(res_dir, "ssl_pretrain_loss.png"))
    plt.close()

    plt.figure(figsize=(10,6))
    plt.plot(df_stats.epoch, df_stats.bpm, label="BPM MAE")
    plt.xlabel("Epoch"); plt.ylabel("BPM MAE"); plt.tight_layout()
    plt.savefig(os.path.join(res_dir, "ssl_pretrain_bpm_error.png"))
    plt.close()

    plt.figure(figsize=(10,6))
    plt.plot(df_stats.epoch, df_stats.steps, label="Steps MAE")
    plt.xlabel("Epoch"); plt.ylabel("Steps MAE"); plt.tight_layout()
    plt.savefig(os.path.join(res_dir, "ssl_pretrain_steps_error.png"))
    plt.close()

    # 7) Best/worst analysis
    df_val = pd.DataFrame(val_errors)
    df_val["total_error"] = df_val.bpm_error + df_val.steps_error
    usr_err = df_val.groupby("user_id").total_error.mean().reset_index()
    best_u  = usr_err.nsmallest(5, "total_error")
    worst_u = usr_err.nlargest(5, "total_error")

    ana_dir = os.path.join(res_dir, "analysis_pretrain")
    os.makedirs(ana_dir, exist_ok=True)
    for _, r in best_u.iterrows():
        recs = df_val[df_val.user_id == r.user_id].to_dict("records")[:2]
        U.plot_prediction_user(
            r.user_id, recs, "best_pretrain",
            cfg.window.predict_windows, cfg.window.size, ana_dir
        )
    for _, r in worst_u.iterrows():
        recs = df_val[df_val.user_id == r.user_id].to_dict("records")[:2]
        U.plot_prediction_user(
            r.user_id, recs, "worst_pretrain",
            cfg.window.predict_windows, cfg.window.size, ana_dir
        )


if __name__ == "__main__":
    main()
