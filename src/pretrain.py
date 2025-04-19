import os, random, warnings, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
from src.config import load_config
import src.utils as U
from src.models import SSLForecastingModel

warnings.filterwarnings("ignore")

def main() -> None:
    cfg = load_config()
    torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # 1 . Load daily LifeSnaps CSV ------------------------------------------
    df = pd.read_csv(cfg.data.lifesnaps_csv, parse_dates=["date"])
    df["datetime"] = pd.to_datetime(df["date"]) + pd.to_timedelta(df["hour"], unit="H")
    df = df[["id","datetime","bpm","steps"]].dropna().sort_values(["id","datetime"])
    df = df.groupby("id").apply(U.scale_per_user).reset_index(drop=True)

    # 2 . Create forecasting samples ----------------------------------------
    samples = U.create_forecasting_samples(
        df,
        window_size     = cfg.window.size,
        input_windows   = cfg.window.input_windows,
        predict_windows = cfg.window.predict_windows
    )
    train_ids, val_ids = train_test_split(df["id"].unique(), test_size=0.2, random_state=42)
    def idx(lst):
        d={}; [d.setdefault(s["user_id"],[]).append(i) for i,s in enumerate(lst)]; return d
    train_data = [s for s in samples if s["user_id"] in train_ids]
    val_data   = [s for s in samples if s["user_id"] in val_ids]

    train_loader = DataLoader(
        U.ForecastDataset(train_data),
        batch_sampler=U.PerUserBatchSampler(idx(train_data), cfg.pretrain.batch_size),
        collate_fn=U.forecasting_collate_fn)
    val_loader   = DataLoader(
        U.ForecastDataset(val_data),
        batch_sampler=U.PerUserBatchSampler(idx(val_data), cfg.pretrain.batch_size),
        collate_fn=U.forecasting_collate_fn)

    # 3 . Weighted loss ------------------------------------------------------
    avg_bpm   = df[df["id"].isin(train_ids)]["bpm"].abs().mean()
    avg_steps = df[df["id"].isin(train_ids)]["steps"].abs().mean()
    alpha = 1/avg_bpm; beta = 1/avg_steps
    alpha, beta = alpha/(alpha+beta), beta/(alpha+beta)

    # 4 . Model, optimiser, scheduler ---------------------------------------
    model = SSLForecastingModel(window_size=cfg.window.size).to(device)
    criterion = torch.nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(model.parameters(), lr=cfg.pretrain.lr, weight_decay=1e-5)
    scheduler = torch.optim.lr_scheduler.StepLR(
        optimizer, step_size=cfg.pretrain.scheduler.step_size,
        gamma=cfg.pretrain.scheduler.gamma)

    # 5 . Training loop ------------------------------------------------------
    best_val, epochs_no_improve = float("inf"), 0
    stats, validation_errors = [], []
    for epoch in range(1, cfg.pretrain.num_epochs+1):
        if epochs_no_improve >= cfg.pretrain.patience: break

        # ---- TRAIN --------------------------------------------------------
        model.train(); train_loss = 0.0
        for bp_in, st_in, cbp, cst, bp_t, st_t, *_ in train_loader:
            bp_in, st_in, cbp, cst, bp_t, st_t = \
                (x.to(device) for x in (bp_in, st_in, cbp, cst, bp_t, st_t))
            optimizer.zero_grad()
            pred_bp, pred_st = model(bp_in, st_in, cbp, cst)
            loss = alpha*criterion(pred_bp, bp_t) + beta*criterion(pred_st, st_t)
            loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step(); train_loss += loss.item()
        scheduler.step()
        train_loss /= len(train_loader)

        # ---- VALIDATION ---------------------------------------------------
        model.eval(); val_loss = tot_bp_err = tot_st_err = 0.0
        val_err_epoch = []
        with torch.no_grad():
            for bp_in, st_in, cbp, cst, bp_t, st_t, uids, bpm_orig, steps_orig, dtimes in val_loader:
                bp_in, st_in, cbp, cst, bp_t, st_t = \
                    (x.to(device) for x in (bp_in, st_in, cbp, cst, bp_t, st_t))
                p_bp_scl, p_st_scl = model(bp_in, st_in, cbp, cst)

                loss_bpm   = criterion(p_bp_scl, bp_t)
                loss_steps = criterion(p_st_scl, st_t)
                val_loss  += alpha*loss_bpm.item() + beta*loss_steps.item()

                p_bp_np, p_st_np = p_bp_scl.cpu().numpy(), p_st_scl.cpu().numpy()
                bp_t_np, st_t_np = bp_t.cpu().numpy(),   st_t.cpu().numpy()
                for i, uid in enumerate(uids):
                    pr_bp = p_bp_np[i].flatten(); pr_st = p_st_np[i].flatten()
                    gt_bp = bp_t_np[i].flatten(); gt_st = st_t_np[i].flatten()
                    pr_bp_un, pr_st_un = U.inverse_transform(uid, pr_bp, pr_st)
                    gt_bp_un, gt_st_un = U.inverse_transform(uid, gt_bp, gt_st)
                    bp_err   = np.mean(np.abs(pr_bp_un - gt_bp_un))
                    steps_err= np.mean(np.abs(pr_st_un - gt_st_un))
                    tot_bp_err += bp_err; tot_st_err += steps_err
                    val_err_epoch.append({
                        'user_id': uid,
                        'epoch': epoch,
                        'bpm_error': bp_err,
                        'steps_error': steps_err,
                        'bpm_pred':   p_bp_np[i],
                        'steps_pred': p_st_np[i],
                        'bpm_true':   bp_t_np[i],
                        'steps_true': st_t_np[i],
                        'datetime': dtimes[i]
                    })
        val_loss /= len(val_loader)
        bpm_mae   = tot_bp_err / len(val_data)
        steps_mae = tot_st_err / len(val_data)

        stats.append(dict(epoch=epoch, train=train_loss, val=val_loss,
                          bpm=bpm_mae, steps=steps_mae))
        validation_errors.extend(val_err_epoch)

        print(f"ep{epoch:03d} tr{train_loss:.4f} vl{val_loss:.4f} "
              f"bpmMAE{bpm_mae:.2f} stepsMAE{steps_mae:.2f}")

        # early‑stop & checkpoint
        if val_loss < best_val:
            best_val, epochs_no_improve = val_loss, 0
            os.makedirs(os.path.dirname(cfg.ssl_ckpt), exist_ok=True)
            torch.save(model.state_dict(), cfg.ssl_ckpt)
            print("  ↳ saved backbone")
        else:
            epochs_no_improve += 1

    # 6 . Save stats + plots --------------------------------------------------
    res_dir = os.path.join(cfg.results_root, "pretrain"); os.makedirs(res_dir, exist_ok=True)
    stats_df = pd.DataFrame(stats)
    stats_df.to_csv(os.path.join(res_dir, "stats_pretrain.csv"), index=False)
    # Loss plot
    plt.figure(figsize=(10,6))
    plt.plot(stats_df["epoch"], stats_df["train"], label="Train")
    plt.plot(stats_df["epoch"], stats_df["val"],   label="Val")
    plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss")
    plt.title("SSL Pretrain Loss"); plt.tight_layout()
    plt.savefig(os.path.join(res_dir, "ssl_pretrain_loss.png")); plt.close()
    # Error plots
    plt.figure(figsize=(10,6)); plt.plot(stats_df["epoch"],stats_df["bpm"]); plt.title("BPM MAE")
    plt.savefig(os.path.join(res_dir,"ssl_pretrain_bpm_error.png")); plt.close()
    plt.figure(figsize=(10,6)); plt.plot(stats_df["epoch"],stats_df["steps"]); plt.title("Steps MAE")
    plt.savefig(os.path.join(res_dir,"ssl_pretrain_steps_error.png")); plt.close()

    # 7 . Best / worst users analysis ---------------------------------------
    val_df = pd.DataFrame(validation_errors)
    val_df["total_err"] = val_df["bpm_error"] + val_df["steps_error"]
    user_err = val_df.groupby("user_id")["total_err"].mean().reset_index()
    best_u = user_err.nsmallest(5, "total_err"); worst_u = user_err.nlargest(5, "total_err")

    ana_dir = os.path.join(res_dir, "analysis_pretrain"); os.makedirs(ana_dir, exist_ok=True)
    for _, row in best_u.iterrows():
        uid = row["user_id"]
        samp = val_df[val_df["user_id"]==uid].to_dict("records")[:2]
        U.plot_prediction_user(uid,samp,"best_pretrain",
                               cfg.window.predict_windows,cfg.window.size,ana_dir)
    for _, row in worst_u.iterrows():
        uid = row["user_id"]
        samp = val_df[val_df["user_id"]==uid].to_dict("records")[:2]
        U.plot_prediction_user(uid,samp,"worst_pretrain",
                               cfg.window.predict_windows,cfg.window.size,ana_dir)

if __name__ == "__main__":
    main()
