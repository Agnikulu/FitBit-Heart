"""
Per‑user fine‑tune **full original functionality**:
 • detailed logs
 • stats CSV
 • loss / error PNGs
 • best / worst prediction plots
 • checkpoint save
"""
import os, random, warnings, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from src.config import load_config
import src.utils as U
from src.models import PersonalizedForecastingModel, partially_unfreeze_backbone

warnings.filterwarnings("ignore")

def main():
    cfg = load_config()
    torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # ---- Load & preprocess identical to original --------------------------
    df_list=[]
    for fp in os.listdir(cfg.data.biosignal_dir):
        if fp.endswith(".csv"):
            df = pd.read_csv(os.path.join(cfg.data.biosignal_dir,fp),parse_dates=["time"])
            if "participant_id" not in df.columns:
                base=os.path.splitext(fp)[0]
                pid=int(base[1:] if base.lower().startswith("p") else base)
                df["participant_id"]=pid
            df_list.append(df)
    if not df_list: raise FileNotFoundError("No CSVs found in biosignal_dir")
    df_raw=pd.concat(df_list,ignore_index=True)
    df_raw.loc[df_raw["data_type"]=="hr","data_type"]="heart_rate"
    df_raw=df_raw[df_raw["data_type"].isin(["heart_rate","steps"])]
    df_raw.drop_duplicates(["participant_id","time","data_type"],inplace=True)

    df_piv=df_raw.pivot(index=["participant_id","time"],
                        columns="data_type",values="value").reset_index()
    df_piv.rename(columns={"participant_id":"id","heart_rate":"bpm"},inplace=True)
    df_piv[["bpm","steps"]]=df_piv[["bpm","steps"]].apply(pd.to_numeric,errors="coerce")
    df_piv.dropna(subset=["bpm","steps"],inplace=True)
    df_piv["datetime"]=pd.to_datetime(df_piv["time"])
    df_piv["date"]=df_piv["datetime"].dt.date; df_piv["hour"]=df_piv["datetime"].dt.hour

    df_sensor=df_piv[["id","datetime","date","hour","bpm","steps"]].copy()
    df_sensor.sort_values(["id","datetime"],inplace=True)

    df_sensor["date"]=pd.to_datetime(df_sensor["date"])
    dfh=(df_sensor.groupby(["id","date","hour"],as_index=False)
         .agg({"bpm":"mean","steps":"sum"}))
    dfh["datetime"]=dfh.apply(
        lambda r: pd.to_datetime(r["date"])+pd.to_timedelta(r["hour"],unit="H"),axis=1)
    dfh=dfh.sort_values(["id","datetime"]).reset_index(drop=True)

    dfh=dfh.groupby("id").apply(U.scale_per_user).reset_index(drop=True)

    samples=U.create_forecasting_samples(
        dfh,
        window_size=cfg.window.size,
        input_windows=cfg.window.input_windows,
        predict_windows=cfg.window.predict_windows)
    print(f"Created {len(samples)} total samples from {dfh['id'].nunique()} users.")

    ssl_ckpt=cfg.ssl_ckpt
    if not os.path.isfile(ssl_ckpt):
        raise FileNotFoundError(f"Pretrained SSL checkpoint not found → {ssl_ckpt}")

    res_root=os.path.join(cfg.results_root,"train"); os.makedirs(res_root,exist_ok=True)

    for uid in sorted(dfh["id"].unique()):
        usr_dir=os.path.join(res_root,f"user_{uid}"); os.makedirs(usr_dir,exist_ok=True)
        usamp=[s for s in samples if s["user_id"]==uid]
        if len(usamp)<10: print(f"[User {uid}] Skipping – <10 samples"); continue
        cut=int(.8*len(usamp))
        tr_loader=DataLoader(U.ForecastDataset(usamp[:cut]),
                             batch_size=cfg.train.batch_size,shuffle=True,
                             collate_fn=U.forecasting_collate_fn)
        vl_loader=DataLoader(U.ForecastDataset(usamp[cut:]),
                             batch_size=cfg.train.batch_size,shuffle=False,
                             collate_fn=U.forecasting_collate_fn)

        u_df=dfh[dfh["id"]==uid]
        alpha=1/max(u_df["bpm"].abs().mean(),1e-6)
        beta =1/max(u_df["steps"].abs().mean(),1e-6)
        alpha,beta=alpha/(alpha+beta), beta/(alpha+beta)

        model=PersonalizedForecastingModel(window_size=cfg.window.size).to(device)
        model.load_state_dict(torch.load(ssl_ckpt,map_location=device),strict=False)
        partially_unfreeze_backbone(model,cfg.train.unfreeze_ratio)

        crit=torch.nn.SmoothL1Loss()
        opt=torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),
                             lr=cfg.train.lr)
        sch=torch.optim.lr_scheduler.StepLR(opt,
            step_size=cfg.train.scheduler.step_size,gamma=cfg.train.scheduler.gamma)

        best,no_imp=float("inf"),0
        stats, val_errs=[],[]
        for ep in range(1,cfg.train.num_epochs+1):
            if no_imp>=cfg.train.patience: break
            # TRAIN
            model.train(); tr=0.
            for bp_in,st_in,cbp,cst,bp_t,st_t,*_ in tr_loader:
                bp_in,st_in,cbp,cst,bp_t,st_t=(x.to(device) for x in
                                               (bp_in,st_in,cbp,cst,bp_t,st_t))
                opt.zero_grad()
                p_bp,p_st=model(bp_in,st_in,cbp,cst)
                loss=alpha*crit(p_bp,bp_t)+beta*crit(p_st,st_t)
                loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                opt.step(); tr+=loss.item()
            tr/=len(tr_loader); sch.step()

            # VAL
            model.eval(); vl=bp_mae=st_mae=0.
            with torch.no_grad():
                for bp_in,st_in,cbp,cst,bp_t,st_t,uids,bp_orig,st_orig,dt in vl_loader:
                    bp_in,st_in,cbp,cst,bp_t,st_t=(x.to(device) for x in
                                                   (bp_in,st_in,cbp,cst,bp_t,st_t))
                    p_bp,p_st=model(bp_in,st_in,cbp,cst)
                    vl+=alpha*crit(p_bp,bp_t).item()+beta*crit(p_st,st_t).item()
                    p_bp_np,p_st_np=p_bp.cpu().numpy(),p_st.cpu().numpy()
                    bp_t_np,st_t_np=bp_t.cpu().numpy(),st_t.cpu().numpy()
                    for i,uid_iter in enumerate(uids):
                        pr_bp=p_bp_np[i].flatten(); pr_st=p_st_np[i].flatten()
                        gt_bp=bp_t_np[i].flatten(); gt_st=st_t_np[i].flatten()
                        pr_bp_un,pr_st_un=U.inverse_transform(uid_iter,pr_bp,pr_st)
                        gt_bp_un,gt_st_un=U.inverse_transform(uid_iter,gt_bp,gt_st)
                        e_bp=np.mean(np.abs(pr_bp_un-gt_bp_un))
                        e_st=np.mean(np.abs(pr_st_un-gt_st_un))
                        bp_mae+=e_bp; st_mae+=e_st
                        val_errs.append({
                            'user_id':uid_iter,'epoch':ep,
                            'bpm_error':e_bp,'steps_error':e_st,
                            'bpm_pred':pr_bp_un.reshape(cfg.window.predict_windows,
                                                        cfg.window.size),
                            'steps_pred':pr_st_un.reshape(cfg.window.predict_windows,
                                                          cfg.window.size),
                            'bpm_true':gt_bp_un.reshape(cfg.window.predict_windows,
                                                        cfg.window.size),
                            'steps_true':gt_st_un.reshape(cfg.window.predict_windows,
                                                          cfg.window.size),
                            'datetime':dt[i]})
            vl/=len(vl_loader)
            bp_mae/=len(vl_loader.dataset); st_mae/=len(vl_loader.dataset)

            print(f"[User {uid}] Epoch={ep}/{cfg.train.num_epochs}, "
                  f"TrainLoss={tr:.4f}, ValLoss={vl:.4f}, "
                  f"BPM_Err={bp_mae:.2f}, Steps_Err={st_mae:.2f}")

            stats.append(dict(epoch=ep,train_loss=tr,val_loss=vl,
                              avg_bpm_error=bp_mae,avg_steps_error=st_mae))
            if vl<best:
                best,no_imp=vl,0
                torch.save(model.state_dict(),os.path.join(usr_dir,"personalized_ssl.pt"))
                print("  ↳ saved new BEST")
            else:
                no_imp+=1

        # Save CSV
        pd.DataFrame(stats).to_csv(os.path.join(usr_dir,"stats.csv"),index=False)
        val_df=pd.DataFrame(val_errs)
        val_df.to_csv(os.path.join(usr_dir,"validation_errors.csv"),index=False)

        # Loss & error plots
        sd=pd.DataFrame(stats)
        plt.figure(figsize=(10,6))
        plt.plot(sd["epoch"],sd["train_loss"],label="Train")
        plt.plot(sd["epoch"],sd["val_loss"],label="Val")
        plt.legend(); plt.xlabel("Epoch"); plt.ylabel("Loss")
        plt.title(f"User {uid} Train/Val Loss")
        plt.tight_layout(); plt.savefig(os.path.join(usr_dir,"loss.png")); plt.close()
        for col,title,fname in [("avg_bpm_error","BPM MAE","bpm_error.png"),
                                ("avg_steps_error","Steps MAE","steps_error.png")]:
            plt.figure(figsize=(10,6))
            plt.plot(sd["epoch"],sd[col]); plt.title(f"User {uid} {title}")
            plt.xlabel("Epoch"); plt.ylabel("MAE")
            plt.tight_layout(); plt.savefig(os.path.join(usr_dir,fname)); plt.close()

        # Best / worst window plots
        if not val_df.empty:
            last_ep=val_df["epoch"].max()
            last=val_df[val_df["epoch"]==last_ep].copy()
            last["tot"]=last["bpm_error"]+last["steps_error"]
            last_sorted=last.sort_values("tot")
            best_s=last_sorted.head(5); worst_s=last_sorted.tail(5)
            ana=os.path.join(usr_dir,"analysis"); os.makedirs(ana,exist_ok=True)
            for i,row in best_s.iterrows():
                U.plot_prediction_user(uid,[row.to_dict()],"best_finetune",
                                       cfg.window.predict_windows,cfg.window.size,ana)
            for i,row in worst_s.iterrows():
                U.plot_prediction_user(uid,[row.to_dict()],"worst_finetune",
                                       cfg.window.predict_windows,cfg.window.size,ana)

if __name__=="__main__":
    main()
