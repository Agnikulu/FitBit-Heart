# src/pretrain.py – keeps ALL original logging, stats, plots, analysis
from __future__ import annotations
import os, random, warnings, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from src.config import load_config
import src.utils as U
from src.models import SSLForecastingModel

warnings.filterwarnings("ignore")

# ------------------------------------------------------------------

def main():
    cfg=load_config(); torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
    dev=torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    # 1) Load LifeSnaps & scale
    df=pd.read_csv(cfg.data.lifesnaps_csv,parse_dates=["date"])
    df["datetime"]=pd.to_datetime(df["date"])+pd.to_timedelta(df["hour"],unit="H")
    df=df[["id","datetime","bpm","steps"]].dropna().sort_values(["id","datetime"])
    df=df.groupby("id").apply(U.scale_per_user).reset_index(drop=True)

    # 2) create samples
    W,P=cfg.window.size,cfg.window.predict_windows
    sam=U.create_forecasting_samples(df,window_size=W,input_windows=cfg.window.input_windows,predict_windows=P)

    tr_ids,vl_ids=train_test_split(df["id"].unique(),test_size=0.2,random_state=cfg.seed)
    tr=[s for s in sam if s["user_id"] in tr_ids]; vl=[s for s in sam if s["user_id"] in vl_ids]
    idx=dict; idx_tr={}; idx_vl={}
    for i,s in enumerate(tr): idx_tr.setdefault(s["user_id"],[]).append(i)
    for i,s in enumerate(vl): idx_vl.setdefault(s["user_id"],[]).append(i)

    tr_loader=DataLoader(U.ForecastDataset(tr),batch_sampler=U.PerUserBatchSampler(idx_tr,cfg.pretrain.batch_size),collate_fn=U.forecasting_collate_fn)
    vl_loader=DataLoader(U.ForecastDataset(vl),batch_sampler=U.PerUserBatchSampler(idx_vl,cfg.pretrain.batch_size),collate_fn=U.forecasting_collate_fn)

    # 3) Weighted α,β based on train originals
    avg_bpm=df[df["id"].isin(tr_ids)]["bpm"].abs().mean(); avg_steps=df[df["id"].isin(tr_ids)]["steps"].abs().mean()
    α,β=1/avg_bpm,1/avg_steps; α,β=α/(α+β),β/(α+β)

    # 4) Model + optim
    model=SSLForecastingModel(window_size=W,predict_windows=P,cfg_model=dict(cfg.model)).to(dev)
    crit=torch.nn.SmoothL1Loss(); opt=torch.optim.Adam(model.parameters(),lr=cfg.pretrain.lr,weight_decay=cfg.pretrain.weight_decay)
    sch=torch.optim.lr_scheduler.ReduceLROnPlateau(opt,mode="min",patience=cfg.pretrain.scheduler.patience,factor=cfg.pretrain.scheduler.factor,min_lr=cfg.pretrain.scheduler.min_lr,verbose=True)

    res_dir=os.path.join(cfg.results_root,"pretrain"); os.makedirs(res_dir,exist_ok=True)
    stats,val_errs=[],[]; best,no_imp=float("inf"),0

    for ep in range(1,cfg.pretrain.num_epochs+1):
        if no_imp>=cfg.pretrain.patience: break
        # ---- train ----
        model.train(); trl=0
        for b in tr_loader:
            bpm_i,st_i,cbp,cst,bp_t,st_t,*_=[x.to(dev) if torch.is_tensor(x) else x for x in b]
            opt.zero_grad(); p_bp,p_st=model(bpm_i,st_i,cbp,cst)
            loss=α*crit(p_bp,bp_t)+β*crit(p_st,st_t); loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1); opt.step(); trl+=loss.item()
        trl/=len(tr_loader)

        # ---- val ----
        model.eval(); vll=tot_bp=tot_st=0
        with torch.no_grad():
            for b in vl_loader:
                bpm_i,st_i,cbp,cst,bp_t,st_t,uids,bp_o,st_o,dt=[x.to(dev) if torch.is_tensor(x) else x for x in b]
                p_bp,p_st=model(bpm_i,st_i,cbp,cst)
                vll+=α*crit(p_bp,bp_t).item()+β*crit(p_st,st_t).item()
                p_bp_np,p_st_np,bp_t_np,st_t_np=[x.cpu().numpy() for x in (p_bp,p_st,bp_t,st_t)]
                for i,uid in enumerate(uids):
                    pr_bp=p_bp_np[i].reshape(P*W); pr_st=p_st_np[i].reshape(P*W); gt_bp=bp_t_np[i].reshape(P*W); gt_st=st_t_np[i].reshape(P*W)
                    pu_bp,pu_st=U.inverse_transform(uid,pr_bp,pr_st)
                    gu_bp,gu_st=U.inverse_transform(uid,gt_bp,gt_st)
                    pu_bp,pu_st,gu_bp,gu_st=[x.reshape(P,W) for x in (pu_bp,pu_st,gu_bp,gu_st)]
                    e_bp=np.mean(np.abs(pu_bp-gu_bp)); e_st=np.mean(np.abs(pu_st-gu_st)); tot_bp+=e_bp; tot_st+=e_st
                    val_errs.append(dict(user_id=uid,epoch=ep,bpm_error=e_bp,steps_error=e_st,bpm_pred=pu_bp,steps_pred=pu_st,bpm_true=gu_bp,steps_true=gu_st,datetime=dt[i]))
        vll/=len(vl_loader); bpm_mae=tot_bp/len(vl); stp_mae=tot_st/len(vl)
        stats.append(dict(epoch=ep,train=trl,val=vll,bpm=bpm_mae,steps=stp_mae))
        print(f"ep{ep:03d} tr{trl:.4f} vl{vll:.4f} {bpm_mae:.2f}bpm {stp_mae:.2f}steps")
        sch.step(vll)
        if vll<best:
            best,no_imp=vll,0; os.makedirs(os.path.dirname(cfg.ssl_ckpt),exist_ok=True); torch.save(model.state_dict(),cfg.ssl_ckpt); print("  ↳ saved backbone")
        else:
            no_imp+=1

    # 5) save stats CSV + plots (identical to original)
    df_stats=pd.DataFrame(stats); df_stats.to_csv(os.path.join(res_dir,"stats_pretrain.csv"),index=False)
    plt.figure(figsize=(10,6)); plt.plot(df_stats.epoch,df_stats.train,label="Train"); plt.plot(df_stats.epoch,df_stats.val,label="Val"); plt.xlabel("Epoch"); plt.ylabel("Loss"); plt.title("SSL Pretrain Loss"); plt.legend(); plt.tight_layout(); plt.savefig(os.path.join(res_dir,"ssl_pretrain_loss.png")); plt.close()
    plt.figure(figsize=(10,6)); plt.plot(df_stats.epoch,df_stats.bpm); plt.xlabel("Epoch"); plt.ylabel("BPM MAE"); plt.tight_layout(); plt.savefig(os.path.join(res_dir,"ssl_pretrain_bpm_error.png")); plt.close()
    plt.figure(figsize=(10,6)); plt.plot(df_stats.epoch,df_stats.steps); plt.xlabel("Epoch"); plt.ylabel("Steps MAE"); plt.tight_layout(); plt.savefig(os.path.join(res_dir,"ssl_pretrain_steps_error.png")); plt.close()

    # 6) per‑user best/worst analysis (unchanged)
    df_val=pd.DataFrame(val_errs); df_val["total_error"]=df_val.bpm_error+df_val.steps_error; usr_err=df_val.groupby("user_id").total_error.mean().reset_index(); best_u=usr_err.nsmallest(5,"total_error"); worst_u=usr_err.nlargest(5,"total_error")
    ana_dir=os.path.join(res_dir,"analysis_pretrain"); os.makedirs(ana_dir,exist_ok=True)
    for _,r in best_u.iterrows():
        recs=df_val[df_val.user_id==r.user_id].to_dict("records")[:2]; U.plot_prediction_user(r.user_id,recs,"best_pretrain",P,W,ana_dir)
    for _,r in worst_u.iterrows():
        recs=df_val[df_val.user_id==r.user_id].to_dict("records")[:2]; U.plot_prediction_user(r.user_id,recs,"worst_pretrain",P,W,ana_dir)

    # 7) auto‑sync README table
    if cfg.autoupdate_readme:
        U.sync_readme_table(os.path.join(res_dir,"stats_pretrain.csv"), tag="PRETRAIN")

if __name__=="__main__":
    main()