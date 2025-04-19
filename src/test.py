"""
Binary classification (drug/craving) with per‑user models.
"""
import os, random, warnings, numpy as np, pandas as pd, torch, matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import Dataset, DataLoader, RandomSampler
from src.config import load_config
import src.utils as U
from src.models import DrugClassifier, partially_unfreeze_backbone

warnings.filterwarnings("ignore")

class CraveDataset(Dataset):
    def __init__(self, S): self.S=S
    def __len__(self):     return len(self.S)
    def __getitem__(self,i):
        s=self.S[i]
        return (torch.tensor(s["bpm"],dtype=torch.float32),
                torch.tensor(s["steps"],dtype=torch.float32),
                s["id"],
                torch.tensor(s["label"],dtype=torch.float32))

def make_samples(df_u, lbl_col, win=6):
    lst=[]
    df_u = df_u.sort_values("datetime").reset_index(drop=True)
    for i in range(0, len(df_u), win):
        chunk = df_u.iloc[i:i+win]
        if len(chunk)<win: break
        lst.append(dict(
            id    = chunk["id"].iat[0],
            bpm   = chunk.get("bpm_scaled",chunk["bpm"]).values,
            steps = chunk.get("steps_scaled",chunk["steps"]).values,
            label = int(chunk[lbl_col].max()==1)
        ))
    return lst

def main():
    cfg = load_config()
    torch.manual_seed(cfg.seed); random.seed(cfg.seed); np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else "cpu")

    df_sensor = U.load_sensor_data(cfg.data.biosignal_dir)
    df_labels = U.load_label_data(cfg.data.label_root)
    df_hour   = U.pivot_label_data(df_labels)
    df        = U.merge_sensors_with_labels(df_sensor, df_hour)
    df        = df.groupby("id").apply(U.scale_per_user).reset_index(drop=True)
    lbl_cols  = [c for c in df.columns if c.endswith("_label")]
    if not lbl_cols: raise SystemExit("No label columns found")

    res_root  = os.path.join(cfg.results_root,"test"); os.makedirs(res_root,exist_ok=True)
    all_res   = []

    for uid in sorted(df["id"].unique()):
        usr_dir = os.path.join(res_root,f"user_{uid}"); os.makedirs(usr_dir, exist_ok=True)
        df_u = df[df["id"]==uid]
        if len(df_u) < 2*cfg.window_size: continue

        for lbl in lbl_cols:
            S = make_samples(df_u,lbl,cfg.window_size)
            if len(S)<15: continue
            y = np.array([s["label"] for s in S])
            if y.sum()<2 or (len(y)-y.sum())<2: continue

            idx = np.arange(len(S))
            idx_t, idx_te = train_test_split(idx,test_size=.15,stratify=y,random_state=42)
            idx_tr, idx_vl = train_test_split(
                idx_t,test_size=.15/.85,stratify=y[idx_t],random_state=42)

            mk_loader = lambda I,sh=True: DataLoader(
                CraveDataset([S[i] for i in I]),
                batch_size=cfg.test.batch_size,
                sampler=RandomSampler(I) if sh else None)
            tr_loader, vl_loader = mk_loader(idx_tr), mk_loader(idx_vl,False)
            te_loader = mk_loader(idx_te,False)

            model = DrugClassifier(window_size=cfg.window_size).to(device)
            ckpt  = os.path.join(cfg.results_root,"train",f"user_{uid}","personalized_ssl.pt")
            if os.path.isfile(ckpt):
                sd = torch.load(ckpt,map_location="cpu"); ms=model.state_dict()
                ms.update({k:v for k,v in sd.items() if k in ms and v.shape==ms[k].shape})
                model.load_state_dict(ms)
            partially_unfreeze_backbone(model,cfg.test.unfreeze_ratio)

            crit = torch.nn.BCEWithLogitsLoss()
            opt  = torch.optim.Adam(filter(lambda p:p.requires_grad,model.parameters()),
                                    lr=cfg.test.lr)
            sch  = torch.optim.lr_scheduler.StepLR(
                opt,step_size=cfg.test.scheduler.step_size,
                gamma=cfg.test.scheduler.gamma)

            best_val,no_imp = float("inf"),0
            lbl_dir = os.path.join(usr_dir,lbl); os.makedirs(lbl_dir,exist_ok=True)
            best_ck = os.path.join(lbl_dir,"best.pt")

            for ep in range(1,cfg.test.num_epochs+1):
                if no_imp>=cfg.test.patience: break
                # train
                model.train(); tr=0.
                for bpm,stp,_id,tgt in tr_loader:
                    bpm,stp,tgt = bpm.to(device),stp.to(device),tgt.to(device)
                    opt.zero_grad(); loss=crit(model(bpm,stp),tgt)
                    loss.backward(); torch.nn.utils.clip_grad_norm_(model.parameters(),1.)
                    opt.step(); tr+=loss.item()
                tr/=len(tr_loader); sch.step()
                # val
                model.eval(); vl=0.; preds=[]; gts=[]
                with torch.no_grad():
                    for bpm,stp,_id,tgt in vl_loader:
                        logit=model(bpm.to(device),stp.to(device))
                        vl+=crit(logit,tgt.to(device)).item()
                        preds += torch.sigmoid(logit).tolist(); gts += tgt.tolist()
                vl/=len(vl_loader); acc=( (np.array(preds)>=.5)==np.array(gts) ).mean()*100
                print(f"[U{uid}|{lbl}] ep{ep:02d} tr{tr:.4f} vl{vl:.4f} acc{acc:5.1f}%")
                if vl<best_val:
                    best_val,no_imp=vl,0; torch.save(model.state_dict(),best_ck)
                else: no_imp+=1
            if not os.path.isfile(best_ck): continue
            model.load_state_dict(torch.load(best_ck,map_location=device))

            # threshold on val
            model.eval(); pv,yv=[],[]
            with torch.no_grad():
                for bpm,stp,_id,tgt in vl_loader:
                    pv += torch.sigmoid(model(bpm.to(device),stp.to(device))).tolist()
                    yv += tgt.tolist()
            pv,yv = np.array(pv),np.array(yv)
            thr = max(np.linspace(0,1,101), key=lambda t:((pv>=t)==yv).mean())

            # test
            pt,yt=[],[]
            with torch.no_grad():
                for bpm,stp,_id,tgt in te_loader:
                    pt += torch.sigmoid(model(bpm.to(device),stp.to(device))).tolist()
                    yt += tgt.tolist()
            pt,yt = np.array(pt),np.array(yt)
            auc = np.nan if len(np.unique(yt))==1 else roc_auc_score(yt,pt)
            pred=(pt>=thr).astype(int); acc=(pred==yt).mean()*100
            tn,fp,fn,tp = confusion_matrix(yt,pred,labels=[0,1]).ravel()
            print(f"[U{uid}|{lbl}] TEST thr={thr:.2f} acc={acc:.2f}% auc={auc:.3f}")

            # plot confusion
            plt.figure(figsize=(3,3)); plt.imshow([[tn,fp],[fn,tp]],cmap="Blues")
            for i,v in enumerate([tn,fp,fn,tp]):
                plt.text(i%2,i//2,str(v),ha="center",va="center",
                         color="white" if v>max(tn,fp,fn,tp)/2 else "black")
            plt.title(f"U{uid} {lbl}\nth={thr:.2f} acc={acc:.1f}%")
            plt.tight_layout(); plt.savefig(os.path.join(lbl_dir,f"cm_{thr:.2f}.png")); plt.close()

            all_res.append(dict(user_id=uid,label_col=lbl,pos=int(yt.sum()),
                                neg=int(len(yt)-yt.sum()),thr=thr,auc=auc,acc=acc,
                                tn=tn,fp=fp,fn=fn,tp=tp))
    if all_res:
        pd.DataFrame(all_res).to_csv(
            os.path.join(res_root,"classification_summary.csv"),index=False)

if __name__ == "__main__":
    main()
