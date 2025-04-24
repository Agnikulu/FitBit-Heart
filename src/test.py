# src/test.py
from __future__ import annotations
import os, random, warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, roc_auc_score
from torch.utils.data import Dataset, DataLoader, RandomSampler
from sklearn.model_selection import train_test_split

from src.config import load_config
import src.utils as U
from src.models import DrugClassifier

warnings.filterwarnings("ignore")

class FocalLoss(nn.Module):
    def __init__(self, gamma: float = 2.0, reduction: str = "mean"):
        super().__init__()
        self.gamma = gamma
        self.bce = nn.BCEWithLogitsLoss(reduction='none')
        self.reduction = reduction

    def forward(self, logits: torch.Tensor, targets: torch.Tensor) -> torch.Tensor:
        bce_loss = self.bce(logits, targets)
        p_t = torch.exp(-bce_loss)
        loss = (1 - p_t) ** self.gamma * bce_loss
        if self.reduction == 'mean': return loss.mean()
        if self.reduction == 'sum': return loss.sum()
        return loss

class CraveDataset(Dataset):
    def __init__(self, samples): self.samples = samples
    def __len__(self): return len(self.samples)
    def __getitem__(self, idx):
        s = self.samples[idx]
        return (
            torch.tensor(s['bpm'], dtype=torch.float32),
            torch.tensor(s['steps'], dtype=torch.float32),
            s['id'],
            torch.tensor(s['label'], dtype=torch.float32),
        )

def make_samples(df_user, label_col, win_size, *, stride):
    out = []
    df_user = df_user.sort_values('datetime').reset_index(drop=True)
    n = len(df_user)
    for i in range(0, n - win_size + 1, stride):
        chunk = df_user.iloc[i:i+win_size]
        out.append({
            'id': chunk['id'].iat[0],
            'bpm': chunk.get('bpm_scaled', chunk['bpm']).values,
            'steps': chunk.get('steps_scaled', chunk['steps']).values,
            'label': int(chunk[label_col].max()==1)
        })
    return out

def main():
    cfg = load_config()
    torch.manual_seed(cfg.seed)
    random.seed(cfg.seed)
    np.random.seed(cfg.seed)
    device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')

    df_sensor = U.load_sensor_data(cfg.data.biosignal_dir)
    df_label = U.load_label_data(cfg.data.label_root)
    df_hour = U.pivot_label_data(df_label)
    df = U.merge_sensors_with_labels(df_sensor, df_hour)
    df = df.groupby('id').apply(U.scale_per_user).reset_index(drop=True)

    lbl_cols = [c for c in df.columns if c.endswith('_label')]
    if not lbl_cols: raise SystemExit('No label columns found – aborting')

    res_root = os.path.join(cfg.results_root, 'test')
    os.makedirs(res_root, exist_ok=True)
    all_results = []

    for uid in sorted(df['id'].unique()):
        user_dir = os.path.join(res_root, f'user_{uid}')
        os.makedirs(user_dir, exist_ok=True)
        df_u = df[df['id']==uid]
        if len(df_u) < 2 * cfg.window.size: continue

        n = len(df_u); cut = int((1-0.15)*n)
        df_tv, df_test = df_u.iloc[:cut], df_u.iloc[cut:]

        for lbl in lbl_cols:
            samples_tv = make_samples(df_tv, lbl, cfg.window.size, stride=1)
            samples_test = make_samples(df_test, lbl, cfg.window.size, stride=cfg.window.size)
            if not samples_test or len(samples_tv)<15: continue
            y_tv = np.array([s['label'] for s in samples_tv])
            if y_tv.sum()<2 or (len(y_tv)-y_tv.sum())<2: continue

            idx_train, idx_val = train_test_split(
                np.arange(len(samples_tv)), test_size=0.176,
                stratify=y_tv, random_state=cfg.seed
            )
            def mk_loader(samps, idxs, shuffle):
                ds = CraveDataset([samps[i] for i in idxs])
                return DataLoader(ds, batch_size=cfg.test.batch_size,
                                  sampler=RandomSampler(ds) if shuffle else None)

            train_loader = mk_loader(samples_tv, idx_train, True)
            val_loader = mk_loader(samples_tv, idx_val, False)
            test_loader = DataLoader(
                CraveDataset(samples_test),
                batch_size=cfg.test.batch_size,
                shuffle=False
            )

            # initialize model and optionally transfer SSL weights
            model = DrugClassifier(
                window_size=cfg.window.size,
                cfg_model=dict(cfg.model)
            ).to(device)
            ckpt = os.path.join(
                cfg.results_root, 'train', f'user_{uid}', 'personalized_ssl.pt'
            )
            if os.path.isfile(ckpt):
                ssl_sd = torch.load(ckpt, map_location='cpu')
                model_sd = model.state_dict()
                matched = {
                    k: v for k, v in ssl_sd.items()
                    if k in model_sd and v.shape == model_sd[k].shape
                }
                model_sd.update(matched)
                model.load_state_dict(model_sd)
                print(f"[U{uid}] Transferred {len(matched)}/{len(model_sd)} layers from SSL")
            else:
                print(f"[U{uid}] No SSL checkpoint – random init")

            # freeze backbone except classifier
            for name, param in model.named_parameters():
                param.requires_grad = 'classifier' in name or 'rnn' in name

            # set up loss, optimizer, scheduler
            criterion = FocalLoss(gamma=2.0)
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

            # training loop
            best_val_loss = float('inf')
            no_improve = 0
            lbl_dir = os.path.join(user_dir, lbl)
            os.makedirs(lbl_dir, exist_ok=True)
            best_ckpt = os.path.join(lbl_dir, 'best.pt')

            for epoch in range(1, cfg.test.num_epochs + 1):
                if no_improve >= cfg.test.patience:
                    break
                # train
                model.train()
                train_loss = 0.0
                for bpm, steps, _, tgt in train_loader:
                    bpm, steps, tgt = bpm.to(device), steps.to(device), tgt.to(device)
                    optimizer.zero_grad()
                    logits = model(bpm, steps)
                    loss = criterion(logits, tgt)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
                    optimizer.step()
                    train_loss += loss.item()
                train_loss /= len(train_loader)

                # validate
                model.eval()
                val_loss = 0.0
                val_preds, val_targets = [], []
                with torch.no_grad():
                    for bpm, steps, _, tgt in val_loader:
                        bpm, steps, tgt = bpm.to(device), steps.to(device), tgt.to(device)
                        logits = model(bpm, steps)
                        val_loss += criterion(logits, tgt).item()
                        val_preds.extend(torch.sigmoid(logits).cpu().tolist())
                        val_targets.extend(tgt.cpu().tolist())
                val_loss /= len(val_loader)
                val_acc = (
                    (np.array(val_preds) >= 0.5) == np.array(val_targets)
                ).mean() * 100
                print(
                    f"[U{uid}|{lbl}] Epoch {epoch} | "
                    f"Train Loss: {train_loss:.4f} | "
                    f"Val Loss: {val_loss:.4f} | "
                    f"Val Acc: {val_acc:.1f}%"
                )

                scheduler.step(val_loss)
                if val_loss < best_val_loss:
                    best_val_loss = val_loss
                    no_improve = 0
                    torch.save(model.state_dict(), best_ckpt)
                else:
                    no_improve += 1

            # load best model
            if os.path.isfile(best_ckpt):
                model.load_state_dict(torch.load(best_ckpt, map_location=device))

            # threshold selection using Youden's J
            all_val_preds, all_val_targets = [], []
            model.eval()
            with torch.no_grad():
                for bpm, steps, _, tgt in val_loader:
                    logits = model(bpm.to(device), steps.to(device))
                    all_val_preds.extend(torch.sigmoid(logits).cpu().tolist())
                    all_val_targets.extend(tgt.cpu().tolist())
            all_val_preds = np.array(all_val_preds)
            all_val_targets = np.array(all_val_targets)
            best_j, best_threshold = -1.0, 0.5
            for t in np.linspace(0.0, 1.0, 101):
                preds_t = (all_val_preds >= t).astype(int)
                tn, fp, fn, tp = confusion_matrix(
                    all_val_targets, preds_t, labels=[0, 1]
                ).ravel()
                sens = tp / (tp + fn) if tp + fn > 0 else 0.0
                spec = tn / (tn + fp) if tn + fp > 0 else 0.0
                j = sens + spec - 1
                if j > best_j:
                    best_j, best_threshold = j, t

            # final test evaluation
            test_preds, test_targets = [], []
            with torch.no_grad():
                for bpm, steps, _, tgt in test_loader:
                    logits = model(bpm.to(device), steps.to(device))
                    test_preds.extend(torch.sigmoid(logits).cpu().tolist())
                    test_targets.extend(tgt.cpu().tolist())
            test_preds = np.array(test_preds)
            test_targets = np.array(test_targets)
            auc_score = (
                np.nan if len(np.unique(test_targets)) < 2 else
                roc_auc_score(test_targets, test_preds)
            )
            test_pred_labels = (test_preds >= best_threshold).astype(int)
            test_acc = (test_pred_labels == test_targets).mean() * 100
            tn, fp, fn, tp = confusion_matrix(
                test_targets, test_pred_labels, labels=[0, 1]
            ).ravel()
            sens_test = tp / (tp + fn) if tp + fn > 0 else np.nan
            spec_test = tn / (tn + fp) if tn + fp > 0 else np.nan
            print(
                f"[U{uid}|{lbl}] TEST | Th={best_threshold:.2f} "
                f"Acc={test_acc:.2f}% | AUC={auc_score:.3f} "
                f"Sens={sens_test:.3f} | Spec={spec_test:.3f}"
            )

            # plot and save confusion matrix
            plt.figure(figsize=(3, 3))
            cmatrix = [[tn, fp], [fn, tp]]
            plt.imshow(cmatrix, cmap='Blues')
            for i, v in enumerate([tn, fp, fn, tp]):
                plt.text(
                    i % 2, i // 2, str(v),
                    ha='center', va='center',
                    color='white' if v > max(tn, fp, fn, tp) / 2 else 'black'
                )
            plt.title(
                f"U{uid} {lbl} Th={best_threshold:.2f} Acc={test_acc:.1f}%"
            )
            plt.tight_layout()
            plt.savefig(os.path.join(lbl_dir, f"cm_{best_threshold:.2f}.png"))
            plt.close()

            # record results
            all_results.append({
                'user_id': uid,
                'label_col': lbl,
                'n_test': len(test_targets),
                'pos': int(test_targets.sum()),
                'neg': int(len(test_targets) - test_targets.sum()),
                'thr': best_threshold,
                'auc': auc_score,
                'acc': test_acc,
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'tp': tp,
                'sensitivity': sens_test,
                'specificity': spec_test
            })

    # save summary CSV and optionally update README
    if all_results:
        df_res = pd.DataFrame(all_results)
        df_res.to_csv(
            os.path.join(res_root, 'classification_summary.csv'),
            index=False
        )
        if cfg.autoupdate_readme:
            U.sync_readme_table(
                os.path.join(res_root, 'classification_summary.csv'),
                tag='CLASSIFICATION'
            )

    # comparison metrics
    df_res = pd.DataFrame(all_results)
    comp_orig = {'Accuracy': 0.786, 'Sensitivity': 0.724, 'Specificity': 0.713, 'AUC': np.nan}
    comp_new = {
        'Accuracy': df_res['acc'].mean() / 100,
        'Sensitivity': df_res['sensitivity'].mean(),
        'Specificity': df_res['specificity'].mean(),
        'AUC': df_res['auc'].dropna().mean()
    }
    comp_list = [
        [m, comp_orig[m], comp_new[m]] for m in ['Accuracy', 'Sensitivity', 'Specificity', 'AUC']
    ]
    pd.DataFrame(comp_list, columns=['Metric', 'Original', 'This Study']).to_csv(
        os.path.join(res_root, 'comparison_metrics.csv'), index=False
    )
    if cfg.autoupdate_readme:
        U.sync_readme_table(
            os.path.join(res_root, 'comparison_metrics.csv'),
            tag='COMPARISON',
            static_head='**Critical Comparison with Prior Work**'
        )

if __name__ == '__main__':
    main()