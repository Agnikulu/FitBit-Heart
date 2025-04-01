# pretrain.py

import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import warnings
import os
import matplotlib.pyplot as plt

from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split

from models import SSLForecastingModel
from utils import (scale_per_user, user_scalers,
                   create_forecasting_samples,
                   ForecastDataset, forecasting_collate_fn,
                   PerUserBatchSampler,
                   plot_prediction_user, select_samples_for_plotting)

warnings.filterwarnings('ignore')

#################################################
# 1. Data Loading & Window Setup
#################################################
df_daily = pd.read_csv("data/lifesnaps.csv", parse_dates=['date'])
df_daily.drop("Unnamed: 0", axis=1, inplace=True)
df = df_daily[["id","date","hour","bpm","steps"]]

df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
df = df.sort_values(by=['id','datetime'])
df = df.dropna().reset_index(drop=True)

# Adjust window sizes from the paper
WINDOW_SIZE = 6
INPUT_WINDOWS = 2
PREDICT_WINDOWS = 1
BATCH_SIZE = 128

#################################################
# 2. Scaling
#################################################
df = df.groupby('id').apply(scale_per_user).reset_index(drop=True)  # populates user_scalers

#################################################
# 3. Create data samples
#################################################
data_samples = create_forecasting_samples(df,
    col_bpm='bpm_scaled', col_steps='steps_scaled',
    window_size=WINDOW_SIZE,
    input_windows=INPUT_WINDOWS,
    predict_windows=PREDICT_WINDOWS
)

unique_ids = df['id'].unique()
train_ids, val_ids = train_test_split(unique_ids, test_size=0.2, random_state=42)
train_data = [s for s in data_samples if s['user_id'] in train_ids]
val_data   = [s for s in data_samples if s['user_id'] in val_ids]

def create_user_dict(samples):
    from collections import defaultdict
    user_dict = defaultdict(list)
    for idx, sample in enumerate(samples):
        user_dict[sample['user_id']].append(idx)
    return user_dict

train_user_dict = create_user_dict(train_data)
val_user_dict   = create_user_dict(val_data)

train_dataset = ForecastDataset(train_data)
val_dataset   = ForecastDataset(val_data)

train_sampler = PerUserBatchSampler(train_user_dict, batch_size=BATCH_SIZE)
val_sampler   = PerUserBatchSampler(val_user_dict, batch_size=BATCH_SIZE)

train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=forecasting_collate_fn)
val_loader   = DataLoader(val_dataset,   batch_sampler=val_sampler,   collate_fn=forecasting_collate_fn)

#################################################
# 4. Weighted Loss
#################################################
avg_bpm   = df[df['id'].isin(train_ids)]['bpm'].abs().mean()
avg_steps = df[df['id'].isin(train_ids)]['steps'].abs().mean()
alpha = 1.0 / avg_bpm
beta  = 1.0 / avg_steps
tot   = alpha + beta
alpha /= tot
beta  /= tot
print(f"Weight for BPM loss: {alpha:.4f}")
print(f"Weight for Steps loss: {beta:.4f}")

#################################################
# 5. Model & Training
#################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

model = SSLForecastingModel(window_size=WINDOW_SIZE).to(device)
criterion = nn.SmoothL1Loss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

num_epochs = 100
patience = 15
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False

save_dir = 'results/saved_models'
os.makedirs(save_dir, exist_ok=True)
model_save_path = os.path.join(save_dir, 'ssl_backbone.pth')

stats = {
    'epoch': [],
    'train_loss': [],
    'val_loss': [],
    'avg_bpm_error': [],
    'avg_steps_error': []
}
validation_errors = []

for epoch in range(num_epochs):
    if early_stop:
        break

    # ----TRAIN----
    model.train()
    epoch_loss = 0.0
    for (bpm_in, steps_in,
         cur_bpm_scl, cur_steps_scl,
         bpm_targ, steps_targ,
         user_ids,
         bpm_targ_orig, steps_targ_orig,
         dtimes) in train_loader:

        bpm_in   = bpm_in.to(device)
        steps_in = steps_in.to(device)
        cur_bpm_scl   = cur_bpm_scl.to(device)
        cur_steps_scl = cur_steps_scl.to(device)
        bpm_targ   = bpm_targ.to(device)
        steps_targ = steps_targ.to(device)

        optimizer.zero_grad()
        bpm_pred, steps_pred = model(bpm_in, steps_in, cur_bpm_scl, cur_steps_scl)

        loss_bpm   = criterion(bpm_pred, bpm_targ)
        loss_steps = criterion(steps_pred, steps_targ)
        loss = alpha * loss_bpm + beta * loss_steps

        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()

    scheduler.step()
    avg_train_loss = epoch_loss / len(train_loader)

    # ----VAL----
    model.eval()
    val_loss = 0.0
    total_bpm_error = 0.0
    total_steps_error = 0.0
    validation_errors_epoch = []

    with torch.no_grad():
        for (bpm_in, steps_in,
             cur_bpm_scl, cur_steps_scl,
             bpm_targ, steps_targ,
             user_ids,
             bpm_targ_orig, steps_targ_orig,
             dtimes) in val_loader:

            bpm_in   = bpm_in.to(device)
            steps_in = steps_in.to(device)
            cur_bpm_scl   = cur_bpm_scl.to(device)
            cur_steps_scl = cur_steps_scl.to(device)
            bpm_targ   = bpm_targ.to(device)
            steps_targ = steps_targ.to(device)

            bpm_pred_scl, steps_pred_scl = model(bpm_in, steps_in, cur_bpm_scl, cur_steps_scl)

            loss_bpm   = criterion(bpm_pred_scl, bpm_targ)
            loss_steps = criterion(steps_pred_scl, steps_targ)
            loss_val   = alpha*loss_bpm + beta*loss_steps
            val_loss  += loss_val.item()

            # compute errors in original scale (MAE)
            bpm_pred_np   = bpm_pred_scl.cpu().numpy()
            steps_pred_np = steps_pred_scl.cpu().numpy()
            bpm_targ_np   = bpm_targ.cpu().numpy()
            steps_targ_np = steps_targ.cpu().numpy()

            Bsize = len(user_ids)
            for i in range(Bsize):
                uid = user_ids[i]
                bp_pred_2d   = bpm_pred_np[i]
                st_pred_2d   = steps_pred_np[i]
                bp_true_2d   = bpm_targ_np[i]
                st_true_2d   = steps_targ_np[i]

                bp_pred_1d = bp_pred_2d.flatten()
                st_pred_1d = st_pred_2d.flatten()
                bp_true_1d = bp_true_2d.flatten()
                st_true_1d = st_true_2d.flatten()

                # Inverse
                from utils import inverse_transform
                bp_pred_unsc, st_pred_unsc = inverse_transform(uid, bp_pred_1d, st_pred_1d)
                bp_true_unsc, st_true_unsc = inverse_transform(uid, bp_true_1d, st_true_1d)

                # MAE
                bpm_err   = np.mean(np.abs(bp_pred_unsc - bp_true_unsc))
                steps_err = np.mean(np.abs(st_pred_unsc - st_true_unsc))
                total_bpm_error   += bpm_err
                total_steps_error += steps_err

                validation_errors_epoch.append({
                    'user_id': uid,
                    'bpm_error': bpm_err,
                    'steps_error': steps_err,
                    'bpm_pred':   bp_pred_unsc.reshape(PREDICT_WINDOWS, WINDOW_SIZE),
                    'steps_pred': st_pred_unsc.reshape(PREDICT_WINDOWS, WINDOW_SIZE),
                    'bpm_true':   bp_true_unsc.reshape(PREDICT_WINDOWS, WINDOW_SIZE),
                    'steps_true': st_true_unsc.reshape(PREDICT_WINDOWS, WINDOW_SIZE),
                    'datetime': dtimes[i]
                })

    avg_val_loss    = val_loss / len(val_loader)
    avg_bpm_error   = total_bpm_error   / len(val_dataset)
    avg_steps_error = total_steps_error / len(val_dataset)

    stats['epoch'].append(epoch+1)
    stats['train_loss'].append(avg_train_loss)
    stats['val_loss'].append(avg_val_loss)
    stats['avg_bpm_error'].append(avg_bpm_error)
    stats['avg_steps_error'].append(avg_steps_error)

    validation_errors.extend(validation_errors_epoch)

    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), model_save_path)
        print(f"Epoch {epoch+1}: Validation loss improved => Model saved.")
    else:
        epochs_no_improve += 1
        print(f"Epoch {epoch+1}: No improvement in val loss.")

    if epochs_no_improve >= patience:
        print(f"Early stopping triggered after {patience} epochs with no improvement.")
        early_stop = True

    print(f"Epoch {epoch+1}/{num_epochs} | "
          f"TrainLoss={avg_train_loss:.4f}, ValLoss={avg_val_loss:.4f}, "
          f"BPM_Err={avg_bpm_error:.2f}, Steps_Err={avg_steps_error:.2f}")

#################################################
# 6. Save training stats & charts
#################################################
results_dir = 'results/pretrain'
os.makedirs(results_dir, exist_ok=True)

import pandas as pd
stats_df = pd.DataFrame(stats)
stats_df.to_csv(os.path.join(results_dir, "stats_pretrain.csv"), index=False)

# Plot training vs. val loss
plt.figure(figsize=(10,6))
plt.plot(stats_df['epoch'], stats_df['train_loss'], label='Train Loss')
plt.plot(stats_df['epoch'], stats_df['val_loss'],   label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch (SSL Pretraining)')
plt.legend()
plt.savefig(os.path.join(results_dir, 'ssl_pretrain_loss.png'))
plt.show()

# Plot BPM Error
plt.figure(figsize=(10,6))
plt.plot(stats_df['epoch'], stats_df['avg_bpm_error'], label='BPM Error')
plt.xlabel('Epoch')
plt.ylabel('MAE (BPM)')
plt.title('BPM Error per Epoch (SSL Pretrain)')
plt.legend()
plt.savefig(os.path.join(results_dir, 'ssl_pretrain_bpm_error.png'))
plt.show()

# Plot Steps Error
plt.figure(figsize=(10,6))
plt.plot(stats_df['epoch'], stats_df['avg_steps_error'], label='Steps Error')
plt.xlabel('Epoch')
plt.ylabel('MAE (Steps)')
plt.title('Steps Error per Epoch (SSL Pretrain)')
plt.legend()
plt.savefig(os.path.join(results_dir, 'ssl_pretrain_steps_error.png'))
plt.show()

print("Pretraining complete. Stats and figures saved.")

# Analyze best/worst users
import pandas as pd
validation_df = pd.DataFrame(validation_errors)
validation_df['total_error'] = validation_df['bpm_error'] + validation_df['steps_error']
user_error_df = validation_df.groupby('user_id')['total_error'].mean().reset_index()
user_error_df_sorted = user_error_df.sort_values(by='total_error', ascending=True)

topN = 5
best_users = user_error_df_sorted.head(topN)
worst_users = user_error_df_sorted.tail(topN)
print("Best Users:\n", best_users)
print("Worst Users:\n", worst_users)

analysis_dir = 'results/pretrain/analysis_pretrain'
os.makedirs(analysis_dir, exist_ok=True)

for _, row in best_users.iterrows():
    uid = row['user_id']
    samples_to_plot = validation_df[validation_df['user_id'] == uid].to_dict(orient='records')[:2]
    plot_prediction_user(uid, samples_to_plot, "best_pretrain", PREDICT_WINDOWS, WINDOW_SIZE, analysis_dir)

for _, row in worst_users.iterrows():
    uid = row['user_id']
    samples_to_plot = validation_df[validation_df['user_id'] == uid].to_dict(orient='records')[:2]
    plot_prediction_user(uid, samples_to_plot, "worst_pretrain", PREDICT_WINDOWS, WINDOW_SIZE, analysis_dir)
