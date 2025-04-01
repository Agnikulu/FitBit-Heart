# train.py

import os
import glob
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler
from sklearn.model_selection import train_test_split

from models import PersonalizedForecastingModel, partially_unfreeze_backbone
from utils import (
    scale_per_user, user_scalers,
    create_forecasting_samples,
    ForecastDataset, forecasting_collate_fn,
    plot_prediction_user,
    select_samples_for_plotting,
    WINDOW_SIZE, INPUT_WINDOWS, PREDICT_WINDOWS,
    inverse_transform  # to revert scaled BPM/Steps
)

warnings.filterwarnings('ignore')

#################################################
# 1) Load your private data (minute-level => hourly)
#################################################
data_folder = "data/Personalized AI Data/Biosignal"
csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
if not csv_files:
    raise ValueError(f"No CSV files found in {data_folder} (check your path or file extension).")

df_list = []
for file in csv_files:
    temp_df = pd.read_csv(file, parse_dates=['time'])
    # If participant_id missing, parse from filename
    if 'participant_id' not in temp_df.columns:
        base = os.path.basename(file)
        name_no_ext = os.path.splitext(base)[0]
        if name_no_ext.lower().startswith('p'):
            pid = name_no_ext[1:]
        else:
            pid = name_no_ext
        temp_df['participant_id'] = int(pid)
    df_list.append(temp_df)

df_raw = pd.concat(df_list, ignore_index=True)
df_raw.loc[df_raw['data_type'] == 'hr', 'data_type'] = 'heart_rate'
df_filtered = df_raw[df_raw['data_type'].isin(['heart_rate','steps'])].copy()
df_filtered.drop_duplicates(subset=['participant_id','time','data_type'], inplace=True)

# Pivot => [id, time, bpm, steps]
df_pivot = df_filtered.pivot(
    index=['participant_id','time'],
    columns='data_type',
    values='value'
).reset_index()

df_pivot.rename(columns={'participant_id': 'id', 'heart_rate': 'bpm'}, inplace=True)
df_pivot['bpm'] = pd.to_numeric(df_pivot['bpm'], errors='coerce')
df_pivot['steps'] = pd.to_numeric(df_pivot['steps'], errors='coerce')
df_pivot.dropna(subset=['bpm','steps'], inplace=True)

df_pivot['datetime'] = pd.to_datetime(df_pivot['time'])
df_pivot['date']     = df_pivot['datetime'].dt.date
df_pivot['hour']     = df_pivot['datetime'].dt.hour

df_sensors = df_pivot[['id','datetime','date','hour','bpm','steps']].copy()
df_sensors = df_sensors.sort_values(by=['id','datetime']).reset_index(drop=True)

def to_hour_datetime(dateval, hourval):
    return pd.to_datetime(dateval) + pd.to_timedelta(hourval, unit='H')

df_sensors['date'] = pd.to_datetime(df_sensors['date'])
df_hourly = df_sensors.groupby(['id','date','hour'], as_index=False).agg({
    'bpm': 'mean',
    'steps': 'sum'
})
df_hourly['datetime'] = df_hourly.apply(
    lambda r: to_hour_datetime(r['date'], r['hour']), axis=1
)
df_hourly = df_hourly[['id','datetime','date','hour','bpm','steps']]
df_hourly = df_hourly.sort_values(by=['id','datetime']).reset_index(drop=True)

df = df_hourly.copy()

#################################################
# 2) Apply scaling per user
#################################################
df = df.groupby('id').apply(scale_per_user).reset_index(drop=True)

#################################################
# 3) Create forecasting samples (windowed)
#################################################
all_samples = create_forecasting_samples(
    df,
    col_bpm='bpm_scaled',
    col_steps='steps_scaled',
    window_size=WINDOW_SIZE,
    input_windows=INPUT_WINDOWS,
    predict_windows=PREDICT_WINDOWS
)

print(f"Created {len(all_samples)} total samples from {df['id'].nunique()} users.")

#################################################
# 4) Load the SSL backbone path
#################################################
ssl_ckpt_path = "results/saved_models/ssl_backbone.pth"
if not os.path.exists(ssl_ckpt_path):
    raise FileNotFoundError(f"Pretrained SSL checkpoint not found at {ssl_ckpt_path}")

#################################################
# 5) Loop over each user ID, train a separate model
#################################################
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

num_epochs = 30
patience = 10
lr = 1e-4

global_stats = []  # to store summary of each user's final results

unique_user_ids = sorted(df['id'].unique())
print(f"Found {len(unique_user_ids)} unique user IDs: {unique_user_ids}")

for user_id in unique_user_ids:
    # Create a user-specific directory for results
    user_dir = os.path.join("results", "train", f"user_{user_id}")
    os.makedirs(user_dir, exist_ok=True)

    # Gather that user's samples
    user_data = [s for s in all_samples if s['user_id'] == user_id]
    if len(user_data) < 10:
        print(f"[User {user_id}] Skipping => not enough samples (<10).")
        continue

    # Train/val split => 80/20
    cutoff = int(0.8 * len(user_data))
    train_data = user_data[:cutoff]
    val_data   = user_data[cutoff:]
    if len(val_data) < 1:
        print(f"[User {user_id}] Skipping => 0 val data.")
        continue

    # Build dataset
    train_ds = ForecastDataset(train_data)
    val_ds   = ForecastDataset(val_data)

    # DataLoaders
    train_loader = DataLoader(
        train_ds,
        batch_size=64,
        shuffle=True,
        collate_fn=forecasting_collate_fn
    )
    val_loader   = DataLoader(
        val_ds,
        batch_size=64,
        shuffle=False,
        collate_fn=forecasting_collate_fn
    )

    # Weighted loss parameters based on user's original data
    user_df = df[df['id'] == user_id]
    avg_bpm   = user_df['bpm'].abs().mean() if len(user_df) > 0 else 1.0
    avg_steps = user_df['steps'].abs().mean() if len(user_df) > 0 else 1.0
    if avg_bpm == 0: avg_bpm = 1.0
    if avg_steps == 0: avg_steps = 1.0
    alpha = 1.0 / avg_bpm
    beta  = 1.0 / avg_steps
    tot = alpha + beta
    alpha /= tot
    beta  /= tot

    # Build model & load SSL backbone
    model = PersonalizedForecastingModel(window_size=WINDOW_SIZE).to(device)
    ssl_state_dict = torch.load(ssl_ckpt_path, map_location=device)
    model.load_state_dict(ssl_state_dict, strict=False)

    # PARTIAL unfreeze
    partially_unfreeze_backbone(model, unfreeze_ratio=0.5)

    criterion = nn.SmoothL1Loss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)

    # We'll track stats like in "pretrain.py"
    stats = {
        'epoch': [],
        'train_loss': [],
        'val_loss': [],
        'avg_bpm_error': [],
        'avg_steps_error': []
    }
    validation_errors = []  # store detailed validation errors for each epoch

    best_val_loss = float('inf')
    epochs_no_improve = 0

    for ep in range(num_epochs):
        if epochs_no_improve >= patience:
            print(f"[User {user_id}] Early stopping at epoch={ep+1}")
            break

        # ---- TRAIN ----
        model.train()
        sum_train_loss = 0.0
        for (bpm_in, steps_in,
             cur_bpm_scl, cur_steps_scl,
             bpm_t, steps_t,
             uids,
             bpm_orig, steps_orig,
             dtimes) in train_loader:

            bpm_in   = bpm_in.to(device)
            steps_in = steps_in.to(device)
            cur_bpm_scl   = cur_bpm_scl.to(device)
            cur_steps_scl = cur_steps_scl.to(device)
            bpm_t   = bpm_t.to(device)
            steps_t = steps_t.to(device)

            optimizer.zero_grad()
            out_bpm, out_steps = model(bpm_in, steps_in, cur_bpm_scl, cur_steps_scl)

            loss_bpm   = criterion(out_bpm, bpm_t)
            loss_steps = criterion(out_steps, steps_t)
            loss       = alpha * loss_bpm + beta * loss_steps
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()

            sum_train_loss += loss.item()

        scheduler.step()
        avg_train_loss = sum_train_loss / len(train_loader) if len(train_loader) > 0 else 0.0

        # ---- VALIDATION ----
        model.eval()
        sum_val_loss = 0.0
        total_bpm_err = 0.0
        total_steps_err = 0.0
        validation_errors_epoch = []

        with torch.no_grad():
            for (bpm_in, steps_in,
                 cur_bpm_scl, cur_steps_scl,
                 bpm_t, steps_t,
                 uids,
                 bpm_orig, steps_orig,
                 dtimes) in val_loader:

                bpm_in   = bpm_in.to(device)
                steps_in = steps_in.to(device)
                cur_bpm_scl   = cur_bpm_scl.to(device)
                cur_steps_scl = cur_steps_scl.to(device)
                bpm_t   = bpm_t.to(device)
                steps_t = steps_t.to(device)

                out_bpm_scl, out_steps_scl = model(bpm_in, steps_in, cur_bpm_scl, cur_steps_scl)

                loss_bpm   = criterion(out_bpm_scl, bpm_t)
                loss_steps = criterion(out_steps_scl, steps_t)
                val_loss   = alpha * loss_bpm + beta * loss_steps
                sum_val_loss += val_loss.item()

                # For BPM/Steps error in original scale
                out_bpm_np   = out_bpm_scl.cpu().numpy()
                out_steps_np = out_steps_scl.cpu().numpy()
                bpm_t_np     = bpm_t.cpu().numpy()
                steps_t_np   = steps_t.cpu().numpy()

                batch_size = len(uids)
                for i in range(batch_size):
                    uid = uids[i]
                    # shape => [PREDICT_WINDOWS, WINDOW_SIZE]
                    pred_bpm_2d   = out_bpm_np[i]
                    pred_steps_2d = out_steps_np[i]
                    true_bpm_2d   = bpm_t_np[i]
                    true_steps_2d = steps_t_np[i]

                    # Flatten to do inverse transform
                    pred_bpm_1d   = pred_bpm_2d.flatten()
                    pred_steps_1d = pred_steps_2d.flatten()
                    true_bpm_1d   = true_bpm_2d.flatten()
                    true_steps_1d = true_steps_2d.flatten()

                    # Revert scaling
                    bpm_pred_unsc, steps_pred_unsc = inverse_transform(uid, pred_bpm_1d, pred_steps_1d)
                    bpm_true_unsc, steps_true_unsc = inverse_transform(uid, true_bpm_1d, true_steps_1d)

                    # MAE
                    bpm_err   = np.mean(np.abs(bpm_pred_unsc - bpm_true_unsc))
                    steps_err = np.mean(np.abs(steps_pred_unsc - steps_true_unsc))

                    total_bpm_err   += bpm_err
                    total_steps_err += steps_err

                    validation_errors_epoch.append({
                        'user_id': uid,
                        'epoch': ep+1,
                        'bpm_error': bpm_err,
                        'steps_error': steps_err,
                        'bpm_pred':   bpm_pred_unsc.reshape(PREDICT_WINDOWS, WINDOW_SIZE),
                        'steps_pred': steps_pred_unsc.reshape(PREDICT_WINDOWS, WINDOW_SIZE),
                        'bpm_true':   bpm_true_unsc.reshape(PREDICT_WINDOWS, WINDOW_SIZE),
                        'steps_true': steps_true_unsc.reshape(PREDICT_WINDOWS, WINDOW_SIZE),
                        'datetime': dtimes[i]
                    })

        avg_val_loss = sum_val_loss / len(val_loader) if len(val_loader) > 0 else 0.0
        # Compute average error across this user's validation set
        val_dataset_size = len(val_ds)
        if val_dataset_size > 0:
            avg_bpm_error   = total_bpm_err / val_dataset_size
            avg_steps_error = total_steps_err / val_dataset_size
        else:
            avg_bpm_error   = 0.0
            avg_steps_error = 0.0

        stats['epoch'].append(ep+1)
        stats['train_loss'].append(avg_train_loss)
        stats['val_loss'].append(avg_val_loss)
        stats['avg_bpm_error'].append(avg_bpm_error)
        stats['avg_steps_error'].append(avg_steps_error)

        validation_errors.extend(validation_errors_epoch)

        print(f"[User {user_id}] Epoch={ep+1}/{num_epochs}, "
              f"TrainLoss={avg_train_loss:.4f}, ValLoss={avg_val_loss:.4f}, "
              f"BPM_Err={avg_bpm_error:.2f}, Steps_Err={avg_steps_error:.2f}")

        # Early stopping check
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1

    # Save final stats for this user in their subdirectory
    user_stats_df = pd.DataFrame(stats)
    user_stats_csv = os.path.join(user_dir, "stats.csv")
    user_stats_df.to_csv(user_stats_csv, index=False)

    # Save detailed validation errors
    validation_csv = os.path.join(user_dir, "validation_errors.csv")
    validation_df = pd.DataFrame(validation_errors)
    validation_df.to_csv(validation_csv, index=False)

    # Plot training vs. validation loss
    plt.figure(figsize=(10,6))
    plt.plot(user_stats_df['epoch'], user_stats_df['train_loss'], label='Train Loss')
    plt.plot(user_stats_df['epoch'], user_stats_df['val_loss'], label='Val Loss')
    plt.xlabel('Epoch')
    plt.ylabel('Loss')
    plt.title(f'User {user_id} Train/Val Loss')
    plt.legend()
    loss_plot_path = os.path.join(user_dir, "loss.png")
    plt.savefig(loss_plot_path)
    plt.close()

    # Plot BPM error
    plt.figure(figsize=(10,6))
    plt.plot(user_stats_df['epoch'], user_stats_df['avg_bpm_error'], label='BPM Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (BPM)')
    plt.title(f'User {user_id} BPM Error per Epoch')
    plt.legend()
    bpm_plot_path = os.path.join(user_dir, "bpm_error.png")
    plt.savefig(bpm_plot_path)
    plt.close()

    # Plot Steps error
    plt.figure(figsize=(10,6))
    plt.plot(user_stats_df['epoch'], user_stats_df['avg_steps_error'], label='Steps Error')
    plt.xlabel('Epoch')
    plt.ylabel('MAE (Steps)')
    plt.title(f'User {user_id} Steps Error per Epoch')
    plt.legend()
    steps_plot_path = os.path.join(user_dir, "steps_error.png")
    plt.savefig(steps_plot_path)
    plt.close()

    # Save final model in the user's directory
    personalized_path = os.path.join(user_dir, "personalized_ssl.pt")
    torch.save(model.state_dict(), personalized_path)
    print(f"[User {user_id}] => saved model => {personalized_path}")

    # Analyze best/worst windows from final epoch
    final_epoch = user_stats_df['epoch'].max()
    final_df = validation_df[validation_df['epoch'] == final_epoch].copy()
    if len(final_df) > 0:
        final_df['total_error'] = final_df['bpm_error'] + final_df['steps_error']
        final_df_sorted = final_df.sort_values('total_error', ascending=True)
        topN = 5
        best_samples = final_df_sorted.head(topN)
        worst_samples = final_df_sorted.tail(topN)

        # Create an analysis subdirectory inside the user's folder
        analysis_dir = os.path.join(user_dir, "analysis")
        os.makedirs(analysis_dir, exist_ok=True)

        # Plot best samples
        for i, row in best_samples.iterrows():
            sample_to_plot = [dict(row)]  # wrapped in a list so that plot_prediction_user can handle it
            plot_prediction_user(
                user_id, sample_to_plot, "best_finetune", PREDICT_WINDOWS, WINDOW_SIZE, analysis_dir
            )

        # Plot worst samples
        for i, row in worst_samples.iterrows():
            sample_to_plot = [dict(row)]
            plot_prediction_user(
                user_id, sample_to_plot, "worst_finetune", PREDICT_WINDOWS, WINDOW_SIZE, analysis_dir
            )

    # Store user-level summary for global comparison
    global_stats.append({
        'user_id': user_id,
        'best_val_loss': best_val_loss,
        'final_epoch': user_stats_df['epoch'].iloc[-1],
        'final_bpm_error': user_stats_df['avg_bpm_error'].iloc[-1],
        'final_steps_error': user_stats_df['avg_steps_error'].iloc[-1]
    })

# Summarize across all users
global_df = pd.DataFrame(global_stats)
if len(global_df) > 0:
    summary_csv = os.path.join("results", "train", "personalized_finetune_summary.csv")
    global_df.to_csv(summary_csv, index=False)
    print("All user training done. Summary =>", summary_csv)
    print(global_df)
else:
    print("No users were trained. Possibly no data matched the criteria.")
