# utils.py

import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset, Sampler
import matplotlib.pyplot as plt
from collections import defaultdict
from sklearn.preprocessing import StandardScaler

# -------------------------
# Window definitions
# (You can override them in each script or place them here if they are truly global)
WINDOW_SIZE = 6
INPUT_WINDOWS = 2
PREDICT_WINDOWS = 1
BATCH_SIZE = 128
# -------------------------


#################################################
# Scaling Utilities
#################################################
user_scalers = {}  # dictionary: user_id -> {bpm_mean, bpm_scale, ...}
def scale_per_user(group, col_bpm='bpm', col_steps='steps'):
    user_id = group.name
    scaler_bpm = StandardScaler()
    scaler_steps = StandardScaler()
    group['bpm_scaled'] = scaler_bpm.fit_transform(group[[col_bpm]]).flatten()
    group['steps_scaled'] = scaler_steps.fit_transform(group[[col_steps]]).flatten()
    user_scalers[user_id] = {
        'bpm_mean': scaler_bpm.mean_[0],
        'bpm_scale': scaler_bpm.scale_[0],
        'steps_mean': scaler_steps.mean_[0],
        'steps_scale': scaler_steps.scale_[0]
    }
    return group

def inverse_transform(user_id, bpm_scaled, steps_scaled):
    """
    Revert scaled BPM and Steps to original scale
    """
    bpm_mean   = user_scalers[user_id]['bpm_mean']
    bpm_scale  = user_scalers[user_id]['bpm_scale']
    steps_mean = user_scalers[user_id]['steps_mean']
    steps_scale= user_scalers[user_id]['steps_scale']

    bpm_original   = bpm_scaled * bpm_scale + bpm_mean
    steps_original = steps_scaled * steps_scale + steps_mean
    return bpm_original, steps_original


#################################################
# Creating Dataset Samples
#################################################
def create_forecasting_samples(df, col_bpm='bpm_scaled', col_steps='steps_scaled',
                              window_size=6, input_windows=2, predict_windows=1):
    """
    Break time series into non-overlapping windows of 'window_size' rows.
    Then slide to form [INPUT_WINDOWS, PREDICT_WINDOWS].
    Return a list of dict samples.
    """
    data_samples = []
    user_groups = df.groupby('id')
    for user_id, group in user_groups:
        group = group.sort_values('datetime').reset_index(drop=True)
        windows = []
        for i in range(0, len(group), window_size):
            chunk = group.iloc[i : i + window_size]
            if len(chunk) == window_size:
                windows.append(chunk)

        for i in range(len(windows) - input_windows - predict_windows + 1):
            in_ws  = windows[i : i + input_windows]
            targ_ws = windows[i + input_windows : i + input_windows + predict_windows]

            bpm_in = np.array([w[col_bpm].values for w in in_ws])
            steps_in = np.array([w[col_steps].values for w in in_ws])

            bpm_t = np.array([tw[col_bpm].values for tw in targ_ws])
            steps_t = np.array([tw[col_steps].values for tw in targ_ws])

            # also store original unscaled if you want
            if 'bpm' in group.columns and 'steps' in group.columns:
                bpm_orig = np.array([tw['bpm'].values for tw in targ_ws])
                steps_orig = np.array([tw['steps'].values for tw in targ_ws])
            else:
                bpm_orig = bpm_t.copy()
                steps_orig = steps_t.copy()

            dt_val = targ_ws[0]['datetime'].values[0]

            data_samples.append({
                'user_id': user_id,
                'bpm_input': bpm_in,
                'steps_input': steps_in,
                'current_bpm_scaled': bpm_t,
                'current_steps_scaled': steps_t,
                'bpm_target': bpm_t,
                'steps_target': steps_t,
                'bpm_target_original': bpm_orig,
                'steps_target_original': steps_orig,
                'datetime': dt_val
            })
    return data_samples


#################################################
# Dataset / Sampler
#################################################
class ForecastDataset(Dataset):
    """
    For forecasting tasks, returns everything needed:
      - Past windows
      - Current windows
      - ...
    """
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        d = self.data_list[idx]
        return (
            torch.tensor(d['bpm_input'], dtype=torch.float32),
            torch.tensor(d['steps_input'], dtype=torch.float32),
            torch.tensor(d['current_bpm_scaled'], dtype=torch.float32),
            torch.tensor(d['current_steps_scaled'], dtype=torch.float32),
            torch.tensor(d['bpm_target'], dtype=torch.float32),
            torch.tensor(d['steps_target'], dtype=torch.float32),
            d['user_id'],
            torch.tensor(d['bpm_target_original'], dtype=torch.float32),
            torch.tensor(d['steps_target_original'], dtype=torch.float32),
            d['datetime']
        )

def forecasting_collate_fn(batch):
    bpm_inputs = torch.stack([b[0] for b in batch])
    steps_inputs = torch.stack([b[1] for b in batch])
    curr_bpm_scl = torch.stack([b[2] for b in batch])
    curr_steps_scl = torch.stack([b[3] for b in batch])
    bpm_targ = torch.stack([b[4] for b in batch])
    steps_targ = torch.stack([b[5] for b in batch])
    user_ids = [b[6] for b in batch]
    bpm_targ_orig = [b[7] for b in batch]
    steps_targ_orig = [b[8] for b in batch]
    datetimes = [b[9] for b in batch]

    return (bpm_inputs, steps_inputs,
            curr_bpm_scl, curr_steps_scl,
            bpm_targ, steps_targ,
            user_ids,
            bpm_targ_orig, steps_targ_orig,
            datetimes)


class PerUserBatchSampler(Sampler):
    """
    Groups all samples by user, yields entire user's samples in mini-batches.
    """
    def __init__(self, user_data_dict, batch_size=128):
        self.user_data = user_data_dict
        self.user_ids = list(user_data_dict.keys())
        self.batch_size = batch_size

    def __iter__(self):
        np.random.shuffle(self.user_ids)
        for uid in self.user_ids:
            indices = self.user_data[uid]
            np.random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                yield indices[i : i + self.batch_size]

    def __len__(self):
        # number of batches across all users
        total_batches = 0
        for uid in self.user_ids:
            n = len(self.user_data[uid])
            total_batches += (n // self.batch_size + (1 if n % self.batch_size != 0 else 0))
        return total_batches


#################################################
# Stats & Analysis
#################################################
def plot_prediction_user(user_id, user_samples, plot_type, predict_windows, window_size, analysis_dir):
    """
    Plots the true vs predicted BPM/Steps for a user for debugging.
    This is the same function from your original code that you
    used in analyzing 'best'/'worst' users.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    os.makedirs(analysis_dir, exist_ok=True)

    for i, sample in enumerate(user_samples):
        bpm_pred_2d   = sample['bpm_pred']
        steps_pred_2d = sample['steps_pred']
        bpm_true_2d   = sample['bpm_true']
        steps_true_2d = sample['steps_true']
        datetime_val  = sample['datetime']

        bpm_pred_1d   = bpm_pred_2d.flatten()
        steps_pred_1d = steps_pred_2d.flatten()
        bpm_true_1d   = bpm_true_2d.flatten()
        steps_true_1d = steps_true_2d.flatten()

        total_time = len(bpm_pred_1d)

        fig, axs = plt.subplots(2, 2, figsize=(15,10))
        fig.suptitle(f'User: {user_id} | Sample {i+1} | DateTime: {datetime_val}', fontsize=16)

        axs[0,0].plot(range(total_time), bpm_true_1d, label='True BPM', marker='o')
        axs[0,0].plot(range(total_time), bpm_pred_1d, label='Predicted BPM', marker='x')
        axs[0,0].set_title('BPM Prediction vs True')
        axs[0,0].legend()

        axs[0,1].plot(range(total_time), steps_true_1d, label='True Steps', marker='o')
        axs[0,1].plot(range(total_time), steps_pred_1d, label='Pred Steps', marker='x')
        axs[0,1].set_title('Steps Prediction vs True')
        axs[0,1].legend()

        axs[1,0].bar(range(total_time), np.abs(bpm_true_1d - bpm_pred_1d))
        axs[1,0].set_title('BPM Absolute Error')

        axs[1,1].bar(range(total_time), np.abs(steps_true_1d - steps_pred_1d))
        axs[1,1].set_title('Steps Absolute Error')

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = f"{plot_type}_user_{user_id}_sample_{i+1}.png"
        plt.savefig(os.path.join(analysis_dir, fname))
        plt.close()


def select_samples_for_plotting(validation_errors_df, user_id, num_samples=2):
    user_samples = validation_errors_df[validation_errors_df['user_id'] == user_id].to_dict(orient='records')
    return user_samples[:num_samples]
