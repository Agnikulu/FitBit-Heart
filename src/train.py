import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch import nn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import warnings
from collections import defaultdict
import os
import matplotlib.pyplot as plt

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================
# 1. Data Loading and Preprocessing
# ============================

# Load your DataFrame
df_daily = pd.read_csv("data/lifesnaps.csv", parse_dates=['date'])
df_daily.drop("Unnamed: 0", axis=1, inplace=True)
df = df_daily[["id", "date", "hour", "bpm", "steps"]]

# Create a datetime column combining date and hour
df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
df = df.sort_values(by=['id', 'datetime'])
df = df.dropna()
df = df.reset_index(drop=True)

# ============================
# 2. Parameter Definitions
# ============================

WINDOW_SIZE = 6         # Number of time steps in each window
INPUT_WINDOWS = 24      # Number of past windows
PREDICT_WINDOWS = 4     # Number of windows to predict (multi-step)
BATCH_SIZE = 128        # Batch size for training and validation

# ============================
# 3. Data Normalization
# ============================

user_scalers = {}

def scale_per_user(group):
    """
    Applies StandardScaler to BPM and Steps data for each user.
    Stores scaler parameters for inverse transformations.
    """
    user_id = group.name
    scaler_bpm = StandardScaler()
    scaler_steps = StandardScaler()
    group['bpm_scaled'] = scaler_bpm.fit_transform(group['bpm'].values.reshape(-1,1)).flatten()
    group['steps_scaled'] = scaler_steps.fit_transform(group['steps'].values.reshape(-1,1)).flatten()
    user_scalers[user_id] = {
        'bpm_mean': scaler_bpm.mean_[0],
        'bpm_scale': scaler_bpm.scale_[0],
        'steps_mean': scaler_steps.mean_[0],
        'steps_scale': scaler_steps.scale_[0]
    }
    return group

# Apply scaling per user
df = df.groupby('id').apply(scale_per_user).reset_index(drop=True)

def inverse_transform(user_id, bpm_scaled, steps_scaled):
    """
    Reverts scaled BPM and Steps data back to original scale 
    using stored scaler parameters.
    """
    bpm_mean = user_scalers[user_id]['bpm_mean']
    bpm_scale = user_scalers[user_id]['bpm_scale']
    steps_mean = user_scalers[user_id]['steps_mean']
    steps_scale = user_scalers[user_id]['steps_scale']
    
    bpm_original = bpm_scaled * bpm_scale + bpm_mean
    steps_original = steps_scaled * steps_scale + steps_mean
    return bpm_original, steps_original

# ============================
# 4. Dataset Preparation
# ============================

def create_data_samples(df):
    """
    Creates data samples with multi-step targets:
      - Breaks time series into non-overlapping windows of size WINDOW_SIZE.
      - Each sample consists of:
          Past: INPUT_WINDOWS consecutive windows
          Future: PREDICT_WINDOWS consecutive windows
      - Instead of storing a single average for each target window, 
        we store ALL time points (i.e. shape [WINDOW_SIZE]) for each target window.
    """
    data_samples = []
    user_groups = df.groupby('id')
    
    for user_id, group in user_groups:
        # Break the time series into non-overlapping windows
        windows = []
        for i in range(0, len(group), WINDOW_SIZE):
            window = group.iloc[i:i+WINDOW_SIZE]
            if len(window) == WINDOW_SIZE:
                windows.append(window)
        
        # Slide over the windows to create samples
        for i in range(len(windows) - INPUT_WINDOWS - PREDICT_WINDOWS + 1):
            input_windows = windows[i : i+INPUT_WINDOWS]                   # Past windows
            target_windows = windows[i+INPUT_WINDOWS : i+INPUT_WINDOWS+PREDICT_WINDOWS]  # Future windows
            
            # Gather scaled BPM and Steps for input
            bpm_input_2d = np.array([w['bpm_scaled'].values for w in input_windows])   # [INPUT_WINDOWS, WINDOW_SIZE]
            steps_input_2d = np.array([w['steps_scaled'].values for w in input_windows])
            
            # Gather scaled targets (BPM & Steps), shape => [PREDICT_WINDOWS, WINDOW_SIZE]
            bpm_target_2d = np.array([tw['bpm_scaled'].values for tw in target_windows])
            steps_target_2d = np.array([tw['steps_scaled'].values for tw in target_windows])
            
            # Also keep original (unscaled) targets
            bpm_target_orig_2d = np.array([tw['bpm'].values for tw in target_windows])
            steps_target_orig_2d = np.array([tw['steps'].values for tw in target_windows])
            
            # We do NOT store just an average, 
            # but the full data for each target window 
            current_bpm_2d = bpm_target_2d.copy()    
            current_steps_2d = steps_target_2d.copy()
            
            # Store the start datetime of the first target window
            datetime_val = target_windows[0]['datetime'].values[0]

            data_samples.append({
                'bpm_input': bpm_input_2d,            # shape [INPUT_WINDOWS, WINDOW_SIZE]
                'steps_input': steps_input_2d,        # shape [INPUT_WINDOWS, WINDOW_SIZE]
                
                'current_bpm_scaled': current_bpm_2d,       # shape [PREDICT_WINDOWS, WINDOW_SIZE]
                'current_steps_scaled': current_steps_2d,   # shape [PREDICT_WINDOWS, WINDOW_SIZE]
                
                'bpm_target': bpm_target_2d,          # shape [PREDICT_WINDOWS, WINDOW_SIZE]
                'steps_target': steps_target_2d,
                'bpm_target_original': bpm_target_orig_2d,
                'steps_target_original': steps_target_orig_2d,
                
                'user_id': user_id,
                'datetime': datetime_val
            })
    return data_samples

# Generate data samples
data_samples = create_data_samples(df)

# ============================
# 5. Train-Validation Split
# ============================

unique_user_ids = df['id'].unique()
train_user_ids, val_user_ids = train_test_split(unique_user_ids, test_size=0.2, random_state=42)

train_samples = [s for s in data_samples if s['user_id'] in train_user_ids]
val_samples = [s for s in data_samples if s['user_id'] in val_user_ids]

# ============================
# 6. Sampler and DataLoader Setup
# ============================

def create_user_datasets(data_samples):
    user_data = defaultdict(list)
    for idx, s in enumerate(data_samples):
        user_data[s['user_id']].append(idx)
    return user_data

train_user_data = create_user_datasets(train_samples)
val_user_data = create_user_datasets(val_samples)

class PerUserBatchSampler(Sampler):
    def __init__(self, user_data, batch_size):
        self.user_data = user_data
        self.user_ids = list(user_data.keys())
        self.batch_size = batch_size

    def __iter__(self):
        np.random.shuffle(self.user_ids)
        for uid in self.user_ids:
            indices = self.user_data[uid]
            np.random.shuffle(indices)
            for i in range(0, len(indices), self.batch_size):
                yield indices[i:i+self.batch_size]

    def __len__(self):
        return sum(
            len(idx_list) // self.batch_size + (1 if len(idx_list) % self.batch_size != 0 else 0)
            for idx_list in self.user_data.values()
        )

class UserDataset(Dataset):
    """
    Each item returns:
        - Past windows of scaled BPM/Steps
        - The FULL set of "current" BPM/Steps for each target window 
          => shape [PREDICT_WINDOWS, WINDOW_SIZE]
        - The target BPM/Steps => shape [PREDICT_WINDOWS, WINDOW_SIZE]
        - The unscaled target BPM/Steps => shape [PREDICT_WINDOWS, WINDOW_SIZE]
        - user_id and datetime
    """
    def __init__(self, data_list):
        self.data_list = data_list
        
    def __len__(self):
        return len(self.data_list)
    
    def __getitem__(self, idx):
        d = self.data_list[idx]
        
        bpm_input = torch.tensor(d['bpm_input'], dtype=torch.float32)   # [INPUT_WINDOWS, WINDOW_SIZE]
        steps_input = torch.tensor(d['steps_input'], dtype=torch.float32)
        
        current_bpm_scaled = torch.tensor(d['current_bpm_scaled'], dtype=torch.float32)
        current_steps_scaled = torch.tensor(d['current_steps_scaled'], dtype=torch.float32)
        
        bpm_target = torch.tensor(d['bpm_target'], dtype=torch.float32)  # [PREDICT_WINDOWS, WINDOW_SIZE]
        steps_target = torch.tensor(d['steps_target'], dtype=torch.float32)
        
        bpm_target_original = torch.tensor(d['bpm_target_original'], dtype=torch.float32)
        steps_target_original = torch.tensor(d['steps_target_original'], dtype=torch.float32)
        
        user_id = d['user_id']
        datetime_val = d['datetime']
        
        return (bpm_input, steps_input,
                current_bpm_scaled, current_steps_scaled,
                bpm_target, steps_target,
                user_id,
                bpm_target_original, steps_target_original,
                datetime_val)

train_dataset = UserDataset(train_samples)
val_dataset = UserDataset(val_samples)

train_sampler = PerUserBatchSampler(train_user_data, BATCH_SIZE)
val_sampler = PerUserBatchSampler(val_user_data, BATCH_SIZE)

def collate_fn(batch):
    bpm_inputs = torch.stack([b[0] for b in batch])             # [B, INPUT_WINDOWS, WINDOW_SIZE]
    steps_inputs = torch.stack([b[1] for b in batch])           # [B, INPUT_WINDOWS, WINDOW_SIZE]
    
    current_bpm_scaled = torch.stack([b[2] for b in batch])     # [B, PREDICT_WINDOWS, WINDOW_SIZE]
    current_steps_scaled = torch.stack([b[3] for b in batch])
    
    bpm_targets = torch.stack([b[4] for b in batch])            # [B, PREDICT_WINDOWS, WINDOW_SIZE]
    steps_targets = torch.stack([b[5] for b in batch])
    
    user_ids = [b[6] for b in batch]
    bpm_targets_original = [b[7] for b in batch]
    steps_targets_original = [b[8] for b in batch]
    datetimes = [b[9] for b in batch]
    
    return (bpm_inputs, steps_inputs,
            current_bpm_scaled, current_steps_scaled,
            bpm_targets, steps_targets,
            user_ids,
            bpm_targets_original, steps_targets_original,
            datetimes)

train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=collate_fn)

# ============================
# 6A. Compute Weights Based on Average Size
# ============================

avg_bpm = df[df['id'].isin(train_user_ids)]['bpm'].abs().mean()
avg_steps = df[df['id'].isin(train_user_ids)]['steps'].abs().mean()

alpha = 1.0 / avg_bpm
beta = 1.0 / avg_steps
tot = alpha + beta
alpha /= tot
beta /= tot

print(f"Weight for BPM loss: {alpha:.4f}")
print(f"Weight for Steps loss: {beta:.4f}")

# ============================
# 7. Model Definition
# ============================

class ForecastingModel(nn.Module):
    """
    Multi-step forecasting model that:
      - Encodes past BPM & Steps via CNN + LSTM.
      - For each predicted window, uses the FULL array [WINDOW_SIZE] for 
        "current" BPM or Steps (rather than a single average).
      - Aggregates that array (e.g., via a small linear or CNN) before fusion.
    """
    def __init__(self):
        super(ForecastingModel, self).__init__()
        
        # CNN + LSTM for BPM (past)
        self.bpm_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.bpm_lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        
        # CNN + LSTM for Steps (past)
        self.steps_cnn = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.steps_lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        
        # We embed each "current" window of size [WINDOW_SIZE]
        self.agg_current_steps = nn.Sequential(
            nn.Linear(WINDOW_SIZE, 16),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.agg_current_bpm = nn.Sequential(
            nn.Linear(WINDOW_SIZE, 16),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # For final fusion
        self.fusion_bpm = nn.Sequential(
            nn.Linear(256 + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, WINDOW_SIZE)  # => predict 6 time steps
        )
        self.fusion_steps = nn.Sequential(
            nn.Linear(256 + 16, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, WINDOW_SIZE)
        )
    
    def forward(self, bpm_input, steps_input, curr_bpm_windows, curr_steps_windows):
        """
        :param bpm_input:      [B, INPUT_WINDOWS, WINDOW_SIZE]
        :param steps_input:    [B, INPUT_WINDOWS, WINDOW_SIZE]
        :param curr_bpm_windows:   [B, PREDICT_WINDOWS, WINDOW_SIZE]
        :param curr_steps_windows: [B, PREDICT_WINDOWS, WINDOW_SIZE]
        
        Output shape => [B, PREDICT_WINDOWS, WINDOW_SIZE] each (for BPM, Steps).
        """
        B = bpm_input.size(0)
        
        # Past BPM => flatten => CNN => LSTM
        bpm_seq = bpm_input.view(B, -1).unsqueeze(1)   # [B, 1, INPUT_WINDOWS*WINDOW_SIZE]
        bpm_cnn_out = self.bpm_cnn(bpm_seq)            # [B, 64, ...]
        bpm_cnn_out = bpm_cnn_out.permute(0, 2, 1)     # [B, ..., 64]
        bpm_lstm_out, _ = self.bpm_lstm(bpm_cnn_out)   # [B, ..., 128]
        bpm_hidden = bpm_lstm_out[:, -1, :]            # [B, 128]
        
        # Past Steps => flatten => CNN => LSTM
        steps_seq = steps_input.view(B, -1).unsqueeze(1) 
        steps_cnn_out = self.steps_cnn(steps_seq)         
        steps_cnn_out = steps_cnn_out.permute(0, 2, 1)    
        steps_lstm_out, _ = self.steps_lstm(steps_cnn_out)
        steps_hidden = steps_lstm_out[:, -1, :]           
        
        # Combine hidden states
        past_features = torch.cat([bpm_hidden, steps_hidden], dim=1)  # [B,256]
        
        # We'll produce PREDICT_WINDOWS outputs, each of size WINDOW_SIZE
        bpm_out_list = []
        steps_out_list = []
        
        for w_idx in range(curr_bpm_windows.size(1)):  # PREDICT_WINDOWS
            # Gather the "current" window [B, WINDOW_SIZE]
            curr_bpm_1win = curr_bpm_windows[:, w_idx, :]
            curr_steps_1win = curr_steps_windows[:, w_idx, :]
            
            # embed them
            curr_bpm_emb = self.agg_current_bpm(curr_bpm_1win)       # [B,16]
            curr_steps_emb = self.agg_current_steps(curr_steps_1win) # [B,16]
            
            # BPM => fuse with steps' embedding
            bpm_fusion_in = torch.cat([past_features, curr_steps_emb], dim=1)  
            bpm_pred_1win = self.fusion_bpm(bpm_fusion_in)  # [B,WINDOW_SIZE]
            
            # Steps => fuse with bpm's embedding
            steps_fusion_in = torch.cat([past_features, curr_bpm_emb], dim=1)
            steps_pred_1win = self.fusion_steps(steps_fusion_in)  # [B,WINDOW_SIZE]
            
            bpm_out_list.append(bpm_pred_1win.unsqueeze(1))     
            steps_out_list.append(steps_pred_1win.unsqueeze(1))
        
        # Concatenate across predicted windows
        bpm_pred_final = torch.cat(bpm_out_list, dim=1)    # [B,PREDICT_WINDOWS,WINDOW_SIZE]
        steps_pred_final = torch.cat(steps_out_list, dim=1)# [B,PREDICT_WINDOWS,WINDOW_SIZE]
        
        return bpm_pred_final, steps_pred_final

# Check GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"Using device: {device}")
model = ForecastingModel().to(device)

# ============================
# 8. Training Setup
# ============================

# --- CHANGED HERE: use MSE loss instead of L1Loss ---
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
model_save_path = os.path.join(save_dir, 'forecasting_backbone.pth')

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
    
    # Training
    model.train()
    epoch_loss = 0.0
    
    for (bpm_input, steps_input,
         curr_bpm_scaled, curr_steps_scaled,
         bpm_target, steps_target,
         user_ids,
         bpm_target_original, steps_target_original,
         datetimes) in train_loader:
        
        bpm_input = bpm_input.to(device)
        steps_input = steps_input.to(device)
        curr_bpm_scaled = curr_bpm_scaled.to(device)    # [B,PREDICT_WINDOWS,WINDOW_SIZE]
        curr_steps_scaled = curr_steps_scaled.to(device)# [B,PREDICT_WINDOWS,WINDOW_SIZE]
        
        bpm_target = bpm_target.to(device)      # [B,PREDICT_WINDOWS,WINDOW_SIZE]
        steps_target = steps_target.to(device)
        
        optimizer.zero_grad()
        bpm_pred, steps_pred = model(bpm_input, steps_input,
                                     curr_bpm_scaled, curr_steps_scaled)
        
        # Weighted MSE in scaled space
        loss_bpm = criterion(bpm_pred, bpm_target)
        loss_steps = criterion(steps_pred, steps_target)
        loss = alpha * loss_bpm + beta * loss_steps
        
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        epoch_loss += loss.item()
    
    scheduler.step()
    avg_train_loss = epoch_loss / len(train_loader)
    
    # Validation
    model.eval()
    val_loss = 0.0
    total_bpm_error = 0.0
    total_steps_error = 0.0
    validation_errors_epoch = []
    
    with torch.no_grad():
        for (bpm_input, steps_input,
             curr_bpm_scaled, curr_steps_scaled,
             bpm_target, steps_target,
             user_ids,
             bpm_target_original, steps_target_original,
             datetimes) in val_loader:
            
            bpm_input = bpm_input.to(device)
            steps_input = steps_input.to(device)
            curr_bpm_scaled = curr_bpm_scaled.to(device)
            curr_steps_scaled = curr_steps_scaled.to(device)
            bpm_target = bpm_target.to(device)
            steps_target = steps_target.to(device)
            
            bpm_pred_scaled, steps_pred_scaled = model(bpm_input, steps_input,
                                                       curr_bpm_scaled, curr_steps_scaled)
            
            # Validation loss in scaled space
            loss_bpm = criterion(bpm_pred_scaled, bpm_target)
            loss_steps = criterion(steps_pred_scaled, steps_target)
            loss = alpha * loss_bpm + beta * loss_steps
            val_loss += loss.item()
            
            # For error in original scale (MAE in BPM/Steps)
            bpm_pred_np = bpm_pred_scaled.cpu().numpy()       # [B,PREDICT_WINDOWS,WINDOW_SIZE]
            steps_pred_np = steps_pred_scaled.cpu().numpy()
            bpm_target_np = bpm_target.cpu().numpy()
            steps_target_np = steps_target.cpu().numpy()
            
            for i in range(len(user_ids)):
                uid = user_ids[i]
                
                bpm_pred_2d = bpm_pred_np[i]     # shape [PREDICT_WINDOWS,WINDOW_SIZE]
                steps_pred_2d = steps_pred_np[i]
                
                bpm_true_2d = bpm_target_np[i]
                steps_true_2d = steps_target_np[i]
                
                # Flatten to do inverse transform
                bpm_pred_1d = bpm_pred_2d.flatten()
                steps_pred_1d = steps_pred_2d.flatten()
                bpm_true_1d = bpm_true_2d.flatten()
                steps_true_1d = steps_true_2d.flatten()
                
                # Inverse
                bpm_pred_unscaled_1d, steps_pred_unscaled_1d = inverse_transform(
                    uid, bpm_pred_1d, steps_pred_1d
                )
                bpm_true_unscaled_1d, steps_true_unscaled_1d = inverse_transform(
                    uid, bpm_true_1d, steps_true_1d
                )
                
                # MAE in original units
                bpm_error = np.mean(np.abs(bpm_pred_unscaled_1d - bpm_true_unscaled_1d))
                steps_error = np.mean(np.abs(steps_pred_unscaled_1d - steps_true_unscaled_1d))
                total_bpm_error += bpm_error
                total_steps_error += steps_error
                
                validation_errors_epoch.append({
                    'user_id': uid,
                    'bpm_error': bpm_error,
                    'steps_error': steps_error,
                    'bpm_pred': bpm_pred_unscaled_1d.reshape(PREDICT_WINDOWS, WINDOW_SIZE),
                    'steps_pred': steps_pred_unscaled_1d.reshape(PREDICT_WINDOWS, WINDOW_SIZE),
                    'bpm_true': bpm_true_unscaled_1d.reshape(PREDICT_WINDOWS, WINDOW_SIZE),
                    'steps_true': steps_true_unscaled_1d.reshape(PREDICT_WINDOWS, WINDOW_SIZE),
                    'datetime': datetimes[i]
                })
    
    avg_val_loss = val_loss / len(val_loader)
    avg_bpm_error = total_bpm_error / len(val_loader.dataset)
    avg_steps_error = total_steps_error / len(val_loader.dataset)
    
    stats['epoch'].append(epoch+1)
    stats['train_loss'].append(avg_train_loss)
    stats['val_loss'].append(avg_val_loss)
    stats['avg_bpm_error'].append(avg_bpm_error)
    stats['avg_steps_error'].append(avg_steps_error)
    
    validation_errors.extend(validation_errors_epoch)
    
    # Early stopping
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), model_save_path)
        print(f"Epoch {epoch+1}: Validation loss improved. Model saved.")
    else:
        epochs_no_improve += 1
        print(f"Epoch {epoch+1}: No improvement in validation loss.")
    
    if epochs_no_improve >= patience:
        print(f"Early stopping triggered after {patience} epochs with no improvement.")
        early_stop = True
    
    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"BPM Error: {avg_bpm_error:.2f}, "
          f"Steps Error: {avg_steps_error:.2f}")

# ============================
# 9. Save Training Statistics
# ============================
results_dir = 'results/train'
os.makedirs(results_dir, exist_ok=True)

stats_df = pd.DataFrame(stats)
stats_df.to_csv(os.path.join(results_dir, "stats.csv"), index=False)

# ============================
# 10. Generate Charts
# ============================
plt.figure(figsize=(10, 6))
plt.plot(stats_df['epoch'], stats_df['train_loss'], label='Train Loss')
plt.plot(stats_df['epoch'], stats_df['val_loss'], label='Val Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch (MSE)')
plt.legend()
plt.savefig(os.path.join(results_dir, 'loss_per_epoch.png'))
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(stats_df['epoch'], stats_df['avg_bpm_error'], label='BPM Error')
plt.xlabel('Epoch')
plt.ylabel('MAE (BPM)')
plt.title('BPM Error per Epoch')
plt.legend()
plt.savefig(os.path.join(results_dir, 'bpm_error_per_epoch.png'))
plt.show()

plt.figure(figsize=(10, 6))
plt.plot(stats_df['epoch'], stats_df['avg_steps_error'], label='Steps Error')
plt.xlabel('Epoch')
plt.ylabel('MAE (Steps)')
plt.title('Steps Error per Epoch')
plt.legend()
plt.savefig(os.path.join(results_dir, 'steps_error_per_epoch.png'))
plt.show()

print(f"Training complete. Figures saved in '{results_dir}'.")

# ============================
# 11. Analyze Validation Errors
# ============================
validation_errors_df = pd.DataFrame(validation_errors)
validation_errors_df['total_error'] = validation_errors_df['bpm_error'] + validation_errors_df['steps_error']

user_error_df = validation_errors_df.groupby('user_id')['total_error'].mean().reset_index()
user_error_df_sorted = user_error_df.sort_values(by='total_error', ascending=True)

top_N = 10
best_users = user_error_df_sorted.head(top_N)
worst_users = user_error_df_sorted.tail(top_N)

print(f"Best Users:\n{best_users}\n")
print(f"Worst Users:\n{worst_users}\n")

analysis_dir = 'results/train/analysis'
os.makedirs(analysis_dir, exist_ok=True)

def plot_prediction_user(user_id, user_samples, plot_type, PREDICT_WINDOWS, WINDOW_SIZE):
    """
    Plots the true vs predicted BPM and Steps for a user when predicting multiple future windows.
    Each window is fully included in the "current" features.
    """
    for i, sample in enumerate(user_samples):
        bpm_pred_2d = sample['bpm_pred']
        steps_pred_2d = sample['steps_pred']
        bpm_true_2d = sample['bpm_true']
        steps_true_2d = sample['steps_true']
        datetime = sample['datetime']
        
        bpm_pred_1d = bpm_pred_2d.flatten()
        steps_pred_1d = steps_pred_2d.flatten()
        bpm_true_1d = bpm_true_2d.flatten()
        steps_true_1d = steps_true_2d.flatten()
        
        total_time = len(bpm_pred_1d)  # = PREDICT_WINDOWS * WINDOW_SIZE
        
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'User: {user_id} | Sample {i+1} | DateTime: {datetime}', fontsize=16)
        
        # BPM
        axs[0,0].plot(range(total_time), bpm_true_1d, label='True BPM', marker='o')
        axs[0,0].plot(range(total_time), bpm_pred_1d, label='Predicted BPM', marker='x')
        axs[0,0].set_title('BPM Prediction vs True')
        axs[0,0].legend()
        
        # Steps
        axs[0,1].plot(range(total_time), steps_true_1d, label='True Steps', marker='o')
        axs[0,1].plot(range(total_time), steps_pred_1d, label='Pred Steps', marker='x', color='orange')
        axs[0,1].set_title('Steps Prediction vs True')
        axs[0,1].legend()
        
        # BPM error
        axs[1,0].bar(range(total_time), np.abs(bpm_true_1d - bpm_pred_1d))
        axs[1,0].set_title('BPM Absolute Error')
        
        # Steps error
        axs[1,1].bar(range(total_time), np.abs(steps_true_1d - steps_pred_1d), color='orange')
        axs[1,1].set_title('Steps Absolute Error')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        fname = f"{plot_type}_user_{user_id}_sample_{i+1}.png"
        plt.savefig(os.path.join(analysis_dir, fname))
        plt.close()
        
        print(f"Saved plot for user={user_id}, sample={i+1} at {fname}")

def select_samples_for_plotting(validation_errors_df, user_id, num_samples=2):
    user_samples = validation_errors_df[validation_errors_df['user_id'] == user_id].to_dict(orient='records')
    return user_samples[:num_samples]

for idx, row in best_users.iterrows():
    uid = row['user_id']
    samples_to_plot = select_samples_for_plotting(validation_errors_df, uid, num_samples=2)
    plot_prediction_user(uid, samples_to_plot, "best", PREDICT_WINDOWS, WINDOW_SIZE)

for idx, row in worst_users.iterrows():
    uid = row['user_id']
    samples_to_plot = select_samples_for_plotting(validation_errors_df, uid, num_samples=2)
    plot_prediction_user(uid, samples_to_plot, "worst", PREDICT_WINDOWS, WINDOW_SIZE)
