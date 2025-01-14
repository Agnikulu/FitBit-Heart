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
    Reverts scaled BPM and Steps data back to original scale using stored scaler parameters.
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
      - For each future window, stores its own average BPM and Steps as "current" features.
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
            input_windows = windows[i : i+INPUT_WINDOWS]                     # Past windows
            target_windows = windows[i+INPUT_WINDOWS : i+INPUT_WINDOWS+PREDICT_WINDOWS]  # Future windows
            
            # Gather scaled BPM and Steps for input
            bpm_input_2d = np.array([w['bpm_scaled'].values for w in input_windows])   # Shape: [INPUT_WINDOWS, WINDOW_SIZE]
            steps_input_2d = np.array([w['steps_scaled'].values for w in input_windows])
            
            # Gather scaled and original BPM and Steps for targets
            bpm_target_2d = np.array([tw['bpm_scaled'].values for tw in target_windows])         # Shape: [PREDICT_WINDOWS, WINDOW_SIZE]
            steps_target_2d = np.array([tw['steps_scaled'].values for tw in target_windows])
            bpm_target_original_2d = np.array([tw['bpm'].values for tw in target_windows])         # Shape: [PREDICT_WINDOWS, WINDOW_SIZE]
            steps_target_original_2d = np.array([tw['steps'].values for tw in target_windows])
            
            # Calculate "current" BPM and Steps for each target window (separately)
            current_bpm_scaleds = np.array([tw['bpm_scaled'].mean() for tw in target_windows])    # Shape: [PREDICT_WINDOWS]
            current_steps_scaleds = np.array([tw['steps_scaled'].mean() for tw in target_windows])  # Shape: [PREDICT_WINDOWS]
    
            # Store the start datetime of the first target window
            datetime_val = target_windows[0]['datetime'].values[0]
    
            data_samples.append({
                'bpm_input': bpm_input_2d,                        # [INPUT_WINDOWS, WINDOW_SIZE]
                'steps_input': steps_input_2d,                    # [INPUT_WINDOWS, WINDOW_SIZE]
                'bpm_target': bpm_target_2d,                      # [PREDICT_WINDOWS, WINDOW_SIZE]
                'steps_target': steps_target_2d,                  # [PREDICT_WINDOWS, WINDOW_SIZE]
                'bpm_target_original': bpm_target_original_2d,     # [PREDICT_WINDOWS, WINDOW_SIZE]
                'steps_target_original': steps_target_original_2d, # [PREDICT_WINDOWS, WINDOW_SIZE]
                'current_bpm_scaled': current_bpm_scaleds,         # [PREDICT_WINDOWS]
                'current_steps_scaled': current_steps_scaleds,     # [PREDICT_WINDOWS]
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

train_samples = [sample for sample in data_samples if sample['user_id'] in train_user_ids]
val_samples = [sample for sample in data_samples if sample['user_id'] in val_user_ids]

# ============================
# 6. Sampler and DataLoader Setup
# ============================

def create_user_datasets(data_samples):
    """
    Organizes data samples by user for the sampler.
    """
    user_data = defaultdict(list)
    for idx, sample in enumerate(data_samples):
        user_data[sample['user_id']].append(idx)
    return user_data

train_user_data = create_user_datasets(train_samples)
val_user_data = create_user_datasets(val_samples)

class PerUserBatchSampler(Sampler):
    """
    Custom sampler to yield batches of samples per user.
    """
    def __init__(self, user_data, batch_size):
        self.user_data = user_data
        self.user_ids = list(user_data.keys())
        self.batch_size = batch_size

    def __iter__(self):
        np.random.shuffle(self.user_ids)  # Shuffle users each epoch
        for user_id in self.user_ids:
            user_samples = self.user_data[user_id]
            np.random.shuffle(user_samples)  # Shuffle samples within the user
            for i in range(0, len(user_samples), self.batch_size):
                yield user_samples[i:i + self.batch_size]

    def __len__(self):
        return sum(
            len(indices) // self.batch_size + (1 if len(indices) % self.batch_size != 0 else 0)
            for indices in self.user_data.values()
        )

class UserDataset(Dataset):
    """
    PyTorch Dataset for user-specific time series data.
    Each item returns:
        - Past windows of scaled BPM/Steps
        - "Current" scaled BPM/Steps for each target window
        - Target windows of scaled BPM/Steps
        - Unscaled target BPM/Steps for analysis
        - user_id
        - datetime
    """
    def __init__(self, data_list):
        self.data = data_list
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        bpm_input = torch.tensor(sample['bpm_input'], dtype=torch.float32)        # [INPUT_WINDOWS, WINDOW_SIZE]
        steps_input = torch.tensor(sample['steps_input'], dtype=torch.float32)    # [INPUT_WINDOWS, WINDOW_SIZE]
        
        # "Current" BPM/Steps => Each target window's average
        current_bpm_scaled = torch.tensor(sample['current_bpm_scaled'], dtype=torch.float32)  # [PREDICT_WINDOWS]
        current_steps_scaled = torch.tensor(sample['current_steps_scaled'], dtype=torch.float32)  # [PREDICT_WINDOWS]
        
        bpm_target = torch.tensor(sample['bpm_target'], dtype=torch.float32)            # [PREDICT_WINDOWS, WINDOW_SIZE]
        steps_target = torch.tensor(sample['steps_target'], dtype=torch.float32)        # [PREDICT_WINDOWS, WINDOW_SIZE]
        
        bpm_target_original = torch.tensor(sample['bpm_target_original'], dtype=torch.float32)  # [PREDICT_WINDOWS, WINDOW_SIZE]
        steps_target_original = torch.tensor(sample['steps_target_original'], dtype=torch.float32)  # [PREDICT_WINDOWS, WINDOW_SIZE]
        
        user_id = sample['user_id']
        datetime_val = sample['datetime']
        
        return (bpm_input, steps_input,
                current_bpm_scaled, current_steps_scaled,
                bpm_target, steps_target,
                user_id,
                bpm_target_original, steps_target_original,
                datetime_val)

# Initialize datasets
train_dataset = UserDataset(train_samples)
val_dataset = UserDataset(val_samples)

# Initialize samplers
train_sampler = PerUserBatchSampler(train_user_data, BATCH_SIZE)
val_sampler = PerUserBatchSampler(val_user_data, BATCH_SIZE)

def collate_fn(batch):
    """
    Custom collate function to handle batching of samples.
    """
    bpm_inputs = torch.stack([s[0] for s in batch])            # [B, INPUT_WINDOWS, WINDOW_SIZE]
    steps_inputs = torch.stack([s[1] for s in batch])          # [B, INPUT_WINDOWS, WINDOW_SIZE]
    current_bpm_scaled = torch.stack([s[2] for s in batch])    # [B, PREDICT_WINDOWS]
    current_steps_scaled = torch.stack([s[3] for s in batch])  # [B, PREDICT_WINDOWS]
    
    bpm_targets = torch.stack([s[4] for s in batch])           # [B, PREDICT_WINDOWS, WINDOW_SIZE]
    steps_targets = torch.stack([s[5] for s in batch])         # [B, PREDICT_WINDOWS, WINDOW_SIZE]
    
    user_ids = [s[6] for s in batch]
    bpm_targets_original = [s[7] for s in batch]               # List of [PREDICT_WINDOWS, WINDOW_SIZE] tensors
    steps_targets_original = [s[8] for s in batch]             # List of [PREDICT_WINDOWS, WINDOW_SIZE] tensors
    datetimes = [s[9] for s in batch]
    
    return (bpm_inputs, steps_inputs,
            current_bpm_scaled, current_steps_scaled,
            bpm_targets, steps_targets,
            user_ids,
            bpm_targets_original, steps_targets_original,
            datetimes)

# Initialize DataLoaders
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=collate_fn)

# ============================
# 6A. Compute Weights Based on Average Size
# ============================

avg_bpm = df[df['id'].isin(train_user_ids)]['bpm'].abs().mean()
avg_steps = df[df['id'].isin(train_user_ids)]['steps'].abs().mean()

alpha = 1.0 / avg_bpm
beta = 1.0 / avg_steps

# Normalize weights so that alpha + beta = 1
total = alpha + beta
alpha /= total
beta /= total

print(f"Weight for BPM loss: {alpha:.4f}")
print(f"Weight for Steps loss: {beta:.4f}")

# ============================
# 7. Model Definition
# ============================

class ForecastingModel(nn.Module):
    """
    Forecasting model with:
      - Separate CNN + LSTM encoders for BPM and Steps (past data).
      - Embeddings for "current" BPM and Steps for each predicted window.
      - Separate fusion and decoding layers for BPM and Steps predictions.
      - Outputs multi-step predictions for BPM and Steps.
    """
    def __init__(self):
        super(ForecastingModel, self).__init__()
        
        # CNN Encoder for BPM
        self.bpm_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # LSTM Encoder for BPM
        self.bpm_lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        
        # CNN Encoder for Steps
        self.steps_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        # LSTM Encoder for Steps
        self.steps_lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        
        # Embeddings for "current" features
        self.current_steps_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        self.current_bpm_fc = nn.Sequential(
            nn.Linear(1, 16),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Fusion layers for BPM and Steps predictions
        self.fusion_bpm = nn.Sequential(
            nn.Linear(256 + 16, 128),  # 256 from past features + 16 from current_steps
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, WINDOW_SIZE)  # Predict WINDOW_SIZE steps for BPM
        )
        self.fusion_steps = nn.Sequential(
            nn.Linear(256 + 16, 128),  # 256 from past features + 16 from current_bpm
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(128, WINDOW_SIZE)  # Predict WINDOW_SIZE steps for Steps
        )

    def forward(self, bpm_input, steps_input, current_bpm_scaled, current_steps_scaled):
        """
        Forward pass for the forecasting model.
        
        :param bpm_input:          [B, INPUT_WINDOWS, WINDOW_SIZE]
        :param steps_input:        [B, INPUT_WINDOWS, WINDOW_SIZE]
        :param current_bpm_scaled: [B, PREDICT_WINDOWS]
        :param current_steps_scaled: [B, PREDICT_WINDOWS]
        
        :return:
            - bpm_pred:    [B, PREDICT_WINDOWS, WINDOW_SIZE]
            - steps_pred:  [B, PREDICT_WINDOWS, WINDOW_SIZE]
        """
        B = bpm_input.size(0)
        
        # Process BPM Input
        bpm_seq = bpm_input.view(B, -1).unsqueeze(1)  # [B, 1, INPUT_WINDOWS * WINDOW_SIZE]
        bpm_cnn_out = self.bpm_cnn(bpm_seq)           # [B, 64, INPUT_WINDOWS * WINDOW_SIZE]
        bpm_cnn_out = bpm_cnn_out.permute(0, 2, 1)    # [B, INPUT_WINDOWS * WINDOW_SIZE, 64]
        bpm_lstm_out, _ = self.bpm_lstm(bpm_cnn_out)  # [B, INPUT_WINDOWS * WINDOW_SIZE, 128]
        bpm_hidden = bpm_lstm_out[:, -1, :]           # [B, 128]
        
        # Process Steps Input
        steps_seq = steps_input.view(B, -1).unsqueeze(1)  # [B, 1, INPUT_WINDOWS * WINDOW_SIZE]
        steps_cnn_out = self.steps_cnn(steps_seq)         # [B, 64, INPUT_WINDOWS * WINDOW_SIZE]
        steps_cnn_out = steps_cnn_out.permute(0, 2, 1)    # [B, INPUT_WINDOWS * WINDOW_SIZE, 64]
        steps_lstm_out, _ = self.steps_lstm(steps_cnn_out)# [B, INPUT_WINDOWS * WINDOW_SIZE, 128]
        steps_hidden = steps_lstm_out[:, -1, :]           # [B, 128]
        
        # Combine BPM and Steps hidden states
        past_features = torch.cat([bpm_hidden, steps_hidden], dim=1)  # [B, 256]
        
        # Initialize lists to collect predictions
        bpm_preds = []
        steps_preds = []
        
        # Iterate over each predicted window
        for w_idx in range(PREDICT_WINDOWS):
            # Extract current_steps for BPM prediction
            curr_steps = current_steps_scaled[:, w_idx].unsqueeze(-1)  # [B, 1]
            curr_steps_emb = self.current_steps_fc(curr_steps)        # [B, 16]
            
            # Fuse past features with current_steps_emb for BPM prediction
            bpm_fusion_input = torch.cat([past_features, curr_steps_emb], dim=1)  # [B, 256 + 16]
            bpm_pred = self.fusion_bpm(bpm_fusion_input)                          # [B, WINDOW_SIZE]
            bpm_preds.append(bpm_pred.unsqueeze(1))                                # [B, 1, WINDOW_SIZE]
            
            # Extract current_bpm for Steps prediction
            curr_bpm = current_bpm_scaled[:, w_idx].unsqueeze(-1)  # [B, 1]
            curr_bpm_emb = self.current_bpm_fc(curr_bpm)           # [B, 16]
            
            # Fuse past features with current_bpm_emb for Steps prediction
            steps_fusion_input = torch.cat([past_features, curr_bpm_emb], dim=1)  # [B, 256 + 16]
            steps_pred = self.fusion_steps(steps_fusion_input)                    # [B, WINDOW_SIZE]
            steps_preds.append(steps_pred.unsqueeze(1))                            # [B, 1, WINDOW_SIZE]
        
        # Concatenate predictions for all windows
        bpm_pred_final = torch.cat(bpm_preds, dim=1)    # [B, PREDICT_WINDOWS, WINDOW_SIZE]
        steps_pred_final = torch.cat(steps_preds, dim=1)# [B, PREDICT_WINDOWS, WINDOW_SIZE]
        
        return bpm_pred_final, steps_pred_final

# Initialize model
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

model = ForecastingModel().to(device)

# ============================
# 8. Training Setup
# ============================

criterion = nn.L1Loss()  # Using Mean Absolute Error (MAE)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

num_epochs = 100  # Adjust as needed

# Early Stopping parameters
patience = 15
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False

# Directory to save models
save_dir = 'results/saved_models'
os.makedirs(save_dir, exist_ok=True)
model_save_path = os.path.join(save_dir, 'forecasting_multistep.pth')

# To track training progress
stats = {
    'epoch': [],
    'train_loss': [],
    'val_loss': [],
    'avg_bpm_error': [],
    'avg_steps_error': []
}

# To store validation errors for analysis
validation_errors = []

for epoch in range(num_epochs):
    if early_stop:
        break
    
    # ---------------------
    # Training Phase
    # ---------------------
    model.train()
    epoch_loss = 0.0
    for (bpm_input, steps_input,
         current_bpm_scaled, current_steps_scaled,
         bpm_target, steps_target,
         user_ids,
         bpm_target_original, steps_target_original,
         datetimes) in train_loader:
        
        bpm_input = bpm_input.to(device)            # [B, INPUT_WINDOWS, WINDOW_SIZE]
        steps_input = steps_input.to(device)        # [B, INPUT_WINDOWS, WINDOW_SIZE]
        current_bpm_scaled = current_bpm_scaled.to(device)    # [B, PREDICT_WINDOWS]
        current_steps_scaled = current_steps_scaled.to(device)# [B, PREDICT_WINDOWS]
        bpm_target = bpm_target.to(device)          # [B, PREDICT_WINDOWS, WINDOW_SIZE]
        steps_target = steps_target.to(device)      # [B, PREDICT_WINDOWS, WINDOW_SIZE]
        
        optimizer.zero_grad()
        bpm_pred, steps_pred = model(
            bpm_input, steps_input,
            current_bpm_scaled, current_steps_scaled
        )
        
        # Compute losses
        loss_bpm = criterion(bpm_pred, bpm_target)       # [B, PREDICT_WINDOWS, WINDOW_SIZE]
        loss_steps = criterion(steps_pred, steps_target)   # [B, PREDICT_WINDOWS, WINDOW_SIZE]
        loss = alpha * loss_bpm + beta * loss_steps
        
        # Backpropagation
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    scheduler.step()
    
    # ---------------------
    # Validation Phase
    # ---------------------
    model.eval()
    val_loss = 0.0
    total_bpm_error = 0.0
    total_steps_error = 0.0
    validation_errors_epoch = []
    
    with torch.no_grad():
        for (bpm_input, steps_input,
             current_bpm_scaled, current_steps_scaled,
             bpm_target, steps_target,
             user_ids,
             bpm_target_original, steps_target_original,
             datetimes) in val_loader:
            
            bpm_input = bpm_input.to(device)
            steps_input = steps_input.to(device)
            current_bpm_scaled = current_bpm_scaled.to(device)
            current_steps_scaled = current_steps_scaled.to(device)
            bpm_target = bpm_target.to(device)
            steps_target = steps_target.to(device)
            
            bpm_pred_scaled, steps_pred_scaled = model(
                bpm_input, steps_input,
                current_bpm_scaled, current_steps_scaled
            )
            
            # Compute scaled loss
            loss_bpm = criterion(bpm_pred_scaled, bpm_target)
            loss_steps = criterion(steps_pred_scaled, steps_target)
            loss = alpha * loss_bpm + beta * loss_steps
            val_loss += loss.item()
            
            # Inverse transform predictions and targets for MAE computation
            bpm_pred_scaled_np = bpm_pred_scaled.cpu().numpy()       # [B, PREDICT_WINDOWS, WINDOW_SIZE]
            steps_pred_scaled_np = steps_pred_scaled.cpu().numpy()   # [B, PREDICT_WINDOWS, WINDOW_SIZE]
            bpm_target_np = bpm_target.cpu().numpy()                 # [B, PREDICT_WINDOWS, WINDOW_SIZE]
            steps_target_np = steps_target.cpu().numpy()             # [B, PREDICT_WINDOWS, WINDOW_SIZE]
            
            for i in range(len(user_ids)):
                uid = user_ids[i]
                
                # Predictions and targets for this sample
                bpm_pred_2d = bpm_pred_scaled_np[i]   # [PREDICT_WINDOWS, WINDOW_SIZE]
                steps_pred_2d = steps_pred_scaled_np[i] # [PREDICT_WINDOWS, WINDOW_SIZE]
                
                bpm_true_2d = bpm_target_np[i]        # [PREDICT_WINDOWS, WINDOW_SIZE]
                steps_true_2d = steps_target_np[i]    # [PREDICT_WINDOWS, WINDOW_SIZE]
                
                # Inverse transform each window
                bpm_pred_1d = bpm_pred_2d.flatten()   # [PREDICT_WINDOWS * WINDOW_SIZE]
                steps_pred_1d = steps_pred_2d.flatten()
                bpm_true_1d = bpm_true_2d.flatten()
                steps_true_1d = steps_true_2d.flatten()
                
                bpm_pred_unscaled_1d, steps_pred_unscaled_1d = inverse_transform(
                    uid, bpm_pred_1d, steps_pred_1d
                )
                bpm_true_unscaled_1d, steps_true_unscaled_1d = inverse_transform(
                    uid, bpm_true_1d, steps_true_1d
                )
                
                # Compute MAE over all predicted steps
                bpm_error = np.mean(np.abs(bpm_pred_unscaled_1d - bpm_true_unscaled_1d))
                steps_error = np.mean(np.abs(steps_pred_unscaled_1d - steps_true_unscaled_1d))
                total_bpm_error += bpm_error
                total_steps_error += steps_error
                
                # Store for analysis
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
    
    # Calculate average losses and errors
    avg_train_loss = epoch_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    avg_bpm_error = total_bpm_error / len(val_loader.dataset)
    avg_steps_error = total_steps_error / len(val_loader.dataset)
    
    # Update stats
    stats['epoch'].append(epoch + 1)
    stats['train_loss'].append(avg_train_loss)
    stats['val_loss'].append(avg_val_loss)
    stats['avg_bpm_error'].append(avg_bpm_error)
    stats['avg_steps_error'].append(avg_steps_error)
    
    # Append validation errors
    validation_errors.extend(validation_errors_epoch)
    
    # Early Stopping Check
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), model_save_path)
        print(f"Epoch {epoch+1}: Validation loss improved. Model saved.")
    else:
        epochs_no_improve += 1
        print(f"Epoch {epoch+1}: No improvement in validation loss.")
    
    if epochs_no_improve >= patience:
        print(f'Early stopping triggered after {patience} epochs with no improvement.')
        early_stop = True
    
    # Print epoch statistics
    print(f"Epoch {epoch+1}/{num_epochs}, "
          f"Train Loss: {avg_train_loss:.4f}, "
          f"Val Loss: {avg_val_loss:.4f}, "
          f"Avg BPM Error: {avg_bpm_error:.2f}, "
          f"Avg Steps Error: {avg_steps_error:.2f}")

# ============================
# 9. Save Training Statistics
# ============================

results_dir = 'results/train'
os.makedirs(results_dir, exist_ok=True)

# Convert stats to DataFrame and save
stats_df = pd.DataFrame(stats)
stats_df.to_csv(os.path.join(results_dir, "stats.csv"), index=False)

# ============================
# 10. Generate Charts
# ============================

# Plot Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(stats_df['epoch'], stats_df['train_loss'], label='Train Loss')
plt.plot(stats_df['epoch'], stats_df['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.savefig(os.path.join(results_dir, 'loss_per_epoch.png'))
plt.show()

# Plot BPM Error
plt.figure(figsize=(10, 6))
plt.plot(stats_df['epoch'], stats_df['avg_bpm_error'], label='BPM Error')
plt.xlabel('Epoch')
plt.ylabel('MAE (BPM)')
plt.title('BPM Error per Epoch')
plt.legend()
plt.savefig(os.path.join(results_dir, 'bpm_error_per_epoch.png'))
plt.show()

# Plot Steps Error
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

# Convert validation_errors to a DataFrame
validation_errors_df = pd.DataFrame(validation_errors)

# Calculate total error (BPM + Steps)
validation_errors_df['total_error'] = validation_errors_df['bpm_error'] + validation_errors_df['steps_error']

# Aggregate errors per user by computing the mean total error
user_error_df = validation_errors_df.groupby('user_id')['total_error'].mean().reset_index()

# Sort users by total_error
user_error_df_sorted = user_error_df.sort_values(by='total_error', ascending=True)

# Select top N best and worst users
top_N = 10
best_users = user_error_df_sorted.head(top_N)
worst_users = user_error_df_sorted.tail(top_N)

print(f"Best Users:\n{best_users}\n")
print(f"Worst Users:\n{worst_users}\n")

# Create a directory to save the analysis plots
analysis_dir = 'results/train/analysis'
os.makedirs(analysis_dir, exist_ok=True)

def plot_prediction_user(user_id, user_samples, plot_type, PREDICT_WINDOWS, WINDOW_SIZE):
    """
    Plots the true vs predicted BPM and Steps for a user when predicting multiple future windows.
    
    Parameters:
        user_id (str): The ID of the user.
        user_samples (list): List of samples (dicts) for the user.
        plot_type (str): 'best' or 'worst' to categorize the plots.
        PREDICT_WINDOWS (int): Number of windows to predict.
        WINDOW_SIZE (int): Number of time steps in each window.
    """
    for i, sample in enumerate(user_samples):
        bpm_pred_2d = sample['bpm_pred']       # [PREDICT_WINDOWS, WINDOW_SIZE]
        steps_pred_2d = sample['steps_pred']   # [PREDICT_WINDOWS, WINDOW_SIZE]
        bpm_true_2d = sample['bpm_true']       # [PREDICT_WINDOWS, WINDOW_SIZE]
        steps_true_2d = sample['steps_true']   # [PREDICT_WINDOWS, WINDOW_SIZE]
        datetime = sample['datetime']
        
        # Debugging: Check shapes
        print(f"Debug: user_id={user_id}, sample={i+1}, "
              f"bpm_pred_2d.shape={bpm_pred_2d.shape} (expected: ({PREDICT_WINDOWS}, {WINDOW_SIZE})))")
        
        # Flatten for continuous plotting
        bpm_pred_1d = bpm_pred_2d.flatten()     # [PREDICT_WINDOWS * WINDOW_SIZE]
        steps_pred_1d = steps_pred_2d.flatten()
        bpm_true_1d = bpm_true_2d.flatten()
        steps_true_1d = steps_true_2d.flatten()
        
        total_time = len(bpm_pred_1d)            # PREDICT_WINDOWS * WINDOW_SIZE
        
        # Create subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'User ID: {user_id} | Sample {i+1} | Datetime: {datetime}', fontsize=16)
        
        # BPM True vs Predicted
        axs[0, 0].plot(range(total_time), bpm_true_1d, label='True BPM', marker='o')
        axs[0, 0].plot(range(total_time), bpm_pred_1d, label='Predicted BPM', marker='x')
        axs[0, 0].set_title('BPM Prediction vs True')
        axs[0, 0].legend()
        
        # Steps True vs Predicted
        axs[0, 1].plot(range(total_time), steps_true_1d, label='True Steps', marker='o')
        axs[0, 1].plot(range(total_time), steps_pred_1d, label='Predicted Steps', marker='x', color='orange')
        axs[0, 1].set_title('Steps Prediction vs True')
        axs[0, 1].legend()
        
        # BPM Absolute Error
        axs[1, 0].bar(range(total_time), np.abs(bpm_true_1d - bpm_pred_1d))
        axs[1, 0].set_title('BPM Absolute Error')
        
        # Steps Absolute Error
        axs[1, 1].bar(range(total_time), np.abs(steps_true_1d - steps_pred_1d), color='orange')
        axs[1, 1].set_title('Steps Absolute Error')
        
        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # Save plot
        plot_filename = f"{plot_type}_user_{user_id}_sample_{i+1}.png"
        plt.savefig(os.path.join(analysis_dir, plot_filename))
        plt.close()
        
        print(f"Saved plot for user={user_id}, sample={i+1} at {plot_filename}")

# Function to select samples to plot
def select_samples_for_plotting(validation_errors_df, user_id, num_samples=2):
    """
    Selects a specified number of samples for a given user to plot.
    
    Parameters:
        validation_errors_df (pd.DataFrame): DataFrame containing validation errors and predictions.
        user_id (str): The ID of the user.
        num_samples (int): Number of samples to select.
        
    Returns:
        list: Selected samples as dictionaries.
    """
    user_samples = validation_errors_df[validation_errors_df['user_id'] == user_id].to_dict(orient='records')
    return user_samples[:num_samples]

# Plot predictions for best and worst users
for idx, row in best_users.iterrows():
    user_id = row['user_id']
    samples_to_plot = select_samples_for_plotting(validation_errors_df, user_id, num_samples=2)
    plot_prediction_user(user_id, samples_to_plot, "best", PREDICT_WINDOWS, WINDOW_SIZE)

for idx, row in worst_users.iterrows():
    user_id = row['user_id']
    samples_to_plot = select_samples_for_plotting(validation_errors_df, user_id, num_samples=2)
    plot_prediction_user(user_id, samples_to_plot, "worst", PREDICT_WINDOWS, WINDOW_SIZE)
