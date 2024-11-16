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

# Preprocess data
df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
df = df.sort_values(by=['id', 'datetime'])
df = df.dropna()
df = df.reset_index(drop=True)

# ============================
# 2. Parameter Definitions
# ============================

WINDOW_SIZE = 24         # Number of time steps in each window
INPUT_WINDOWS = 3       # Number of input windows to consider for prediction
PREDICT_WINDOW = 1      # Number of windows to predict
BATCH_SIZE = 128         # Batch size for training and validation

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

# Function to inverse transform scaled data
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

class UserDataset(Dataset):
    """
    PyTorch Dataset for user-specific time series data.
    """
    def __init__(self, data_list):
        self.data = data_list
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        bpm_input = torch.tensor(sample['bpm_input'], dtype=torch.float32)
        steps_input = torch.tensor(sample['steps_input'], dtype=torch.float32)
        bpm_target = torch.tensor(sample['bpm_target'], dtype=torch.float32)
        steps_target = torch.tensor(sample['steps_target'], dtype=torch.float32)
        user_id = sample['user_id']
        bpm_target_original = torch.tensor(sample['bpm_target_original'], dtype=torch.float32)
        steps_target_original = torch.tensor(sample['steps_target_original'], dtype=torch.float32)
        return bpm_input, steps_input, bpm_target, steps_target, user_id, bpm_target_original, steps_target_original

def create_data_samples(df):
    """
    Creates data samples by sliding a window of INPUT_WINDOWS to predict the PREDICT_WINDOW.
    Ensures that windows are non-overlapping and ordered per user.
    """
    data_samples = []
    user_groups = df.groupby('id')
    for user_id, group in user_groups:
        windows = []
        # Create non-overlapping windows
        for i in range(0, len(group), WINDOW_SIZE):
            window = group.iloc[i:i+WINDOW_SIZE]
            if len(window) == WINDOW_SIZE:
                windows.append(window)
        # Create samples with INPUT_WINDOWS and PREDICT_WINDOW
        for i in range(len(windows) - INPUT_WINDOWS - PREDICT_WINDOW + 1):
            input_windows = windows[i:i+INPUT_WINDOWS]
            target_window = windows[i+INPUT_WINDOWS]
            data_samples.append({
                'bpm_input': np.array([w['bpm_scaled'].values for w in input_windows]),
                'steps_input': np.array([w['steps_scaled'].values for w in input_windows]),
                'bpm_target': target_window['bpm_scaled'].values,
                'steps_target': target_window['steps_scaled'].values,
                'bpm_target_original': target_window['bpm'].values,
                'steps_target_original': target_window['steps'].values,
                'user_id': user_id
            })
    return data_samples

# Generate data samples
data_samples = create_data_samples(df)

# ============================
# 5. Train-Validation Split
# ============================

# Implement user-level train-validation split to prevent data leakage
unique_user_ids = df['id'].unique()
train_user_ids, val_user_ids = train_test_split(unique_user_ids, test_size=0.2, random_state=42)

# Filter data_samples based on user IDs
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

# Organize training and validation samples by user
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
        return sum(len(indices) // self.batch_size + (1 if len(indices) % self.batch_size != 0 else 0) 
                   for indices in self.user_data.values())

# Instantiate datasets
train_dataset = UserDataset(train_samples)
val_dataset = UserDataset(val_samples)

# Instantiate samplers
train_sampler = PerUserBatchSampler(train_user_data, batch_size=BATCH_SIZE)
val_sampler = PerUserBatchSampler(val_user_data, batch_size=BATCH_SIZE)

# Define collate function
def collate_fn(batch):
    """
    Custom collate function to handle batching of samples.
    """
    bpm_inputs = torch.stack([s[0] for s in batch])
    steps_inputs = torch.stack([s[1] for s in batch])
    bpm_targets = torch.stack([s[2] for s in batch])
    steps_targets = torch.stack([s[3] for s in batch])
    user_ids = [s[4] for s in batch]
    bpm_targets_original = [s[5] for s in batch]
    steps_targets_original = [s[6] for s in batch]
    return bpm_inputs, steps_inputs, bpm_targets, steps_targets, user_ids, bpm_targets_original, steps_targets_original

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=collate_fn)

# ============================
# 6A. Compute Weights Based on Average Size
# ============================

# Calculate the average absolute value of BPM and steps in the training set
avg_bpm = df[df['id'].isin(train_user_ids)]['bpm'].abs().mean()
avg_steps = df[df['id'].isin(train_user_ids)]['steps'].abs().mean()

# Set weights inversely proportional to the average size
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
    Forecasting model with separate 1D CNN and LSTM encoders for BPM and Steps,
    followed by separate decoders for each modality.
    """
    def __init__(self):
        super(ForecastingModel, self).__init__()
        # CNN Encoder for BPM
        self.bpm_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),  # Added Dropout
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3)   # Added Dropout
        )
        # LSTM Encoder for BPM
        self.bpm_lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        
        # CNN Encoder for Steps
        self.steps_cnn = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3),  # Added Dropout
            nn.Conv1d(32, 64, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Dropout(0.3)   # Added Dropout
        )
        # LSTM Encoder for Steps
        self.steps_lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        
        # Fully Connected Layers for Fusion
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),  # Added Dropout
            # Removed the combined decoder
        )
        
        # Separate Decoders for BPM and Steps
        self.bpm_decoder = nn.Linear(128, WINDOW_SIZE)   # Predict BPM
        self.steps_decoder = nn.Linear(128, WINDOW_SIZE) # Predict Steps

    def forward(self, bpm_input, steps_input):
        """
        Forward pass for the forecasting model.
        """
            # Reshape BPM input: [batch_size, INPUT_WINDOWS, WINDOW_SIZE] -> [batch_size, 1, INPUT_WINDOWS * WINDOW_SIZE]
        bpm_seq = bpm_input.view(bpm_input.size(0), -1).contiguous().unsqueeze(1)  # Shape: [B, 1, 6]
        
        # CNN Encoder for BPM
        bpm_cnn_output = self.bpm_cnn(bpm_seq)  # Shape: [B, 64, 6]
        bpm_cnn_output = bpm_cnn_output.permute(0, 2, 1)  # Shape: [B, 6, 64]
        
        # LSTM Encoder for BPM
        bpm_lstm_output, _ = self.bpm_lstm(bpm_cnn_output)  # Shape: [B, 6, 128]
        bpm_hidden = bpm_lstm_output[:, -1, :]  # Shape: [B, 128]
        
        # Reshape Steps input similarly: [B, 1, 6]
        steps_seq = steps_input.view(steps_input.size(0), -1).contiguous().unsqueeze(1)  # Shape: [B, 1, 6]
        
        # CNN Encoder for Steps
        steps_cnn_output = self.steps_cnn(steps_seq)  # Shape: [B, 64, 6]
        steps_cnn_output = steps_cnn_output.permute(0, 2, 1)  # Shape: [B, 6, 64]
        
        # LSTM Encoder for Steps
        steps_lstm_output, _ = self.steps_lstm(steps_cnn_output)  # Shape: [B, 6, 128]
        steps_hidden = steps_lstm_output[:, -1, :]  # Shape: [B, 128]
        
        # Concatenate the hidden states from BPM and Steps
        combined_features = torch.cat((bpm_hidden, steps_hidden), dim=1)  # Shape: [B, 256]
        
        # Fully Connected Layers for Fusion
        fused_features = self.fc(combined_features)  # Shape: [B, 128]
        
        # Separate Decoders
        bpm_pred = self.bpm_decoder(fused_features)    # Shape: [B, WINDOW_SIZE]
        steps_pred = self.steps_decoder(fused_features) # Shape: [B, WINDOW_SIZE]
        
        return bpm_pred, steps_pred

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Instantiate the model and move it to the device
model = ForecastingModel().to(device)

# ============================
# 8. Training Setup
# ============================

# Define the loss function and optimizer
criterion = nn.L1Loss()  # Using Mean Absolute Error (MAE) Loss

# Initialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight_decay for regularization

# Initialize the learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

num_epochs = 100

# ============================
# 9. Training and Validation Loop 
# ============================

# Early Stopping Parameters
patience = 15
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False

# Directory to save the best model
save_dir = 'results/saved_models'
os.makedirs(save_dir, exist_ok=True)
model_save_path = os.path.join(save_dir, 'forecasting_backbone.pth')

# Initialize statistics dictionary
stats = {
    'epoch': [],
    'train_loss': [],
    'val_loss': [],
    'avg_bpm_error': [],
    'avg_steps_error': []
}

for epoch in range(num_epochs):
    if early_stop:
        break
    
    # ---------------------
    # Training Phase
    # ---------------------
    model.train()
    epoch_loss = 0.0
    for bpm_input, steps_input, bpm_target, steps_target, _, _, _ in train_loader:
        optimizer.zero_grad()
        
        # Move tensors to the device
        bpm_input = bpm_input.to(device)
        steps_input = steps_input.to(device)
        bpm_target = bpm_target.to(device)
        steps_target = steps_target.to(device)
        
        # Forward pass
        bpm_pred, steps_pred = model(bpm_input, steps_input)
        
        # Compute loss with weights
        loss_bpm = criterion(bpm_pred, bpm_target)
        loss_steps = criterion(steps_pred, steps_target)
        loss = alpha * loss_bpm + beta * loss_steps
        
        # Backward pass and optimization
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_loss += loss.item()
    
    # Update learning rate
    scheduler.step()
    
    # ---------------------
    # Validation Phase
    # ---------------------
    model.eval()
    val_loss = 0.0
    total_bpm_error = 0.0
    total_steps_error = 0.0
    with torch.no_grad():
        for bpm_input, steps_input, bpm_target, steps_target, user_id, bpm_target_original, steps_target_original in val_loader:
            # Move tensors to the device
            bpm_input = bpm_input.to(device)
            steps_input = steps_input.to(device)
            bpm_target = bpm_target.to(device)
            steps_target = steps_target.to(device)
            
            # Forward pass
            bpm_pred_scaled, steps_pred_scaled = model(bpm_input, steps_input)
            
            # Compute loss
            loss_bpm = criterion(bpm_pred_scaled, bpm_target)
            loss_steps = criterion(steps_pred_scaled, steps_target)
            loss = alpha * loss_bpm + beta * loss_steps
            val_loss += loss.item()
            
            # Inverse transform predictions and targets for error calculation
            for i in range(len(user_id)):
                uid = user_id[i]
                bpm_pred_scaled_np = bpm_pred_scaled[i].cpu().numpy()
                steps_pred_scaled_np = steps_pred_scaled[i].cpu().numpy()
                bpm_pred_unscaled, steps_pred_unscaled = inverse_transform(
                    uid,
                    bpm_pred_scaled_np,
                    steps_pred_scaled_np
                )
                bpm_target_unscaled = bpm_target_original[i].numpy()
                steps_target_unscaled = steps_target_original[i].numpy()
                
                # Compute Mean Absolute Error (MAE)
                bpm_error = np.mean(np.abs(bpm_pred_unscaled - bpm_target_unscaled))
                steps_error = np.mean(np.abs(steps_pred_unscaled - steps_target_unscaled))
                total_bpm_error += bpm_error
                total_steps_error += steps_error
    
    # ---------------------
    # Calculate Average Losses and Errors
    # ---------------------
    avg_train_loss = epoch_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    avg_bpm_error = total_bpm_error / len(val_loader.dataset)
    avg_steps_error = total_steps_error / len(val_loader.dataset)
    
    # Log statistics
    stats['epoch'].append(epoch + 1)
    stats['train_loss'].append(avg_train_loss)
    stats['val_loss'].append(avg_val_loss)
    stats['avg_bpm_error'].append(avg_bpm_error)
    stats['avg_steps_error'].append(avg_steps_error)
    
    # ---------------------
    # Check for Improvement
    # ---------------------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), model_save_path)
    else:
        epochs_no_improve += 1
    
    # Check if early stopping should be triggered
    if epochs_no_improve >= patience:
        print(f'Early stopping triggered after {patience} epochs with no improvement.')
        early_stop = True
    
    # Print epoch statistics
    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Loss: {avg_train_loss:.4f}, '
          f'Val Loss: {avg_val_loss:.4f}, '
          f'Avg BPM Error: {avg_bpm_error:.2f}, '
          f'Avg Steps Error: {avg_steps_error:.2f}')

# ============================
# 10. Generate Charts
# ============================

# Create a directory to save the figures
results_dir = 'results/train'
os.makedirs(results_dir, exist_ok=True)

# Convert statistics to DataFrame
stats_df = pd.DataFrame(stats)

# Save DataFrame
stats_df.to_csv(os.path.join(results_dir, "stats.csv"))

# Plot and save Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(stats_df['epoch'], stats_df['train_loss'], label='Train Loss')
plt.plot(stats_df['epoch'], stats_df['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.savefig(os.path.join(results_dir, 'loss_per_epoch.png'))
plt.show()

# Plot and save BPM Error
plt.figure(figsize=(10, 6))
plt.plot(stats_df['epoch'], stats_df['avg_bpm_error'], label='BPM Error')
plt.xlabel('Epoch')
plt.ylabel('BPM Error')
plt.title('BPM Error per Epoch')
plt.legend()
plt.savefig(os.path.join(results_dir, 'bpm_error_per_epoch.png'))
plt.show()

# Plot and save Steps Error
plt.figure(figsize=(10, 6))
plt.plot(stats_df['epoch'], stats_df['avg_steps_error'], label='Steps Error')
plt.xlabel('Epoch')
plt.ylabel('Steps Error')
plt.title('Steps Error per Epoch')
plt.legend()
plt.savefig(os.path.join(results_dir, 'steps_error_per_epoch.png'))
plt.show()

print(f"Training complete. Figures saved in '{results_dir}'.")
