import pandas as pd
import numpy as np
import torch
from torch.utils.data import Dataset, DataLoader, Sampler
from torch import nn
from sklearn.preprocessing import RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
import warnings
from collections import defaultdict
import os
import matplotlib.pyplot as plt
import random
import seaborn as sns

# ============================
# 1. Configuration and Parameters
# ============================

# Set random seeds for reproducibility
SEED = 42
random.seed(SEED)
np.random.seed(SEED)
torch.manual_seed(SEED)
torch.cuda.manual_seed_all(SEED)

# Parameters
WINDOW_SIZE = 24          # Number of time steps in each window
INPUT_WINDOWS = 3         # Number of input windows to consider for prediction 
PREDICT_WINDOW = 1        # Number of windows to predict
BATCH_SIZE = 64           # Batch size for training and validation
NUM_EPOCHS = 150          # Number of training epochs
PATIENCE = 25             # Patience for early stopping
LEARNING_RATE = 0.001     # Learning rate for optimizer
WEIGHT_DECAY = 1e-5       # Weight decay for optimizer
SPIKE_THRESH = 1.5        # Threshold for spike detection (not used with percentile-based method)
SPIKE_WEIGHT = None       # We'll compute SPIKE_WEIGHT based on class imbalance
SPIKE_WEIGHT_CAP = 5.0    # Cap for SPIKE_WEIGHT to prevent overcompensation
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')  # Device configuration

# Directories
DATA_PATH = "data/lifesnaps.csv"  # Path to your data CSV
SAVE_DIR = 'results/saved_models'
RESULTS_DIR = 'results/train'
ANALYSIS_DIR = os.path.join(RESULTS_DIR, 'analysis')
os.makedirs(SAVE_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(ANALYSIS_DIR, exist_ok=True)

# ============================
# 2. Data Loading and Preprocessing
# ============================

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# Load your DataFrame
df_daily = pd.read_csv(DATA_PATH, parse_dates=['date'])
if 'Unnamed: 0' in df_daily.columns:
    df_daily.drop("Unnamed: 0", axis=1, inplace=True)  # Drop unnecessary column if exists
df = df_daily[["id", "date", "hour", "bpm", "steps"]]

# Preprocess data
df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
df = df.sort_values(by=['id', 'datetime']).reset_index(drop=True)
df = df.dropna().reset_index(drop=True)

# ============================
# 3. Data Normalization with RobustScaler
# ============================

user_scalers = {}

def scale_per_user(group):
    """
    Applies RobustScaler to BPM and Steps data for each user.
    Stores scaler parameters for inverse transformations.
    """
    user_id = group.name
    scaler_bpm = RobustScaler()
    scaler_steps = RobustScaler()
    group['bpm_scaled'] = scaler_bpm.fit_transform(group['bpm'].values.reshape(-1,1)).flatten()
    group['steps_scaled'] = scaler_steps.fit_transform(group['steps'].values.reshape(-1,1)).flatten()
    user_scalers[user_id] = {
        'bpm_median': scaler_bpm.center_[0],
        'bpm_scale': scaler_bpm.scale_[0],
        'steps_median': scaler_steps.center_[0],
        'steps_scale': scaler_steps.scale_[0]
    }
    return group

# Apply scaling per user with RobustScaler
df = df.groupby('id').apply(scale_per_user).reset_index(drop=True)

def inverse_transform(user_id, bpm_scaled, steps_scaled):
    """
    Reverts scaled BPM and Steps data back to original scale using stored scaler parameters.
    """
    bpm_median = user_scalers[user_id]['bpm_median']
    bpm_scale = user_scalers[user_id]['bpm_scale']
    steps_median = user_scalers[user_id]['steps_median']
    steps_scale = user_scalers[user_id]['steps_scale']
    bpm_original = bpm_scaled * bpm_scale + bpm_median
    steps_original = steps_scaled * steps_scale + steps_median
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
        spike_labels = torch.tensor(sample['spike_labels'], dtype=torch.float32)
        user_id = sample['user_id']
        bpm_target_original = torch.tensor(sample['bpm_target_original'], dtype=torch.float32)
        steps_target_original = torch.tensor(sample['steps_target_original'], dtype=torch.float32)
        datetime = sample.get('datetime', None)
        hour = sample.get('hour', 0)
        day_of_week = sample.get('day_of_week', 0)
        weight = sample.get('weight', 1.0)
        return bpm_input, steps_input, bpm_target, steps_target, spike_labels, user_id, bpm_target_original, steps_target_original, datetime, weight, hour, day_of_week

def create_data_samples(df):
    """
    Creates data samples by sliding a window of INPUT_WINDOWS to predict the PREDICT_WINDOW.
    Labels each time step in the target window with spike labels and includes time-based features.
    """
    data_samples = []
    user_groups = df.groupby('id')
    
    # Precompute percentiles per user for spike detection
    user_stats = df.groupby('id')['steps'].agg(
        mean='mean',
        std='std',
        lower_percentile=lambda x: np.percentile(x, 10),
        upper_percentile=lambda x: np.percentile(x, 90)
    ).to_dict('index')
    
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
            steps_target = target_window['steps'].values
            # Define spike thresholds using percentiles
            lower_threshold = user_stats[user_id]['lower_percentile']  # 10th percentile
            upper_threshold = user_stats[user_id]['upper_percentile']  # 90th percentile
            # Label spikes
            spike_labels = ((steps_target > upper_threshold) | (steps_target < lower_threshold)).astype(float)
            # Extract time-based features from the target window
            datetime_np = target_window['datetime'].values[0]
            # Convert numpy.datetime64 to pandas.Timestamp
            datetime = pd.to_datetime(datetime_np)
            hour = datetime.hour
            day_of_week = datetime.dayofweek
            data_samples.append({
                'bpm_input': np.array([w['bpm_scaled'].values for w in input_windows]),
                'steps_input': np.array([w['steps_scaled'].values for w in input_windows]),
                'bpm_target': target_window['bpm_scaled'].values,
                'steps_target': target_window['steps_scaled'].values,
                'spike_labels': spike_labels,
                'bpm_target_original': target_window['bpm'].values,
                'steps_target_original': target_window['steps'].values,
                'user_id': user_id,
                'datetime': datetime,  # Now a pandas.Timestamp
                'hour': hour,
                'day_of_week': day_of_week
            })
    return data_samples

# Generate data samples
data_samples = create_data_samples(df)

# ============================
# 5. Analyze Class Imbalance
# ============================

# Calculate total number of spike and no-spike samples
total_spike_labels = np.concatenate([sample['spike_labels'] for sample in data_samples])
num_spikes = np.sum(total_spike_labels)
num_no_spikes = len(total_spike_labels) - num_spikes

print(f"Total samples: {len(data_samples)}")
print(f"Total time steps (spike labels): {len(total_spike_labels)}")
print(f"Number of spike samples: {int(num_spikes)}")
print(f"Number of no-spike samples: {int(num_no_spikes)}")
print(f"Spike sample ratio: {num_spikes / len(total_spike_labels):.4f}")

# Calculate SPIKE_WEIGHT based on class imbalance
spike_ratio = num_spikes / len(total_spike_labels)
no_spike_ratio = num_no_spikes / len(total_spike_labels)

# Set SPIKE_WEIGHT to inverse of spike ratio
SPIKE_WEIGHT = no_spike_ratio / spike_ratio
# Cap SPIKE_WEIGHT to prevent overcompensation
SPIKE_WEIGHT = min(SPIKE_WEIGHT, SPIKE_WEIGHT_CAP)
print(f"Computed SPIKE_WEIGHT (capped): {SPIKE_WEIGHT:.2f}")

# ============================
# 6. Train-Validation Split
# ============================

# Implement user-level train-validation split to prevent data leakage
unique_user_ids = df['id'].unique()
train_user_ids, val_user_ids = train_test_split(unique_user_ids, test_size=0.2, random_state=SEED)

# Filter data_samples based on user IDs
train_samples = [sample for sample in data_samples if sample['user_id'] in train_user_ids]
val_samples = [sample for sample in data_samples if sample['user_id'] in val_user_ids]

# ============================
# 7. Oversample Spike Samples in Training Data
# ============================

from sklearn.utils import resample

def balance_dataset(train_samples):
    """
    Balances the training dataset by oversampling spike samples.
    """
    spike_samples = [sample for sample in train_samples if sample['spike_labels'].sum() > 0]
    non_spike_samples = [sample for sample in train_samples if sample['spike_labels'].sum() == 0]
    print(f"Original spike samples: {len(spike_samples)}, non-spike samples: {len(non_spike_samples)}")
    
    # Oversample spike samples to match the number of non-spike samples
    spike_samples_oversampled = resample(spike_samples,
                                         replace=True,
                                         n_samples=len(non_spike_samples),
                                         random_state=SEED)
    
    # Combine and shuffle
    train_samples_balanced = spike_samples_oversampled + non_spike_samples
    random.shuffle(train_samples_balanced)
    return train_samples_balanced

# Balance the training dataset
train_samples_balanced = balance_dataset(train_samples)

# ============================
# 7A. DataLoader Setup
# ============================

def create_user_datasets(data_samples):
    """
    Organizes data samples by user for the sampler.
    """
    user_data = defaultdict(list)
    for idx, sample in enumerate(data_samples):
        user_data[sample['user_id']].append(idx)
    return user_data

# Define collate function
def collate_fn(batch):
    """
    Custom collate function to handle batching of samples.
    """
    bpm_inputs = torch.stack([s[0] for s in batch])
    steps_inputs = torch.stack([s[1] for s in batch])
    bpm_targets = torch.stack([s[2] for s in batch])
    steps_targets = torch.stack([s[3] for s in batch])
    spike_labels = torch.stack([s[4] for s in batch])
    user_ids = [s[5] for s in batch]
    bpm_targets_original = [s[6] for s in batch]
    steps_targets_original = [s[7] for s in batch]
    datetimes = [s[8] for s in batch]
    weights = [s[9] for s in batch]
    hours = torch.tensor([s[10] for s in batch], dtype=torch.long)
    days_of_week = torch.tensor([s[11] for s in batch], dtype=torch.long)
    return (bpm_inputs, steps_inputs, bpm_targets, steps_targets, spike_labels, user_ids,
            bpm_targets_original, steps_targets_original, datetimes, weights, hours, days_of_week)

# Define PerUserBatchSampler
class PerUserBatchSampler(Sampler):
    """
    Custom sampler to yield batches containing samples from only one user,
    while preserving temporal order within each user's data.
    """
    def __init__(self, user_data, batch_size):
        self.user_data = user_data  # Mapping from user_id to list of indices
        self.batch_size = batch_size
        self.batches = []
        for user_id, indices in self.user_data.items():
            # Do not shuffle indices to maintain temporal order
            # Split indices into batches of size up to batch_size
            user_batches = [indices[i:i+self.batch_size] for i in range(0, len(indices), self.batch_size)]
            self.batches.extend(user_batches)
        # Optionally shuffle batches to mix up the order of users
        random.shuffle(self.batches)
    
    def __iter__(self):
        for batch in self.batches:
            yield batch
    
    def __len__(self):
        return len(self.batches)

# Create training dataset and DataLoader
train_dataset = UserDataset(train_samples_balanced)
train_user_data = create_user_datasets(train_samples_balanced)
train_sampler = PerUserBatchSampler(train_user_data, batch_size=BATCH_SIZE)
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)

# Create validation dataset and DataLoader
val_dataset = UserDataset(val_samples)
val_user_data = create_user_datasets(val_samples)
val_sampler = PerUserBatchSampler(val_user_data, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=collate_fn)

# ============================
# 8. Model Definition
# ============================

class ResidualBlock(nn.Module):
    """
    Residual Block for CNN Encoders.
    """
    def __init__(self, in_channels, out_channels, kernel_size=3, padding=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size, padding=padding)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size, padding=padding)
        self.dropout = nn.Dropout(0.3)
        if in_channels != out_channels:
            self.residual = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        else:
            self.residual = nn.Identity()
    
    def forward(self, x):
        residual = self.residual(x)
        out = self.conv1(x)
        out = self.relu(out)
        out = self.dropout(out)
        out = self.conv2(out)
        out = self.dropout(out)
        out += residual
        out = self.relu(out)
        return out

class ForecastingModel(nn.Module):
    """
    Forecasting model with separate CNN and LSTM encoders for BPM and Steps,
    Attention mechanisms, Residual Connections, Fusion layers, Decoders,
    and a Classification head for per-time-step spike detection.
    """
    def __init__(self):
        super(ForecastingModel, self).__init__()
        # CNN Encoder for BPM with Residual Blocks
        self.bpm_cnn = nn.Sequential(
            ResidualBlock(INPUT_WINDOWS, 32),
            ResidualBlock(32, 64)
        )
        # LSTM Encoder for BPM
        self.bpm_lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        
        # CNN Encoder for Steps with Residual Blocks
        self.steps_cnn = nn.Sequential(
            ResidualBlock(INPUT_WINDOWS, 32),
            ResidualBlock(32, 64)
        )
        # LSTM Encoder for Steps
        self.steps_lstm = nn.LSTM(input_size=64, hidden_size=128, num_layers=2, batch_first=True)
        
        # Attention Layers
        self.bpm_attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        self.steps_attention = nn.MultiheadAttention(embed_dim=128, num_heads=4, batch_first=True)
        
        # Fully Connected Layers for Fusion
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3),
        )
        
        # Separate Decoders for BPM and Steps
        self.bpm_decoder = nn.Linear(128, WINDOW_SIZE)    # Predict BPM
        self.steps_decoder = nn.Linear(128, WINDOW_SIZE)  # Predict Steps
        
        # Classification Head for per-time-step Spike Detection
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, WINDOW_SIZE),  # Output per time step
            # Note: We'll use BCEWithLogitsLoss or FocalLoss, so no Sigmoid here
        )
    
    def forward(self, bpm_input, steps_input, hour, day_of_week):
        """
        Forward pass for the forecasting model.
        """
        # Input Dimensions: [B, C, L] where C=INPUT_WINDOWS, L=WINDOW_SIZE
        bpm_seq = bpm_input  # Shape: [B, 3, 24]
        
        # CNN Encoder for BPM
        bpm_cnn_output = self.bpm_cnn(bpm_seq)  # Shape: [B, 64, 24]
        
        # LSTM Encoder for BPM
        bpm_lstm_output, _ = self.bpm_lstm(bpm_cnn_output.permute(0, 2, 1))  # Shape: [B, 24, 128]
        
        # Attention for BPM
        bpm_attn_output, _ = self.bpm_attention(bpm_lstm_output, bpm_lstm_output, bpm_lstm_output)  # Shape: [B, 24, 128]
        bpm_attn_output = bpm_attn_output.mean(dim=1)  # Aggregate over time steps
        
        # CNN Encoder for Steps
        steps_seq = steps_input  # Shape: [B, 3, 24]
        steps_cnn_output = self.steps_cnn(steps_seq)  # Shape: [B, 64, 24]
        
        # LSTM Encoder for Steps
        steps_lstm_output, _ = self.steps_lstm(steps_cnn_output.permute(0, 2, 1))  # Shape: [B, 24, 128]
        
        # Attention for Steps
        steps_attn_output, _ = self.steps_attention(steps_lstm_output, steps_lstm_output, steps_lstm_output)  # Shape: [B, 24, 128]
        steps_attn_output = steps_attn_output.mean(dim=1)  # Aggregate over time steps
        
        # Concatenate the attention outputs from BPM and Steps
        combined_features = torch.cat((bpm_attn_output, steps_attn_output), dim=1)  # Shape: [B, 256]
        
        # Fully Connected Layers for Fusion
        fused_features = self.fc(combined_features)  # Shape: [B, 128]
        
        # Separate Decoders
        bpm_pred = self.bpm_decoder(fused_features)      # Shape: [B, WINDOW_SIZE]
        steps_pred = self.steps_decoder(fused_features)  # Shape: [B, WINDOW_SIZE]
        
        # Classification Head
        spike_logits = self.classifier(fused_features)   # Shape: [B, WINDOW_SIZE]
        
        return bpm_pred, steps_pred, spike_logits

# Instantiate the model and move it to the device
model = ForecastingModel().to(DEVICE)
print(f'Using device: {DEVICE}')

# ============================
# 9. Loss Function Definition
# ============================

class FocalLoss(nn.Module):
    def __init__(self, alpha=0.99, gamma=2.0, reduction='mean'):
        """
        Focal Loss for binary classification.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha  # Weighting factor for positive class (spike)
        self.gamma = gamma
        self.reduction = reduction
    
    def forward(self, inputs, targets):
        """
        Forward pass for Focal Loss.
        """
        BCE_loss = nn.functional.binary_cross_entropy_with_logits(inputs, targets, reduction='none')
        pt = torch.exp(-BCE_loss)
        # Correct calculation of alpha_t
        alpha_t = self.alpha * targets + (1 - self.alpha) * (1 - targets)
        F_loss = alpha_t * (1 - pt) ** self.gamma * BCE_loss
        
        if self.reduction == 'mean':
            return F_loss.mean()
        else:
            return F_loss.sum()

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=1.0, gamma=2.0, focal_alpha=0.99):
        """
        Combined Loss for regression and classification tasks.
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # Weight for regression loss
        self.beta = beta    # Weight for classification loss
        self.regression_loss = nn.MSELoss(reduction='mean')  # For BPM and Steps
        self.classification_loss = FocalLoss(alpha=focal_alpha, gamma=gamma, reduction='mean')
        
    def forward(self, bpm_pred, bpm_target, steps_pred, steps_target, spike_logits, spike_label):
        """
        Compute the combined loss.
        """
        # Regression Loss
        loss_bpm = self.regression_loss(bpm_pred, bpm_target)
        loss_steps = self.regression_loss(steps_pred, steps_target)
        loss_regression = (loss_bpm + loss_steps) / 2  # Average to balance
        
        # Classification Loss (per time step)
        loss_classification = self.classification_loss(spike_logits, spike_label)
        
        # Total Loss
        total_loss = self.alpha * loss_regression + self.beta * loss_classification
        return total_loss

# Instantiate the loss function with adjusted alpha
criterion = CombinedLoss(alpha=1.0, beta=1.0, gamma=2.0, focal_alpha=0.99)

# ============================
# 10. Optimizer and Scheduler Setup
# ============================

# Initialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Initialize the learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# ============================
# 11. Training and Validation Loop
# ============================

# Early Stopping Parameters
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False

# Model save path
model_save_path = os.path.join(SAVE_DIR, 'forecasting_backbone.pth')

# Statistics Dictionary
stats = {
    'epoch': [],
    'train_loss': [],
    'val_loss': [],
    'avg_bpm_error': [],
    'avg_steps_error': [],
    'spike_accuracy': [],
    'spike_precision': [],
    'spike_recall': [],
    'spike_f1': [],
    'best_threshold': []
}

# Validation Errors List
validation_errors = []

for epoch in range(NUM_EPOCHS):
    if early_stop:
        break
    
    # ---------------------
    # Training Phase
    # ---------------------
    model.train()
    epoch_loss = 0.0
    for batch in train_loader:
        (bpm_input, steps_input, bpm_target, steps_target, spike_labels, _, _, _, _, _, hours, days_of_week) = batch
        optimizer.zero_grad()
        
        # Move tensors to the device
        bpm_input = bpm_input.to(DEVICE)
        steps_input = steps_input.to(DEVICE)
        bpm_target = bpm_target.to(DEVICE)
        steps_target = steps_target.to(DEVICE)
        spike_labels = spike_labels.to(DEVICE)
        hours = hours.to(DEVICE)
        days_of_week = days_of_week.to(DEVICE)
        
        # Forward pass
        bpm_pred, steps_pred, spike_logits = model(bpm_input, steps_input, hours, days_of_week)
        
        # Ensure tensor shapes match
        if spike_logits.shape != spike_labels.shape:
            spike_labels = spike_labels.view_as(spike_logits)
        
        # Compute combined loss
        loss = criterion(bpm_pred, bpm_target, steps_pred, steps_target, spike_logits, spike_labels)
        
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
    all_spike_probs = []
    all_spike_labels = []
    
    # Reset validation errors list at each epoch
    validation_errors_epoch = []
    
    with torch.no_grad():
        for batch in val_loader:
            (bpm_input, steps_input, bpm_target, steps_target, spike_labels,
             user_id, bpm_target_original, steps_target_original, datetimes,
             _, hours, days_of_week) = batch
            # Move tensors to the device
            bpm_input = bpm_input.to(DEVICE)
            steps_input = steps_input.to(DEVICE)
            bpm_target = bpm_target.to(DEVICE)
            steps_target = steps_target.to(DEVICE)
            spike_labels = spike_labels.to(DEVICE)
            hours = hours.to(DEVICE)
            days_of_week = days_of_week.to(DEVICE)
            
            # Forward pass
            bpm_pred_scaled, steps_pred_scaled, spike_logits = model(bpm_input, steps_input, hours, days_of_week)
            spike_prob = torch.sigmoid(spike_logits)
            
            # Ensure tensor shapes match
            if spike_logits.shape != spike_labels.shape:
                spike_labels = spike_labels.view_as(spike_logits)
            
            # Compute combined loss
            loss = criterion(bpm_pred_scaled, bpm_target, steps_pred_scaled, steps_target, spike_logits, spike_labels)
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
                
                # Collect spike probabilities and labels for metrics
                all_spike_probs.extend(spike_prob[i].cpu().numpy())
                all_spike_labels.extend(spike_labels[i].cpu().numpy())
                
                # Store detailed error information
                validation_errors_epoch.append({
                    'user_id': uid,
                    'bpm_error': bpm_error,
                    'steps_error': steps_error,
                    'bpm_pred': bpm_pred_unscaled,
                    'steps_pred': steps_pred_unscaled,
                    'bpm_true': bpm_target_unscaled,
                    'steps_true': steps_target_unscaled,
                    'datetime': datetimes[i]
                })
    
    # Aggregate errors
    avg_train_loss = epoch_loss / len(train_loader)
    avg_val_loss = val_loss / len(val_loader)
    avg_bpm_error = total_bpm_error / len(val_loader.dataset)
    avg_steps_error = total_steps_error / len(val_loader.dataset)
    
    # Compute spike metrics
    spike_probs = np.array(all_spike_probs)
    spike_labels = np.array(all_spike_labels)
    
    # Adjust the threshold for spike prediction
    thresholds = np.linspace(0.1, 0.9, 9)
    best_f1 = 0
    best_threshold = 0.5
    for thresh in thresholds:
        spike_preds = (spike_probs >= thresh).astype(int)
        f1 = f1_score(spike_labels, spike_preds, zero_division=0)
        if f1 > best_f1:
            best_f1 = f1
            best_threshold = thresh
    
    # Use the best threshold
    spike_preds = (spike_probs >= best_threshold).astype(int)
    spike_accuracy = accuracy_score(spike_labels, spike_preds)
    spike_precision = precision_score(spike_labels, spike_preds, zero_division=0)
    spike_recall = recall_score(spike_labels, spike_preds, zero_division=0)
    spike_f1 = f1_score(spike_labels, spike_preds, zero_division=0)
    
    # Log statistics
    stats['epoch'].append(epoch + 1)
    stats['train_loss'].append(avg_train_loss)
    stats['val_loss'].append(avg_val_loss)
    stats['avg_bpm_error'].append(avg_bpm_error)
    stats['avg_steps_error'].append(avg_steps_error)
    stats['spike_accuracy'].append(spike_accuracy)
    stats['spike_precision'].append(spike_precision)
    stats['spike_recall'].append(spike_recall)
    stats['spike_f1'].append(spike_f1)
    stats['best_threshold'].append(best_threshold)
    
    # Append epoch's validation errors to the main list
    validation_errors.extend(validation_errors_epoch)
    
    # ---------------------
    # Check for Improvement
    # ---------------------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), model_save_path)
        print(f'Epoch {epoch+1}: Validation loss decreased. Saving model to {model_save_path}')
    else:
        epochs_no_improve += 1
    
    # Check if early stopping should be triggered
    if epochs_no_improve >= PATIENCE:
        print(f'Early stopping triggered after {PATIENCE} epochs with no improvement.')
        early_stop = True
    
    # Print epoch statistics
    print(f'Epoch {epoch+1}/{NUM_EPOCHS}, '
          f'Train Loss: {avg_train_loss:.4f}, '
          f'Val Loss: {avg_val_loss:.4f}, '
          f'Avg BPM Error: {avg_bpm_error:.2f}, '
          f'Avg Steps Error: {avg_steps_error:.2f}, '
          f'Spike Acc: {spike_accuracy:.4f}, '
          f'Spike Precision: {spike_precision:.4f}, '
          f'Spike Recall: {spike_recall:.4f}, '
          f'Spike F1: {spike_f1:.4f}, '
          f'Best Threshold: {best_threshold:.2f}') 
    
    # ---------------------
    # Visualize Spike Probability Distribution
    # ---------------------
    if (epoch + 1) % 10 == 0 or epoch == NUM_EPOCHS - 1:
        plt.figure(figsize=(10, 6))
        sns.histplot(spike_probs[spike_labels == 1], color='red', label='Actual Spikes', bins=50, kde=True)
        sns.histplot(spike_probs[spike_labels == 0], color='blue', label='Actual No-Spikes', bins=50, kde=True)
        plt.xlabel('Predicted Spike Probability')
        plt.ylabel('Frequency')
        plt.title(f'Spike Probability Distribution at Epoch {epoch+1}')
        plt.legend()
        plt.savefig(os.path.join(ANALYSIS_DIR, f'spike_probability_distribution_epoch_{epoch+1}.png'))
        plt.close()

# ============================
# 12. Generate Charts
# ============================

# Convert statistics to DataFrame
stats_df = pd.DataFrame(stats)

# Save DataFrame
stats_df.to_csv(os.path.join(RESULTS_DIR, "stats.csv"), index=False)

# Plot and save Training and Validation Loss
plt.figure(figsize=(10, 6))
plt.plot(stats_df['epoch'], stats_df['train_loss'], label='Train Loss')
plt.plot(stats_df['epoch'], stats_df['val_loss'], label='Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss per Epoch')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'loss_per_epoch.png'))
plt.close()

# Plot and save BPM Error
plt.figure(figsize=(10, 6))
plt.plot(stats_df['epoch'], stats_df['avg_bpm_error'], label='BPM Error')
plt.xlabel('Epoch')
plt.ylabel('BPM Error')
plt.title('BPM Error per Epoch')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'bpm_error_per_epoch.png'))
plt.close()

# Plot and save Steps Error
plt.figure(figsize=(10, 6))
plt.plot(stats_df['epoch'], stats_df['avg_steps_error'], label='Steps Error')
plt.xlabel('Epoch')
plt.ylabel('Steps Error')
plt.title('Steps Error per Epoch')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'steps_error_per_epoch.png'))
plt.close()

# Plot and save Spike Detection Accuracy
plt.figure(figsize=(10, 6))
plt.plot(stats_df['epoch'], stats_df['spike_accuracy'], label='Spike Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.title('Spike Detection Accuracy per Epoch')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'spike_accuracy_per_epoch.png'))
plt.close()

# Plot and save Spike Detection F1 Score
plt.figure(figsize=(10, 6))
plt.plot(stats_df['epoch'], stats_df['spike_f1'], label='Spike F1 Score')
plt.xlabel('Epoch')
plt.ylabel('F1 Score')
plt.title('Spike Detection F1 Score per Epoch')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'spike_f1_per_epoch.png'))
plt.close()

# Plot and save Spike Detection Precision
plt.figure(figsize=(10, 6))
plt.plot(stats_df['epoch'], stats_df['spike_precision'], label='Spike Precision')
plt.xlabel('Epoch')
plt.ylabel('Precision')
plt.title('Spike Detection Precision per Epoch')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'spike_precision_per_epoch.png'))
plt.close()

# Plot and save Spike Detection Recall
plt.figure(figsize=(10, 6))
plt.plot(stats_df['epoch'], stats_df['spike_recall'], label='Spike Recall')
plt.xlabel('Epoch')
plt.ylabel('Recall')
plt.title('Spike Detection Recall per Epoch')
plt.legend()
plt.savefig(os.path.join(RESULTS_DIR, 'spike_recall_per_epoch.png'))
plt.close()

print(f"Training complete. Figures saved in '{RESULTS_DIR}'.")

# ============================
# 13. Analyze Validation Errors
# ============================

# Convert validation_errors to a DataFrame
validation_errors_df = pd.DataFrame(validation_errors)

# Calculate total error (BPM + Steps)
validation_errors_df['total_error'] = validation_errors_df['bpm_error'] + validation_errors_df['steps_error']

# Check the size of validation_errors
print(f"Number of validation error records: {len(validation_errors_df)}")

# Check for any missing or NaN values
print(f"Number of NaN values in total_error: {validation_errors_df['total_error'].isna().sum()}")

# Define the number of top samples to select
top_N_samples = 10  # Adjust as needed

# Sort validation errors by total_error
validation_errors_sorted = validation_errors_df.sort_values(by='total_error', ascending=True)

# Select top N samples with lowest total_error (best samples)
best_samples = validation_errors_sorted.head(top_N_samples)

# Select top N samples with highest total_error (worst samples)
worst_samples = validation_errors_sorted.tail(top_N_samples)

print(f"Selected Best Samples:\n{best_samples[['user_id', 'total_error']]}\n")
print(f"Selected Worst Samples:\n{worst_samples[['user_id', 'total_error']]}\n")

# Function to plot predictions for individual samples
def plot_prediction_sample(sample, index, category):
    """
    Plots the true vs predicted BPM and Steps for a single sample.
    """
    bpm_pred = sample['bpm_pred']
    steps_pred = sample['steps_pred']
    bpm_true = sample['bpm_true']
    steps_true = sample['steps_true']
    datetime = sample['datetime']
    user_id = sample['user_id']
    total_error = sample['total_error']

    # Create subplots
    fig, axs = plt.subplots(2, 2, figsize=(15, 10))
    fig.suptitle(f'User ID: {user_id} | Sample {index+1} | Datetime: {datetime}\nTotal Error: {total_error:.2f}', fontsize=16)

    # Plot BPM Prediction vs True
    axs[0, 0].plot(range(WINDOW_SIZE), bpm_true, label='True BPM', marker='o')
    axs[0, 0].plot(range(WINDOW_SIZE), bpm_pred, label='Predicted BPM', marker='x')
    axs[0, 0].set_title('BPM Prediction vs True')
    axs[0, 0].legend()

    # Plot Steps Prediction vs True
    axs[0, 1].plot(range(WINDOW_SIZE), steps_true, label='True Steps', marker='o')
    axs[0, 1].plot(range(WINDOW_SIZE), steps_pred, label='Predicted Steps', marker='x', color='orange')
    axs[0, 1].set_title('Steps Prediction vs True')
    axs[0, 1].legend()

    # Plot BPM Absolute Error
    axs[1, 0].bar(range(WINDOW_SIZE), np.abs(bpm_true - bpm_pred))
    axs[1, 0].set_title('BPM Absolute Error')

    # Plot Steps Absolute Error
    axs[1, 1].bar(range(WINDOW_SIZE), np.abs(steps_true - steps_pred), color='orange')
    axs[1, 1].set_title('Steps Absolute Error')

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plot_filename = os.path.join(ANALYSIS_DIR, f'{category}_sample_{index+1}_user_{user_id}.png')
    plt.savefig(plot_filename)
    plt.close()

    print(f'Saved plot for {category.capitalize()} Sample {index+1}, User {user_id} at {plot_filename}')

# ============================
# 14. Plot Best and Worst Samples
# ============================

# Plot the best samples
for idx, sample in best_samples.iterrows():
    plot_prediction_sample(sample, index=idx, category='best')

# Plot the worst samples
for idx, sample in worst_samples.iterrows():
    plot_prediction_sample(sample, index=idx, category='worst')

# ============================
# 15. Identify and Visualize Patterns in Errors
# ============================

# Add hour of day if not already present
if 'hour' not in validation_errors_df.columns:
    validation_errors_df['hour'] = validation_errors_df['datetime'].dt.hour

# Plot total error by hour of day
plt.figure(figsize=(12, 6))
plt.scatter(validation_errors_df['hour'], validation_errors_df['total_error'], alpha=0.5)
plt.xlabel('Hour of Day')
plt.ylabel('Total Error (BPM + Steps)')
plt.title('Total Prediction Error by Hour of Day')
plt.savefig(os.path.join(ANALYSIS_DIR, 'error_by_hour.png'))
plt.close()
print(f'Saved error by hour plot at {os.path.join(ANALYSIS_DIR, "error_by_hour.png")}')
