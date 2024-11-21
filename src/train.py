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
BATCH_SIZE = 64          # Batch size for training and validation
NUM_EPOCHS = 150          # Number of training epochs
PATIENCE = 25             # Patience for early stopping
LEARNING_RATE = 0.001     # Learning rate for optimizer
WEIGHT_DECAY = 1e-5       # Weight decay for optimizer
SPIKE_WEIGHT = 2.0        # Weight for spike samples in loss function
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
        user_id = sample['user_id']
        bpm_target_original = torch.tensor(sample['bpm_target_original'], dtype=torch.float32)
        steps_target_original = torch.tensor(sample['steps_target_original'], dtype=torch.float32)
        datetime = sample.get('datetime', None)
        hour = sample.get('hour', 0)
        day_of_week = sample.get('day_of_week', 0)
        weight = sample.get('weight', 1.0)
        contains_spike = sample.get('contains_spike', False)
        return bpm_input, steps_input, bpm_target, steps_target, user_id, bpm_target_original, steps_target_original, datetime, weight, hour, day_of_week, contains_spike

def create_data_samples(df):
    """
    Creates data samples by sliding a window of INPUT_WINDOWS to predict the PREDICT_WINDOW.
    Labels samples that contain spikes and includes time-based features.
    """
    data_samples = []
    user_groups = df.groupby('id')
    
    # Precompute mean and std steps per user for spike detection
    user_stats = df.groupby('id')['steps'].agg(['mean', 'std']).to_dict('index')
    
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
            # Define spike threshold
            user_mean = user_stats[user_id]['mean']
            user_std = user_stats[user_id]['std']
            spike_threshold = user_mean + 3 * user_std
            # Check if any step count in the target window exceeds the threshold
            contains_spike = np.any(steps_target > spike_threshold)
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
                'bpm_target_original': target_window['bpm'].values,
                'steps_target_original': target_window['steps'].values,
                'user_id': user_id,
                'datetime': datetime,  # Now a pandas.Timestamp
                'contains_spike': contains_spike,
                'hour': hour,
                'day_of_week': day_of_week
            })
    return data_samples

# Generate data samples
data_samples = create_data_samples(df)

# ============================
# 5. Train-Validation Split
# ============================

# Implement user-level train-validation split to prevent data leakage
unique_user_ids = df['id'].unique()
train_user_ids, val_user_ids = train_test_split(unique_user_ids, test_size=0.2, random_state=SEED)

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

class BalancedPerUserBatchSampler(Sampler):
    """
    Custom sampler to yield balanced batches of spike and non-spike samples per user.
    Ensures that each batch contains an equal number of spike and non-spike samples.
    """
    def __init__(self, user_data, data_samples, batch_size):
        self.user_data = user_data
        self.data_samples = data_samples
        self.batch_size = batch_size
        self.samples_per_class = batch_size // 2  # Half spikes, half non-spikes
    
    def __iter__(self):
        for user_id, indices in self.user_data.items():
            # Separate spike and non-spike indices
            spike_indices = [idx for idx in indices if self.data_samples[idx]['contains_spike']]
            non_spike_indices = [idx for idx in indices if not self.data_samples[idx]['contains_spike']]
            
            # Calculate the number of batches
            num_spikes = len(spike_indices)
            num_non_spikes = len(non_spike_indices)
            num_batches = min(num_spikes, num_non_spikes) // self.samples_per_class
            
            # Shuffle the indices
            random.shuffle(spike_indices)
            random.shuffle(non_spike_indices)
            
            for i in range(num_batches):
                batch_spikes = spike_indices[i * self.samples_per_class : (i + 1) * self.samples_per_class]
                batch_non_spikes = non_spike_indices[i * self.samples_per_class : (i + 1) * self.samples_per_class]
                batch = batch_spikes + batch_non_spikes
                random.shuffle(batch)  # Shuffle within the batch
                yield batch
            
            # Handle remaining samples by oversampling if necessary
            remaining_spikes = spike_indices[num_batches * self.samples_per_class:]
            remaining_non_spikes = non_spike_indices[num_batches * self.samples_per_class:]
            if remaining_spikes or remaining_non_spikes:
                # To maintain balance, oversample the minority class
                if len(remaining_spikes) < self.samples_per_class and len(remaining_non_spikes) < self.samples_per_class:
                    # Both classes have fewer samples than required for a batch
                    # Skip this incomplete batch
                    continue
                elif len(remaining_spikes) < self.samples_per_class:
                    # Oversample spikes
                    oversample_spikes = remaining_spikes * (self.samples_per_class // len(remaining_spikes) + 1)
                    batch_spikes = oversample_spikes[:self.samples_per_class]
                    batch_non_spikes = remaining_non_spikes[:self.samples_per_class]
                elif len(remaining_non_spikes) < self.samples_per_class:
                    # Oversample non-spikes
                    oversample_non_spikes = remaining_non_spikes * (self.samples_per_class // len(remaining_non_spikes) + 1)
                    batch_non_spikes = oversample_non_spikes[:self.samples_per_class]
                    batch_spikes = remaining_spikes[:self.samples_per_class]
                else:
                    # Both classes have enough samples
                    batch_spikes = remaining_spikes[:self.samples_per_class]
                    batch_non_spikes = remaining_non_spikes[:self.samples_per_class]
                
                batch = batch_spikes + batch_non_spikes
                random.shuffle(batch)
                yield batch
    
    def __len__(self):
        total_batches = 0
        for user_id, indices in self.user_data.items():
            spike_count = len([idx for idx in indices if self.data_samples[idx]['contains_spike']])
            non_spike_count = len(indices) - spike_count
            total_batches += min(spike_count, non_spike_count) // (self.batch_size // 2)
        return total_batches

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
    datetimes = [s[7] for s in batch]
    weights = [s[8] for s in batch]
    hours = torch.tensor([s[9] for s in batch], dtype=torch.long)
    days_of_week = torch.tensor([s[10] for s in batch], dtype=torch.long)
    contains_spike = torch.tensor([float(s[11]) for s in batch], dtype=torch.float32)  # Binary label
    return bpm_inputs, steps_inputs, bpm_targets, steps_targets, user_ids, bpm_targets_original, steps_targets_original, datetimes, weights, hours, days_of_week, contains_spike

# ============================
# 6B. Balanced Sampling of Spike Samples in Training Set
# ============================

# Separate spike and non-spike samples
spike_train_samples = [sample for sample in train_samples if sample['contains_spike']]
non_spike_train_samples = [sample for sample in train_samples if not sample['contains_spike']]

print(f"Number of training samples: {len(train_samples)}")
print(f" - Spike samples: {len(spike_train_samples)}")
print(f" - Non-spike samples: {len(non_spike_train_samples)}")

# Determine oversampling factor to balance spike and non-spike samples
if len(spike_train_samples) > 0:
    oversample_factor = int(len(non_spike_train_samples) / len(spike_train_samples))
    oversample_factor = max(oversample_factor, 1)  # Ensure at least 1
else:
    oversample_factor = 1
print(f"Oversampling spike samples by factor of {oversample_factor}")

# Oversample spike samples
augmented_spike_train_samples = spike_train_samples * oversample_factor
augmented_train_samples = non_spike_train_samples + augmented_spike_train_samples

print(f"Total training samples after oversampling: {len(augmented_train_samples)}")

# Shuffle the augmented training samples
random.shuffle(augmented_train_samples)

# Assign higher weight to spike samples
for sample in augmented_train_samples:
    sample['weight'] = SPIKE_WEIGHT if sample['contains_spike'] else 1.0  # Assign higher weight to spike samples

# Update the training dataset
train_dataset = UserDataset(augmented_train_samples)
train_sampler = BalancedPerUserBatchSampler(create_user_datasets(augmented_train_samples), augmented_train_samples, batch_size=BATCH_SIZE)
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)

# Update the validation dataset
val_dataset = UserDataset(val_samples)
val_sampler = BalancedPerUserBatchSampler(create_user_datasets(val_samples), val_samples, batch_size=BATCH_SIZE)
val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=collate_fn)

# ============================
# 7. Model Definition
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
    and a Classification head for spike detection.
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
        
        # Classification Head for Spike Detection
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1),
            nn.Sigmoid()  # Outputs probability between 0 and 1
        )
        
        # Embedding for Time-Based Features
        self.hour_embedding = nn.Embedding(24, 8)   # Embedding for hour (0-23)
        self.day_embedding = nn.Embedding(7, 8)     # Embedding for day of week (0-6)
    
    def forward(self, bpm_input, steps_input, hour, day_of_week):
        """
        Forward pass for the forecasting model.
        """
        # Correct Input Dimensions: [B, C, L] where C=INPUT_WINDOWS, L=WINDOW_SIZE
        # No permutation needed if bpm_input is [B, INPUT_WINDOWS, WINDOW_SIZE]
        bpm_seq = bpm_input  # Shape: [B, 3, 24]
        
        # CNN Encoder for BPM
        bpm_cnn_output = self.bpm_cnn(bpm_seq)  # Shape: [B, 64, 24]
        
        # LSTM Encoder for BPM
        bpm_lstm_output, _ = self.bpm_lstm(bpm_cnn_output.permute(0, 2, 1))  # Shape: [B, 24, 128]
        bpm_hidden = bpm_lstm_output[:, -1, :]  # Shape: [B, 128]
        
        # Attention for BPM
        bpm_attn_output, _ = self.bpm_attention(bpm_hidden.unsqueeze(1), bpm_lstm_output, bpm_lstm_output)  # Shape: [B, 1, 128]
        bpm_attn_output = bpm_attn_output.squeeze(1)  # Shape: [B, 128]
        
        # CNN Encoder for Steps
        steps_seq = steps_input  # Shape: [B, 3, 24]
        steps_cnn_output = self.steps_cnn(steps_seq)  # Shape: [B, 64, 24]
        
        # LSTM Encoder for Steps
        steps_lstm_output, _ = self.steps_lstm(steps_cnn_output.permute(0, 2, 1))  # Shape: [B, 24, 128]
        steps_hidden = steps_lstm_output[:, -1, :]  # Shape: [B, 128]
        
        # Attention for Steps
        steps_attn_output, _ = self.steps_attention(steps_hidden.unsqueeze(1), steps_lstm_output, steps_lstm_output)  # Shape: [B, 1, 128]
        steps_attn_output = steps_attn_output.squeeze(1)  # Shape: [B, 128]
        
        # Concatenate the attention outputs from BPM and Steps
        combined_features = torch.cat((bpm_attn_output, steps_attn_output), dim=1)  # Shape: [B, 256]
        
        # Fully Connected Layers for Fusion
        fused_features = self.fc(combined_features)  # Shape: [B, 128]
        
        # Separate Decoders
        bpm_pred = self.bpm_decoder(fused_features)      # Shape: [B, 24]
        steps_pred = self.steps_decoder(fused_features)  # Shape: [B, 24]
        
        # Process time-based features
        hour_embedded = self.hour_embedding(hour.long())           # Shape: [B, 8]
        day_embedded = self.day_embedding(day_of_week.long())      # Shape: [B, 8]
        time_features = torch.cat((hour_embedded, day_embedded), dim=1)  # Shape: [B, 16]
        
        # Classification Head
        spike_prob = self.classifier(fused_features).squeeze(1)   # Shape: [B]
        
        return bpm_pred, steps_pred, spike_prob

# Instantiate the model and move it to the device
model = ForecastingModel().to(DEVICE)
print(f'Using device: {DEVICE}')

# ============================
# 8. Loss Function Definition
# ============================

class FocalLoss(nn.Module):
    def __init__(self, alpha=1.0, gamma=2.0, reduction='mean'):
        """
        Focal Loss for binary classification.
        """
        super(FocalLoss, self).__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction
        self.bce_loss = nn.BCELoss(reduction='none')
    
    def forward(self, inputs, targets):
        """
        Forward pass for Focal Loss.
        """
        bce_loss = self.bce_loss(inputs, targets)
        pt = torch.where(targets == 1, inputs, 1 - inputs)
        focal_loss = self.alpha * (1 - pt) ** self.gamma * bce_loss
        
        if self.reduction == 'mean':
            return focal_loss.mean()
        elif self.reduction == 'sum':
            return focal_loss.sum()
        else:
            return focal_loss

class CombinedLoss(nn.Module):
    def __init__(self, alpha=1.0, beta=2.0, gamma=2.0):
        """
        Combined Loss for regression and classification tasks.
        """
        super(CombinedLoss, self).__init__()
        self.alpha = alpha  # Weight for regression loss
        self.beta = beta    # Weight for classification loss
        self.regression_loss = nn.MSELoss(reduction='mean')  # For BPM and Steps
        self.classification_loss = FocalLoss(alpha=1.0, gamma=gamma, reduction='mean')  # For spike detection
    
    def forward(self, bpm_pred, bpm_target, steps_pred, steps_target, spike_prob, spike_label, step_weights, spike_weight=1.0):
        """
        Compute the combined loss.
        """
        # Regression Loss
        loss_bpm = self.regression_loss(bpm_pred, bpm_target)
        loss_steps = self.regression_loss(steps_pred, steps_target)
        loss_regression = (loss_bpm + loss_steps) / 2  # Average to balance
        
        # Classification Loss
        loss_classification = self.classification_loss(spike_prob, spike_label)
        loss_classification = loss_classification * spike_weight  # Weighting spikes
        
        # Total Loss
        total_loss = self.alpha * loss_regression + self.beta * loss_classification
        return total_loss

# Instantiate the loss function with increased beta to prioritize classification
criterion = CombinedLoss(alpha=1.0, beta=2.0, gamma=2.0)

# ============================
# 9. Optimizer and Scheduler Setup
# ============================

# Initialize the optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)

# Initialize the learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

# ============================
# 10. Training and Validation Loop
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
    'spike_f1': []
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
        bpm_input, steps_input, bpm_target, steps_target, _, _, _, _, weights, hours, days_of_week, contains_spike = batch
        optimizer.zero_grad()
        
        # Move tensors to the device
        bpm_input = bpm_input.to(DEVICE)
        steps_input = steps_input.to(DEVICE)
        bpm_target = bpm_target.to(DEVICE)
        steps_target = steps_target.to(DEVICE)
        hours = hours.to(DEVICE)
        days_of_week = days_of_week.to(DEVICE)
        contains_spike = contains_spike.to(DEVICE)
        
        # Ensure weights are converted to tensors
        weights = torch.as_tensor(weights, dtype=torch.float32).to(DEVICE)
        
        # Forward pass
        bpm_pred, steps_pred, spike_prob = model(bpm_input, steps_input, hours, days_of_week)
        
        # Compute combined loss
        loss = criterion(bpm_pred, bpm_target, steps_pred, steps_target, spike_prob, contains_spike, weights, spike_weight=SPIKE_WEIGHT)
        
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
            bpm_input, steps_input, bpm_target, steps_target, user_id, bpm_target_original, steps_target_original, datetimes, weights, hours, days_of_week, contains_spike = batch
            # Move tensors to the device
            bpm_input = bpm_input.to(DEVICE)
            steps_input = steps_input.to(DEVICE)
            bpm_target = bpm_target.to(DEVICE)
            steps_target = steps_target.to(DEVICE)
            hours = hours.to(DEVICE)
            days_of_week = days_of_week.to(DEVICE)
            contains_spike = contains_spike.to(DEVICE)
            
            # Ensure weights are converted to tensors
            weights = torch.as_tensor(weights, dtype=torch.float32).to(DEVICE)
            
            # Forward pass
            bpm_pred_scaled, steps_pred_scaled, spike_prob = model(bpm_input, steps_input, hours, days_of_week)
            
            # Compute combined loss
            loss = criterion(bpm_pred_scaled, bpm_target, steps_pred_scaled, steps_target, spike_prob, contains_spike, weights, spike_weight=SPIKE_WEIGHT)
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
                all_spike_probs.append(spike_prob[i].cpu().numpy())
                all_spike_labels.append(contains_spike[i].cpu().numpy())
                
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
    spike_preds = (spike_probs >= 0.5).astype(int)
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
    
    # Append epoch's validation errors to the main list
    validation_errors.extend(validation_errors_epoch)
    
    # ---------------------
    # Check for Improvement
    # ---------------------
    if avg_val_loss < best_val_loss:
        best_val_loss = avg_val_loss
        epochs_no_improve = 0
        torch.save(model.state_dict(), model_save_path)
        print(f'Validation loss decreased. Saving model to {model_save_path}')
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
          f'Spike Acc: {spike_accuracy:.2f}, '
          f'Spike Precision: {spike_precision:.2f}, '
          f'Spike Recall: {spike_recall:.2f}, '
          f'Spike F1: {spike_f1:.2f}')

# ============================
# 11. Generate Charts
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
# 12. Analyze Validation Errors
# ============================

# Convert validation_errors to a DataFrame
validation_errors_df = pd.DataFrame(validation_errors)

# Calculate total error (BPM + Steps)
validation_errors_df['total_error'] = validation_errors_df['bpm_error'] + validation_errors_df['steps_error']

# Check the size of validation_errors
print(f"Number of validation error records: {len(validation_errors_df)}")

# Check for any missing or NaN values
print(f"Number of NaN values in total_error: {validation_errors_df['total_error'].isna().sum()}")

# Aggregate errors per user by computing the mean total error
user_error_df = validation_errors_df.groupby('user_id')['total_error'].mean().reset_index()

# Check the number of unique users
num_unique_users = user_error_df['user_id'].nunique()
print(f"Number of unique users in validation set: {num_unique_users}")

# Sort users by total_error
user_error_df_sorted = user_error_df.sort_values(by='total_error', ascending=True)

# Check unique total_error values and their counts
unique_errors = user_error_df_sorted['total_error'].unique()
num_unique_errors = len(unique_errors)
print(f"Number of unique total_error values: {num_unique_errors}")
print(f"Unique total_error values: {unique_errors}")

# Adjust top_N based on the number of unique users
top_N = 10
half_users = num_unique_users // 2
if num_unique_users < 2 * top_N:
    top_N = half_users
    print(f"Adjusted top_N to {top_N} due to insufficient unique users.")

# Select top N best and top N worst users
best_users = user_error_df_sorted.head(top_N)
worst_users = user_error_df_sorted.tail(top_N)

print(f"Selected Best Users:\n{best_users}\n")
print(f"Selected Worst Users:\n{worst_users}\n")

# Display the first and last few entries to verify sorting
print("First few users with lowest total_error (Best Users):")
print(user_error_df_sorted.head(5))
print("\nLast few users with highest total_error (Worst Users):")
print(user_error_df_sorted.tail(5))

# Function to plot predictions without input windows
def plot_prediction_user(user_id, user_samples, index, category):
    """
    Plots the true vs predicted BPM and Steps for a user.
    """
    # Iterate over user's samples and plot them
    for i, sample in enumerate(user_samples):
        bpm_pred = sample['bpm_pred']
        steps_pred = sample['steps_pred']
        bpm_true = sample['bpm_true']
        steps_true = sample['steps_true']
        datetime = sample['datetime']

        # Create subplots
        fig, axs = plt.subplots(2, 2, figsize=(15, 10))
        fig.suptitle(f'User ID: {user_id} | Sample {i+1} | Datetime: {datetime}', fontsize=16)

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
        plot_filename = os.path.join(ANALYSIS_DIR, f'{category}_user_{user_id}_sample_{i+1}.png')
        plt.savefig(plot_filename)
        plt.close()

        print(f'Saved plot for {category} User {user_id}, Sample {i+1} at {plot_filename}')

# Plot predictions for best users
for idx, row in best_users.iterrows():
    user_id = row['user_id']
    # Get all samples for this user
    user_samples = validation_errors_df[validation_errors_df['user_id'] == user_id].to_dict(orient='records')
    # Limit the number of samples to plot per user if needed
    samples_to_plot = user_samples[:2]  # Adjust as needed
    for sample_idx, sample in enumerate(samples_to_plot):
        plot_prediction_user(user_id, [sample], sample_idx, "best")

# Plot predictions for worst users
for idx, row in worst_users.iterrows():
    user_id = row['user_id']
    # Get all samples for this user
    user_samples = validation_errors_df[validation_errors_df['user_id'] == user_id].to_dict(orient='records')
    # Limit the number of samples to plot per user if needed
    samples_to_plot = user_samples[:2]  # Adjust as needed
    for sample_idx, sample in enumerate(samples_to_plot):
        plot_prediction_user(user_id, [sample], sample_idx, "worst")

# ============================
# 13. Identify and Visualize Patterns in Errors
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
