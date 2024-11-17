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
from sklearn.metrics import confusion_matrix, accuracy_score

# Suppress warnings for cleaner output
warnings.filterwarnings('ignore')

# ============================
# 1. Data Loading and Preprocessing
# ============================

# Load your new DataFrame with 'heartCondition'
df = pd.read_csv("data/myheartcounts.csv", parse_dates=['date'])

# Preprocess data
df['datetime'] = pd.to_datetime(df['date']) + pd.to_timedelta(df['hour'], unit='h')
df = df.sort_values(by=['id', 'datetime'])
df = df.dropna()
df = df.reset_index(drop=True)

# ============================
# 2. Parameter Definitions
# ============================

WINDOW_SIZE = 2        # Number of time steps in each window
INPUT_WINDOWS = 3      # Number of input windows to consider for prediction
PREDICT_WINDOW = 1     # Number of windows to predict
BATCH_SIZE = 128       # Batch size for training and validation

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
    Includes heart condition label for classification.
    """
    def __init__(self, data_list, user_labels):
        self.data = data_list
        self.user_labels = user_labels  # Dictionary mapping user_id to label
                
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        sample = self.data[idx]
        bpm_input = torch.tensor(sample['bpm_input'], dtype=torch.float32)
        steps_input = torch.tensor(sample['steps_input'], dtype=torch.float32)
        user_id = sample['user_id']
        heart_condition = torch.tensor(self.user_labels[user_id], dtype=torch.float32)  # Binary label
        return bpm_input, steps_input, user_id, heart_condition

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
                'user_id': user_id
            })
    return data_samples

# ============================
# 5. Balancing the Dataset
# ============================

# Extract unique users with their heartCondition
user_labels_df = df.groupby('id')['heartCondition'].first().reset_index()

# Ensure that 'heartCondition' is consistent per user
assert user_labels_df['heartCondition'].nunique() == 2 or user_labels_df['heartCondition'].nunique() == 1, "Inconsistent heartCondition labels per user."

# Separate users by class
heart_condition_users = user_labels_df[user_labels_df['heartCondition'] == True]['id'].tolist()
no_heart_condition_users = user_labels_df[user_labels_df['heartCondition'] == False]['id'].tolist()

# Determine number of users to undersample
num_heart_condition = len(heart_condition_users)
num_no_heart_condition = len(no_heart_condition_users)

if num_no_heart_condition > num_heart_condition:
    no_heart_condition_users = train_test_split(no_heart_condition_users, 
                                               train_size=num_heart_condition, 
                                               random_state=42)[0]

# Combine balanced user lists
balanced_users = heart_condition_users + no_heart_condition_users
balanced_labels = {user_id:1.0 for user_id in heart_condition_users}
balanced_labels.update({user_id:0.0 for user_id in no_heart_condition_users})

# Filter the dataframe to include only balanced users
df_balanced = df[df['id'].isin(balanced_users)].reset_index(drop=True)

# ============================
# 6. Train-Validation-Test Split
# ============================

# Split users into train, val, test (e.g., 70%, 15%, 15%)
train_users, temp_users = train_test_split(
    balanced_users, test_size=0.3, stratify=[balanced_labels[user] for user in balanced_users], random_state=42)

val_users, test_users = train_test_split(
    temp_users, test_size=0.5, stratify=[balanced_labels[user] for user in temp_users], random_state=42)

# Create label dictionaries for each split
train_labels = {user_id:balanced_labels[user_id] for user_id in train_users}
val_labels = {user_id:balanced_labels[user_id] for user_id in val_users}
test_labels = {user_id:balanced_labels[user_id] for user_id in test_users}

# ============================
# 7. Create Data Samples
# ============================

# Generate data samples
data_samples = create_data_samples(df_balanced)

# Assign samples to splits based on user_id
train_samples = [sample for sample in data_samples if sample['user_id'] in train_users]
val_samples = [sample for sample in data_samples if sample['user_id'] in val_users]
test_samples = [sample for sample in data_samples if sample['user_id'] in test_users]

# ============================
# 8. Sampler and DataLoader Setup
# ============================

def create_user_datasets(data_samples):
    """
    Organizes data samples by user for the sampler.
    """
    user_data = defaultdict(list)
    for idx, sample in enumerate(data_samples):
        user_data[sample['user_id']].append(idx)
    return user_data

# Organize training, validation, and test samples by user
train_user_data = create_user_datasets(train_samples)
val_user_data = create_user_datasets(val_samples)
test_user_data = create_user_datasets(test_samples)

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
train_dataset = UserDataset(train_samples, train_labels)
val_dataset = UserDataset(val_samples, val_labels)
test_dataset = UserDataset(test_samples, test_labels)

# Instantiate samplers
train_sampler = PerUserBatchSampler(train_user_data, batch_size=BATCH_SIZE)
val_sampler = PerUserBatchSampler(val_user_data, batch_size=BATCH_SIZE)
test_sampler = PerUserBatchSampler(test_user_data, batch_size=BATCH_SIZE)

# Define collate function
def collate_fn(batch):
    """
    Custom collate function to handle batching of samples.
    """
    bpm_inputs = torch.stack([s[0] for s in batch])
    steps_inputs = torch.stack([s[1] for s in batch])
    user_ids = [s[2] for s in batch]
    heart_conditions = torch.stack([s[3] for s in batch])  # Binary labels
    return bpm_inputs, steps_inputs, user_ids, heart_conditions

# Create DataLoaders
train_loader = DataLoader(train_dataset, batch_sampler=train_sampler, collate_fn=collate_fn)
val_loader = DataLoader(val_dataset, batch_sampler=val_sampler, collate_fn=collate_fn)
test_loader = DataLoader(test_dataset, batch_sampler=test_sampler, collate_fn=collate_fn)

# ============================
# 9. Compute Weights Based on Average Size
# ============================

# Calculate the average absolute value of BPM and steps in the training set
avg_bpm = df_balanced[df_balanced['id'].isin(train_users)]['bpm'].abs().mean()
avg_steps = df_balanced[df_balanced['id'].isin(train_users)]['steps'].abs().mean()

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
# 10. Updated Model Definition without Decoders
# ============================

class ClassifierModel(nn.Module):
    """
    Forecasting model with separate 1D CNN and LSTM encoders for BPM and Steps,
    followed by a classifier for heart condition prediction.
    Decoders are removed to focus solely on classification.
    """
    def __init__(self, backbone_path="results/saved_models/forecasting_backbone.pth"):
        super(ClassifierModel, self).__init__()
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
        
        # Fully Connected Layers for Fusion
        self.fc = nn.Sequential(
            nn.Linear(256, 128),
            nn.ReLU(),
            nn.Dropout(0.3)
        )
        
        # Classifier for Heart Condition
        self.classifier = nn.Sequential(
            nn.Linear(128, 64),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(64, 1)  # Output logit for binary classification
        )
        
        # Load pretrained backbone weights if provided
        if backbone_path and os.path.exists(backbone_path):
            self.load_backbone(backbone_path)
            print(f"Loaded backbone weights from {backbone_path}")
        
    def load_backbone(self, backbone_path):
        """
        Loads the backbone weights (forecasting model) from a saved state dict,
        excluding the classifier layers to avoid size mismatches.
        """
        backbone_state = torch.load(backbone_path, map_location='cpu')
        
        # Exclude decoder parameters if any
        backbone_state = {k: v for k, v in backbone_state.items() if not k.startswith('bpm_decoder.') 
                                                       and not k.startswith('steps_decoder.') 
                                                       and not k.startswith('classifier.')}
        
        # Load the state dict with strict=False to ignore missing keys (classifier layers)
        self.load_state_dict(backbone_state, strict=False)
        
    def forward(self, bpm_input, steps_input):
        """
        Forward pass for the classification model.
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
        
        # Classifier
        heart_condition_logit = self.classifier(fused_features)  # Shape: [B, 1]
        
        return heart_condition_logit

# ============================
# 11. Model Initialization and Loading
# ============================

# Check if GPU is available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Instantiate the model and move it to the device
backbone_path = 'results/saved_models/forecasting_backbone.pth'  # Update path if different
model = ClassifierModel(backbone_path=backbone_path).to(device)

# ============================
# 12. Training Setup
# ============================

# Define the loss functions
classification_criterion = nn.BCEWithLogitsLoss() # Binary Cross-Entropy Loss for classification

# Initialize the optimizer (fine-tuning all parameters)
optimizer = torch.optim.Adam(model.parameters(), lr=0.001, weight_decay=1e-5)  # Added weight_decay for regularization

# Initialize the learning rate scheduler
scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=20, gamma=0.1)

num_epochs = 100

# ============================
# 13. Training and Validation Loop
# ============================

# Early Stopping Parameters
patience = 15
best_val_loss = float('inf')
epochs_no_improve = 0
early_stop = False

# Directory to save the best model
save_dir = 'results/saved_models'
os.makedirs(save_dir, exist_ok=True)
model_save_path = os.path.join(save_dir, 'heart_condition_classifier.pth')

# Initialize statistics dictionary
stats = {
    'epoch': [],
    'train_classification_loss': [],
    'val_classification_loss': [],
    'classification_accuracy': []
}

for epoch in range(num_epochs):
    if early_stop:
        break
    
    # ---------------------
    # Training Phase
    # ---------------------
    model.train()
    epoch_classification_loss = 0.0
    correct_predictions = 0
    total_predictions = 0
    for bpm_input, steps_input, user_id, heart_condition in train_loader:
        optimizer.zero_grad()
        
        # Move tensors to the device
        bpm_input = bpm_input.to(device)            # Shape: [B, INPUT_WINDOWS, WINDOW_SIZE]
        steps_input = steps_input.to(device)        # Shape: [B, INPUT_WINDOWS, WINDOW_SIZE]
        heart_condition = heart_condition.to(device) # Shape: [B]
        
        # Forward pass
        heart_condition_logit = model(bpm_input, steps_input).squeeze(-1)  # Shape: [B]
        
        # Compute classification loss
        classification_loss = classification_criterion(heart_condition_logit, heart_condition)
        
        # Backward pass and optimization
        classification_loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        
        epoch_classification_loss += classification_loss.item()
        
        # Compute training accuracy
        predictions = torch.sigmoid(heart_condition_logit) >= 0.5
        correct_predictions += (predictions.float() == heart_condition).sum().item()
        total_predictions += heart_condition.size(0)
    
    # Update learning rate
    scheduler.step()
    
    # ---------------------
    # Validation Phase
    # ---------------------
    model.eval()
    val_classification_loss = 0.0
    val_correct_predictions = 0
    val_total_predictions = 0
    with torch.no_grad():
        for bpm_input, steps_input, user_id, heart_condition in val_loader:
            # Move tensors to the device
            bpm_input = bpm_input.to(device)
            steps_input = steps_input.to(device)
            heart_condition = heart_condition.to(device)
            
            # Forward pass
            heart_condition_logit = model(bpm_input, steps_input).squeeze(-1)  # Shape: [B]
            
            # Compute classification loss
            classification_loss = classification_criterion(heart_condition_logit, heart_condition)
            val_classification_loss += classification_loss.item()
            
            # Compute validation accuracy
            predictions = torch.sigmoid(heart_condition_logit) >= 0.5
            val_correct_predictions += (predictions.float() == heart_condition).sum().item()
            val_total_predictions += heart_condition.size(0)
    
    # ---------------------
    # Calculate Average Losses and Accuracies
    # ---------------------
    avg_train_classification_loss = epoch_classification_loss / len(train_loader)
    avg_val_classification_loss = val_classification_loss / len(val_loader)
    train_accuracy = (correct_predictions / total_predictions) * 100
    val_accuracy = (val_correct_predictions / val_total_predictions) * 100
    
    # ---------------------
    # Check for Improvement
    # ---------------------
    if avg_val_classification_loss < best_val_loss:
        best_val_loss = avg_val_classification_loss
        epochs_no_improve = 0
        # Save the best model
        torch.save(model.state_dict(), model_save_path)
    else:
        epochs_no_improve += 1
    
    # Check if early stopping should be triggered
    if epochs_no_improve >= patience:
        print(f'Early stopping triggered after {patience} epochs with no improvement.')
        early_stop = True
    
    # ---------------------
    # Log epoch statistics
    # ---------------------
    stats['epoch'].append(epoch + 1)
    stats['train_classification_loss'].append(avg_train_classification_loss)
    stats['val_classification_loss'].append(avg_val_classification_loss)
    stats['classification_accuracy'].append(val_accuracy)
    
    # ---------------------
    # Print epoch statistics
    # ---------------------
    print(f'Epoch {epoch+1}/{num_epochs}, '
          f'Train Classification Loss: {avg_train_classification_loss:.4f}, '
          f'Val Classification Loss: {avg_val_classification_loss:.4f}, '
          f'Validation Accuracy: {val_accuracy:.2f}%')

# ============================
# 14. Save Training Statistics and Generate Plots
# ============================

# Create a directory to save the figures
results_dir = 'results/test'
os.makedirs(results_dir, exist_ok=True)

# Convert statistics to DataFrame
stats_df = pd.DataFrame(stats)

# Save DataFrame
stats_df.to_csv(os.path.join(results_dir, "training_stats.csv"), index=False)

# Plot and save Classification Loss
plt.figure(figsize=(10, 6))
plt.plot(stats_df['epoch'], stats_df['train_classification_loss'], label='Train Classification Loss')
plt.plot(stats_df['epoch'], stats_df['val_classification_loss'], label='Validation Classification Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Classification Loss per Epoch')
plt.legend()
plt.savefig(os.path.join(results_dir, 'classification_loss.png'))
plt.close()

# Plot and save Classification Accuracy
plt.figure(figsize=(10, 6))
plt.plot(stats_df['epoch'], stats_df['classification_accuracy'], label='Validation Classification Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy (%)')
plt.title('Validation Classification Accuracy per Epoch')
plt.legend()
plt.savefig(os.path.join(results_dir, 'classification_accuracy.png'))
plt.close()

print(f"\nTraining complete. Figures saved in '{results_dir}'.")

# ============================
# 15. Evaluation on Test Set 
# ============================

# Initialize test statistics dictionary
test_stats = {
    "Metric": ["Test Classification Loss", "Classification Accuracy (%)"],
    "Value": []
}

# Initialize lists for confusion matrix
all_true_labels = []
all_pred_labels = []

# Variables to track test performance
test_classification_loss = 0.0
total_classification_correct = 0
total_samples = 0

model.eval()

with torch.no_grad():
    for bpm_input, steps_input, user_id, heart_condition in test_loader:
        # Move tensors to device
        bpm_input, steps_input = bpm_input.to(device), steps_input.to(device)
        heart_condition = heart_condition.to(device)
        
        # Forward pass
        heart_condition_logit = model(bpm_input, steps_input).squeeze(-1)  # Shape: [B]
        
        # Compute classification loss
        classification_loss = classification_criterion(heart_condition_logit, heart_condition)
        test_classification_loss += classification_loss.item()
        
        # Compute predictions
        predictions = torch.sigmoid(heart_condition_logit) >= 0.5
        total_classification_correct += (predictions.float() == heart_condition).sum().item()
        total_samples += heart_condition.size(0)
        
        # Collect labels for confusion matrix
        all_true_labels.extend(heart_condition.cpu().numpy())
        all_pred_labels.extend(predictions.cpu().numpy())

# Calculate averages
avg_test_classification_loss = test_classification_loss / len(test_loader)
classification_accuracy = (total_classification_correct / total_samples) * 100

# Update test statistics
test_stats["Value"] = [
    avg_test_classification_loss,
    classification_accuracy
]

# Create a DataFrame for test statistics
test_stats_df = pd.DataFrame(test_stats)
test_stats_file = os.path.join(results_dir, "test_summary.csv")
test_stats_df.to_csv(test_stats_file, index=False)

print("\nTest Phase Summary Statistics:")
print(test_stats_df)

# ============================
# 16. Confusion Matrix and Plot
# ============================

# Generate confusion matrix
cm = confusion_matrix(all_true_labels, all_pred_labels)

# Create confusion matrix DataFrame for better readability
cm_df = pd.DataFrame(
    cm, 
    index=['Actual No Heart Condition', 'Actual Heart Condition'],
    columns=['Predicted No Heart Condition', 'Predicted Heart Condition']
)

# Save confusion matrix as CSV
confusion_matrix_file = os.path.join(results_dir, "confusion_matrix.csv")
cm_df.to_csv(confusion_matrix_file)

# Plot confusion matrix as a heatmap
plt.figure(figsize=(8, 6))
plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
plt.title("Confusion Matrix")
plt.colorbar()

# Add labels and ticks
tick_marks = np.arange(len(cm_df))
plt.xticks(tick_marks, cm_df.columns, rotation=45)
plt.yticks(tick_marks, cm_df.index)

# Add values to the heatmap
thresh = cm.max() / 2.0
for i, j in np.ndindex(cm.shape):
    plt.text(j, i, format(cm[i, j], "d"),
             horizontalalignment="center",
             color="white" if cm[i, j] > thresh else "black")

plt.ylabel("True Label")
plt.xlabel("Predicted Label")
plt.tight_layout()

# Save the confusion matrix plot
confusion_matrix_plot_file = os.path.join(results_dir, "confusion_matrix.png")
plt.savefig(confusion_matrix_plot_file)
plt.close()

print(f"\nConfusion matrix saved as CSV to {confusion_matrix_file} and as PNG to {confusion_matrix_plot_file}.")