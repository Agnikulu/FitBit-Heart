# test.py

import os
import glob
import warnings
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from sklearn.metrics import confusion_matrix, accuracy_score

from models import DrugClassifier
from utils import scale_per_user, user_scalers, WINDOW_SIZE  # from your `utils.py`

warnings.filterwarnings('ignore')


##############################################
# 1) LOADING & PREPROCESSING HOURLY DATA + LABELS
##############################################
"""
We assume:
 - You have your minute-level sensor data in /Biosignal
 - You have label files in /Label that indicate usage events
 - We merge them into an hourly 'df_merged' => columns [id, datetime, bpm, steps, label]
 - We do a final grouping by user + scaling => columns [bpm_scaled, steps_scaled, label]
"""
def load_and_merge_data():
    # 1A) Load hospital data, pivot to hourly BPM/Steps
    data_folder = "/home/agnik/uh-internship/data/Personalized AI Data/Biosignal"
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))

    df_list = []
    for file in csv_files:
        tmp = pd.read_csv(file, parse_dates=['time'])
        if 'participant_id' not in tmp.columns:
            base = os.path.basename(file)
            name_no_ext = os.path.splitext(base)[0]
            if name_no_ext.lower().startswith('p'):
                pid = name_no_ext[1:]
            else:
                pid = name_no_ext
            tmp['participant_id'] = int(pid)
        df_list.append(tmp)

    df_raw = pd.concat(df_list, ignore_index=True)
    df_raw.loc[df_raw['data_type'] == 'hr','data_type'] = 'heart_rate'

    # Filter for heart_rate & steps
    df_filt = df_raw[df_raw['data_type'].isin(['heart_rate','steps'])].copy()
    df_filt = df_filt.drop_duplicates(subset=['participant_id','time','data_type'])

    # Pivot => [id, time, bpm, steps]
    df_pivot = df_filt.pivot(
        index=['participant_id','time'],
        columns='data_type',
        values='value'
    ).reset_index()

    df_pivot.rename(columns={'participant_id':'id','heart_rate':'bpm'}, inplace=True)
    df_pivot['bpm'] = pd.to_numeric(df_pivot['bpm'], errors='coerce')
    df_pivot['steps'] = pd.to_numeric(df_pivot['steps'], errors='coerce')
    df_pivot.dropna(subset=['bpm','steps'], inplace=True)

    # Add date/hour
    df_pivot['datetime'] = pd.to_datetime(df_pivot['time'])
    df_pivot['date'] = df_pivot['datetime'].dt.date
    df_pivot['hour'] = df_pivot['datetime'].dt.hour

    df_sensors = df_pivot[['id','datetime','date','hour','bpm','steps']].copy()
    df_sensors = df_sensors.sort_values(by=['id','datetime']).reset_index(drop=True)

    # Hourly aggregation
    def to_hour_datetime(dateval, hourval):
        return pd.to_datetime(dateval) + pd.to_timedelta(hourval, unit='H')

    df_sensors['date'] = pd.to_datetime(df_sensors['date'])
    df_hourly = df_sensors.groupby(['id','date','hour'], as_index=False).agg({
        'bpm':'mean',
        'steps':'sum'
    })
    df_hourly['datetime'] = df_hourly.apply(
        lambda row: to_hour_datetime(row['date'], row['hour']), axis=1
    )
    df_hourly = df_hourly[['id','datetime','date','hour','bpm','steps']]
    df_hourly = df_hourly.sort_values(by=['id','datetime']).reset_index(drop=True)

    # 1B) Load usage label files from /Label
    labels_dir = "/home/agnik/uh-internship/data/Personalized AI Data/Label"
    all_label_rows = []

    subdirs = glob.glob(os.path.join(labels_dir, "ID*"))
    for sd in subdirs:
        pid_str = os.path.basename(sd)
        p_id = int(pid_str.replace("ID", ""))  # e.g. "ID14" => 14

        label_files = glob.glob(os.path.join(sd, "*.csv")) + glob.glob(os.path.join(sd, "*.xlsx"))
        for lf in label_files:
            base = os.path.basename(lf).lower()
            df_label = None
            if lf.endswith(".csv"):
                try:
                    df_label = pd.read_csv(lf)
                except:
                    continue
            elif lf.endswith(".xlsx"):
                try:
                    df_label = pd.read_excel(lf)
                except:
                    continue

            if df_label is None or df_label.empty:
                continue

            # check known time col
            time_col = None
            if 'hawaii_use_time' in df_label.columns:
                time_col = 'hawaii_use_time'
            elif 'hawaii_createdat_time' in df_label.columns:
                time_col = 'hawaii_createdat_time'
            else:
                continue

            df_label[time_col] = pd.to_datetime(df_label[time_col], errors='coerce')
            df_label.dropna(subset=[time_col], inplace=True)

            for idx, row in df_label.iterrows():
                label_str = str(row.get('crave_use_none_label','')).lower().strip()
                event_dt  = row[time_col]
                if label_str in ['use','crave']:
                    label_val = 1
                else:
                    label_val = 0

                all_label_rows.append({
                    'id': p_id,
                    'datetime': event_dt,
                    'label': label_val
                })

    df_labels = pd.DataFrame(all_label_rows)
    df_labels.dropna(subset=['datetime'], inplace=True)
    df_labels = df_labels.sort_values(['id','datetime']).reset_index(drop=True)

    df_labels['date'] = df_labels['datetime'].dt.date
    df_labels['hour'] = df_labels['datetime'].dt.hour
    df_labels['date'] = pd.to_datetime(df_labels['date'])
    df_labels_hour = df_labels.groupby(['id','date','hour'], as_index=False)['label'].max()
    df_labels_hour['datetime'] = df_labels_hour.apply(
        lambda row: to_hour_datetime(row['date'], row['hour']), axis=1
    )
    df_labels_hour = df_labels_hour[['id','datetime','label']].sort_values(['id','datetime'])
    df_labels_hour.reset_index(drop=True, inplace=True)

    # Merge sensors with labels
    df_sensors_hour = df_hourly.copy()
    df_sensors_hour['datetime'] = df_sensors_hour['datetime'].dt.tz_localize(None)
    df_labels_hour['datetime'] = df_labels_hour['datetime'].dt.tz_localize(None)

    df_merged = pd.merge(df_sensors_hour, df_labels_hour, on=['id','datetime'], how='left')
    df_merged['label'] = df_merged['label'].fillna(0).astype(int)
    return df_merged


##############################################
# 2) CREATE Classification Dataset (Windowed)
##############################################
class CravingDataset(Dataset):
    """
    Each item => (bpm_input, steps_input, user_id, label).
    We'll define chunk windows of length=WINDOW_SIZE for features,
    label=1 if any usage in that chunk, else 0.
    """
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        d = self.data_list[idx]
        # shape => [WINDOW_SIZE]
        bpm_in   = torch.tensor(d['bpm_input'], dtype=torch.float32)
        steps_in = torch.tensor(d['steps_input'], dtype=torch.float32)
        label    = torch.tensor(d['label'], dtype=torch.float32)
        uid      = d['id']
        return bpm_in, steps_in, uid, label


def create_classification_samples(df_merged, window_size=6):
    """
    For each (id):
     1) sort by datetime,
     2) break into non-overlapping windows of size=window_size,
     3) label=1 if ANY row in that chunk => label=1,
     4) store [bpm_scaled, steps_scaled].
    """
    data_samples = []
    for uid, group in df_merged.groupby('id'):
        group = group.sort_values('datetime').reset_index(drop=True)

        # Break into non-overlapping windows
        chunked = []
        for i in range(0, len(group), window_size):
            chunk = group.iloc[i:i+window_size]
            if len(chunk) == window_size:
                chunked.append(chunk)

        for ch in chunked:
            chunk_label = 1 if (ch['label'].max() == 1) else 0
            if 'bpm_scaled' in ch.columns:
                bpm_arr   = ch['bpm_scaled'].values
                steps_arr = ch['steps_scaled'].values
            else:
                # fallback
                bpm_arr   = ch['bpm'].values
                steps_arr = ch['steps'].values

            data_samples.append({
                'id': uid,
                'bpm_input': bpm_arr,
                'steps_input': steps_arr,
                'label': chunk_label,
                'datetime_start': ch['datetime'].iloc[0],
                'datetime_end':   ch['datetime'].iloc[-1]
            })
    return data_samples


##############################################
# 3) MAIN SCRIPT
##############################################
def main():
    # 1) load + merge
    df_merged = load_and_merge_data()
    print("Merged shape:", df_merged.shape)
    print("Columns:", df_merged.columns.tolist())

    # 2) scale => 'bpm_scaled','steps_scaled'
    df_merged = df_merged.groupby('id').apply(scale_per_user).reset_index(drop=True)

    # 3) create classification samples
    data_samples = create_classification_samples(df_merged, window_size=WINDOW_SIZE)
    print("Created classification samples:", len(data_samples))

    # We'll store final test results for each user
    global_results = []

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unique_ids = sorted(df_merged['id'].unique())

    for user_id in unique_ids:
        # Create a user-specific directory for test results
        user_test_dir = os.path.join("results", "test", f"user_{user_id}")
        os.makedirs(user_test_dir, exist_ok=True)

        user_data = [s for s in data_samples if s['id'] == user_id]
        if len(user_data) < 15:
            print(f"[User {user_id}] Skipping => not enough classification samples (<15).")
            continue

        # 4) train/val/test split => e.g. 70/15/15
        total_count = len(user_data)
        train_cut = int(0.70 * total_count)
        val_cut   = int(0.85 * total_count)
        user_train = user_data[:train_cut]
        user_val   = user_data[train_cut:val_cut]
        user_test  = user_data[val_cut:]

        if len(user_val) < 1 or len(user_test) < 1:
            print(f"[User {user_id}] Skipping => not enough val/test data.")
            continue

        # Build dataset
        train_ds = CravingDataset(user_train)
        val_ds   = CravingDataset(user_val)
        test_ds  = CravingDataset(user_test)

        from torch.utils.data import RandomSampler
        train_loader = DataLoader(train_ds, batch_size=32, sampler=RandomSampler(train_ds))
        val_loader   = DataLoader(val_ds, batch_size=32, sampler=RandomSampler(val_ds))
        test_loader  = DataLoader(test_ds, batch_size=32)

        # 5) Build classifier => load user forecasting checkpoint from saved models
        # Expect model to be stored in "results/saved_models/user_{user_id}/drug_classifier.pt"
        user_model_dir = os.path.join("results", "train", f"user_{user_id}")
        user_ckpt = os.path.join(user_model_dir, "personalized_ssl.pt")
        if not os.path.exists(user_ckpt):
            print(f"[User {user_id}] No saved model checkpoint at {user_ckpt} => skip classification.")
            continue

        classifier = DrugClassifier(window_size=WINDOW_SIZE).to(device)

        # Load forecasting weights into classifier's CNN+LSTM
        pers_sd = torch.load(user_ckpt, map_location=device)
        classifier.load_state_dict(pers_sd, strict=False)

        # Freeze the CNN+LSTM, only train classifier head
        for name, param in classifier.named_parameters():
            if 'classifier' in name:
                param.requires_grad = True
            else:
                param.requires_grad = False

        # Weighted BCE
        labels_train = [x['label'] for x in user_train]
        pos_count = sum(labels_train)
        neg_count = len(labels_train) - pos_count
        if pos_count == 0:
            pos_weight_val = 1.0
        else:
            pos_weight_val = neg_count / float(pos_count)

        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([pos_weight_val], dtype=torch.float).to(device))
        optimizer = torch.optim.Adam(classifier.classifier.parameters(), lr=1e-3)
        scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
        num_epochs = 20
        patience = 5

        best_val_loss = float('inf')
        epochs_no_improve = 0

        # We'll store train/val stats
        STATS = {
            'epoch': [],
            'train_loss': [],
            'val_loss': [],
            'val_acc': []
        }

        # 6) Classification training loop
        for epoch in range(num_epochs):
            if epochs_no_improve >= patience:
                print(f"[User {user_id}] Early stop at epoch={epoch+1}")
                break

            # -- TRAIN --
            classifier.train()
            sum_train_loss = 0.0
            for (bpm_in, steps_in, pids, labels) in train_loader:
                bpm_in   = bpm_in.to(device)
                steps_in = steps_in.to(device)
                labels   = labels.to(device)

                optimizer.zero_grad()
                logits = classifier(bpm_in, steps_in)
                loss   = criterion(logits, labels)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                optimizer.step()
                sum_train_loss += loss.item()

            scheduler.step()
            avg_train_loss = sum_train_loss / len(train_loader)

            # -- VAL --
            classifier.eval()
            sum_val_loss = 0.0
            val_preds = []
            val_true  = []
            with torch.no_grad():
                for (bpm_in, steps_in, pids, labels) in val_loader:
                    bpm_in   = bpm_in.to(device)
                    steps_in = steps_in.to(device)
                    labels   = labels.to(device)

                    logits = classifier(bpm_in, steps_in)
                    loss = criterion(logits, labels)
                    sum_val_loss += loss.item()

                    probs = torch.sigmoid(logits)
                    preds = (probs >= 0.5).float()
                    val_preds.extend(preds.cpu().numpy())
                    val_true.extend(labels.cpu().numpy())

            avg_val_loss = sum_val_loss / len(val_loader)
            val_preds = np.array(val_preds)
            val_true  = np.array(val_true)
            val_acc   = (val_preds == val_true).mean() * 100.0

            if avg_val_loss < best_val_loss:
                best_val_loss = avg_val_loss
                epochs_no_improve = 0
                # save best classifier for this user in their test directory
                best_path = os.path.join(user_test_dir, "drug_classifier.pth")
                torch.save(classifier.state_dict(), best_path)
            else:
                epochs_no_improve += 1

            STATS['epoch'].append(epoch+1)
            STATS['train_loss'].append(avg_train_loss)
            STATS['val_loss'].append(avg_val_loss)
            STATS['val_acc'].append(val_acc)

            print(f"[User {user_id}] Epoch={epoch+1}/{num_epochs}, "
                  f"TrainLoss={avg_train_loss:.4f}, ValLoss={avg_val_loss:.4f}, ValAcc={val_acc:.2f}%")

        # 7) Save train/val stats in user test directory
        stats_df = pd.DataFrame(STATS)
        stats_path = os.path.join(user_test_dir, "train_stats.csv")
        stats_df.to_csv(stats_path, index=False)

        # 8) Load best model => final test evaluation
        classifier.load_state_dict(torch.load(best_path, map_location=device))
        classifier.eval()

        test_preds = []
        test_true  = []
        with torch.no_grad():
            for (bpm_in, steps_in, pids, labels) in DataLoader(test_ds, batch_size=32):
                bpm_in   = bpm_in.to(device)
                steps_in = steps_in.to(device)
                labels   = labels.to(device)

                logits = classifier(bpm_in, steps_in)
                probs = torch.sigmoid(logits)
                preds = (probs >= 0.5).float()

                test_preds.extend(preds.cpu().numpy())
                test_true.extend(labels.cpu().numpy())

        test_preds = np.array(test_preds, dtype=int)
        test_true  = np.array(test_true,  dtype=int)
        test_acc   = (test_preds == test_true).mean() * 100.0

        cm = confusion_matrix(test_true, test_preds, labels=[0, 1])
        print(f"[User {user_id}] => TEST Accuracy: {test_acc:.2f}%")
        print("Confusion Matrix (Test):")
        print(cm)

        # Plot confusion matrix in user test directory
        plt.figure(figsize=(5,4))
        plt.imshow(cm, cmap='Blues')
        plt.title(f"User {user_id} Confusion Matrix (Test)")
        plt.colorbar()
        thresh = cm.max() / 2
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, cm[i, j],
                         ha="center", va="center",
                         color="white" if cm[i, j] > thresh else "black")
        plt.xlabel("Predicted")
        plt.ylabel("True")
        plt.tight_layout()
        cm_fname = os.path.join(user_test_dir, "confusion_matrix.png")
        plt.savefig(cm_fname)
        plt.close()

        # store global result
        tn, fp, fn, tp = cm.ravel()
        global_results.append({
            'user_id': user_id,
            'test_acc': test_acc,
            'tn': tn,
            'fp': fp,
            'fn': fn,
            'tp': tp
        })

    # 9) Summaries across all users
    if len(global_results) > 0:
        results_df = pd.DataFrame(global_results)
        results_csv = os.path.join("results", "test", "classification_summary.csv")
        results_df.to_csv(results_csv, index=False)
        print(f"Saved test summary => {results_csv}")

        best_df  = results_df.sort_values('test_acc', ascending=False).head(5)
        worst_df = results_df.sort_values('test_acc', ascending=True).head(5)
        print("\nTop 5 Users by Test Accuracy:\n", best_df)
        print("\nWorst 5 Users by Test Accuracy:\n", worst_df)
    else:
        print("No classification results to summarize.")

    print("Done with final drug classification on test sets!")

if __name__ == "__main__":
    main()
