import os
import glob
import warnings
import random
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import matplotlib.pyplot as plt

from torch.utils.data import Dataset, DataLoader
from torch.utils.data import RandomSampler
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, roc_auc_score

###############################################################################
# Adjust these placeholders with your actual code/paths/models as needed
###############################################################################
WINDOW_SIZE = 6  # e.g. 6-hour window
def scale_per_user(df):
    """Simple Z-score scaling per user on bpm/steps."""
    df = df.sort_values('datetime').copy()
    df['bpm_scaled'] = (df['bpm'] - df['bpm'].mean()) / (df['bpm'].std() + 1e-9)
    df['steps_scaled'] = (df['steps'] - df['steps'].mean()) / (df['steps'].std() + 1e-9)
    return df

class DrugClassifier(nn.Module):
    """
    Replace this with your actual CNN+LSTM or other arch.
    For demonstration, a simple MLP on the concatenation of
    [bpm_in, steps_in] -> dimension = 2 * window_size.
    """
    def __init__(self, window_size=6):
        super().__init__()
        self.window_size = window_size
        self.features = nn.Sequential(
            nn.Linear(2 * window_size, 64),
            nn.ReLU(),
            nn.Linear(64, 32),
            nn.ReLU()
        )
        self.classifier = nn.Linear(32, 1)

    def forward(self, bpm_in, steps_in):
        # shape (batch_size, window_size)
        x = torch.cat([bpm_in, steps_in], dim=1)  # shape = (batch_size, 2*window_size)
        feats = self.features(x)
        logits = self.classifier(feats).squeeze(-1)
        return logits

###############################################################################
# 1) LOAD SENSOR DATA => HOURLY
###############################################################################
def load_sensor_data():
    """
    Reads minute-level sensor data from data/Personalized AI Data/Biosignal/*.csv
    Aggregates to hourly resolution => [id, datetime, date, hour, bpm, steps].
    """
    data_folder = "data/Personalized AI Data/Biosignal"
    csv_files = glob.glob(os.path.join(data_folder, "*.csv"))
    if not csv_files:
        raise ValueError(f"No CSV files found in '{data_folder}'.")

    df_list = []
    for file in csv_files:
        tmp = pd.read_csv(file, parse_dates=['time'])
        # If 'participant_id' missing, parse from filename
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

    # filter for heart_rate & steps
    df_filt = df_raw[df_raw['data_type'].isin(['heart_rate','steps'])].copy()
    df_filt.drop_duplicates(subset=['participant_id','time','data_type'], inplace=True)

    # pivot => [id, time, bpm, steps]
    df_pivot = df_filt.pivot(
        index=['participant_id','time'],
        columns='data_type',
        values='value'
    ).reset_index()

    df_pivot.rename(columns={'participant_id':'id','heart_rate':'bpm'}, inplace=True)
    df_pivot['bpm'] = pd.to_numeric(df_pivot['bpm'], errors='coerce')
    df_pivot['steps'] = pd.to_numeric(df_pivot['steps'], errors='coerce')
    df_pivot.dropna(subset=['bpm','steps'], inplace=True)

    df_pivot['datetime'] = pd.to_datetime(df_pivot['time'])
    df_pivot['date'] = df_pivot['datetime'].dt.date
    df_pivot['hour'] = df_pivot['datetime'].dt.hour

    df_sensors = df_pivot[['id','datetime','date','hour','bpm','steps']].copy()
    df_sensors.sort_values(by=['id','datetime'], inplace=True)
    df_sensors.reset_index(drop=True, inplace=True)

    # group to hourly
    df_sensors['date'] = pd.to_datetime(df_sensors['date'])
    def to_hour_datetime(d, h):
        return pd.to_datetime(d) + pd.to_timedelta(h, unit='H')
    df_hourly = df_sensors.groupby(['id','date','hour'], as_index=False).agg({
        'bpm':'mean',
        'steps':'sum'
    })
    df_hourly['datetime'] = df_hourly.apply(lambda r: to_hour_datetime(r['date'], r['hour']), axis=1)
    df_hourly.sort_values(['id','datetime'], inplace=True)
    df_hourly.reset_index(drop=True, inplace=True)

    return df_hourly

###############################################################################
# 2) LOAD LABEL DATA (Fruits => "substances"), pivot to hour-level
###############################################################################
def load_label_data():
    """
    Reads label data from data/Personalized AI Data/Label/ID*/...
    We expect columns: 
      - 'substance_fruit_label' => e.g. "Carrot", "Coconut", etc.
      - 'crave_use_none_label' => "Use", "Crave", or "None"
      - time col among 'hawaii_use_time', 'hawaii_createdat_time', 'created_at'
    We only keep rows with "Use" or "Crave".
    """
    labels_dir = "data/Personalized AI Data/Label"
    all_rows = []

    subdirs = glob.glob(os.path.join(labels_dir, "ID*"))
    for sd in subdirs:
        pid_str = os.path.basename(sd)  # e.g. "ID5"
        try:
            p_id = int(pid_str.replace("ID",""))
        except:
            continue

        label_files = glob.glob(os.path.join(sd, "*.csv")) + glob.glob(os.path.join(sd, "*.xlsx"))
        for lf in label_files:
            df_label = None
            if lf.endswith(".csv"):
                try:
                    df_label = pd.read_csv(lf)
                except:
                    continue
            else:
                # xlsx
                try:
                    df_label = pd.read_excel(lf)
                except:
                    continue

            if df_label is None or df_label.empty:
                continue

            # pick time col
            time_col = None
            for c in ["hawaii_use_time","hawaii_createdat_time","created_at"]:
                if c in df_label.columns:
                    time_col = c
                    break
            if not time_col:
                continue

            df_label[time_col] = pd.to_datetime(df_label[time_col], errors='coerce')
            df_label.dropna(subset=[time_col], inplace=True)

            for _, row in df_label.iterrows():
                fruit_str = str(row.get('substance_fruit_label','')).lower().strip()
                label_str = str(row.get('crave_use_none_label','')).lower().strip()
                dtime = row[time_col]

                if label_str not in ["use","crave"]:
                    # skip "None" or anything else
                    continue

                all_rows.append({
                    'id': p_id,
                    'datetime': dtime,
                    'fruit_substance': fruit_str,  # e.g. "carrot", "coconut", "almond"...
                    'label_str': label_str        # "use" or "crave"
                })

    df_labels = pd.DataFrame(all_rows)
    df_labels.sort_values(['id','datetime'], inplace=True)
    df_labels.reset_index(drop=True, inplace=True)
    return df_labels

def pivot_label_data(df_labels):
    """
    For each unique fruit_substance => create columns <fruit>_use_label, <fruit>_crave_label
    Then group by hour => take 'max' within each hour to get a 1/0 label.
    """
    if df_labels.empty:
        return df_labels

    all_fruits = sorted(df_labels['fruit_substance'].unique())
    label_types = ["use","crave"]

    # create columns
    for f in all_fruits:
        for lbl in label_types:
            colname = f"{f}_{lbl}_label"
            df_labels[colname] = 0

    # set 1 if row matches that fruit + label_str
    for idx, row in df_labels.iterrows():
        f = row['fruit_substance']
        l = row['label_str']
        col = f"{f}_{l}_label"
        if col in df_labels.columns:
            df_labels.at[idx, col] = 1

    # group by hour
    df_labels['date'] = df_labels['datetime'].dt.date
    df_labels['hour'] = df_labels['datetime'].dt.hour
    df_labels['date'] = pd.to_datetime(df_labels['date'])

    agg_dict = {}
    for f in all_fruits:
        for lbl in label_types:
            c = f"{f}_{lbl}_label"
            agg_dict[c] = 'max'

    df_hour_labels = df_labels.groupby(['id','date','hour'], as_index=False).agg(agg_dict)

    def to_hour_datetime(d, h):
        return pd.to_datetime(d) + pd.to_timedelta(h, unit='H')
    df_hour_labels['datetime'] = df_hour_labels.apply(lambda r: to_hour_datetime(r['date'], r['hour']), axis=1)

    return df_hour_labels

def merge_sensors_with_labels(df_sensors_hour, df_hour_labels):
    """
    Merge sensor data with wide label columns. Fill missing labels => 0.
    """
    if df_hour_labels.empty:
        return df_sensors_hour.copy()

    # ensure tz naive
    df_sensors_hour['datetime'] = df_sensors_hour['datetime'].dt.tz_localize(None)
    df_hour_labels['datetime'] = df_hour_labels['datetime'].dt.tz_localize(None)

    df_merged = pd.merge(df_sensors_hour, df_hour_labels, on=['id','datetime'], how='left')
    label_cols = [c for c in df_hour_labels.columns if c.endswith('_label')]
    for c in label_cols:
        if c not in df_merged.columns:
            df_merged[c] = 0
    df_merged[label_cols] = df_merged[label_cols].fillna(0).astype(int)
    return df_merged

###############################################################################
# 3) CREATE NON-OVERLAPPING WINDOWED DATA
###############################################################################
class CravingDataset(Dataset):
    """Each sample => (bpm_input, steps_input, user_id, label)."""
    def __init__(self, data_list):
        self.data_list = data_list

    def __len__(self):
        return len(self.data_list)

    def __getitem__(self, idx):
        d = self.data_list[idx]
        bpm_in   = torch.tensor(d['bpm_input'], dtype=torch.float32)     # shape [WINDOW_SIZE]
        steps_in = torch.tensor(d['steps_input'], dtype=torch.float32)   # shape [WINDOW_SIZE]
        label    = torch.tensor(d['label'], dtype=torch.float32)
        uid      = d['id']
        return bpm_in, steps_in, uid, label

def create_classification_samples(df_user, label_col, window_size=6):
    """
    Slice user data in non-overlapping windows of size=window_size.
    label=1 if ANY row in chunk => label_col=1, else 0.
    Use scaled columns if present, else raw.
    """
    samples = []
    df_user = df_user.sort_values('datetime').reset_index(drop=True)
    n = len(df_user)
    for i in range(0, n, window_size):
        chunk = df_user.iloc[i : i+window_size]
        if len(chunk) < window_size:
            break
        chunk_label = 1 if (chunk[label_col].max() == 1) else 0

        if 'bpm_scaled' in chunk.columns and 'steps_scaled' in chunk.columns:
            bpm_arr   = chunk['bpm_scaled'].values
            steps_arr = chunk['steps_scaled'].values
        else:
            bpm_arr   = chunk['bpm'].values
            steps_arr = chunk['steps'].values

        samples.append({
            'id': chunk['id'].iloc[0],
            'bpm_input': bpm_arr,
            'steps_input': steps_arr,
            'label': chunk_label,
            'datetime_start': chunk['datetime'].iloc[0],
            'datetime_end':   chunk['datetime'].iloc[-1]
        })
    return samples

###############################################################################
# 4) MAIN
###############################################################################
def main():
    warnings.filterwarnings('ignore')

    # 1) sensor data => hourly
    df_sensors_hour = load_sensor_data()
    print("Sensor data shape:", df_sensors_hour.shape)

    # 2) label data => pivot
    df_labels_raw = load_label_data()
    df_hour_labels = pivot_label_data(df_labels_raw)
    print("Label data shape (hourly):", df_hour_labels.shape)

    # 3) merge => df_merged
    df_merged = merge_sensors_with_labels(df_sensors_hour, df_hour_labels)

    # scale
    df_merged = df_merged.groupby('id').apply(scale_per_user).reset_index(drop=True)

    # find label cols
    label_cols = [c for c in df_merged.columns if c.endswith('_label')]
    if not label_cols:
        print("No label columns found => no classification tasks.")
        return
    label_cols = sorted(label_cols)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    unique_ids = sorted(df_merged['id'].unique())

    global_results = []

    for user_id in unique_ids:
        user_dir = os.path.join("results", "test", f"user_{user_id}")
        os.makedirs(user_dir, exist_ok=True)

        df_user = df_merged[df_merged['id']==user_id].copy()
        if len(df_user) < 2*WINDOW_SIZE:
            print(f"[User {user_id}] => not enough rows for windows.")
            continue

        for label_col in label_cols:
            # create windowed samples
            data_samples = create_classification_samples(df_user, label_col, WINDOW_SIZE)
            if len(data_samples) < 15:
                print(f"[User {user_id} | {label_col}] => not enough classification windows (<15).")
                continue

            labels_all = [s['label'] for s in data_samples]
            pos_count = sum(labels_all)
            neg_count = len(data_samples) - pos_count

            # Fix for “least populated class in y has only 1 member”
            # => skip if <2 positives OR <2 negatives
            if pos_count < 2:
                print(f"[User {user_id} | {label_col}] => skipping => not enough positive samples (pos_count={pos_count}).")
                continue
            if neg_count < 2:
                print(f"[User {user_id} | {label_col}] => skipping => not enough negative samples (neg_count={neg_count}).")
                continue

            # Now we can do stratified splits safely.
            X_idx = np.arange(len(data_samples))
            y_lbl = np.array(labels_all)

            # first => 15% test
            trainval_idx, test_idx = train_test_split(
                X_idx,
                test_size=0.15,
                shuffle=True,
                stratify=y_lbl,
                random_state=42
            )
            if len(trainval_idx)==0 or len(test_idx)==0:
                print(f"[User {user_id} | {label_col}] => skipping => no data after test split.")
                continue

            # second => val from trainval
            val_size = 0.15 / 0.85  # ~17.6% of total
            train_idx, val_idx = train_test_split(
                trainval_idx,
                test_size=val_size,
                shuffle=True,
                stratify=y_lbl[trainval_idx],
                random_state=42
            )

            user_train = [data_samples[i] for i in train_idx]
            user_val   = [data_samples[i] for i in val_idx]
            user_test  = [data_samples[i] for i in test_idx]

            if len(user_train)==0 or len(user_val)==0 or len(user_test)==0:
                print(f"[User {user_id} | {label_col}] => skipping => not enough train/val/test data.")
                continue

            # build dataset/dataloaders
            train_ds = CravingDataset(user_train)
            val_ds   = CravingDataset(user_val)
            test_ds  = CravingDataset(user_test)

            train_loader = DataLoader(train_ds, batch_size=32, sampler=RandomSampler(train_ds))
            val_loader   = DataLoader(val_ds, batch_size=32, sampler=RandomSampler(val_ds))
            test_loader  = DataLoader(test_ds, batch_size=32)

            # build classifier
            classifier = DrugClassifier(window_size=WINDOW_SIZE).to(device)
            # (Optionally load pretrained checkpoint for user)

            criterion = nn.BCEWithLogitsLoss()
            optimizer = torch.optim.Adam(classifier.parameters(), lr=1e-3)
            scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)
            num_epochs = 20
            patience = 5

            best_val_loss = float('inf')
            epochs_no_improve = 0

            label_subdir = os.path.join(user_dir, label_col)
            os.makedirs(label_subdir, exist_ok=True)
            best_model_path = os.path.join(label_subdir, "best_model.pth")

            STATS = {
                'epoch': [],
                'train_loss': [],
                'val_loss': [],
                'val_acc_05': []
            }

            # train loop
            for epoch in range(num_epochs):
                if epochs_no_improve >= patience:
                    print(f"[User {user_id} | {label_col}] => Early stop at epoch {epoch+1}")
                    break

                # --- train ---
                classifier.train()
                sum_train_loss = 0.0
                for (bpm_in, steps_in, pids, labels) in train_loader:
                    bpm_in   = bpm_in.to(device)
                    steps_in = steps_in.to(device)
                    labels   = labels.to(device)

                    optimizer.zero_grad()
                    logits = classifier(bpm_in, steps_in)
                    loss = criterion(logits, labels)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                    optimizer.step()
                    sum_train_loss += loss.item()

                scheduler.step()
                avg_train_loss = sum_train_loss / len(train_loader)

                # --- val ---
                classifier.eval()
                sum_val_loss = 0.0
                val_preds_05 = []
                val_true = []
                with torch.no_grad():
                    for (bpm_in, steps_in, pids, labels) in val_loader:
                        bpm_in   = bpm_in.to(device)
                        steps_in = steps_in.to(device)
                        labels   = labels.to(device)

                        logits = classifier(bpm_in, steps_in)
                        loss = criterion(logits, labels)
                        sum_val_loss += loss.item()

                        probs = torch.sigmoid(logits).cpu().numpy()
                        preds_05 = (probs >= 0.5).astype(int)
                        val_preds_05.extend(preds_05)
                        val_true.extend(labels.cpu().numpy())

                avg_val_loss = sum_val_loss / len(val_loader)
                val_preds_05 = np.array(val_preds_05)
                val_true     = np.array(val_true)
                val_acc_05   = (val_preds_05 == val_true).mean() * 100.0

                if avg_val_loss < best_val_loss:
                    best_val_loss = avg_val_loss
                    epochs_no_improve = 0
                    torch.save(classifier.state_dict(), best_model_path)
                else:
                    epochs_no_improve += 1

                STATS['epoch'].append(epoch+1)
                STATS['train_loss'].append(avg_train_loss)
                STATS['val_loss'].append(avg_val_loss)
                STATS['val_acc_05'].append(val_acc_05)

                print(f"[User {user_id} | {label_col}] epoch={epoch+1}/{num_epochs}, "
                      f"trainLoss={avg_train_loss:.4f}, valLoss={avg_val_loss:.4f}, "
                      f"valAcc@0.5={val_acc_05:.2f}%")

            # save stats
            stats_df = pd.DataFrame(STATS)
            stats_df.to_csv(os.path.join(label_subdir, "train_val_stats.csv"), index=False)

            # 5) Evaluate test w/ best threshold from val
            classifier.load_state_dict(torch.load(best_model_path, map_location=device))
            classifier.eval()

            # find best threshold from val
            val_probs = []
            val_lbls  = []
            with torch.no_grad():
                for (bpm_in, steps_in, pids, labels) in val_loader:
                    bpm_in   = bpm_in.to(device)
                    steps_in = steps_in.to(device)
                    l_cpu    = labels.cpu().numpy()
                    logits   = classifier(bpm_in, steps_in)
                    probs    = torch.sigmoid(logits).cpu().numpy()

                    val_probs.extend(probs)
                    val_lbls.extend(l_cpu)

            val_probs = np.array(val_probs)
            val_lbls  = np.array(val_lbls)
            thresholds = np.linspace(0.0, 1.0, 101)
            best_thr = 0.5
            best_acc_val = 0.0
            for thr in thresholds:
                preds_thr = (val_probs >= thr).astype(int)
                acc_thr   = (preds_thr == val_lbls).mean() * 100.0
                if acc_thr > best_acc_val:
                    best_acc_val = acc_thr
                    best_thr = thr

            # test
            test_probs = []
            test_lbls  = []
            with torch.no_grad():
                for (bpm_in, steps_in, pids, labels) in test_loader:
                    bpm_in   = bpm_in.to(device)
                    steps_in = steps_in.to(device)
                    l_cpu    = labels.cpu().numpy()
                    logits   = classifier(bpm_in, steps_in)
                    probs    = torch.sigmoid(logits).cpu().numpy()

                    test_probs.extend(probs)
                    test_lbls.extend(l_cpu)

            test_probs = np.array(test_probs)
            test_lbls  = np.array(test_lbls)

            if len(np.unique(test_lbls)) == 1:
                auc_val = float('nan')
            else:
                auc_val = roc_auc_score(test_lbls, test_probs)

            test_preds = (test_probs >= best_thr).astype(int)
            test_acc   = (test_preds == test_lbls).mean() * 100.0
            cm         = confusion_matrix(test_lbls, test_preds, labels=[0,1])
            tn, fp, fn, tp = cm.ravel()

            print(f"\n[User {user_id} | {label_col}] => TEST thr={best_thr:.2f}")
            print(f" - AUC={auc_val:.4f}, Accuracy={test_acc:.2f}%")
            print("Confusion Matrix [0,1]:\n", cm)

            plt.figure(figsize=(4,3))
            plt.imshow(cm, cmap='Blues')
            plt.title(f"U{user_id} {label_col}\nth={best_thr:.2f} Acc={test_acc:.1f}%")
            plt.colorbar()
            for i in range(cm.shape[0]):
                for j in range(cm.shape[1]):
                    plt.text(j, i, cm[i,j],
                             ha='center', va='center',
                             color='white' if cm[i,j] > cm.max()/2 else 'black')
            plt.xlabel("Predicted")
            plt.ylabel("True")
            plt.tight_layout()
            out_cm = os.path.join(label_subdir, f"cm_{best_thr:.2f}.png")
            plt.savefig(out_cm)
            plt.close()

            global_results.append({
                'user_id': user_id,
                'label_col': label_col,
                'pos_count': pos_count,
                'neg_count': neg_count,
                'best_thr': best_thr,
                'auc': auc_val,
                'test_acc': test_acc,
                'tn': tn,
                'fp': fp,
                'fn': fn,
                'tp': tp
            })

    if global_results:
        results_df = pd.DataFrame(global_results)
        out_path = os.path.join("results", "test", "classification_summary.csv")
        results_df.to_csv(out_path, index=False)
        print(f"\nSaved final summary => {out_path}")
    else:
        print("No classification results produced.")

    print("\nDone with final classification!\n")

if __name__ == "__main__":
    main()
