# **Comprehensive Technical Report: Personalized Biosignal Analysis for Substance Use Detection**  

---

## **1. Data Processing Pipeline**  

### **1.1 Data Ingestion and Merging**  
- **Sources**:  
  - **LifeSnaps Dataset**: Minute-level Fitbit readings collected unobtrusively for more than 4 months by n=71 participants, under the European H2020 RAIS project.
  - **Biosignal Data**: Minute-level Fitbit readings stored in `/Biosignal` CSV files.  
  - **Label Data**: EMA-reported substance use events stored in `/Label` subdirectories (e.g., `ID5/ID5_Crave.csv`).  
- **Key Steps**:  
  1. **Biosignal Aggregation**:  
     - Raw data pivoted to hourly format (`datetime`, `bpm`, `steps`).  
     - Hourly aggregation: `bpm` averaged, `steps` summed.  
  2. **Label Merging**:  
     - EMA timestamps truncated to hourly intervals.  
     - Binary labels (`1` if any usage/craving event in the hour; `0` otherwise).  
     - Merged with biosignals via `(id, datetime)` keys.  
  3. **User-Specific Scaling**:  
     - `StandardScaler` applied per user for `bpm` and `steps` (prevents inter-user variability from skewing models).  
     - Scaling parameters stored in `user_scalers` for inverse transformations.  

### **1.2 Windowing Strategies**  
- **Classification**:  
  - **6-hour non-overlapping windows** labeled `1` if any hour contains substance use.  
  - Input shape: `[6, 2]` (6 hours × 2 features: `bpm_scaled`, `steps_scaled`).  
  - **Class Balancing**: Downsampling applied to training data by randomly removing excess majority-class samples while preserving all minority samples. Excess samples added to test set.
  - **Per-Substance Models**: Separate binary classifiers for each substance (e.g., "carrot", "melon") and event type ("use"/"crave").  
- **Forecasting**:  
  - **Input**: 2 consecutive windows (12 hours total).  
  - **Target**: 1 subsequent window (6 hours).  
  - Temporal alignment enforced via strict `datetime` sorting.  

---

## **2. Model Architectures**  
 
### **2.1 SSLForecastingModel (Self-Supervised Learning)**
- **Objective**: Predict future biosignals (BPM/steps) without labels.  
- **Structure**:  
  - **BPM Encoder**:  
    ```python
    nn.Sequential(
      nn.Conv1d(1, 32, kernel_size=3, padding=1),  # Input: [B, 1, 12]
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Conv1d(32, 64, kernel_size=3, padding=1),
      nn.ReLU(),
      nn.Dropout(0.3)
    )
    ```
    → LSTM(64 → 128, 2 layers).  
  - **Steps Encoder**: Identical structure to BPM encoder.  
  - **Fusion**: Concatenated LSTM outputs (`[B, 256]`) + current window embeddings → linear heads for BPM/steps prediction.  
 
### **2.2 PersonalizedForecastingModel**
- **Fine-Tuning**:  
  - Loads pretrained SSL weights.  
  - Unfreezes 50% of layers (e.g., last CNN/LSTM layers) via `partially_unfreeze_backbone()`.  
  - Optimizes user-specific patterns using per-user data splits (80% train, 20% val).  
 
### **2.3 DrugClassifier**
- **Architecture**:  
  - Reuses CNN+LSTM backbone from SSLForecastingModel.  
  - Classification head:  
    ```python
    nn.Sequential(
      nn.Linear(256, 128),  # Fused features
      nn.ReLU(),
      nn.Dropout(0.3),
      nn.Linear(128, 1)     # Sigmoid for binary output
    )
    ```
- **Training**: Partially unfreezes 30% of backbone via partially_unfreeze_backbone(model, unfreeze_ratio=0.3) in init_classifier_for_user, keeps remaining layers frozen.

---

## **3. Training Procedures**

### **3.1 SSL Pretraining**
- **Dataset**: `lifesnaps.csv` (external dataset for general biosignal patterns).  
- **Hyperparameters**:  
  - Optimizer: Adam (`lr=0.001`, `weight_decay=1e-5`).  
  - Loss: Weighted SmoothL1 (`alpha=0.85` for BPM, `beta=0.15` for steps).  
  - Scheduler: StepLR (γ=0.1 every 20 epochs).  
- **Results (from `stats_pretrain.csv`)**:
  | Epoch | Train Loss | Val Loss | BPM MAE | Steps MAE |
  |-------|------------|----------|---------|-----------|
  | 1     | 0.307      | 0.220    | 6.75    | 357.1     |
  | 50    | 0.203      | 0.164    | 5.54    | 301.4     |
  - **Trend**: Steady decline in MAE, indicating effective pretraining.

### **3.2 Personalized Fine-Tuning**
- **Key Metrics (from `personalized_finetune_summary.csv`)**:

  | User ID | Best Val Loss | BPM MAE | Steps MAE |
  |---------|---------------|---------|-----------|
  | 5       | 0.28          | 6.23    | 240.06    |
  | 9       | 0.24          | 7.83    | 701.65    |
  | 10      | 0.27          | 6.78    | 284.57    |
  | 12      | 0.30          | 5.76    | 745.27    |
  | 13      | 0.18          | 7.17    | 277.99    |
  | 14      | 0.28          | 8.53    | 426.57    |
  | 15      | 0.22          | 4.98    | 373.50    |
  | 18      | 0.23          | 8.71    | 171.47    |
  | 19      | 0.23          | 4.11    | 166.48    |
  | 20      | 0.16          | 3.27    | 406.88    |
  | 25      | 0.21          | 5.80    | 265.61    |
  | 27      | 0.34          | 7.03    | 300.60    |
  | 28      | 0.24          | 6.77    | 339.08    |
  | 29      | 0.15          | 6.09    | 380.86    |
  | 31      | 0.49          | 11.65   | 307.88    |
  | 32      | 0.19          | 5.06    | 377.27    |
  | 33      | 0.31          | 6.70    | 334.12    |
  | 35      | 0.12          | 5.96    | 317.57    |

Notes:
  - **Insight**: High variability in performance (e.g., User 31’s BPM MAE=11.65 vs. User 20’s 3.27). Likely due to data quality (e.g., missing sensor readings).
  - **Insight**: Often negligible impact in affecting model performance. May need to unfreeze more weights or employ a different transfer learning strategy.

### **3.3 Classification Results (Per-Substance Models)**  
- **Strategy**:  
  - Separate binary classifiers for **use** and **crave** events per substance (e.g., "carrot_use_label", "melon_crave_label").  
  - Only users with ≥2 positive samples for a substance-event pair are included.  
  - Thresholds selected via validation split (70/15/15 train/val/test).  

**Full Classification Results** (`classification_summary.csv`):

| user_id | label_col | pos_count | neg_count | best_thr | auc | test_acc | tn | fp | fn | tp |
|---|---|---|---|---|---|---|---|---|---|---|
| 5 | methamphetamine_crave_label | 5 | 24 | 0.41 | 0.75 | 60.00 | 3 | 1 | 1 | 0 |
| 9 | methamphetamine_crave_label | 3 | 39 | 0.46 | — | 100.00 | 7 | 0 | 0 | 0 |
| 10 | cannabis_crave_label | 18 | 59 | 0.29 | 0.63 | 75.00 | 7 | 2 | 1 | 2 |
| 10 | cannabis_use_label | 18 | 59 | 0.25 | 0.11 | 58.33 | 7 | 2 | 3 | 0 |
| 10 | nicotine_crave_label | 16 | 61 | 0.27 | 0.50 | 50.00 | 6 | 4 | 2 | 0 |
| 10 | nicotine_use_label | 16 | 61 | 0.22 | 0.75 | 58.33 | 6 | 4 | 1 | 1 |
| 12 | alcohol_use_label | 2 | 85 | 0.18 | — | 85.71 | 12 | 2 | 0 | 0 |
| 12 | ghb_crave_label | 2 | 85 | 0.21 | — | 100.00 | 14 | 0 | 0 | 0 |
| 12 | ghb_use_label | 6 | 81 | 0.21 | 0.92 | 92.86 | 13 | 0 | 1 | 0 |
| 12 | methamphetamine_crave_label | 13 | 74 | 0.33 | 0.83 | 78.57 | 11 | 1 | 2 | 0 |
| 12 | methamphetamine_use_label | 34 | 53 | 0.42 | 0.36 | 35.71 | 5 | 4 | 5 | 0 |
| 12 | nicotine_crave_label | 11 | 76 | 0.27 | 0.46 | 85.71 | 12 | 0 | 2 | 0 |
| 12 | nicotine_use_label | 4 | 83 | 0.28 | 0.31 | 92.86 | 13 | 0 | 1 | 0 |
| 13 | alcohol_use_label | 9 | 45 | 0.42 | 0.00 | 88.89 | 8 | 0 | 1 | 0 |
| 13 | cannabis_crave_label | 2 | 52 | 0.28 | — | 100.00 | 9 | 0 | 0 | 0 |
| 13 | cannabis_use_label | 6 | 48 | 0.33 | 0.50 | 88.89 | 8 | 0 | 1 | 0 |
| 13 | nicotine_use_label | 28 | 26 | 0.46 | 0.80 | 77.78 | 2 | 2 | 0 | 5 |
| 14 | cannabis_crave_label | 18 | 70 | 0.42 | 0.91 | 78.57 | 11 | 0 | 3 | 0 |
| 14 | cannabis_use_label | 48 | 40 | 0.00 | 0.50 | 57.14 | 0 | 6 | 0 | 8 |
| 15 | cannabis_crave_label | 32 | 56 | 0.35 | 0.27 | 35.71 | 3 | 6 | 3 | 2 |
| 15 | cannabis_use_label | 47 | 41 | 0.00 | 0.53 | 50.00 | 0 | 7 | 0 | 7 |
| 18 | cannabis_crave_label | 25 | 64 | 0.29 | 0.73 | 57.14 | 5 | 5 | 1 | 3 |
| 18 | cannabis_use_label | 3 | 86 | 0.20 | — | 100.00 | 14 | 0 | 0 | 0 |
| 19 | alcohol_crave_label | 4 | 49 | 0.25 | 0.29 | 75.00 | 6 | 1 | 1 | 0 |
| 19 | alcohol_use_label | 6 | 47 | 0.28 | 1.00 | 87.50 | 7 | 0 | 1 | 0 |
| 19 | methamphetamine_crave_label | 4 | 49 | 0.20 | 0.86 | 87.50 | 6 | 1 | 0 | 1 |
| 19 | methamphetamine_use_label | 26 | 27 | 0.47 | 0.63 | 62.50 | 1 | 3 | 0 | 4 |
| 20 | methamphetamine_crave_label | 10 | 28 | 0.55 | 0.50 | 66.67 | 4 | 0 | 2 | 0 |
| 20 | methamphetamine_use_label | 9 | 29 | 0.41 | 1.00 | 33.33 | 1 | 4 | 0 | 1 |
| 20 | nicotine_crave_label | 8 | 30 | 0.45 | 0.40 | 66.67 | 4 | 1 | 1 | 0 |
| 20 | nicotine_use_label | 8 | 30 | 0.46 | 0.80 | 83.33 | 4 | 1 | 0 | 1 |
| 25 | alcohol_use_label | 4 | 59 | 0.30 | 0.67 | 90.00 | 9 | 0 | 1 | 0 |
| 27 | methamphetamine_crave_label | 13 | 50 | 0.27 | 0.88 | 80.00 | 7 | 1 | 1 | 1 |
| 27 | methamphetamine_use_label | 14 | 49 | 0.41 | 0.38 | 80.00 | 8 | 0 | 2 | 0 |
| 27 | nicotine_crave_label | 20 | 43 | 0.42 | 0.71 | 60.00 | 5 | 2 | 2 | 1 |
| 27 | nicotine_use_label | 13 | 50 | 0.50 | 0.81 | 80.00 | 8 | 0 | 2 | 0 |
| 28 | alcohol_use_label | 12 | 68 | 0.22 | 0.30 | 58.33 | 7 | 3 | 2 | 0 |
| 28 | caffeine_use_label | 12 | 68 | 0.18 | 0.30 | 75.00 | 9 | 1 | 2 | 0 |
| 28 | cannabis_use_label | 2 | 78 | 0.21 | — | 100.00 | 12 | 0 | 0 | 0 |
| 28 | coffee_use_label | 7 | 73 | 0.19 | 0.73 | 91.67 | 11 | 0 | 1 | 0 |
| 33 | nicotine_use_label | 44 | 8 | 0.00 | 1.00 | 87.50 | 0 | 1 | 0 | 7 |
| 35 | nicotine_use_label | 29 | 20 | 0.49 | 0.87 | 75.00 | 2 | 1 | 1 | 4 |
| 35 | opioid_use_label | 8 | 41 | 0.47 | 0.14 | 87.50 | 7 | 0 | 1 | 0 |

**Aggregate Metrics**

| Metric | Value |
|--------|-------|
| **Average AUC** (37 valid models) | **0.60** |
| **Average Accuracy** (all 43 models) | **75.23 %** |

---

## **4 . Key Observations**  

1. **Best‑case AUC = 1.00** achieved by  
   - *alcohol_use* for **User 19**  
   - *methamphetamine_use* for **User 20**  
2. **Worst‑case AUC ≈ 0.11** (*cannabis_use*, User 10) – heavy class imbalance and noisy HR patterns.  
3. Thresholds span **0.00 → 0.55**; models with `thr ≈ 0.00` maximise recall at the cost of precision.  
4. Substances with very few positives (e.g. opioid_use) still reach high accuracy due to skewed negative class.  

---

## **5. Critical Comparison with Original Paper**  

| **Aspect**         | **Paper**                          | **This Implementation**         |  
|--------------------|------------------------------------|---------------------------------|  
| **Features**       | HR, steps, SpO₂, HRV, sleep        | HR, steps only                  |  
| **Windowing**      | 12-hour windows                    | 6-hour windows                  |  
| **SSL Approach**   | Contrastive learning               | Future biosignal prediction     |  
| **Classification** | 1D-CNN + Brier score               | CNN-LSTM + BCEWithLogits        |  
| **AUC**            | 0.729 (SSL)                        | ~0.60 (current run)             |  
| **Accuracy**       | ~70%                               | ~75.23% (average, dynamic thr)  |  

--- 