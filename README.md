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
- **Training**: Freezes backbone, trains only classifier head with BCEWithLogits loss.  

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

### **3.3 Classification Results (Updated)**  
In this latest run, we applied **downsampling** on the training set to address class imbalance:
- We **randomly remove** excess majority-class samples until positives and negatives are balanced.  
- We then append those “extra” majority samples to the **test** set (so no data is discarded).  
- **Thresholds** are selected on the **validation** split (70/15/15 train/val/test). For each user, we sweep thresholds from 0.0 to 1.0 in increments of 0.01, picking the threshold that yields **highest validation accuracy**.  

**New Classification Table** (by user):

| user_id | val_threshold | auc                          | accuracy                        | tn  | fp | fn | tp |
|---------|--------------|-------------------------------|---------------------------------|-----|----|----|----|
| 5       | 0.54         | *(blank)*                    | 100.0                           | 15  | 0  | 0  | 0  |
| 9       | 0.52         | 0.8064516129032258           | 96.875                          | 31  | 0  | 1  | 0  |
| 10      | 0.49         | 0.6274509803921569           | 69.56521739130434               | 0   | 6  | 1  | 16 |
| 12      | 0.52         | 0.696969696969697            | 21.428571428571427              | 6   | 0  | 22 | 0  |
| 13      | 0.51         | 0.8034188034188033           | 72.72727272727273               | 5   | 4  | 2  | 11 |
| 14      | 0.44         | 0.5064935064935066           | 79.3103448275862                | 2   | 5  | 1  | 21 |
| 15      | 0.43         | 0.5709459459459459           | 82.22222222222221               | 0   | 8  | 0  | 37 |
| 18      | 0.55         | 0.7999999999999999           | 84.375                          | 27  | 0  | 5  | 0  |
| 19      | 0.53         | 0.2777777777777778           | 28.57142857142857               | 6   | 0  | 15 | 0  |
| 20      | 0.53         | 0.6666666666666667           | 62.5                            | 5   | 1  | 2  | 0  |
| 25      | 0.54         | 0.4666666666666667           | 93.47826086956522               | 43  | 2  | 1  | 0  |
| 27      | 0.53         | 0.6410256410256411           | 50.0                            | 8   | 1  | 10 | 3  |
| 28      | 0.53         | 0.711111111111111            | 77.77777777777779               | 14  | 1  | 3  | 0  |
| 29      | 0.46         | *(blank)*                    | 80.0                            | 4   | 1  | 0  | 0  |
| 31      | 0.48         | *(blank)*                    | 100.0                           | 4   | 0  | 0  | 0  |
| 32      | 0.33         | *(blank)*                    | 86.66666666666667               | 13  | 2  | 0  | 0  |
| 33      | 0.42         | 0.9117647058823529           | 94.44444444444444               | 0   | 2  | 0  | 34 |
| 35      | 0.49         | 0.3939393939393939           | 50.0                            | 1   | 2  | 5  | 6  |

- **Average AUC** (across users with numeric AUC): **~0.63**  
- **Average Accuracy** (across all 18 users above): **~73.9%**  

**Key Observations**:
- AUC is blank when all test samples fall in the same class or are insufficient for a proper ROC curve.  
- Accuracy can be high (e.g., users 5 or 31 at 100%), but that sometimes reflects extreme imbalance.  
- The dynamic thresholding approach helps avoid cherry-picking on the test set by using validation for threshold selection.  

---

## **4. Critical Comparison with Original Paper**  

| **Aspect**         | **Paper**                          | **This Implementation**         |  
|--------------------|------------------------------------|---------------------------------|  
| **Features**       | HR, steps, SpO₂, HRV, sleep        | HR, steps only                  |  
| **Windowing**      | 12-hour windows                    | 6-hour windows                  |  
| **SSL Approach**   | Contrastive learning               | Future biosignal prediction     |  
| **Classification** | 1D-CNN + Brier score               | CNN-LSTM + BCEWithLogits        |  
| **AUC**            | 0.729 (SSL)                        | ~0.63 (current run)             |  
| **Accuracy**       | ~70%                               | ~73.9% (average, dynamic thr)   |  

---

## **5. Limitations & Recommendations**  

### **5.1 Identified Issues**  
1. **Feature Deficiency**:  
   - Missing SpO₂, HRV, and sleep data limits model performance.  
2. **Class Imbalance**:  
   - Downsampling helps but does not fully solve rare-usage cases (some remain heavily imbalanced).  
3. **Thresholding**:  
   - Some users produce borderline or empty confusion matrices if usage is extremely sparse.  

### **5.2 Proposed Fixes**  
1. **Expand Features**: Ingest all Fitbit modalities (SpO₂, HRV).  
2. **Hybrid Imbalance Handling**: Combine downsampling with focal loss or SMOTE.  
3. **Multi-Window Aggregation**: Consider 12 or 24-hour windows to capture slower physiological trends.  
4. **Refined Thresholding**: Evaluate additional metrics (F1, precision/recall) or multiple cutoffs for each user.  

---
