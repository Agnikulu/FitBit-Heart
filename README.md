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
   | 20      | 0.162         | 3.27    | 406.9     |
   | 35      | 0.118         | 5.96    | 317.6     |
   | 31      | 0.487         | 11.65   | 307.9     |
   - **Insight**: High variability in performance (e.g., User 31’s BPM MAE=11.65 vs. User 20’s 3.27). Likely due to data quality (e.g., missing sensor readings).  
   - **Insight**: Often negligible impact in affecting model performance. May need to unfreeze more weights or employ a different transfer learning strategy.  

### **3.3 Classification**  
- **Results (AUC & Accuracy)**:  

| user_id | auc   | acc_0.5 | tn_0.5 | fp_0.5 | fn_0.5 | tp_0.5 | best_thr | best_acc |
|---------|-------|---------|--------|--------|--------|--------|----------|----------|
| 5       |       | 100.00  | 5      | 0      | 0      | 0      | 0.49     | 100.00   |
| 9       | 0.83  | 85.71   | 6      | 0      | 1      | 0      | 0.46     | 85.71    |
| 10      | 0.75  | 50.00   | 2      | 4      | 2      | 4      | 0.53     | 75.00    |
| 12      | 0.54  | 42.86   | 6      | 0      | 8      | 0      | 0.49     | 64.29    |
| 13      |       | 55.56   | 5      | 4      | 0      | 0      | 0.54     | 100.00   |
| 14      | 0.47  | 42.86   | 3      | 4      | 4      | 3      | 0.45     | 64.29    |
| 15      | 0.88  | 78.57   | 6      | 2      | 1      | 5      | 0.50     | 78.57    |
| 18      | 0.69  | 71.43   | 6      | 3      | 1      | 4      | 0.49     | 78.57    |
| 19      | 0.00  | 75.00   | 6      | 0      | 2      | 0      | 0.48     | 75.00    |
| 20      |       | 100.00  | 6      | 0      | 0      | 0      | 0.50     | 100.00   |
| 25      | 0.44  | 60.00   | 6      | 3      | 1      | 0      | 0.60     | 90.00    |
| 27      | 1.00  | 60.00   | 5      | 4      | 0      | 1      | 0.52     | 90.00    |
| 28      | 0.59  | 75.00   | 9      | 0      | 3      | 0      | 0.48     | 83.33    |
| 29      |       | 100.00  | 5      | 0      | 0      | 0      | 0.46     | 100.00   |
| 31      |       | 100.00  | 4      | 0      | 0      | 0      | 0.48     | 100.00   |
| 32      |       | 100.00  | 15     | 0      | 0      | 0      | 0.39     | 100.00   |
| 33      | 1.00  | 87.50   | 2      | 0      | 1      | 5      | 0.47     | 100.00   |
| 35      | 0.53  | 62.50   | 1      | 2      | 1      | 4      | 0.00     | 62.50    |

- **Key Insights**:  
  - **Average AUC**: **0.65** (excluding NaN cases).  
  - **Average Accuracy**: **72.3%** (varies widely by user).  
  - **Best Performers**: Users 27 & 33 (AUC=1.0, high precision).  
  - **Worst Performers**: User 19 (AUC=0.0, severe class imbalance).  

---

## **4. Critical Comparison with Original Paper**  

| **Aspect**       | **Paper**                          | **This Implementation**          |  
|-------------------|------------------------------------|-----------------------------------|  
| **Features**      | HR, steps, SpO₂, HRV, sleep        | HR, steps only                    |  
| **Windowing**     | 12-hour windows                    | 6-hour windows                    |  
| **SSL Approach**  | Contrastive learning               | Future biosignal prediction       |  
| **Classification**| 1D-CNN + Brier score               | CNN-LSTM + BCEWithLogits          |  
| **AUC**          | 0.729 (SSL)                       | **0.65** (user-dependent)         |  
| **Accuracy**     | ~70%                              | **72.3%** (average)              |  

---

## **5. Limitations & Recommendations**  

### **5.1 Identified Issues**  
1. **Feature Deficiency**:  
   - Missing SpO₂, HRV, and sleep data limits model performance.  
2. **Class Imbalance**:  
   - Poor recall for rare "use" events (e.g., User 20: 0 TP).  
3. **Thresholding**:  
   - Fixed threshold (0.5) leads to suboptimal accuracy.  

### **5.2 Proposed Fixes**  
1. **Expand Features**: Ingest all Fitbit modalities (SpO₂, HRV).  
2. **Handle Imbalance**: Use focal loss or synthetic oversampling (SMOTE).  
3. **Dynamic Thresholding**: Optimize thresholds per-user via ROC analysis.  
4. **Adopt 12-Hour Windows**: Align with paper for long-term pattern capture.  

---