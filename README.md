**Comprehensive Technical Report: Code Methodology and Results Analysis**

---

### **1. Data Processing Pipeline**

#### **1.1 Data Ingestion and Merging**
- **Sources**:
  - **Biosignal Data**: Minute-level Fitbit readings (heart rate, steps) stored in `/Biosignal` CSV files.  
  - **Label Data**: EMA-reported substance use events stored in `/Label` subdirectories (e.g., `ID14/events.csv`).  
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
     - Scaling parameters stored in `user_scalers` for inverse transformations during error calculation.  

#### **1.2 Windowing Strategies**
- **Classification**:  
  - **6-hour non-overlapping windows** labeled `1` if any hour contains substance use.  
  - Input shape: `[6, 2]` (6 hours × 2 features: `bpm_scaled`, `steps_scaled`).  
- **Forecasting**:  
  - **Input**: 2 consecutive windows (12 hours total).  
  - **Target**: 1 subsequent window (6 hours).  
  - Temporal alignment enforced via strict `datetime` sorting.  

---

### **2. Model Architectures**

#### **2.1 SSLForecastingModel (Self-Supervised Learning)**
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

#### **2.2 PersonalizedForecastingModel**
- **Fine-Tuning**:  
  - Loads pretrained SSL weights.  
  - Unfreezes 50% of layers (e.g., last CNN/LSTM layers) via `partially_unfreeze_backbone()`.  
  - Optimizes user-specific patterns using per-user data splits (80% train, 20% val).  

#### **2.3 DrugClassifier**
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

### **3. Training Procedures**

#### **3.1 SSL Pretraining**
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

#### **3.2 Personalized Fine-Tuning**
- **Key Metrics (from `personalized_finetune_summary.csv`)**:
  | User ID | Best Val Loss | BPM MAE | Steps MAE |
  |---------|---------------|---------|-----------|
  | 20      | 0.162         | 3.27    | 406.9     |
  | 35      | 0.118         | 5.96    | 317.6     |
  | 31      | 0.487         | 11.65   | 307.9     |
  - **Insight**: High variability in performance (e.g., User 31’s BPM MAE=11.65 vs. User 20’s 3.27). Likely due to data quality (e.g., missing sensor readings).  

#### **3.3 Classification**
- **Results (from `classification_summary.csv`)**:
  | User ID | Test Acc (%) | TP | FP | TN | FN |
  |---------|--------------|----|----|----|----|
  | 29      | 100          | 0  | 0  | 5  | 0  |
  | 32      | 100          | 0  | 0  | 15 | 0  |
  | 19      | 12.5         | 1  | 6  | 0  | 1  |
  - **Analysis**:  
    - **Perfect Scores (Users 29, 32)**: Likely overfitting due to small sample size (e.g., User 29: 5 TN, 0 TP).  
    - **Poor Performers (User 19, 20)**: Severe class imbalance (e.g., User 20: 6 FP, 0 TP).  

---

### **4. Critical Comparison with Paper**

#### **4.1 Data and Preprocessing**
| **Aspect**       | **Paper**                          | **Code**                          |
|-------------------|------------------------------------|-----------------------------------|
| **Features**      | HR, steps, SpO₂, HRV, sleep        | HR, steps only                    |
| **Windowing**     | 12-hour windows                    | 6-hour windows                    |
| **Scaling**       | User-specific                      | User-specific (matches paper)     |
| **Label Alignment**| EMA + 1-hour intervals             | EMA + hourly merge (matches)      |

#### **4.2 Model Design**
| **Component**     | **Paper**                          | **Code**                          |
|-------------------|------------------------------------|-----------------------------------|
| **SSL**           | Contrastive learning on biosignals | Forecasting future biosignals     |
| **Classification**| 1D-CNN + Brier score               | CNN-LSTM + BCEWithLogits          |
| **Thresholding**  | Dynamic ROC-based selection        | Fixed threshold (0.5)             |

#### **4.3 Performance**
| **Metric**        | **Paper**              | **Code**               |
|-------------------|------------------------|------------------------|
| **AUC**           | 0.729 (SSL)            | N/A (not implemented)  |
| **MAE (BPM)**     | Not reported           | 3.27–11.65             |
| **Accuracy**      | ~70%                   | 0–100% (user-dependent)|

---

### **5. Limitations and Recommendations**
1. **Feature Deficiency**:  
   - **Issue**: Code excludes SpO₂, HRV, and sleep metrics.  
   - **Fix**: Ingest all Fitbit modalities and implement RF-based feature selection.  

2. **Evaluation Gaps**:  
   - **Issue**: No AUC/threshold optimization.  
   - **Fix**: Add ROC analysis and Brier score for classification.  

3. **Class Imbalance**:  
   - **Issue**: Poor recall for rare "use" events (e.g., User 20: 0% accuracy).  
   - **Fix**: Implement focal loss or oversampling.  

4. **Window Size**:  
   - **Issue**: 6-hour windows may miss long-term patterns.  
   - **Fix**: Adopt 12-hour windows to align with the paper.  

---

### **6. Conclusion**
The codebase provides a robust framework for personalized biosignal analysis but diverges from the paper in feature diversity, model simplicity, and evaluation rigor. By integrating missing modalities, refining evaluation metrics, and adopting dynamic thresholding, the code can achieve clinical-grade performance as described in the paper.  