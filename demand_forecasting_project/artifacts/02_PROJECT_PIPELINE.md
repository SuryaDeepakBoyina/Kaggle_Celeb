# 🗺️ Project Pipeline Architecture
## Demand Forecasting – Micro-Fulfillment Hubs

> **Purpose:** Full pipeline diagram and stage-by-stage description for all phases of the project.

---

## Pipeline Overview

```
┌─────────────────────────────────────────────────────────────────────────┐
│                    DEMAND FORECASTING PIPELINE                          │
└─────────────────────────────────────────────────────────────────────────┘

  [RAW DATA]                    [PROCESSING]               [OUTPUT]
  ────────────                  ────────────               ────────
  orders_train.csv ──┐
  orders_test.csv  ──┼──► MERGE with hub_metadata ──► merged_train
  hub_metadata.csv ──┘                             ──► merged_test

  merged_train ──► EDA ──────────────────────────► plots/ (11+ PNGs)
                                                 ──► EDA observations dict

  merged_train ──┐
  merged_test  ──┴──► FEATURE ENGINEERING ──────► train_featured
                                               ──► test_featured

  train_featured ──► TIME-BASED SPLIT ──────────► df_train (80%)
                                               ──► df_val   (20%)

  df_train ──► LightGBM.fit() ──────────────────► model_lgb
  df_train ──► XGBoost.fit()  ──────────────────► model_xgb

  model_lgb ──► predict(df_val) ─────────────────► val_preds_lgb
  model_xgb ──► predict(df_val) ─────────────────► val_preds_xgb
  0.5×lgb + 0.5×xgb ────────────────────────────► val_preds_ens

  val_preds_* ──► evaluate_model() ─────────────► RMSE, RMSPE, MAE

  best_model ──► predict(test_featured) ─────────► test_preds
  test_preds ──► round + clip + IsOpen=0 mask ───► predictions.csv
```

---

## Phase 1: EDA (Exploratory Data Analysis)

**File:** `src/eda.py` → `perform_eda()`
**Notebook section:** Cell 2

| Step | What Happens | Output |
|---|---|---|
| 1.1 Basic Info | Shape, columns, date ranges, hub counts | Console output |
| 1.2 Missing Values | Per-column NaN counts for train & test | results dict |
| 1.3 Target Distribution | Histogram, log histogram, boxplot; skewness, zero% | `01_target_distribution.png` |
| 1.4 Time-Series | Total daily volume, top-6 hub time series | `02_daily_total_timeseries.png`, `03_sample_hub_timeseries.png` |
| 1.5 Weekly Patterns | Mean/median volume by Weekday | `04_weekday_pattern.png` |
| 1.6 Monthly Patterns | Mean volume by month | `05_monthly_pattern.png` |
| 1.7 Promo/Holiday/School | Mean volume by PromoActive, RegionalHoliday, SchoolClosureFlag | `06_promo_holiday_effects.png` |
| 1.8 AppSessions Correlation | Scatter + Pearson r | `07_appsessions_vs_ordervolume.png` |
| 1.9 Hub Metadata | Volume by HubFormat, AssortmentTier, LoyaltyProgram | `08-10_hub_*.png` |
| 1.10 Correlation Heatmap | All numeric features vs OrderVolume | `11_correlation_heatmap.png` |
| 1.11 Outlier Analysis | IQR-based upper fence, outlier count | results dict |
| 1.12 Key Findings | Auto-generated insights list | results['key_findings'] |

---

## Phase 2: Feature Engineering

**File:** `src/features.py` → `create_features()`
**Notebook section:** Cells 3a & 3b

### Critical Design Decisions

**Q: How do we get lag features for the test set?**
**A:** Concatenate train+test (sorted by HubID + Date), compute lags on the combined series, then split back. Test rows naturally inherit lag values computed from train.

```python
combined = pd.concat([train_with_dummy_target, test])
combined_feat = create_features(combined, is_train=True)
train_feat = combined_feat[combined_feat['_set'] == 'train']
test_feat  = combined_feat[combined_feat['_set'] == 'test']
```

### Feature Groups Built

| Group | Features | Count |
|---|---|---|
| Temporal | year, month, day, day_of_year, week_of_year, quarter, is_weekend, cyclical sin/cos | ~10 |
| Lags | lag_1, lag_7, lag_14, lag_21, lag_28 | 5 |
| Rolling | rolling_mean/std (7,14,28), rolling_max/min_7, momentum_7 | 9 |
| Hub Metadata | competitor_distance_log, _binned, competitor_tenure_days, has_loyalty, loyalty_age_years, has_loyalty_interval | 6 |
| Operational | IsOpen, PromoActive, RegionalHoliday, SchoolClosureFlag, AppSessions, app_sessions_log | 6 |
| Interactions | promo_weekend, promo_holiday, promo_school, format_tier, sessions_promo | 5 |
| Categoricals | HubFormat codes, AssortmentTier codes, format_tier codes | 3 |
| **TOTAL** | | **~44 features** |

---

## Phase 3: Model Training

**File:** `src/model.py` → `train_model()`
**Notebook section:** Cells 5a & 5b

### Preprocessing Before Training
1. **Remove IsOpen=0 rows** from training (trivially zero, shouldn't teach model zero patterns)
2. **Drop rows with NaN lag features** (first 28 days per hub lack full lag history)
3. Keep all val rows for realistic RMSPE evaluation

### Model Configurations

**LightGBM (Primary)**
```
n_estimators    = 2000
learning_rate   = 0.03
num_leaves      = 63
max_depth       = -1 (unlimited)
subsample       = 0.8
colsample_bytree= 0.8
min_child_samples= 20
reg_alpha       = 0.1  (L1)
reg_lambda      = 0.1  (L2)
early_stopping  = 50 rounds
```

**XGBoost (Comparison)**
```
n_estimators    = 2000
learning_rate   = 0.03
max_depth       = 6
subsample       = 0.8
colsample_bytree= 0.8
early_stopping  = 50 rounds
```

---

## Phase 4: Evaluation

**File:** `src/model.py` → `evaluate_model()`
**Notebook section:** Cell 6

### Metrics Computed
| Metric | Formula | Target |
|---|---|---|
| RMSPE | `sqrt(mean(((y_true-y_pred)/y_true)^2))` — zeros excluded | < 0.30 (good), < 0.25 (excellent) |
| RMSE | `sqrt(mean((y_true-y_pred)^2))` | Minimize |
| MAE | `mean(|y_true-y_pred|)` | Minimize |

### Plots Generated
- `12_actual_vs_predicted.png` – scatter for LGB, XGB, Ensemble
- `13_residuals.png` – residual distribution + residuals vs fitted
- `14_feature_importance.png` – top 25 features for both models

### Decision Logic
- Compare RMSPE of LGB, XGB, Ensemble (0.5×LGB + 0.5×XGB)
- Use whichever has lowest RMSPE on validation for final test predictions

---

## Phase 5: Predictions & Submission

**Notebook section:** Cells 8 & 9

### Post-Processing Rules
1. **Clip negatives**: `np.clip(preds, 0, None)`
2. **Force zero for closed hubs**: `preds[IsOpen == 0] = 0`
3. **Round to integer**: `np.round(preds).astype(int)`

### Output Files
| File | Description |
|---|---|
| `outputs/predictions.csv` | Final submission (`Id, OrderVolume`) |
| `outputs/validation_metrics.csv` | RMSE/RMSPE/MAE for all models |
| `outputs/feature_importance_lgb.csv` | LightGBM feature rankings |
| `outputs/feature_importance_xgb.csv` | XGBoost feature rankings |
| `outputs/model_artifacts/lightgbm_model.pkl` | Saved LightGBM model |
| `outputs/model_artifacts/xgboost_model.pkl` | Saved XGBoost model |
| `outputs/plots/` | 14+ visualization PNGs |

---

## File Structure

```
demand_forecasting_project/
├── data/
│   ├── orders_train.csv
│   ├── orders_test.csv
│   ├── hub_metadata.csv
│   └── sample_submission.csv
├── src/
│   ├── __init__.py
│   ├── utils.py        ← load_data(), setup_logging()
│   ├── eda.py          ← perform_eda()
│   ├── features.py     ← create_features(), get_feature_cols()
│   └── model.py        ← train_model(), evaluate_model(), predict(), save_model()
├── notebooks/
│   └── main_workflow.ipynb   ← END-TO-END RUNNABLE NOTEBOOK
├── outputs/
│   ├── plots/          ← EDA + evaluation PNGs
│   ├── model_artifacts/← .pkl model files
│   ├── predictions.csv
│   ├── validation_metrics.csv
│   ├── feature_importance_lgb.csv
│   └── feature_importance_xgb.csv
├── artifacts/
│   ├── 01_EDA_OBSERVATIONS.md    ← This file's companion
│   ├── 02_PROJECT_PIPELINE.md    ← This file
│   └── 03_CONTEXT_FOR_LLM.md    ← Pass-to-LLM context package
├── requirements.txt
└── README.md
```

---

## Running the Project

```bash
# 1. Create & activate venv
python -m venv .venv
source .venv/bin/activate  # Linux/Mac

# 2. Install dependencies
pip install -r requirements.txt

# 3. Launch notebook
cd notebooks
jupyter notebook main_workflow.ipynb

# 4. Run all cells sequentially
# OR run as script:
cd ..
jupyter nbconvert --to notebook --execute notebooks/main_workflow.ipynb
```
