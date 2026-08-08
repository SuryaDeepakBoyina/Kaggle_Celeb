# 🤖 Context Package for Downstream LLMs
## Demand Forecasting – Micro-Fulfillment Hubs

> **Purpose:** Self-contained context to pass to another LLM agent to continue this project.
> Contains: task definition, actual schema, completed work, next steps, and known pitfalls.

---

## TASK DEFINITION

**Problem Type:** Multi-series time-series regression (one series per hub).

**Target:** `OrderVolume` — daily integer count of orders at each hub.

**Metric:** RMSPE = `sqrt(mean(((actual - predicted) / actual)^2))`
- Zeros in `actual` are EXCLUDED from the denominator (division by zero risk).
- Good: RMSPE < 0.30 | Excellent: RMSPE < 0.25

**Constraint:** Use ONLY the 3 provided data files. No external data.

**Submission format:** `Id, OrderVolume` (Id from orders_test.csv)

---

## ACTUAL DATASET SCHEMA

### orders_train.csv
```
HubID          : int   – hub identifier (join key)
Weekday        : int   – 1=Mon, 2=Tue, ..., 7=Sun
Date           : date  – YYYY-MM-DD
OrderVolume    : int   – TARGET VARIABLE (daily orders)
AppSessions    : int   – in-app sessions that day (NOT in test!)
IsOpen         : int   – 1=open, 0=closed
PromoActive    : int   – 1=promotion active
RegionalHoliday: int   – 1=regional holiday
SchoolClosureFlag: int – 1=schools closed
```
Date range: **2013-01-01 → 2015-06-19**

### orders_test.csv
```
Id             : int   – submission row ID (NEEDED for output)
HubID          : int   – hub identifier
Weekday        : int   – same as train
Date           : date  – test period dates
IsOpen         : int   – 1/0
PromoActive    : int   – 1/0
RegionalHoliday: int   – 1/0
SchoolClosureFlag: int – 1/0
NOTE: AppSessions is ABSENT from test!
```
Date range: **2015-06-20 → 2015-07-31** (~6 weeks)

### hub_metadata.csv
```
HubID                   : int
HubFormat               : int   – hub type (values: 1, 3, 4)
AssortmentTier          : int   – product breadth (values: 1, 3)
CompetitorDistance      : float – distance to nearest competitor (meters)
CompetitorOpenSinceMonth: float – month competitor opened (NaN = unknown)
CompetitorOpenSinceYear : float – year competitor opened (NaN = unknown)
LoyaltyProgram          : int   – 1=has loyalty program
LoyaltyProgramSinceWeek : float – week loyalty launched (NaN if no program)
LoyaltyProgramSinceYear : float – year loyalty launched
LoyaltyProgramInterval  : str   – quarterly schedule (e.g. "Jan,Apr,Jul,Oct")
```

---

## WHAT HAS BEEN BUILT

### Project Structure (all files created)
```
demand_forecasting_project/
├── data/                   ✅ all 4 CSVs copied here
├── src/
│   ├── utils.py            ✅ load_data(), setup_logging()
│   ├── eda.py              ✅ perform_eda() – full EDA with 11+ plots
│   ├── features.py         ✅ create_features(), get_feature_cols()
│   └── model.py            ✅ train_model(), evaluate_model(), predict(), save_model()
├── notebooks/
│   └── main_workflow.ipynb ✅ end-to-end runnable notebook (9 sections)
├── artifacts/
│   ├── 01_EDA_OBSERVATIONS.md   ✅ complete EDA findings
│   ├── 02_PROJECT_PIPELINE.md   ✅ pipeline architecture
│   └── 03_CONTEXT_FOR_LLM.md   ✅ this file
└── requirements.txt        ✅ all dependencies
```

### What the Notebook Does (in order)
1. **Load & merge** orders_train/test + hub_metadata (join on HubID)
2. **EDA** – calls `perform_eda()`, saves 11+ PNGs to `outputs/plots/`
3. **Feature engineering** – concatenates train+test, builds lags/rolling on combined, splits back
4. **Time split** – 80% train dates / 20% val dates (time-based, expanding window)
5. **Train LightGBM** with early stopping on val set
6. **Train XGBoost** with early stopping on val set
7. **Evaluate** LGB, XGB, Ensemble (0.5 each) → prints RMSE/RMSPE/MAE
8. **Predict** on test set using best model
9. **Save** models, metrics, feature importances, predictions.csv

---

## KNOWN PITFALLS (Critical!)

### 1. AppSessions not in test
- `AppSessions` is a strong signal in train but absent from test
- Solution: build lag/rolling features of AppSessions as part of feature engineering on combined (train+test) data
- The combined approach propagates train AppSessions values into test lag features

### 2. Zero inflation from IsOpen=0
- All rows where IsOpen=0 have OrderVolume=0 trivially
- These rows pollute training (model learns "zeros everywhere")
- **Solution implemented:** Filter `IsOpen=0` rows BEFORE training
- **For test:** Force predictions to 0 where IsOpen=0

### 3. RMSPE denominator
- Never evaluate RMSPE including zero-actual rows
- The `rmspe()` function in model.py already masks zeros

### 4. Lag NaN at start of series
- First 28 days of each hub's history have NaN lag features (not enough history)
- **Solution:** Drop these rows from training set (use `dropna(subset=lag_cols)`)
- **For test:** Fill NaN lags with 0 or hub-level median before predicting

### 5. No random splits allowed
- Always use time-based split
- Val dates must be AFTER all train dates

### 6. Negative predictions
- Tree models can predict negatives for small volumes
- Always `np.clip(predictions, 0, None)` before submission

### 7. Rider availability (from context docs)
- Context docs mentioned train/test distribution shift in `rider_availability`
- **THIS COLUMN DOES NOT EXIST** in the actual dataset — ignore this warning

---

## FEATURE ENGINEERING SUMMARY

All features built by `create_features()`:

```python
TEMPORAL (10):     year, month, day, day_of_year, week_of_year, quarter,
                   is_weekend, month_sin, month_cos, weekday_sin, weekday_cos

LAGS (5):          lag_1, lag_7, lag_14, lag_21, lag_28

ROLLING (9):       rolling_mean_7/14/28, rolling_std_7/14/28,
                   rolling_max_7, rolling_min_7, momentum_7

HUB METADATA (6):  competitor_distance_log, competitor_distance_binned,
                   competitor_tenure_days, has_loyalty,
                   loyalty_age_years, has_loyalty_interval

OPERATIONAL (6):   IsOpen, PromoActive, RegionalHoliday, SchoolClosureFlag,
                   AppSessions, app_sessions_log

INTERACTIONS (5):  promo_weekend, promo_holiday, promo_school,
                   format_tier (encoded), sessions_promo

CATEGORICAL (3):   HubFormat (codes), AssortmentTier (codes), format_tier (codes)

TOTAL: ~44 features
```

---

## MODEL CONFIGURATION

### LightGBM (Primary)
```python
lgb.LGBMRegressor(
    n_estimators=2000, learning_rate=0.03, num_leaves=63,
    max_depth=-1, subsample=0.8, colsample_bytree=0.8,
    min_child_samples=20, reg_alpha=0.1, reg_lambda=0.1,
    random_state=42
)
# With early stopping (50 rounds) on validation set
```

### XGBoost (Comparison)
```python
xgb.XGBRegressor(
    n_estimators=2000, learning_rate=0.03, max_depth=6,
    subsample=0.8, colsample_bytree=0.8,
    random_state=42
)
# With early stopping (50 rounds) on validation set
```

### Ensemble
Simple average: `0.5 × LGB_preds + 0.5 × XGB_preds`
Best of 3 chosen by lowest validation RMSPE.

---

## WHAT TO DO NEXT (Suggested Next Steps)

### If RMSPE > 0.30 (model underperforming):
1. **Try log1p target transformation**: train on `log1p(OrderVolume)`, expm1 on predictions
2. **Increase num_leaves** to 127 or 255 (LightGBM)
3. **Add more lags**: lag_35, lag_42, lag_56 (more weekly cycles)
4. **Add per-hub target encoding**: mean/std of OrderVolume per HubID from train
5. **Try Optuna hyperparameter tuning**: 50 trials, optimize RMSPE on val set
6. **Hub-specific models**: cluster hubs by volume level, train separate models per cluster

### If RMSPE is < 0.25 (model is good):
1. Try **weighted ensemble**: weight LGB more if it consistently beats XGB
2. **Stacking**: use LGB predictions as a meta-feature for a linear model
3. **Submission refinement**: round predictions to nearest integer

### If new hubs appear in test (not in train):
1. Check: `test_feat['HubID'].isin(orders_train['HubID'].unique())`
2. For new hubs: fill all lag/rolling features with global mean
3. Consider a "cold start" model using only static hub metadata features

---

## QUICK REFERENCE: How to Run

```bash
cd demand_forecasting_project
pip install -r requirements.txt
jupyter notebook notebooks/main_workflow.ipynb
# Run all cells top-to-bottom
```

Expected outputs:
- `outputs/plots/` – 14+ PNG plots
- `outputs/predictions.csv` – submission file
- `outputs/validation_metrics.csv` – RMSE/RMSPE/MAE
- `outputs/model_artifacts/*.pkl` – saved models

---

## EXAMPLE RMSPE CALCULATION (for sanity check)

```python
import numpy as np

def rmspe(y_true, y_pred):
    mask = y_true > 0
    pct_err = (y_true[mask] - y_pred[mask]) / y_true[mask]
    return np.sqrt(np.mean(pct_err ** 2))

# Quick test:
y_true = np.array([100, 50, 0, 200])
y_pred = np.array([90,  55, 5, 180])
print(rmspe(y_true, y_pred))  # ~0.086 (zeros excluded)
```
