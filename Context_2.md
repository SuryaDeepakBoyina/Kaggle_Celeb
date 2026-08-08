# 🚀 Final Combined Report & Agent Execution Blueprint

---

## PART 1: MASTER CONTEXT DOCUMENT (For Your Agent)

```
═══════════════════════════════════════════════════════════
PROJECT: Company_X Micro-Fulfillment Demand Forecasting
TYPE: Multi-Series Time-Series Regression
EVALUATION: RMSPE (Root Mean Squared Percentage Error)
CONSTRAINT: Held-out future time period (NO random splits)
RULE: No external data. Only 3 provided CSV files.
═══════════════════════════════════════════════════════════

1. BUSINESS PROBLEM
───────────────────────────────────────────────────────────
Company_X operates 800+ micro-fulfillment hubs delivering
groceries in 15-30 minutes. Demand forecasting errors are
catastrophic:
  - Overstocking → perishables spoil, space wasted
  - Understocking → orders cancelled, trust lost

Goal: Predict DAILY order volume PER HUB for a future period.
Business Use: Replenishment, staffing, fleet allocation.

2. DATA FILES
───────────────────────────────────────────────────────────
FILE 1: orders_train.csv
  - Role: Historical training data
  - Key Columns: Store, Date, OrderVolume (TARGET),
    RiderAvailability, OperationalStatus, PromotionActivity,
    DayOfWeek, HolidayIndicator
  - NOTE: RiderAvailability here is ACTUAL (known after day)

FILE 2: orders_test.csv
  - Role: Future prediction period
  - Structure: Same as train, but OrderVolume is MISSING
  - ⚠️ CRITICAL: RiderAvailability and OperationalStatus
    are PLANNED/SCHEDULED values (estimates), NOT actuals.
    This creates a train-test distribution shift.

FILE 3: hub_metadata.csv
  - Role: Static/semi-static hub attributes
  - Columns: Store, Format, AssortmentTier,
    CompetitorDistance, CompetitorTenure,
    LoyaltyProgramStatus, LaunchDate

JOIN KEY: "Store" across all three files.

3. EVALUATION METRIC: RMSPE
───────────────────────────────────────────────────────────
Formula: sqrt(mean(((actual - predicted) / actual)^2))

RISKS:
  - If actual = 0 → division by zero (undefined)
  - If actual is very small → tiny absolute errors create
    massive percentage errors
  - Heavily penalizes poor predictions on low-volume days

STRATEGY:
  - Check for zeros in OrderVolume immediately
  - Consider log1p(OrderVolume) transformation
  - Monitor low-volume hubs separately
  - Consider weighted loss or clipping

4. CRITICAL VULNERABILITIES IDENTIFIED
───────────────────────────────────────────────────────────
VULNERABILITY 1: Train-Test Distribution Shift
  - Train: RiderAvailability = ACTUAL riders who showed up
  - Test: RiderAvailability = PLANNED riders (estimate)
  - Risk: Model overfits to actual rider patterns that
    won't exist in test
  - Mitigation: Use rider features cautiously, create
    ratios/normalized versions, monitor feature importance

VULNERABILITY 2: Zero/Low Volume Days
  - RMSPE explodes near zero
  - Hubs may have closed days (operational_status)
  - Mitigation: Separate closed days, log-transform target

VULNERABILITY 3: New Hubs (Cold Start)
  - LaunchDate tells us hub age
  - New hubs have no history → lag features are NaN
  - Mitigation: Create hub_age feature, use metadata to
    predict new hub baselines

VULNERABILITY 4: Data Leakage
  - Rolling features must be shifted (no peeking)
  - Validation must be strictly chronological
  - Mitigation: Always shift(1) before rolling windows

5. MODEL STRATEGY
───────────────────────────────────────────────────────────
PRIMARY MODEL: LightGBM
  - Handles mixed data (numeric + categorical)
  - Fast training, handles missing values natively
  - Dominates tabular forecasting competitions

SECONDARY: XGBoost (for comparison/blending)

SKIP: LSTM (too slow for 2-hour sprint, poor with static
features, requires heavy preprocessing)

OPTIONAL FINAL: Simple ensemble (LightGBM + XGBoost average)
or blend with 7-day moving average baseline

6. FEATURE ENGINEERING PLAN
───────────────────────────────────────────────────────────
CATEGORY A: Temporal/Calendar
  - day_of_week (given), is_weekend (derive)
  - month, week_of_year (extract from Date)
  - holiday_indicator (given)
  - Cyclical encoding: sin/cos for day_of_week, month

CATEGORY B: Lag Features (Per Store)
  - lag_1, lag_7, lag_14, lag_21, lag_28
  - These capture weekly seasonality and trends

CATEGORY C: Rolling Statistics (Per Store, SHIFTED)
  - rolling_mean_7, rolling_std_7
  - rolling_mean_14, rolling_mean_28
  - rolling_min_7, rolling_max_7
  - MUST shift(1) before calculating to avoid leakage

CATEGORY D: Hub Metadata (Static)
  - hub_age = (Date - LaunchDate).days
  - CompetitorDistance (binned: near/medium/far)
  - CompetitorTenure (years)
  - Format, AssortmentTier (encode)
  - LoyaltyProgramStatus (binary)

CATEGORY E: Operational (USE WITH CAUTION)
  - RiderAvailability (monitor importance)
  - OperationalStatus (binary: open/closed)
  - PromotionActivity (binary or intensity)
  - Interaction: Promotion × is_weekend

CATEGORY F: Interactions
  - promo × holiday
  - promo × day_of_week
  - hub_age × promotion (new hubs respond differently)

7. VALIDATION STRATEGY
───────────────────────────────────────────────────────────
  - NO random train_test_split
  - Use TimeSeriesSplit or manual chronological split
  - Train: first 80% of dates
  - Validation: last 20% of dates
  - Test: provided future dates
  - Evaluate on validation using RMSPE before touching test
```

---

## PART 2: PROJECT STRUCTURE (What Agent Should Create)

```
demand_forecasting/
│
├── .venv/                          # Virtual environment
├── requirements.txt                # All dependencies
├── README.md                       # Project overview
│
├── data/
│   ├── raw/                        # Original CSVs go here
│   │   ├── orders_train.csv
│   │   ├── orders_test.csv
│   │   └── hub_metadata.csv
│   └── processed/                  # Cleaned/featured data
│       ├── train_featured.csv
│       └── test_featured.csv
│
├── notebooks/
│   └── 01_eda_and_feature_engineering.ipynb   # Main notebook
│
├── src/                            # Helper modules
│   ├── __init__.py
│   ├── data_loader.py              # Load & merge 3 CSVs
│   ├── eda.py                      # All EDA functions
│   ├── feature_engineering.py      # Lag, rolling, temporal
│   └── validation.py               # Time-based split + RMSPE
│
└── outputs/
    ├── plots/                       # Save EDA charts
    └── models/                      # Save trained models later
```

---

## PART 3: ENVIRONMENT SETUP INSTRUCTIONS

```bash
# Step 1: Create project folder
mkdir demand_forecasting
cd demand_forecasting

# Step 2: Create virtual environment
python -m venv .venv

# Step 3: Activate virtual environment
# Windows:
.venv\Scripts\activate
# Mac/Linux:
source .venv/bin/activate

# Step 4: Install dependencies
pip install pandas numpy matplotlib seaborn scikit-learn lightgbm xgboost jupyter notebook tqdm

# Step 5: Create folder structure
mkdir -p data/raw data/processed notebooks src outputs/plots outputs/models

# Step 6: Launch Jupyter
jupyter notebook
```

**requirements.txt content:**
```
pandas>=2.0.0
numpy>=1.24.0
matplotlib>=3.7.0
seaborn>=0.12.0
scikit-learn>=1.3.0
lightgbm>=4.0.0
xgboost>=2.0.0
jupyter>=1.0.0
tqdm>=4.65.0
```

---

## PART 4: DETAILED AGENT PROMPT (Copy-Paste This)

```
═══════════════════════════════════════════════════════════
AGENT INSTRUCTION: BUILD EDA & FEATURE ENGINEERING MODULE
═══════════════════════════════════════════════════════════

You are a Senior Data Scientist building a demand forecasting
pipeline for Company_X micro-fulfillment hubs.

PROJECT CONTEXT: [Paste PART 1 above]

YOUR TASK: Create the following files with clean, modular,
well-documented Python code.

───────────────────────────────────────────────────────────
FILE 1: src/data_loader.py
───────────────────────────────────────────────────────────
Create functions to:
  - load_orders_train(path) → DataFrame
  - load_orders_test(path) → DataFrame
  - load_hub_metadata(path) → DataFrame
  - merge_all_data(train, test, metadata) → merged DataFrames
    (join on "Store" column)
  - Validate: check for missing Store IDs, date gaps,
    duplicate rows

───────────────────────────────────────────────────────────
FILE 2: src/eda.py
───────────────────────────────────────────────────────────
Create functions for:

  a) target_analysis(df):
     - Plot histogram of OrderVolume
     - Count and report zeros
     - Detect outliers (IQR method)
     - Plot time series for 5 random hubs
     - Report skewness and kurtosis

  b) temporal_analysis(df):
     - Plot average OrderVolume by day_of_week
     - Plot holiday vs non-holiday comparison
     - Plot monthly trend
     - Check for weekly seasonality (autocorrelation plot)

  c) operational_analysis(df):
     - Plot RiderAvailability vs OrderVolume (scatter)
     - Compare rider distribution: train vs test
       (⚠️ check for distribution shift)
     - Plot OperationalStatus value counts
     - Promotion lift analysis: avg volume with/without promo

  d) hub_metadata_analysis(df):
     - Plot OrderVolume by Format
     - Plot OrderVolume by AssortmentTier
     - Plot hub_age vs average OrderVolume
     - Plot CompetitorDistance vs OrderVolume
     - LoyaltyProgramStatus impact

  e) rmspe_risk_analysis(df):
     - Identify hubs with average volume < 10
     - Count days with volume < 5 per hub
     - Flag hubs at high RMSPE risk
     - Suggest: which hubs need special treatment

  f) data_quality_report(df):
     - Missing values per column (%)
     - Duplicate rows count
     - Date range coverage per hub
     - Train vs test date overlap check (should be ZERO)

───────────────────────────────────────────────────────────
FILE 3: src/feature_engineering.py
───────────────────────────────────────────────────────────
Create functions for:

  a) create_temporal_features(df):
     - Extract: month, week_of_year, day_of_month
     - Create: is_weekend (binary)
     - Cyclical encoding: day_of_week_sin, day_of_week_cos
     - Cyclical encoding: month_sin, month_cos

  b) create_lag_features(df, lags=[1,7,14,21,28]):
     - For each lag: shift OrderVolume by N days PER STORE
     - Column name: f"lag_{n}"
     - ⚠️ Ensure no data leakage (only past data)

  c) create_rolling_features(df, windows=[7,14,28]):
     - rolling_mean, rolling_std, rolling_min, rolling_max
     - PER STORE, and MUST shift(1) before rolling
     - Column names: f"rolling_mean_{w}", f"rolling_std_{w}"

  d) create_hub_features(df, metadata):
     - hub_age = (Date - LaunchDate).days
     - Encode: Format, AssortmentTier (LabelEncoder)
     - Binary: LoyaltyProgramStatus
     - Bin: CompetitorDistance into [near, medium, far]

  e) create_interaction_features(df):
     - promo_x_weekend = PromotionActivity * is_weekend
     - promo_x_holiday = PromotionActivity * HolidayIndicator
     - rider_ratio = RiderAvailability / rolling_mean_7_rider

  f) build_full_pipeline(train_df, test_df, metadata_df):
     - Call all above functions in order
     - Return: train_featured, test_featured
     - Print shape before/after
     - Log any columns with >50% missing

───────────────────────────────────────────────────────────
FILE 4: src/validation.py
───────────────────────────────────────────────────────────
Create functions for:

  a) time_based_split(df, val_days=14):
     - Split by Date: train = all except last val_days
     - Validation = last val_days
     - Return: train_split, val_split
     - ⚠️ NO random shuffle

  b) rmspe_score(actual, predicted):
     - Implement RMSPE formula
     - Handle zeros: add epsilon (1e-8) to denominator
     - Return: float score

  c) validate_features(train_featured, val_featured):
     - Check: no NaN in lag features for validation
     - Check: feature distributions match
     - Report: any feature with train mean ≠ val mean > 2x

───────────────────────────────────────────────────────────
FILE 5: notebooks/01_eda_and_feature_engineering.ipynb
───────────────────────────────────────────────────────────
Structure the notebook as:

  Cell 1: Title + Import all modules
  Cell 2: Load data using data_loader.py
  Cell 3: Basic shape/dtypes/preview
  Cell 4: Run target_analysis()
  Cell 5: Run temporal_analysis()
  Cell 6: Run operational_analysis()
  Cell 7: Run hub_metadata_analysis()
  Cell 8: Run rmspe_risk_analysis()
  Cell 9: Run data_quality_report()
  Cell 10: KEY FINDINGS SUMMARY (markdown cell)
  Cell 11: Run build_full_pipeline()
  Cell 12: Run time_based_split()
  Cell 13: Verify features on validation set
  Cell 14: Save processed data to data/processed/
  Cell 15: NEXT STEPS (markdown cell)

───────────────────────────────────────────────────────────
CODING STANDARDS:
───────────────────────────────────────────────────────────
  - Type hints on all functions
  - Docstrings on all functions
  - Use logging (not print) for pipeline steps
  - Handle edge cases: empty DataFrames, missing columns
  - All plots must have titles, axis labels, and be saved
    to outputs/plots/
  - Use tqdm for any loop over 800+ hubs
  - Make functions idempotent (safe to re-run)

───────────────────────────────────────────────────────────
CRITICAL RULES:
───────────────────────────────────────────────────────────
  1. NEVER use random train_test_split
  2. ALWAYS shift(1) before rolling calculations
  3. ALWAYS check for zeros before RMSPE calculation
  4. ALWAYS compare rider_availability train vs test
  5. Treat rider_availability as UNRELIABLE in test
  6. No external data sources allowed
  7. Join key is "Store" everywhere
```

---

## PART 5: WHAT YOU NEED TO DO NOW

| Step | Action |
|------|--------|
| 1 | Create the folder structure above |
| 2 | Create `.venv` and install requirements |
| 3 | Download the 3 CSV files into `data/raw/` |
| 4 | Paste the **Agent Prompt (Part 4)** to your agent |
| 5 | Once agent generates the code, run the notebook |
| 6 | Come back here with EDA outputs if you need help interpreting |

---

Once you have the data downloaded and the agent has built the structure, we will move into **Phase 2: Running EDA and interpreting the findings together**. I will help you decide which features to keep, which to drop, and how to handle the rider availability shift.

**Go build the structure, drop the data files in, and let's execute!** 🚀
