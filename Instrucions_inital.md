# 🎯 **INSTRUCTIONS FOR YOUR AI AGENT**

## Project Setup & Execution Steps

---

### Step 1: Project Structure

```
demand_forecasting_project/
│
├── .venv/                      # Virtual environment
├── data/
│   ├── orders_train.csv        # Place downloaded files here
│   ├── orders_test.csv
│   └── hub_metadata.csv
│
├── src/
│   ├── __init__.py
│   ├── eda.py                  # EDA functions
│   ├── features.py             # Feature engineering
│   ├── model.py                # Model training & evaluation
│   └── utils.py                # Helper functions
│
├── notebooks/
│   └── main_workflow.ipynb     # Main Jupyter Notebook
│
├── outputs/
│   ├── plots/                  # Visualizations
│   ├── predictions.csv         # Final predictions
│   └── model_artifacts/        # Saved models
│
├── requirements.txt
└── README.md
```

---

### Step 2: Virtual Environment Setup

```bash
# Create virtual environment
python -m venv .venv

# Activate (Windows)
.venv\Scripts\activate

# Activate (Mac/Linux)
source .venv/bin/activate

# Install requirements
pip install -r requirements.txt
```

---

### Step 3: `requirements.txt`

```
# Core Data Science
pandas==2.0.3
numpy==1.24.3
scipy==1.11.1

# Visualization
matplotlib==3.7.2
seaborn==0.12.2
plotly==5.15.0

# Machine Learning
scikit-learn==1.3.0
lightgbm==4.0.0
xgboost==2.0.0

# Utilities
jupyter==1.0.0
tqdm==4.65.0
joblib==1.3.1
python-dotenv==1.0.0

# Optional (for advanced)
optuna==3.3.0  # Hyperparameter optimization
```

---

### Step 4: Main Notebook Workflow (`main_workflow.ipynb`)

```python
# ============================================
# 1. IMPORTS & CONFIGURATION
# ============================================
import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path('.').resolve() / 'src'))

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

from eda import perform_eda
from features import create_features
from model import train_model, evaluate_model, predict
from utils import setup_logging, load_data

# Configuration
DATA_PATH = Path('../data')
OUTPUT_PATH = Path('../outputs')
PLOT_PATH = OUTPUT_PATH / 'plots'
PLOT_PATH.mkdir(parents=True, exist_ok=True)

# ============================================
# 2. LOAD DATA
# ============================================
print("Loading data...")
orders_train = pd.read_csv(DATA_PATH / 'orders_train.csv')
orders_test = pd.read_csv(DATA_PATH / 'orders_test.csv')
hub_metadata = pd.read_csv(DATA_PATH / 'hub_metadata.csv')

print(f"Train shape: {orders_train.shape}")
print(f"Test shape: {orders_test.shape}")
print(f"Metadata shape: {hub_metadata.shape}")

# ============================================
# 3. DATA MERGING
# ============================================
# Merge train with metadata
train = orders_train.merge(hub_metadata, on='Store', how='left')

# Merge test with metadata
test = orders_test.merge(hub_metadata, on='Store', how='left')

# ============================================
# 4. EDA (Phase 1)
# ============================================
from eda import perform_eda

eda_results = perform_eda(
    train=train,
    test=test,
    target='OrderVolume',
    hub_id='Store',
    date_col='Date',
    plot_path=PLOT_PATH
)

# ============================================
# 5. FEATURE ENGINEERING (Phase 2)
# ============================================
from features import create_features

# Create features for training
train_featured = create_features(
    df=train,
    date_col='Date',
    hub_id='Store',
    target_col='OrderVolume',
    is_train=True
)

# Create features for test (no target)
test_featured = create_features(
    df=test,
    date_col='Date',
    hub_id='Store',
    target_col=None,
    is_train=False
)

# ============================================
# 6. TRAIN/VALIDATION SPLIT (Time-based)
# ============================================
# Sort by date
train_featured = train_featured.sort_values(['Store', 'Date'])

# Get unique dates
unique_dates = train_featured['Date'].unique()
split_idx = int(len(unique_dates) * 0.8)
train_dates = unique_dates[:split_idx]
val_dates = unique_dates[split_idx:]

train_df = train_featured[train_featured['Date'].isin(train_dates)]
val_df = train_featured[train_featured['Date'].isin(val_dates)]

print(f"Train period: {train_dates[0]} to {train_dates[-1]}")
print(f"Val period: {val_dates[0]} to {val_dates[-1]}")

# ============================================
# 7. MODEL TRAINING (Phase 3)
# ============================================
from model import train_model, evaluate_model

# Define features to use (exclude non-feature columns)
exclude_cols = ['Store', 'Date', 'OrderVolume', 'launch_date']
feature_cols = [col for col in train_df.columns if col not in exclude_cols]

X_train = train_df[feature_cols]
y_train = train_df['OrderVolume']
X_val = val_df[feature_cols]
y_val = val_df['OrderVolume']

# Train LightGBM
model_lgb = train_model(
    X_train=X_train,
    y_train=y_train,
    model_type='lightgbm',
    params={
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'num_leaves': 31,
        'max_depth': -1,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
)

# Evaluate
val_preds = model_lgb.predict(X_val)
rmse, rmspe = evaluate_model(y_val, val_preds)
print(f"RMSE: {rmse:.2f}")
print(f"RMSPE: {rmspe:.4f}")

# ============================================
# 8. PREDICTIONS (Phase 4)
# ============================================
# Predict on test data
test_preds = model_lgb.predict(test_featured[feature_cols])

# Create submission
submission = pd.DataFrame({
    'Store': test_featured['Store'],
    'Date': test_featured['Date'],
    'OrderVolume': test_preds
})

# Save predictions
submission.to_csv(OUTPUT_PATH / 'predictions.csv', index=False)

# ============================================
# 9. REFINEMENT (Phase 5 - Optional)
# ============================================
# Try XGBoost for comparison
from model import train_model

model_xgb = train_model(
    X_train=X_train,
    y_train=y_train,
    model_type='xgboost',
    params={
        'n_estimators': 1000,
        'learning_rate': 0.05,
        'max_depth': 6,
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'random_state': 42
    }
)

xgb_preds = model_xgb.predict(X_val)
_, xgb_rmspe = evaluate_model(y_val, xgb_preds)
print(f"XGBoost RMSPE: {xgb_rmspe:.4f}")

# ============================================
# 10. SAVE ARTIFACTS
# ============================================
import joblib

joblib.dump(model_lgb, OUTPUT_PATH / 'model_artifacts' / 'lightgbm_model.pkl')
joblib.dump(feature_cols, OUTPUT_PATH / 'model_artifacts' / 'feature_cols.pkl')

print("✅ Pipeline complete!")
```

---

### Step 5: Helper Module Files

#### `src/eda.py`

```python
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path

def perform_eda(train, test, target, hub_id, date_col, plot_path):
    """
    Perform comprehensive EDA on the dataset
    """
    results = {}
    
    # 1. Basic info
    print("\n=== BASIC INFO ===")
    print(f"Train shape: {train.shape}")
    print(f"Test shape: {test.shape}")
    print(f"Train columns: {train.columns.tolist()}")
    print(f"Test columns: {test.columns.tolist()}")
    
    # 2. Missing values
    print("\n=== MISSING VALUES ===")
    train_missing = train.isnull().sum()
    test_missing = test.isnull().sum()
    print("Train missing:\n", train_missing[train_missing > 0])
    print("Test missing:\n", test_missing[test_missing > 0])
    results['train_missing'] = train_missing
    results['test_missing'] = test_missing
    
    # 3. Target distribution
    print("\n=== TARGET DISTRIBUTION ===")
    print(train[target].describe())
    
    # Plot target distribution
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    sns.histplot(train[target], bins=50, ax=axes[0])
    axes[0].set_title(f'Distribution of {target}')
    sns.boxplot(y=train[target], ax=axes[1])
    axes[1].set_title(f'Boxplot of {target}')
    plt.tight_layout()
    plt.savefig(plot_path / 'target_distribution.png')
    plt.close()
    results['target_dist'] = train[target].describe()
    
    # 4. Zero values in target
    zero_count = (train[target] == 0).sum()
    zero_pct = zero_count / len(train) * 100
    print(f"\nZeros in {target}: {zero_count} ({zero_pct:.2f}%)")
    results['zero_count'] = zero_count
    results['zero_pct'] = zero_pct
    
    # 5. Time patterns
    print("\n=== TIME PATTERNS ===")
    train['Date'] = pd.to_datetime(train[date_col])
    test['Date'] = pd.to_datetime(test[date_col])
    
    # Sample one hub for time series plot
    sample_hub = train[hub_id].value_counts().index[0]
    sample_data = train[train[hub_id] == sample_hub].sort_values(date_col)
    
    fig, ax = plt.subplots(figsize=(14, 5))
    ax.plot(sample_data[date_col], sample_data[target])
    ax.set_title(f'{target} for Hub {sample_hub} over time')
    ax.set_xlabel('Date')
    ax.set_ylabel('Order Volume')
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(plot_path / 'time_series_sample.png')
    plt.close()
    
    # 6. Day of week patterns
    if 'day_of_week' in train.columns:
        dow_avg = train.groupby('day_of_week')[target].mean().reindex([0,1,2,3,4,5,6])
        fig, ax = plt.subplots(figsize=(10, 5))
        dow_avg.plot(kind='bar', ax=ax)
        ax.set_title(f'Average {target} by Day of Week')
        ax.set_xlabel('Day of Week (0=Mon, 6=Sun)')
        ax.set_ylabel('Average Order Volume')
        plt.tight_layout()
        plt.savefig(plot_path / 'day_of_week_pattern.png')
        plt.close()
        results['dow_pattern'] = dow_avg.to_dict()
    
    # 7. Train-Test shift in rider_availability
    if 'rider_availability' in train.columns and 'rider_availability' in test.columns:
        fig, ax = plt.subplots(figsize=(10, 5))
        train['rider_availability'].hist(alpha=0.5, label='Train', bins=30, ax=ax)
        test['rider_availability'].hist(alpha=0.5, label='Test', bins=30, ax=ax)
        ax.set_title('Rider Availability Distribution: Train vs Test')
        ax.legend()
        plt.savefig(plot_path / 'rider_shift.png')
        plt.close()
        
        # Statistical test
        from scipy import stats
        ks_stat, ks_p = stats.ks_2samp(
            train['rider_availability'].dropna(),
            test['rider_availability'].dropna()
        )
        print(f"\nRider availability KS test: statistic={ks_stat:.3f}, p-value={ks_p:.4f}")
        results['rider_ks_stat'] = ks_stat
        results['rider_ks_p'] = ks_p
    
    # 8. Hub metadata exploration
    print("\n=== HUB METADATA ===")
    print(f"Total hubs: {train[hub_id].nunique()}")
    print(f"Avg orders per hub: {train.groupby(hub_id)[target].mean().mean():.2f}")
    
    # Hub volume distribution
    hub_avg = train.groupby(hub_id)[target].mean().sort_values()
    fig, ax = plt.subplots(figsize=(12, 4))
    hub_avg.hist(bins=50, ax=ax)
    ax.set_title('Distribution of Average Order Volume by Hub')
    ax.set_xlabel('Average Order Volume')
    plt.savefig(plot_path / 'hub_volume_distribution.png')
    plt.close()
    
    results['n_hubs'] = train[hub_id].nunique()
    results['avg_hub_volume'] = hub_avg.mean()
    
    print("\n✅ EDA Complete!")
    return results
```

#### `src/features.py`

```python
import pandas as pd
import numpy as np
from tqdm import tqdm

def create_features(df, date_col, hub_id, target_col=None, is_train=True):
    """
    Create all features for the model
    
    Parameters:
    -----------
    df : pd.DataFrame
        Input dataframe
    date_col : str
        Name of date column
    hub_id : str
        Name of hub identifier column
    target_col : str
        Name of target column (for training data)
    is_train : bool
        Whether this is training data (has target)
    
    Returns:
    --------
    pd.DataFrame with engineered features
    """
    
    # Make a copy
    df_copy = df.copy()
    
    # Ensure date is datetime
    df_copy[date_col] = pd.to_datetime(df_copy[date_col])
    
    # Sort by hub and date
    df_copy = df_copy.sort_values([hub_id, date_col])
    
    # ============================================
    # 1. TEMPORAL FEATURES
    # ============================================
    df_copy['year'] = df_copy[date_col].dt.year
    df_copy['month'] = df_copy[date_col].dt.month
    df_copy['day'] = df_copy[date_col].dt.day
    df_copy['day_of_year'] = df_copy[date_col].dt.dayofyear
    df_copy['week_of_year'] = df_copy[date_col].dt.isocalendar().week.astype(int)
    df_copy['quarter'] = df_copy[date_col].dt.quarter
    
    # Is weekend
    if 'day_of_week' not in df_copy.columns:
        df_copy['day_of_week'] = df_copy[date_col].dt.dayofweek
    df_copy['is_weekend'] = df_copy['day_of_week'].isin([5, 6]).astype(int)
    
    # ============================================
    # 2. LAG FEATURES (PER HUB)
    # ============================================
    if is_train and target_col:
        # Lags for training data
        for lag in [1, 7, 14, 21, 28]:
            df_copy[f'lag_{lag}'] = df_copy.groupby(hub_id)[target_col].shift(lag)
        
        # Rolling statistics
        for window in [7, 14, 28]:
            df_copy[f'rolling_mean_{window}'] = df_copy.groupby(hub_id)[target_col].transform(
                lambda x: x.shift(1).rolling(window).mean()
            )
            df_copy[f'rolling_std_{window}'] = df_copy.groupby(hub_id)[target_col].transform(
                lambda x: x.shift(1).rolling(window).std()
            )
        
        # Rolling max/min
        df_copy[f'rolling_max_7'] = df_copy.groupby(hub_id)[target_col].transform(
            lambda x: x.shift(1).rolling(7).max()
        )
        df_copy[f'rolling_min_7'] = df_copy.groupby(hub_id)[target_col].transform(
            lambda x: x.shift(1).rolling(7).min()
        )
    
    # ============================================
    # 3. HUB METADATA FEATURES
    # ============================================
    if 'launch_date' in df_copy.columns:
        df_copy['launch_date'] = pd.to_datetime(df_copy['launch_date'])
        df_copy['hub_age_days'] = (df_copy[date_col] - df_copy['launch_date']).dt.days
        df_copy['hub_age_months'] = df_copy['hub_age_days'] / 30.44
        df_copy['hub_age_years'] = df_copy['hub_age_days'] / 365.25
        
        # Hub maturity categories
        df_copy['hub_maturity'] = pd.cut(
            df_copy['hub_age_days'],
            bins=[-1, 30, 90, 365, 10000],
            labels=['new', 'recent', 'established', 'mature']
        )
    
    # ============================================
    # 4. COMPETITION FEATURES
    # ============================================
    if 'competitor_distance' in df_copy.columns:
        # Binned competitor distance
        df_copy['competitor_distance_binned'] = pd.cut(
            df_copy['competitor_distance'],
            bins=[-1, 500, 1000, 2000, 5000, 100000],
            labels=['very_close', 'close', 'medium', 'far', 'very_far']
        )
    
    if 'competitor_tenure' in df_copy.columns:
        df_copy['competitor_tenure_binned'] = pd.cut(
            df_copy['competitor_tenure'],
            bins=[-1, 365, 730, 1825, 10000],
            labels=['new', 'recent', 'established', 'old']
        )
    
    # ============================================
    # 5. OPERATIONAL FEATURES
    # ============================================
    if 'promotion_activity' in df_copy.columns:
        # Promotion interactions
        df_copy['promotion_weekend'] = df_copy['promotion_activity'] * df_copy['is_weekend']
        if 'holiday_indicator' in df_copy.columns:
            df_copy['promotion_holiday'] = df_copy['promotion_activity'] * df_copy['holiday_indicator']
    
    if 'operational_status' in df_copy.columns:
        # Binary flag for fully operational
        df_copy['is_fully_operational'] = (df_copy['operational_status'] == 'open').astype(int)
    
    # ============================================
    # 6. ENCODE CATEGORICALS
    # ============================================
    categorical_cols = [
        'format', 'assortment_tier', 'hub_maturity', 
        'competitor_distance_binned', 'competitor_tenure_binned',
        'operational_status', 'loyalty_program_status'
    ]
    
    for col in categorical_cols:
        if col in df_copy.columns:
            # Convert to string and then to category
            df_copy[col] = df_copy[col].astype(str).astype('category')
    
    return df_copy
```

#### `src/model.py`

```python
import numpy as np
import pandas as pd
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error

def rmspe(y_true, y_pred):
    """
    Calculate Root Mean Squared Percentage Error
    
    Parameters:
    -----------
    y_true : array-like
        Actual values
    y_pred : array-like
        Predicted values
    
    Returns:
    --------
    float: RMSPE value
    """
    # Handle zeros to avoid division by zero
    mask = y_true > 0
    if mask.sum() == 0:
        return np.inf
    
    y_true_masked = y_true[mask]
    y_pred_masked = y_pred[mask]
    
    # Calculate percentage errors
    pct_error = (y_true_masked - y_pred_masked) / y_true_masked
    
    # Calculate RMSPE
    rmspe = np.sqrt(np.mean(pct_error ** 2))
    return rmspe

def evaluate_model(y_true, y_pred):
    """
    Evaluate model with RMSE and RMSPE
    """
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    rmspe_val = rmspe(y_true, y_pred)
    return rmse, rmspe_val

def train_model(X_train, y_train, model_type='lightgbm', params=None):
    """
    Train a model
    
    Parameters:
    -----------
    X_train : pd.DataFrame
        Training features
    y_train : pd.Series or np.array
        Training target
    model_type : str
        'lightgbm' or 'xgboost'
    params : dict
        Model parameters
    
    Returns:
    --------
    Trained model
    """
    
    # Default parameters
    if params is None:
        params = {}
    
    if model_type == 'lightgbm':
        default_params = {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'num_leaves': 31,
            'max_depth': -1,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbose': -1
        }
        default_params.update(params)
        
        model = lgb.LGBMRegressor(**default_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            callbacks=[lgb.early_stopping(50), lgb.log_evaluation(0)]
        )
        
    elif model_type == 'xgboost':
        default_params = {
            'n_estimators': 1000,
            'learning_rate': 0.05,
            'max_depth': 6,
            'subsample': 0.8,
            'colsample_bytree': 0.8,
            'random_state': 42,
            'n_jobs': -1,
            'verbosity': 0
        }
        default_params.update(params)
        
        model = xgb.XGBRegressor(**default_params)
        model.fit(
            X_train, y_train,
            eval_set=[(X_train, y_train)],
            verbose=False
        )
    
    return model
```

---

### Step 6: Data Download Instructions

**Your Task:** 
1. Download the 3 files from the competition/data source:
   - `orders_train.csv`
   - `orders_test.csv`
   - `hub_metadata.csv`

2. Place them in the `data/` folder of the project structure.

3. Then run the Jupyter Notebook workflow.

---

### Step 7: Execution Order

```bash
# 1. Setup project structure
mkdir -p demand_forecasting_project/{data,src,notebooks,outputs/{plots,model_artifacts}}

# 2. Create virtual environment
cd demand_forecasting_project
python -m venv .venv
source .venv/bin/activate  # or .venv\Scripts\activate on Windows

# 3. Install requirements
pip install -r requirements.txt

# 4. Place data files in data/ folder

# 5. Launch Jupyter
jupyter notebook notebooks/main_workflow.ipynb

# 6. Run cells sequentially from top to bottom
```

---

## 🎯 Summary: What Your Agent Should Do

1. **Create the project structure** as specified above
2. **Generate `requirements.txt`** with all dependencies
3. **Create the helper modules** (`eda.py`, `features.py`, `model.py`)
4. **Generate the main notebook** with the complete workflow
5. **Wait for you** to download and place the data files
6. **Execute the notebook** to:
   - Perform EDA
   - Engineer features
   - Train LightGBM & XGBoost models
   - Evaluate with RMSPE
   - Generate predictions
   - Save outputs

---

**Ready to go!** Download your data, and let me know when you're ready to execute. 🚀
