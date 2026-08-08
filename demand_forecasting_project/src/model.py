"""
model.py - Model training, evaluation, and prediction utilities.

Models supported: LightGBM (primary), XGBoost (secondary).
Metric: RMSPE (Root Mean Squared Percentage Error).
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import joblib
import lightgbm as lgb
import xgboost as xgb
from sklearn.metrics import mean_squared_error
from pathlib import Path


# ═══════════════════════════════════════════════════════════════════════════
# METRIC
# ═══════════════════════════════════════════════════════════════════════════

def rmspe(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    """
    Root Mean Squared Percentage Error.

    Rows where y_true == 0 are excluded (division-by-zero risk).
    Returns np.inf if no non-zero rows remain.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    mask = y_true > 0
    if mask.sum() == 0:
        return np.inf

    pct_err = (y_true[mask] - y_pred[mask]) / y_true[mask]
    return float(np.sqrt(np.mean(pct_err ** 2)))


def evaluate_model(y_true, y_pred) -> dict:
    """
    Return a dict with RMSE, RMSPE, and MAE.
    """
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)

    rmse_val  = float(np.sqrt(mean_squared_error(y_true, y_pred)))
    rmspe_val = rmspe(y_true, y_pred)
    mae_val   = float(np.mean(np.abs(y_true - y_pred)))

    print(f"  RMSE  : {rmse_val:.4f}")
    print(f"  RMSPE : {rmspe_val:.4f}  (target < 0.30)")
    print(f"  MAE   : {mae_val:.4f}")

    return {"rmse": rmse_val, "rmspe": rmspe_val, "mae": mae_val}


# ═══════════════════════════════════════════════════════════════════════════
# TRAINING
# ═══════════════════════════════════════════════════════════════════════════

_LGBM_DEFAULTS = {
    "n_estimators":    1000,
    "learning_rate":   0.05,
    "num_leaves":      63,
    "max_depth":       -1,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "min_child_samples": 20,
    "reg_alpha":       0.1,
    "reg_lambda":      0.1,
    "random_state":    42,
    "n_jobs":          -1,
    "verbose":         -1,
}

_XGB_DEFAULTS = {
    "n_estimators":    1000,
    "learning_rate":   0.05,
    "max_depth":       6,
    "subsample":       0.8,
    "colsample_bytree":0.8,
    "min_child_weight":5,
    "reg_alpha":       0.1,
    "reg_lambda":      1.0,
    "random_state":    42,
    "n_jobs":          -1,
    "verbosity":       0,
}


def train_model(
    X_train,
    y_train,
    X_val=None,
    y_val=None,
    model_type: str = "lightgbm",
    params: dict = None,
    early_stopping_rounds: int = 50,
):
    """
    Train LightGBM or XGBoost with optional early stopping.

    Parameters
    ----------
    X_train / y_train : training features / target
    X_val   / y_val   : validation set for early stopping (optional)
    model_type        : 'lightgbm' or 'xgboost'
    params            : override default hyperparameters
    early_stopping_rounds : patience for early stopping

    Returns
    -------
    fitted model object
    """
    params = params or {}

    if model_type == "lightgbm":
        cfg = {**_LGBM_DEFAULTS, **params}
        model = lgb.LGBMRegressor(**cfg)

        callbacks = [lgb.log_evaluation(100)]
        if X_val is not None:
            callbacks.append(lgb.early_stopping(early_stopping_rounds, verbose=False))
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=callbacks,
            )
        else:
            model.fit(X_train, y_train, callbacks=callbacks)

    elif model_type == "xgboost":
        cfg = {**_XGB_DEFAULTS, **params}
        model = xgb.XGBRegressor(**cfg)

        if X_val is not None:
            model.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                early_stopping_rounds=early_stopping_rounds,
                verbose=100,
            )
        else:
            model.fit(X_train, y_train, verbose=False)

    else:
        raise ValueError(f"Unknown model_type='{model_type}'. Use 'lightgbm' or 'xgboost'.")

    print(f"  [{model_type}] Training complete.")
    return model


# ═══════════════════════════════════════════════════════════════════════════
# PREDICTION & PERSISTENCE
# ═══════════════════════════════════════════════════════════════════════════

def predict(model, X) -> np.ndarray:
    """Predict and clip negatives to 0."""
    preds = model.predict(X)
    return np.clip(preds, 0, None)


def save_model(model, path: Path, feature_cols: list = None):
    """Save model and optionally feature column list."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    joblib.dump(model, path)
    print(f"  [save] model → {path}")
    if feature_cols is not None:
        fc_path = path.with_suffix(".feature_cols.pkl")
        joblib.dump(feature_cols, fc_path)
        print(f"  [save] feature_cols → {fc_path}")


def load_model(path: Path):
    """Load a saved model."""
    return joblib.load(path)


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE IMPORTANCE
# ═══════════════════════════════════════════════════════════════════════════

def get_feature_importance(model, feature_cols: list, model_type: str = "lightgbm") -> pd.DataFrame:
    """Return a sorted DataFrame of feature importances."""
    if model_type == "lightgbm":
        imp = model.feature_importances_
    elif model_type == "xgboost":
        imp = model.feature_importances_
    else:
        return pd.DataFrame()

    fi = (
        pd.DataFrame({"feature": feature_cols, "importance": imp})
        .sort_values("importance", ascending=False)
        .reset_index(drop=True)
    )
    return fi
