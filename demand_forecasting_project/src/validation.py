"""
validation.py - Time-based validation utilities for demand forecasting.

Implements:
  - time_based_split()    : chronological train/val split (NO random shuffle)
  - rmspe_score()         : RMSPE metric with zero-handling
  - validate_features()   : sanity checks on engineered features
  - rmspe_by_hub()        : per-hub RMSPE breakdown for error analysis
"""

import logging
import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import Tuple, Dict, Optional

logger = logging.getLogger(__name__)


# ═══════════════════════════════════════════════════════════════════════════
# 1. TIME-BASED SPLIT
# ═══════════════════════════════════════════════════════════════════════════

def time_based_split(
    df: pd.DataFrame,
    date_col: str = "Date",
    val_fraction: float = 0.20,
    val_days: Optional[int] = None,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    """
    Split a DataFrame chronologically into train and validation sets.

    NEVER uses random shuffling. Always uses time ordering.

    Parameters
    ----------
    df           : full featured DataFrame (must contain date_col)
    date_col     : name of the date column
    val_fraction : fraction of unique dates to use as validation (default 0.20)
    val_days     : if set, use this many trailing days for validation
                   (overrides val_fraction)

    Returns
    -------
    train_split, val_split : pd.DataFrame, pd.DataFrame
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])

    unique_dates = np.sort(df[date_col].unique())
    n_dates = len(unique_dates)

    if n_dates == 0:
        raise ValueError("No dates found in DataFrame.")

    if val_days is not None:
        if val_days >= n_dates:
            raise ValueError(f"val_days={val_days} >= total unique dates={n_dates}")
        cutoff_idx = n_dates - val_days
    else:
        cutoff_idx = int(n_dates * (1 - val_fraction))

    train_dates = unique_dates[:cutoff_idx]
    val_dates   = unique_dates[cutoff_idx:]

    train_split = df[df[date_col].isin(train_dates)].copy()
    val_split   = df[df[date_col].isin(val_dates)].copy()

    logger.info(
        f"[split] Train: {pd.Timestamp(train_dates[0]).date()} → "
        f"{pd.Timestamp(train_dates[-1]).date()} ({len(train_split):,} rows)"
    )
    logger.info(
        f"[split] Val  : {pd.Timestamp(val_dates[0]).date()} → "
        f"{pd.Timestamp(val_dates[-1]).date()} ({len(val_split):,} rows)"
    )

    print(f"  Train: {pd.Timestamp(train_dates[0]).date()} → "
          f"{pd.Timestamp(train_dates[-1]).date()}  ({len(train_split):,} rows)")
    print(f"  Val  : {pd.Timestamp(val_dates[0]).date()} → "
          f"{pd.Timestamp(val_dates[-1]).date()}  ({len(val_split):,} rows)")

    return train_split, val_split


# ═══════════════════════════════════════════════════════════════════════════
# 2. RMSPE METRIC
# ═══════════════════════════════════════════════════════════════════════════

def rmspe_score(
    actual: np.ndarray,
    predicted: np.ndarray,
    epsilon: float = 0.0,
) -> float:
    """
    Root Mean Squared Percentage Error.

    Formula: sqrt( mean( ((actual - predicted) / actual)^2 ) )

    Zero-handling strategy
    ----------------------
    By default (epsilon=0.0), rows where actual == 0 are EXCLUDED
    from the calculation entirely (division by zero is undefined).

    If epsilon > 0 (e.g. 1e-8), it is added to the denominator instead
    of excluding zeros — use only if the competition rules require it.

    Parameters
    ----------
    actual      : ground-truth values
    predicted   : model predictions
    epsilon     : small constant added to denominator (default 0 = exclude zeros)

    Returns
    -------
    float : RMSPE value (lower is better; 0.0 = perfect)
    """
    actual    = np.asarray(actual,    dtype=float)
    predicted = np.asarray(predicted, dtype=float)

    if epsilon > 0:
        denom   = actual + epsilon
        pct_err = (actual - predicted) / denom
        return float(np.sqrt(np.mean(pct_err ** 2)))
    else:
        mask = actual > 0
        if mask.sum() == 0:
            logger.warning("[rmspe] No non-zero actual values — returning inf")
            return np.inf
        pct_err = (actual[mask] - predicted[mask]) / actual[mask]
        return float(np.sqrt(np.mean(pct_err ** 2)))


def rmspe_by_hub(
    df: pd.DataFrame,
    actual_col: str = "OrderVolume",
    pred_col: str   = "pred",
    hub_id: str     = "HubID",
    plot_path: Path = Path("outputs/plots"),
) -> pd.DataFrame:
    """
    Compute per-hub RMSPE and return a sorted DataFrame.

    Also saves a plot of the worst 20 hubs.

    Parameters
    ----------
    df         : DataFrame containing actual, predicted, and hub columns
    actual_col : column name of ground-truth values
    pred_col   : column name of predictions
    hub_id     : hub identifier column
    plot_path  : directory to save the plot

    Returns
    -------
    pd.DataFrame with columns [hub_id, 'rmspe', 'n_rows', 'avg_actual']
    sorted by rmspe descending.
    """
    plot_path = Path(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)

    results = []
    for hub, grp in df.groupby(hub_id):
        score = rmspe_score(grp[actual_col].values, grp[pred_col].values)
        results.append({
            hub_id:       hub,
            "rmspe":      score,
            "n_rows":     len(grp),
            "avg_actual": grp[actual_col].mean(),
        })

    hub_rmspe = (
        pd.DataFrame(results)
        .sort_values("rmspe", ascending=False)
        .reset_index(drop=True)
    )

    print(f"\n  Top 10 worst hubs by RMSPE:")
    print(hub_rmspe.head(10).to_string(index=False))

    # Plot worst 20
    worst20 = hub_rmspe.head(20)
    fig, ax = plt.subplots(figsize=(12, 6))
    bars = ax.barh(
        worst20[hub_id].astype(str)[::-1],
        worst20["rmspe"][::-1],
        color="tomato",
    )
    ax.axvline(0.30, color="navy",   lw=1.5, ls="--", label="Good threshold (0.30)")
    ax.axvline(0.25, color="green",  lw=1.5, ls="--", label="Excellent (0.25)")
    ax.set_title("Worst 20 Hubs by RMSPE (Validation Set)")
    ax.set_xlabel("RMSPE")
    ax.set_ylabel("HubID")
    ax.legend(fontsize=9)
    plt.tight_layout()
    _save(fig, plot_path / "18_rmspe_by_hub.png")

    return hub_rmspe


# ═══════════════════════════════════════════════════════════════════════════
# 3. FEATURE VALIDATION
# ═══════════════════════════════════════════════════════════════════════════

def validate_features(
    train_featured: pd.DataFrame,
    val_featured: pd.DataFrame,
    lag_cols: Optional[list] = None,
    drift_threshold: float = 2.0,
) -> Dict:
    """
    Sanity-check engineered features between train and validation sets.

    Checks
    ------
    1. No NaN in lag features in the validation set.
    2. Feature distribution drift: flag any column where
       abs(train_mean / val_mean) > drift_threshold or vice versa.
    3. Column presence match (train vs val must have same columns).

    Parameters
    ----------
    train_featured   : engineered training DataFrame
    val_featured     : engineered validation DataFrame
    lag_cols         : list of lag column names (auto-detected if None)
    drift_threshold  : ratio threshold to flag drift (default 2.0 = 2×)

    Returns
    -------
    dict with keys: 'lag_nan_issues', 'drifted_features', 'missing_cols'
    """
    print("\n" + "="*60)
    print("FEATURE VALIDATION")
    print("="*60)

    report = {}

    # ── 1. Column presence ───────────────────────────────────────────────
    train_cols = set(train_featured.columns)
    val_cols   = set(val_featured.columns)
    missing_in_val   = train_cols - val_cols
    missing_in_train = val_cols   - train_cols

    if missing_in_val:
        print(f"  ⚠️  Columns in train but NOT in val : {missing_in_val}")
    if missing_in_train:
        print(f"  ⚠️  Columns in val but NOT in train : {missing_in_train}")
    if not missing_in_val and not missing_in_train:
        print(f"  ✅ Column sets match ({len(train_cols)} columns)")

    report["missing_cols"] = {
        "missing_in_val":   list(missing_in_val),
        "missing_in_train": list(missing_in_train),
    }

    # ── 2. Lag NaN check in validation ──────────────────────────────────
    if lag_cols is None:
        lag_cols = [c for c in val_featured.columns if c.startswith("lag_")]

    lag_nan_issues = {}
    for col in lag_cols:
        if col not in val_featured.columns:
            continue
        n_nan = val_featured[col].isna().sum()
        if n_nan > 0:
            pct = 100 * n_nan / len(val_featured)
            lag_nan_issues[col] = {"n_nan": n_nan, "pct": pct}
            print(f"  ⚠️  {col}: {n_nan:,} NaN in val ({pct:.1f}%)")

    if not lag_nan_issues:
        print(f"  ✅ No NaN in lag features in validation set")

    report["lag_nan_issues"] = lag_nan_issues

    # ── 3. Feature drift ─────────────────────────────────────────────────
    shared_num_cols = list(
        set(train_featured.select_dtypes(include=[np.number]).columns)
        & set(val_featured.select_dtypes(include=[np.number]).columns)
    )

    drifted = []
    for col in shared_num_cols:
        t_mean = train_featured[col].mean()
        v_mean = val_featured[col].mean()

        if t_mean == 0 and v_mean == 0:
            continue
        if t_mean == 0 or v_mean == 0:
            drifted.append({"feature": col, "train_mean": t_mean, "val_mean": v_mean,
                            "ratio": np.inf})
            continue

        ratio = max(abs(t_mean / v_mean), abs(v_mean / t_mean))
        if ratio > drift_threshold:
            drifted.append({"feature": col, "train_mean": round(t_mean, 4),
                            "val_mean": round(v_mean, 4), "ratio": round(ratio, 2)})

    if drifted:
        drift_df = pd.DataFrame(drifted).sort_values("ratio", ascending=False)
        print(f"\n  ⚠️  {len(drifted)} features with mean ratio > {drift_threshold}x:")
        print(drift_df.to_string(index=False))
    else:
        print(f"  ✅ No significant feature drift detected (threshold={drift_threshold}x)")

    report["drifted_features"] = drifted

    print("\n  ✅ Feature validation complete")
    return report


# ═══════════════════════════════════════════════════════════════════════════
# HELPER
# ═══════════════════════════════════════════════════════════════════════════

def _save(fig, path: Path):
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {path.name}")
