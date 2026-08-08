"""
features.py - Feature Engineering for Demand Forecasting.

Actual dataset columns:
  orders_train : HubID, Weekday, Date, OrderVolume, AppSessions,
                 IsOpen, PromoActive, RegionalHoliday, SchoolClosureFlag
  hub_metadata : HubID, HubFormat, AssortmentTier, CompetitorDistance,
                 CompetitorOpenSinceMonth, CompetitorOpenSinceYear,
                 LoyaltyProgram, LoyaltyProgramSinceWeek,
                 LoyaltyProgramSinceYear, LoyaltyProgramInterval
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def create_features(
    df: pd.DataFrame,
    date_col: str = "Date",
    hub_id: str = "HubID",
    target_col: str = "OrderVolume",
    is_train: bool = True,
) -> pd.DataFrame:
    """
    Build all engineered features.

    Parameters
    ----------
    df         : merged DataFrame (orders + hub_metadata)
    date_col   : date column name
    hub_id     : hub identifier column name
    target_col : target column (None for test set)
    is_train   : True → compute lag / rolling features using target column

    Returns
    -------
    pd.DataFrame with all engineered columns appended.
    """
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col])
    df = df.sort_values([hub_id, date_col]).reset_index(drop=True)

    print(f"[features] Building features – is_train={is_train}, shape={df.shape}")

    # ── 1. Temporal / Calendar Features ─────────────────────────────────
    df = _temporal_features(df, date_col)

    # ── 2. Lag Features (only on training data; requires target) ─────────
    if is_train and target_col and target_col in df.columns:
        df = _lag_features(df, hub_id, target_col)
        df = _rolling_features(df, hub_id, target_col)

    # ── 3. Hub Metadata Features ─────────────────────────────────────────
    df = _hub_metadata_features(df, date_col)

    # ── 4. Operational / Contextual Features ────────────────────────────
    df = _operational_features(df)

    # ── 5. Interaction Features ──────────────────────────────────────────
    df = _interaction_features(df)

    # ── 6. Encode Categoricals ───────────────────────────────────────────
    df = _encode_categoricals(df)

    print(f"[features] Done – final shape={df.shape}, "
          f"new cols added={df.shape[1] - 9}")  # approx baseline cols
    return df


# ═══════════════════════════════════════════════════════════════════════════
# FEATURE BUILDERS
# ═══════════════════════════════════════════════════════════════════════════

def _temporal_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Extract calendar-based features from the date column."""
    d = df[date_col]
    df["year"]        = d.dt.year
    df["month"]       = d.dt.month
    df["day"]         = d.dt.day
    df["day_of_year"] = d.dt.dayofyear
    df["week_of_year"]= d.dt.isocalendar().week.astype(int)
    df["quarter"]     = d.dt.quarter

    # Weekday (dataset already has Weekday 1-7; derive is_weekend from it)
    # 6=Sat, 7=Sun in this dataset (Weekday starts from 1=Mon based on data)
    if "Weekday" in df.columns:
        df["is_weekend"] = df["Weekday"].isin([6, 7]).astype(int)
    else:
        df["is_weekend"] = d.dt.dayofweek.isin([5, 6]).astype(int)

    # Cyclical encoding of month & weekday (better for tree models too)
    df["month_sin"]   = np.sin(2 * np.pi * df["month"] / 12)
    df["month_cos"]   = np.cos(2 * np.pi * df["month"] / 12)
    if "Weekday" in df.columns:
        df["weekday_sin"] = np.sin(2 * np.pi * df["Weekday"] / 7)
        df["weekday_cos"] = np.cos(2 * np.pi * df["Weekday"] / 7)

    return df


def _lag_features(df: pd.DataFrame, hub_id: str, target_col: str) -> pd.DataFrame:
    """
    Per-hub lag features.
    IMPORTANT: shift() prevents data leakage – lag_1 = yesterday's value.
    """
    grp = df.groupby(hub_id)[target_col]

    for lag in [1, 7, 14, 21, 28]:
        df[f"lag_{lag}"] = grp.shift(lag)

    return df


def _rolling_features(df: pd.DataFrame, hub_id: str, target_col: str) -> pd.DataFrame:
    """
    Per-hub rolling window statistics.
    We shift by 1 BEFORE rolling to prevent leakage.
    """
    grp = df.groupby(hub_id)[target_col]

    for window in [7, 14, 28]:
        shifted = grp.shift(1)
        df[f"rolling_mean_{window}"] = shifted.transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f"rolling_std_{window}"] = shifted.transform(
            lambda x: x.rolling(window, min_periods=1).std()
        )

    df["rolling_max_7"] = grp.shift(1).transform(
        lambda x: x.rolling(7, min_periods=1).max()
    )
    df["rolling_min_7"] = grp.shift(1).transform(
        lambda x: x.rolling(7, min_periods=1).min()
    )

    # Momentum: lag_1 vs rolling_mean_7 ratio
    if "lag_1" in df.columns and "rolling_mean_7" in df.columns:
        denom = df["rolling_mean_7"].replace(0, np.nan)
        df["momentum_7"] = df["lag_1"] / denom

    return df


def _hub_metadata_features(df: pd.DataFrame, date_col: str) -> pd.DataFrame:
    """Derive features from hub metadata columns."""

    # ── Competitor features ──────────────────────────────────────────────
    if "CompetitorDistance" in df.columns:
        df["competitor_distance_log"] = np.log1p(
            df["CompetitorDistance"].fillna(df["CompetitorDistance"].median())
        )
        df["competitor_distance_binned"] = pd.cut(
            df["CompetitorDistance"],
            bins=[-1, 500, 1000, 2000, 5000, 1e9],
            labels=[0, 1, 2, 3, 4],  # numeric labels for tree models
        ).astype(float)

    # Competitor open date → tenure in days
    if "CompetitorOpenSinceYear" in df.columns and "CompetitorOpenSinceMonth" in df.columns:
        # Build competitor open date (use day=1)
        valid = (
            df["CompetitorOpenSinceYear"].notna()
            & df["CompetitorOpenSinceMonth"].notna()
        )
        comp_open = pd.to_datetime(
            {
                "year":  df.loc[valid, "CompetitorOpenSinceYear"].astype(int),
                "month": df.loc[valid, "CompetitorOpenSinceMonth"].astype(int),
                "day":   1,
            }
        )
        df.loc[valid, "competitor_tenure_days"] = (
            df.loc[valid, date_col] - comp_open
        ).dt.days
        df["competitor_tenure_days"] = df["competitor_tenure_days"].fillna(-1)

    # ── Loyalty program ──────────────────────────────────────────────────
    if "LoyaltyProgram" in df.columns:
        df["has_loyalty"] = df["LoyaltyProgram"].fillna(0).astype(int)

    if "LoyaltyProgramSinceYear" in df.columns:
        ref_year = df[date_col].dt.year
        ly_year  = df["LoyaltyProgramSinceYear"].fillna(ref_year)
        df["loyalty_age_years"] = (ref_year - ly_year).clip(lower=0)

    # Loyalty program has interval (quarterly, etc.) – binary flag
    if "LoyaltyProgramInterval" in df.columns:
        df["has_loyalty_interval"] = df["LoyaltyProgramInterval"].notna().astype(int)

    return df


def _operational_features(df: pd.DataFrame) -> pd.DataFrame:
    """Process operational/event flags."""

    # IsOpen – already binary; fill NaN with 1 (assume open if unknown)
    if "IsOpen" in df.columns:
        df["IsOpen"] = df["IsOpen"].fillna(1).astype(int)

    # PromoActive
    if "PromoActive" in df.columns:
        df["PromoActive"] = df["PromoActive"].fillna(0).astype(int)

    # RegionalHoliday
    if "RegionalHoliday" in df.columns:
        df["RegionalHoliday"] = df["RegionalHoliday"].fillna(0).astype(int)

    # SchoolClosureFlag
    if "SchoolClosureFlag" in df.columns:
        df["SchoolClosureFlag"] = df["SchoolClosureFlag"].fillna(0).astype(int)

    # AppSessions – fill missing with 0 (hub closed)
    if "AppSessions" in df.columns:
        df["AppSessions"] = df["AppSessions"].fillna(0)
        df["app_sessions_log"] = np.log1p(df["AppSessions"])

    return df


def _interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Cross-product interaction terms."""

    # Promo × Weekend
    if "PromoActive" in df.columns and "is_weekend" in df.columns:
        df["promo_weekend"] = df["PromoActive"] * df["is_weekend"]

    # Promo × Holiday
    if "PromoActive" in df.columns and "RegionalHoliday" in df.columns:
        df["promo_holiday"] = df["PromoActive"] * df["RegionalHoliday"]

    # Promo × SchoolClosure
    if "PromoActive" in df.columns and "SchoolClosureFlag" in df.columns:
        df["promo_school"] = df["PromoActive"] * df["SchoolClosureFlag"]

    # Format × AssortmentTier
    if "HubFormat" in df.columns and "AssortmentTier" in df.columns:
        df["format_tier"] = (
            df["HubFormat"].astype(str) + "_" + df["AssortmentTier"].astype(str)
        )

    # AppSessions × PromoActive
    if "AppSessions" in df.columns and "PromoActive" in df.columns:
        df["sessions_promo"] = df["AppSessions"] * df["PromoActive"]

    return df


def _encode_categoricals(df: pd.DataFrame) -> pd.DataFrame:
    """Label-encode object / category columns for LightGBM / XGBoost."""
    cat_cols = df.select_dtypes(include=["object", "category"]).columns.tolist()

    # Exclude identifier / date columns
    exclude = {"Date", "HubID"}
    cat_cols = [c for c in cat_cols if c not in exclude]

    for col in cat_cols:
        df[col] = df[col].astype(str).astype("category").cat.codes

    return df


# ═══════════════════════════════════════════════════════════════════════════
# UTILITY: list feature columns to use for modelling
# ═══════════════════════════════════════════════════════════════════════════

def get_feature_cols(df: pd.DataFrame) -> list:
    """
    Return columns suitable for model input.
    Excludes identifiers, raw date, and the target.
    """
    exclude = {
        "HubID", "Id", "Date", "OrderVolume",
        # raw metadata fields kept for reference
        "CompetitorOpenSinceMonth", "CompetitorOpenSinceYear",
        "LoyaltyProgramSinceWeek", "LoyaltyProgramSinceYear",
        "LoyaltyProgramInterval",
    }
    return [c for c in df.columns if c not in exclude]
