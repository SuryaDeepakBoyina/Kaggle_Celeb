"""
eda.py - Comprehensive Exploratory Data Analysis for Demand Forecasting.

Column mapping (actual dataset):
  orders_train : HubID, Weekday, Date, OrderVolume, AppSessions,
                 IsOpen, PromoActive, RegionalHoliday, SchoolClosureFlag
  orders_test  : Id, HubID, Weekday, Date, IsOpen, PromoActive,
                 RegionalHoliday, SchoolClosureFlag
  hub_metadata : HubID, HubFormat, AssortmentTier, CompetitorDistance,
                 CompetitorOpenSinceMonth, CompetitorOpenSinceYear,
                 LoyaltyProgram, LoyaltyProgramSinceWeek,
                 LoyaltyProgramSinceYear, LoyaltyProgramInterval
"""

import warnings
warnings.filterwarnings("ignore")

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")           # non-interactive backend - safe for scripts
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import seaborn as sns
from pathlib import Path
from scipy import stats

# ── colour palette ──────────────────────────────────────────────────────────
PALETTE = sns.color_palette("husl", 8)
sns.set_theme(style="whitegrid", palette=PALETTE)


# ═══════════════════════════════════════════════════════════════════════════
# MAIN ENTRY POINT
# ═══════════════════════════════════════════════════════════════════════════

def perform_eda(
    train: pd.DataFrame,
    test: pd.DataFrame,
    target: str = "OrderVolume",
    hub_id: str = "HubID",
    date_col: str = "Date",
    plot_path: Path = Path("outputs/plots"),
) -> dict:
    """
    Run the full EDA pipeline and save all plots.

    Parameters
    ----------
    train     : merged train DataFrame (orders_train + hub_metadata)
    test      : merged test  DataFrame (orders_test  + hub_metadata)
    target    : name of the target column
    hub_id    : name of the hub identifier column
    date_col  : name of the date column
    plot_path : directory where PNG files are saved

    Returns
    -------
    results : dict  -  summary statistics & key findings
    """
    plot_path = Path(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)

    results = {}

    # ── ensure datetime ──────────────────────────────────────────────────
    train = train.copy()
    test  = test.copy()
    train[date_col] = pd.to_datetime(train[date_col])
    test[date_col]  = pd.to_datetime(test[date_col])

    # ── 1. Basic shape & schema ──────────────────────────────────────────
    print("\n" + "="*60)
    print("1. BASIC INFO")
    print("="*60)
    print(f"  Train : {train.shape[0]:,} rows  x  {train.shape[1]} cols")
    print(f"  Test  : {test.shape[0]:,} rows  x  {test.shape[1]} cols")
    print(f"\n  Train columns : {train.columns.tolist()}")
    print(f"  Test  columns : {test.columns.tolist()}")
    print(f"\n  Date range (train): {train[date_col].min().date()} → {train[date_col].max().date()}")
    print(f"  Date range (test) : {test[date_col].min().date()} → {test[date_col].max().date()}")
    print(f"  Unique hubs (train): {train[hub_id].nunique()}")
    print(f"  Unique hubs (test) : {test[hub_id].nunique()}")
    results["train_shape"] = train.shape
    results["test_shape"]  = test.shape
    results["date_range_train"] = (train[date_col].min(), train[date_col].max())
    results["date_range_test"]  = (test[date_col].min(),  test[date_col].max())
    results["n_hubs_train"] = train[hub_id].nunique()

    # ── 2. Missing values ────────────────────────────────────────────────
    print("\n" + "="*60)
    print("2. MISSING VALUES")
    print("="*60)
    train_miss = train.isnull().sum()
    test_miss  = test.isnull().sum()
    print("  Train:\n", train_miss[train_miss > 0].to_string() if train_miss.any() else "  None")
    print("  Test:\n",  test_miss[test_miss > 0].to_string()  if test_miss.any()  else "  None")
    results["train_missing"] = train_miss
    results["test_missing"]  = test_miss

    # ── 3. Target distribution ───────────────────────────────────────────
    print("\n" + "="*60)
    print("3. TARGET DISTRIBUTION")
    print("="*60)
    tgt = train[target]
    print(tgt.describe().to_string())

    zero_count = (tgt == 0).sum()
    zero_pct   = zero_count / len(tgt) * 100
    skew       = tgt.skew()
    print(f"\n  Zeros : {zero_count:,}  ({zero_pct:.1f}%)")
    print(f"  Skewness : {skew:.2f}")
    results.update({"target_desc": tgt.describe(), "zero_count": zero_count,
                    "zero_pct": zero_pct, "target_skew": skew})

    # Plot
    fig, axes = plt.subplots(1, 3, figsize=(16, 4))
    sns.histplot(tgt, bins=60, ax=axes[0], color=PALETTE[0], kde=True)
    axes[0].set_title("OrderVolume – Raw Distribution")

    sns.histplot(np.log1p(tgt), bins=60, ax=axes[1], color=PALETTE[1], kde=True)
    axes[1].set_title("log1p(OrderVolume) – Log Distribution")

    sns.boxplot(y=tgt, ax=axes[2], color=PALETTE[2])
    axes[2].set_title("OrderVolume – Boxplot")

    plt.tight_layout()
    _save(fig, plot_path / "01_target_distribution.png")

    # ── 4. Time-series – full fleet overview ─────────────────────────────
    print("\n" + "="*60)
    print("4. TIME-SERIES PATTERNS")
    print("="*60)

    daily_total = train.groupby(date_col)[target].sum().reset_index()

    fig, ax = plt.subplots(figsize=(15, 4))
    ax.plot(daily_total[date_col], daily_total[target], lw=0.8, color=PALETTE[0])
    ax.set_title("Total Daily OrderVolume – All Hubs")
    ax.set_xlabel("Date"); ax.set_ylabel("Sum of OrderVolume")
    plt.xticks(rotation=30)
    plt.tight_layout()
    _save(fig, plot_path / "02_daily_total_timeseries.png")

    # Sample hub time-series
    top_hubs = train.groupby(hub_id)[target].mean().nlargest(6).index.tolist()
    fig, axes = plt.subplots(3, 2, figsize=(16, 10), sharex=True)
    for ax, hub in zip(axes.flatten(), top_hubs):
        sub = train[train[hub_id] == hub].sort_values(date_col)
        ax.plot(sub[date_col], sub[target], lw=0.7)
        ax.set_title(f"Hub {hub}")
        ax.set_ylabel("OrderVolume")
    plt.suptitle("Time-Series – Top 6 Hubs by Avg Volume", fontsize=13)
    plt.tight_layout()
    _save(fig, plot_path / "03_sample_hub_timeseries.png")

    # ── 5. Weekly seasonality ────────────────────────────────────────────
    print("\n" + "="*60)
    print("5. WEEKLY & MONTHLY SEASONALITY")
    print("="*60)

    # Weekday (Weekday col in dataset: 1=Mon ... 7=Sun based on context)
    if "Weekday" in train.columns:
        dow_map = {1:"Mon", 2:"Tue", 3:"Wed", 4:"Thu", 5:"Fri", 6:"Sat", 7:"Sun"}
        dow_avg = train.groupby("Weekday")[target].agg(["mean","median"]).rename(columns={"mean":"Mean","median":"Median"})
        dow_avg.index = [dow_map.get(i, i) for i in dow_avg.index]

        fig, axes = plt.subplots(1, 2, figsize=(14, 4))
        dow_avg["Mean"].plot(kind="bar", ax=axes[0], color=PALETTE[0])
        axes[0].set_title("Mean OrderVolume by Weekday")
        axes[0].set_ylabel("Avg OrderVolume"); axes[0].tick_params(axis='x', rotation=30)
        dow_avg["Median"].plot(kind="bar", ax=axes[1], color=PALETTE[3])
        axes[1].set_title("Median OrderVolume by Weekday")
        axes[1].tick_params(axis='x', rotation=30)
        plt.tight_layout()
        _save(fig, plot_path / "04_weekday_pattern.png")
        results["dow_pattern"] = dow_avg.to_dict()
        print(dow_avg.to_string())

    # Monthly pattern
    train["_month"] = train[date_col].dt.month
    month_avg = train.groupby("_month")[target].mean()
    fig, ax = plt.subplots(figsize=(10, 4))
    month_avg.plot(kind="bar", ax=ax, color=PALETTE[2])
    ax.set_title("Mean OrderVolume by Month")
    ax.set_xlabel("Month"); ax.set_ylabel("Avg OrderVolume")
    ax.xaxis.set_major_formatter(mticker.FixedFormatter(
        ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]))
    plt.tight_layout()
    _save(fig, plot_path / "05_monthly_pattern.png")
    train.drop(columns=["_month"], inplace=True)

    # ── 6. Promo & Holiday effects ───────────────────────────────────────
    print("\n" + "="*60)
    print("6. PROMO / HOLIDAY / SCHOOL-CLOSURE EFFECTS")
    print("="*60)

    effect_cols = [c for c in ["PromoActive", "RegionalHoliday", "SchoolClosureFlag"] if c in train.columns]
    if effect_cols:
        fig, axes = plt.subplots(1, len(effect_cols), figsize=(5*len(effect_cols), 5))
        if len(effect_cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, effect_cols):
            grp = train.groupby(col)[target].mean()
            grp.plot(kind="bar", ax=ax, color=PALETTE[4])
            ax.set_title(f"Mean OrderVolume\nby {col}")
            ax.set_xlabel(col); ax.set_ylabel("Avg OrderVolume")
            ax.tick_params(axis='x', rotation=0)
            print(f"  {col}:\n{grp.to_string()}\n")
        plt.tight_layout()
        _save(fig, plot_path / "06_promo_holiday_effects.png")

    # IsOpen effect
    if "IsOpen" in train.columns:
        open_stats = train.groupby("IsOpen")[target].describe()
        print("  IsOpen stats:\n", open_stats.to_string())
        results["isopen_stats"] = open_stats

    # ── 7. AppSessions vs OrderVolume correlation ────────────────────────
    if "AppSessions" in train.columns:
        print("\n" + "="*60)
        print("7. APP SESSIONS vs ORDER VOLUME")
        print("="*60)
        corr_val = train[[target, "AppSessions"]].corr().iloc[0, 1]
        print(f"  Pearson correlation: {corr_val:.4f}")
        results["appsessions_corr"] = corr_val

        # Scatter sample for speed
        sample = train.sample(min(5000, len(train)), random_state=42)
        fig, ax = plt.subplots(figsize=(8, 5))
        ax.scatter(sample["AppSessions"], sample[target], alpha=0.15, s=5, color=PALETTE[5])
        ax.set_title(f"AppSessions vs OrderVolume  (r={corr_val:.3f})")
        ax.set_xlabel("AppSessions"); ax.set_ylabel("OrderVolume")
        plt.tight_layout()
        _save(fig, plot_path / "07_appsessions_vs_ordervolume.png")

    # ── 8. Hub metadata analysis ─────────────────────────────────────────
    print("\n" + "="*60)
    print("8. HUB METADATA ANALYSIS")
    print("="*60)

    hub_avg = train.groupby(hub_id)[target].mean()
    print(f"  Avg hub volume – mean: {hub_avg.mean():.1f}, "
          f"std: {hub_avg.std():.1f}, "
          f"min: {hub_avg.min():.1f}, max: {hub_avg.max():.1f}")
    results["hub_avg_stats"] = hub_avg.describe().to_dict()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    hub_avg.hist(bins=50, ax=axes[0], color=PALETTE[0])
    axes[0].set_title("Avg OrderVolume Distribution across Hubs")
    axes[0].set_xlabel("Avg Volume")

    hub_avg.sort_values().reset_index(drop=True).plot(ax=axes[1], color=PALETTE[1])
    axes[1].set_title("Sorted Avg Volume per Hub (Hub Ranking)")
    axes[1].set_xlabel("Hub Rank"); axes[1].set_ylabel("Avg Volume")
    plt.tight_layout()
    _save(fig, plot_path / "08_hub_volume_distribution.png")

    # HubFormat & AssortmentTier
    for col in ["HubFormat", "AssortmentTier"]:
        if col in train.columns:
            grp = train.groupby(col)[target].mean().sort_index()
            print(f"\n  Mean OrderVolume by {col}:\n{grp.to_string()}")
            fig, ax = plt.subplots(figsize=(8, 4))
            grp.plot(kind="bar", ax=ax, color=PALETTE[6])
            ax.set_title(f"Mean OrderVolume by {col}")
            ax.tick_params(axis='x', rotation=0)
            plt.tight_layout()
            _save(fig, plot_path / f"09_{col.lower()}_analysis.png")

    # LoyaltyProgram
    if "LoyaltyProgram" in train.columns:
        lp_stats = train.groupby("LoyaltyProgram")[target].agg(["mean","count"])
        print(f"\n  LoyaltyProgram stats:\n{lp_stats.to_string()}")

    # CompetitorDistance
    if "CompetitorDistance" in train.columns:
        fig, ax = plt.subplots(figsize=(10, 4))
        ax.scatter(train["CompetitorDistance"], train[target], alpha=0.05, s=3, color=PALETTE[3])
        ax.set_title("CompetitorDistance vs OrderVolume")
        ax.set_xlabel("CompetitorDistance (m)"); ax.set_ylabel("OrderVolume")
        plt.tight_layout()
        _save(fig, plot_path / "10_competitor_distance_vs_volume.png")

    # ── 9. Correlation heatmap ───────────────────────────────────────────
    print("\n" + "="*60)
    print("9. CORRELATION HEATMAP (numeric features)")
    print("="*60)

    num_cols = train.select_dtypes(include=[np.number]).columns.tolist()
    if len(num_cols) > 1:
        corr_matrix = train[num_cols].corr()
        target_corr = corr_matrix[target].drop(target).sort_values(ascending=False)
        print(f"\n  Top correlations with {target}:\n{target_corr.head(10).to_string()}")
        results["top_correlations"] = target_corr.to_dict()

        fig, ax = plt.subplots(figsize=(max(10, len(num_cols)), max(8, len(num_cols)-2)))
        sns.heatmap(corr_matrix, annot=True, fmt=".2f", cmap="coolwarm",
                    center=0, linewidths=0.5, ax=ax, annot_kws={"size": 7})
        ax.set_title("Feature Correlation Heatmap")
        plt.tight_layout()
        _save(fig, plot_path / "11_correlation_heatmap.png")

    # ── 10. Outlier analysis ─────────────────────────────────────────────
    print("\n" + "="*60)
    print("10. OUTLIER ANALYSIS")
    print("="*60)
    Q1, Q3 = tgt.quantile(0.25), tgt.quantile(0.75)
    IQR  = Q3 - Q1
    upper_bound = Q3 + 3 * IQR
    outliers = tgt[tgt > upper_bound]
    print(f"  IQR upper fence (3x): {upper_bound:.1f}")
    print(f"  Outlier rows: {len(outliers):,}  ({100*len(outliers)/len(tgt):.2f}%)")
    results["outlier_count"] = len(outliers)
    results["outlier_pct"]   = 100 * len(outliers) / len(tgt)

    # ── 11. Key findings summary ─────────────────────────────────────────
    print("\n" + "="*60)
    print("11. KEY FINDINGS SUMMARY")
    print("="*60)
    findings = _generate_findings(results, tgt)
    for i, f in enumerate(findings, 1):
        print(f"  [{i}] {f}")
    results["key_findings"] = findings

    print("\n✅ EDA Complete! All plots saved to:", plot_path)
    return results


# ═══════════════════════════════════════════════════════════════════════════
# HELPERS
# ═══════════════════════════════════════════════════════════════════════════

def _save(fig, path: Path):
    fig.savefig(path, dpi=120, bbox_inches="tight")
    plt.close(fig)
    print(f"  [plot] saved → {path.name}")


def _generate_findings(results: dict, tgt: pd.Series) -> list:
    findings = []

    zero_pct = results.get("zero_pct", 0)
    if zero_pct > 10:
        findings.append(
            f"HIGH zero rate ({zero_pct:.1f}%) in OrderVolume – "
            "likely driven by IsOpen=0 days. RMSPE must exclude zeros (division by zero risk)."
        )
    elif zero_pct > 0:
        findings.append(
            f"Low zero rate ({zero_pct:.1f}%) – mainly closed-hub days. "
            "Filter IsOpen=0 rows before training or treat separately."
        )

    skew = results.get("target_skew", 0)
    if abs(skew) > 1.0:
        findings.append(
            f"Target is right-skewed (skew={skew:.2f}). "
            "Consider log1p transformation for training."
        )

    if results.get("appsessions_corr", 0) > 0.3:
        findings.append(
            "AppSessions is positively correlated with OrderVolume – "
            "strong feature candidate."
        )

    n_hubs = results.get("n_hubs_train", 0)
    findings.append(
        f"Dataset covers {n_hubs} hubs – multi-series forecasting. "
        "Hub-level lag & rolling features are critical."
    )

    if results.get("outlier_pct", 0) > 1:
        findings.append(
            f"Outliers: {results['outlier_count']:,} rows ({results['outlier_pct']:.1f}%) "
            "above 3×IQR upper fence. Investigate before training."
        )

    findings.append(
        "No rider_availability in actual dataset – context docs referenced planned vs actual "
        "rider shift. NOT applicable here. Focus on AppSessions, Promo, Holiday, Weekday."
    )

    findings.append(
        "Validation must be time-based (expanding window). "
        "Last ~20% of dates → validation set. NO random splits."
    )

    return findings


# ═══════════════════════════════════════════════════════════════════════════
# SUPPLEMENTARY ANALYSIS FUNCTIONS
# ═══════════════════════════════════════════════════════════════════════════

def rmspe_risk_analysis(
    train: pd.DataFrame,
    target: str = "OrderVolume",
    hub_id: str = "HubID",
    plot_path: Path = Path("outputs/plots"),
) -> dict:
    """
    Identify hubs at high risk of inflating RMSPE due to low / near-zero volumes.

    RMSPE = mean( ((y_true - y_pred) / y_true)^2 )^0.5, so small y_true values
    cause the per-row error to explode.  Hubs whose average volume is very low
    (< 10 units) or that frequently record near-zero days (volume < 5) are
    flagged as high-risk.

    Parameters
    ----------
    train     : merged train DataFrame containing hub_id and target columns
    target    : name of the target column  (default: 'OrderVolume')
    hub_id    : name of the hub identifier column  (default: 'HubID')
    plot_path : directory where PNG files are saved

    Returns
    -------
    dict with keys
        high_risk_hubs       : list of HubIDs whose avg volume < 10
        low_vol_hub_count    : number of such hubs (int)
        zero_vol_days_per_hub: pd.Series  – days with volume < 5, indexed by HubID
    """
    plot_path = Path(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("RMSPE RISK ANALYSIS")
    print("="*60)

    train = train.copy()

    # ── per-hub average volume ────────────────────────────────────────────
    hub_avg = train.groupby(hub_id)[target].mean().sort_values()

    # High-risk: avg volume below threshold
    RISK_THRESHOLD = 10
    high_risk = hub_avg[hub_avg < RISK_THRESHOLD]
    high_risk_hubs   = high_risk.index.tolist()
    low_vol_hub_count = len(high_risk_hubs)

    print(f"  Hubs with avg {target} < {RISK_THRESHOLD}: {low_vol_hub_count}")
    if low_vol_hub_count:
        print(f"  Avg volume range in high-risk group: "
              f"{high_risk.min():.2f} – {high_risk.max():.2f}")

    # ── near-zero day counts per hub ─────────────────────────────────────
    NEAR_ZERO = 5
    near_zero_mask = train[target] < NEAR_ZERO
    zero_vol_days_per_hub = (
        train[near_zero_mask]
        .groupby(hub_id)
        .size()
        .rename(f"days_{target}_lt_{NEAR_ZERO}")
        .sort_values(ascending=False)
    )
    print(f"  Hubs with at least one day where {target} < {NEAR_ZERO}: "
          f"{len(zero_vol_days_per_hub)}")
    print(f"  Max near-zero days for a single hub: "
          f"{zero_vol_days_per_hub.max() if len(zero_vol_days_per_hub) else 0}")

    # ── bar plot – bottom-20 lowest avg-volume hubs ───────────────────────
    bottom20 = hub_avg.head(20)
    fig, ax = plt.subplots(figsize=(12, 5))
    bottom20.plot(kind="bar", ax=ax, color=PALETTE[7], edgecolor="white", linewidth=0.4)
    ax.axhline(y=RISK_THRESHOLD, color="red", linestyle="--", linewidth=1.2,
               label=f"Risk threshold ({RISK_THRESHOLD})")
    ax.set_title("Bottom-20 Hubs by Avg OrderVolume (RMSPE Risk)")
    ax.set_xlabel(hub_id)
    ax.set_ylabel(f"Avg {target}")
    ax.tick_params(axis="x", rotation=45)
    ax.legend()
    plt.tight_layout()
    _save(fig, plot_path / "15_rmspe_risk_hubs.png")

    print(f"  High-risk hub IDs: {high_risk_hubs[:10]}"
          f"{'  …' if len(high_risk_hubs) > 10 else ''}")

    return {
        "high_risk_hubs":        high_risk_hubs,
        "low_vol_hub_count":     low_vol_hub_count,
        "zero_vol_days_per_hub": zero_vol_days_per_hub,
    }


def data_quality_report(
    train: pd.DataFrame,
    test: pd.DataFrame,
    hub_id: str = "HubID",
    date_col: str = "Date",
    plot_path: Path = Path("outputs/plots"),
) -> dict:
    """
    Produce a comprehensive data-quality report for the train and test sets.

    Checks performed
    ----------------
    * Missing-value percentage per column (train & test)
    * Duplicate row counts
    * Per-hub date coverage: min date, max date, number of distinct days
    * Train / test date overlap (should be zero – WARNING printed if any)

    A heatmap of missing-value percentages is saved as
    ``16_missing_values_heatmap.png``.

    Parameters
    ----------
    train     : merged train DataFrame
    test      : merged test  DataFrame
    hub_id    : name of the hub identifier column  (default: 'HubID')
    date_col  : name of the date column            (default: 'Date')
    plot_path : directory where PNG files are saved

    Returns
    -------
    dict with keys
        train_missing_pct : pd.Series – missing % per column in train
        test_missing_pct  : pd.Series – missing % per column in test
        train_duplicates  : int  – duplicate row count in train
        test_duplicates   : int  – duplicate row count in test
        date_overlap      : set  – dates that appear in both train and test
    """
    plot_path = Path(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("DATA QUALITY REPORT")
    print("="*60)

    train = train.copy()
    test  = test.copy()
    train[date_col] = pd.to_datetime(train[date_col])
    test[date_col]  = pd.to_datetime(test[date_col])

    # ── missing values ────────────────────────────────────────────────────
    train_missing_pct = (train.isnull().mean() * 100).round(2)
    test_missing_pct  = (test.isnull().mean()  * 100).round(2)

    print("\n  Train – missing % per column:")
    print(train_missing_pct[train_missing_pct > 0].to_string()
          if train_missing_pct.any() else "    None")
    print("\n  Test – missing % per column:")
    print(test_missing_pct[test_missing_pct > 0].to_string()
          if test_missing_pct.any() else "    None")

    # ── duplicates ────────────────────────────────────────────────────────
    train_duplicates = int(train.duplicated().sum())
    test_duplicates  = int(test.duplicated().sum())
    print(f"\n  Duplicate rows – train: {train_duplicates:,}  |  test: {test_duplicates:,}")

    # ── per-hub date coverage ─────────────────────────────────────────────
    hub_coverage = (
        train.groupby(hub_id)[date_col]
        .agg(min_date="min", max_date="max", day_count="nunique")
    )
    print("\n  Per-hub date coverage (train) – first 5 hubs:")
    print(hub_coverage.head().to_string())
    print(f"  Day-count stats: min={hub_coverage['day_count'].min()} "
          f"max={hub_coverage['day_count'].max()} "
          f"mean={hub_coverage['day_count'].mean():.1f}")

    # ── train / test date overlap ─────────────────────────────────────────
    train_dates = set(train[date_col].dt.normalize().unique())
    test_dates  = set(test[date_col].dt.normalize().unique())
    date_overlap = train_dates & test_dates

    if date_overlap:
        print(f"\n  ⚠️  WARNING: {len(date_overlap)} date(s) appear in BOTH "
              f"train and test – potential data leakage!")
        print(f"  Overlapping dates (first 5): "
              f"{sorted(date_overlap)[:5]}")
    else:
        print("\n  ✅ No date overlap between train and test.")

    # ── heatmap of missing % (combined view) ─────────────────────────────
    # Align columns for a side-by-side heatmap
    all_cols = sorted(set(train.columns) | set(test.columns))
    miss_df  = pd.DataFrame({
        "train": train_missing_pct.reindex(all_cols, fill_value=0),
        "test":  test_missing_pct.reindex(all_cols,  fill_value=0),
    }).T  # shape: (2, n_cols)

    fig, ax = plt.subplots(figsize=(max(10, len(all_cols) * 0.6), 3))
    sns.heatmap(
        miss_df,
        annot=True, fmt=".1f", cmap="YlOrRd",
        vmin=0, vmax=100,
        linewidths=0.5, ax=ax,
        annot_kws={"size": 8},
        cbar_kws={"label": "Missing %"},
    )
    ax.set_title("Missing Value % per Column  (train vs test)")
    ax.set_xlabel("Column")
    ax.tick_params(axis="x", rotation=45)
    plt.tight_layout()
    _save(fig, plot_path / "16_missing_values_heatmap.png")

    return {
        "train_missing_pct": train_missing_pct,
        "test_missing_pct":  test_missing_pct,
        "train_duplicates":  train_duplicates,
        "test_duplicates":   test_duplicates,
        "date_overlap":      date_overlap,
    }


def autocorrelation_plot(
    train: pd.DataFrame,
    hub_id: str = "HubID",
    target: str = "OrderVolume",
    date_col: str = "Date",
    n_lags: int = 28,
    plot_path: Path = Path("outputs/plots"),
) -> pd.Series:
    """
    Compute and plot autocorrelation of the target time series for the hub
    with the most observations.

    Autocorrelation at lag k is the Pearson correlation between the series
    and its k-step lagged version.  Lags 1 through ``n_lags`` are computed
    and plotted as a bar chart with ±1.96/√N confidence bands.

    The plot is saved as ``17_autocorrelation.png``.

    Parameters
    ----------
    train     : merged train DataFrame
    hub_id    : name of the hub identifier column  (default: 'HubID')
    target    : name of the target column           (default: 'OrderVolume')
    date_col  : name of the date column             (default: 'Date')
    n_lags    : number of lags to compute           (default: 28)
    plot_path : directory where PNG files are saved

    Returns
    -------
    pd.Series  – autocorrelation values indexed by lag (1 … n_lags)
    """
    plot_path = Path(plot_path)
    plot_path.mkdir(parents=True, exist_ok=True)

    print("\n" + "="*60)
    print("AUTOCORRELATION ANALYSIS")
    print("="*60)

    train = train.copy()
    train[date_col] = pd.to_datetime(train[date_col])

    # ── pick the hub with the most data points ────────────────────────────
    best_hub = train.groupby(hub_id).size().idxmax()
    hub_series = (
        train[train[hub_id] == best_hub]
        .sort_values(date_col)
        .set_index(date_col)[target]
    )
    n_obs = len(hub_series)
    print(f"  Selected hub: {best_hub}  ({n_obs} observations)")

    # ── compute autocorrelation for lags 1 … n_lags ───────────────────────
    acf_values = {
        lag: hub_series.autocorr(lag=lag)
        for lag in range(1, n_lags + 1)
    }
    acf_series = pd.Series(acf_values, name="autocorrelation")
    acf_series.index.name = "lag"

    print(f"  Lag-1  autocorr : {acf_series.iloc[0]:.4f}")
    print(f"  Lag-7  autocorr : {acf_series.get(7, float('nan')):.4f}")
    print(f"  Lag-14 autocorr : {acf_series.get(14, float('nan')):.4f}")
    print(f"  Lag-28 autocorr : {acf_series.get(28, float('nan')):.4f}")

    # ── confidence band ±1.96 / sqrt(N) ──────────────────────────────────
    conf_band = 1.96 / np.sqrt(n_obs)

    # ── plot ──────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(12, 5))
    lags = acf_series.index.tolist()
    colors = [
        PALETTE[0] if abs(v) > conf_band else PALETTE[4]
        for v in acf_series.values
    ]
    ax.bar(lags, acf_series.values, color=colors, edgecolor="white", linewidth=0.4)
    ax.axhline(y=0, color="black", linewidth=0.8)
    ax.axhline(y=conf_band,  color="red", linestyle="--", linewidth=1,
               label=f"±1.96/√N ({conf_band:.3f})")
    ax.axhline(y=-conf_band, color="red", linestyle="--", linewidth=1)
    ax.set_title(
        f"Autocorrelation – Hub {best_hub}\n"
        f"(bars outside red band are statistically significant)"
    )
    ax.set_xlabel("Lag (days)")
    ax.set_ylabel("Autocorrelation")
    ax.set_xticks(lags)
    ax.legend()
    plt.tight_layout()
    _save(fig, plot_path / "17_autocorrelation.png")

    return acf_series
