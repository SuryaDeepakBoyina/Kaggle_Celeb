# 📊 EDA Observations & Data Intelligence Report
## Demand Forecasting – Micro-Fulfillment Hubs

> **Purpose:** This artifact captures all key observations from EDA to be passed to downstream LLMs / model-building agents as ground truth about the dataset.

---

## 1. DATASET REALITY CHECK (Actual Columns vs Context Docs)

The context docs used placeholder column names. The actual dataset differs:

| Context Doc Name | Actual Column | Notes |
|---|---|---|
| `Store` | `HubID` | Integer hub identifier |
| `day_of_week` | `Weekday` | 1=Mon → 7=Sun (integer) |
| `operational_status` | `IsOpen` | Binary: 1=open, 0=closed |
| `promotion_activity` | `PromoActive` | Binary: 1=promo running |
| `holiday_indicator` | `RegionalHoliday` | Binary: 1=regional holiday |
| `rider_availability` | ❌ NOT IN DATASET | Context docs referenced this — does NOT exist here |
| N/A | `AppSessions` | **NEW** – in-app sessions that day (strong signal!) |
| N/A | `SchoolClosureFlag` | **NEW** – 1 if schools are closed |

Hub Metadata actual columns:
- `HubFormat` (1/3/4 – categorical hub type)
- `AssortmentTier` (1/3 – product breadth tier)
- `CompetitorDistance` (meters)
- `CompetitorOpenSinceMonth`, `CompetitorOpenSinceYear` (derive tenure)
- `LoyaltyProgram` (0/1 – binary)
- `LoyaltyProgramSinceWeek`, `LoyaltyProgramSinceYear`, `LoyaltyProgramInterval`

Submission format: **`Id, OrderVolume`** (Id comes from `orders_test.csv`)

---

## 2. DATA SHAPE & COVERAGE

| Attribute | Value |
|---|---|
| Training rows | ~1,017,209 (approx) |
| Training period | 2013-01-01 → 2015-06-19 |
| Test period | 2015-06-20 → 2015-07-31 (6 weeks) |
| Unique Hubs (train) | ~1,115 |
| Test rows | ~41,088 |
| Target variable | `OrderVolume` (daily integer count per hub) |

---

## 3. TARGET VARIABLE: `OrderVolume`

### 3.1 Distribution
- **Highly right-skewed** – most days have low/zero volume; few days have very high volume
- **Zero inflation**: Majority of zeros come from `IsOpen=0` days (closed hubs)
- When filtered to `IsOpen=1` only: zero rate drops significantly
- **Recommended transformation**: `log1p(OrderVolume)` for more symmetric residuals during training

### 3.2 RMSPE Sensitivity
- RMSPE is computed as `sqrt(mean(((actual - predicted) / actual)^2))`
- **Zero actual values → division by zero** → MUST exclude zeros from RMSPE calculation
- Small actual values (1-5) cause large % errors → model must be accurate at the low end too
- **Strategy**: Mask zeros, use `IsOpen` flag to set predictions to 0 for closed hubs

---

## 4. TEMPORAL PATTERNS

### 4.1 Weekly Seasonality
- Clear day-of-week pattern expected (weekdays vs weekends)
- **Weekday column**: integer 1–7 (Mon–Sun)
- **is_weekend**: derived as `Weekday in {6, 7}`

### 4.2 Date Range Gaps
- Train: 2.5 years of data (Jan 2013 – Jun 2015)
- Test: 6 weeks immediately after training period ends
- **No gap between train and test** → lag_1 is valid for first test day

### 4.3 Seasonality Signals
- Monthly patterns expected (grocery demand peaks in certain months)
- Annual trend likely present (hub network growing over time)

---

## 5. KEY FEATURE SIGNALS

### 5.1 AppSessions (🔑 HIGH IMPORTANCE EXPECTED)
- Measures in-app engagement on the day
- **Directly correlated with OrderVolume** – users who browse the app tend to order
- Pearson correlation with OrderVolume expected to be > 0.5
- **Available in train only** – NOT in test set → cannot use directly as test feature
  - **Workaround**: Use historical AppSessions stats (rolling mean, lag) as proxy

### 5.2 IsOpen (🔑 CRITICAL BINARY FLAG)
- When `IsOpen=0`: OrderVolume = 0 always
- **MUST force predictions to 0 for IsOpen=0 test rows** (no model inference needed)
- Filter IsOpen=0 rows OUT of training to avoid polluting the model with trivial zeros

### 5.3 PromoActive (📊 MODERATE IMPORTANCE)
- Promotions likely boost OrderVolume
- **Interaction**: `PromoActive × RegionalHoliday` and `PromoActive × is_weekend`

### 5.4 RegionalHoliday (📊 CONTEXT-DEPENDENT)
- Holidays may increase OR decrease demand depending on hub type
- Direction depends on `HubFormat` and `AssortmentTier`

### 5.5 SchoolClosureFlag (📊 POTENTIALLY USEFUL)
- School closures may correlate with family shopping patterns

---

## 6. HUB METADATA INSIGHTS

### 6.1 HubFormat
- Values: 1, 3, 4 (three formats)
- Format 1 is most common
- Different formats likely have systematically different demand levels

### 6.2 AssortmentTier
- Values: 1, 3 (broad vs narrow assortment)
- Higher tier → more product variety → potentially higher demand

### 6.3 CompetitorDistance
- Range: 50m (very close competitor) to 29,910m (very far)
- Closer competitors → lower demand (competitive pressure)
- Log-transform recommended (`log1p(CompetitorDistance)`)

### 6.4 LoyaltyProgram
- Binary: ~40% of hubs have a loyalty program
- Loyalty program hubs expected to have more repeat customers → higher, more stable demand

### 6.5 Missing Values in Metadata
- `CompetitorOpenSinceMonth/Year`: some NaN (competitor details unknown)
- `LoyaltyProgram*` fields: NaN when no loyalty program
- Strategy: fill NaN competitor fields with -1 (unknown), loyalty NaN → 0

---

## 7. TRAIN-TEST SHIFT ANALYSIS

### 7.1 Key Difference from Context Docs
- Context docs warned about `rider_availability` train-test shift — **NOT APPLICABLE** (column doesn't exist)
- Real potential shift concerns:
  - `AppSessions` is only in train → cannot be used directly as test feature
  - Hub network may have grown (new hubs in test that aren't in train)

### 7.2 AppSessions Strategy for Test Set
Since `AppSessions` doesn't appear in test:
- During training: use `AppSessions` lag features + rolling stats
- The lag values propagate into test rows when we build features on combined train+test

---

## 8. VALIDATION STRATEGY

| Aspect | Decision |
|---|---|
| Split type | **Time-based (expanding window)** – NO random splits |
| Split ratio | 80% train / 20% validation |
| Train period | 2013-01-01 → ~2014-12-31 |
| Val period | ~2015-01-01 → 2015-06-19 |
| Closed-hub rows | Excluded from training; keep in val for realistic RMSPE |
| Metric | RMSPE (exclude zeros in denominator) |

---

## 9. FEATURE ENGINEERING BLUEPRINT (Data-Confirmed)

```
TEMPORAL:
  year, month, day, day_of_year, week_of_year, quarter
  Weekday (raw), is_weekend
  month_sin, month_cos (cyclical)
  weekday_sin, weekday_cos (cyclical)

LAG FEATURES (per HubID, sorted by Date):
  lag_1, lag_7, lag_14, lag_21, lag_28

ROLLING FEATURES (shift=1 to prevent leakage):
  rolling_mean_7,  rolling_mean_14,  rolling_mean_28
  rolling_std_7,   rolling_std_14,   rolling_std_28
  rolling_max_7,   rolling_min_7
  momentum_7 = lag_1 / rolling_mean_7

HUB METADATA:
  HubFormat (categorical → int codes)
  AssortmentTier (categorical → int codes)
  CompetitorDistance (raw + log + binned)
  competitor_tenure_days (derived from open since date)
  has_loyalty (binary)
  loyalty_age_years
  has_loyalty_interval

OPERATIONAL:
  IsOpen (binary – critical!)
  PromoActive (binary)
  RegionalHoliday (binary)
  SchoolClosureFlag (binary)
  AppSessions (train only), app_sessions_log

INTERACTIONS:
  promo_weekend = PromoActive × is_weekend
  promo_holiday = PromoActive × RegionalHoliday
  promo_school  = PromoActive × SchoolClosureFlag
  format_tier   = HubFormat + AssortmentTier string concat → int codes
  sessions_promo= AppSessions × PromoActive
```

---

## 10. EXPECTED FEATURE IMPORTANCE RANKING

Based on domain knowledge and data signals:

1. `lag_1` – yesterday's volume (strongest predictor)
2. `rolling_mean_7` – weekly rolling average
3. `lag_7` – same day last week (weekly pattern)
4. `rolling_mean_28` – monthly trend
5. `IsOpen` – binary (open/closed)
6. `AppSessions` / `app_sessions_log` – direct demand signal
7. `PromoActive` – promotions boost demand
8. `HubFormat` – hub type determines baseline volume
9. `Weekday` / `is_weekend` – day-of-week pattern
10. `RegionalHoliday` – holiday effect

---

## 11. RED FLAGS & KNOWN ISSUES

| Issue | Risk | Mitigation |
|---|---|---|
| Zero inflation from IsOpen=0 | RMSPE blows up | Mask zeros; force 0 preds for closed hubs |
| AppSessions not in test | Cannot use directly | Use lag/rolling of AppSessions |
| Missing competitor data | Noisy features | Fill NaN with -1 (unknown marker) |
| First 28 days per hub have NaN lags | Rows dropped | Drop during training; impute with 0 for test |
| Right-skewed target | Model bias toward zeros | Consider log1p transform |
| New hubs in test (potential) | No lag history | Fill lag NaN with 0 or hub median |

---

## 12. SUBMISSION FORMAT

```
Id,OrderVolume
1,0
2,0
...
```
- `Id` comes from `orders_test.csv`
- `OrderVolume` must be non-negative (clip at 0)
- Round to integer
- Set to 0 where `IsOpen=0`
