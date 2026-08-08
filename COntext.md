Excellent! I'll combine **my analysis + the other LLM's critical insights** into one **comprehensive, battle-tested Context Report** and then give you **clean, actionable instructions** for your AI Agent to build the Jupyter Notebook project.

---

# 📊 **COMPREHENSIVE CONTEXT REPORT**
## Demand Forecasting for Micro-Fulfillment Hubs

---

### 1. BUSINESS CONTEXT

**Company:** Company_X operates 800+ micro-fulfillment hubs delivering groceries/essentials in **15–30 minutes**.

**The Core Problem:** Demand forecasting errors are **catastrophically expensive**:
- **Overstocking** → Perishables spoil, limited hub space wasted → Higher COGS
- **Understocking** → Order cancellations, lost customer trust → Lower LTV

**Why This Matters:** Unlike traditional retail, micro-fulfillment has:
- **Tiny storage capacity** (can't hold excess inventory)
- **Perishable goods** (spoilage risk)
- **Ultra-fast delivery** (no time to restock mid-day)

---

### 2. MACHINE LEARNING TASK

| **Aspect** | **Details** |
|------------|-------------|
| **Task Type** | Multi-Series Time-Series Regression |
| **Target Variable** | `OrderVolume` (daily order count per hub) |
| **Prediction Horizon** | Daily (next day(s)) - NOT 15-minute intervals |
| **Evaluation Metric** | **RMSPE** (Root Mean Squared Percentage Error) |
| **Validation Rule** | **Held-out future period** - NO random splits! |
| **Constraint** | **No external data** - Use ONLY the 3 provided files |

**RMSPE Formula:**
```
RMSPE = sqrt(mean(((actual - predicted) / actual)^2))
```
⚠️ **Critical**: Highly sensitive to **zeros and small values** - division by zero possible!

---

### 3. DATA ARCHITECTURE

| **File** | **Content** | **Key Columns** | **Join Key** |
|----------|-------------|-----------------|--------------|
| `orders_train.csv` | Historical daily orders + operational context | `Store`, `Date`, `OrderVolume` (target), `rider_availability`, `operational_status`, `promotion_activity`, `day_of_week`, `holiday_indicator` | `Store` |
| `orders_test.csv` | Future dates for prediction | Same structure, but `OrderVolume` MISSING. **CRITICAL:** Rider values are **PLANNED** (not actual) | `Store` |
| `hub_metadata.csv` | Static hub attributes | `Store`, `format`, `assortment_tier`, `competitor_distance`, `competitor_tenure`, `loyalty_program_status`, `launch_date` | `Store` |

---

### 4. 🚨 CRITICAL INSIGHTS & HIDDEN TRAPS

#### A. The Rider Availability Trap (Train-Test Distribution Shift)

| **In Training** | **In Test** |
|-----------------|-------------|
| `rider_availability` = **ACTUAL** riders who showed up | `rider_availability` = **PLANNED/SCHEDULED** riders (estimate made BEFORE the day) |

**The Danger:**
- Model learns: "More riders = More orders" (causal relationship in training)
- In test: Planned riders may be systematically different (planners over-schedule on weekends)
- Results in **train-test distribution shift** → Poor generalization

**Our Strategy:**
- Use `rider_availability` **cautiously** (monitor feature importance)
- Consider it a **weak feature**, not primary driver
- Engineer proxy: `rider_availability` vs rolling average of rider availability

#### B. The RMSPE Zero Problem

| **Scenario** | **Impact on RMSPE** |
|--------------|---------------------|
| Actual = 0 | **Division by zero** → Infinite error |
| Actual = 1, Predicted = 2 | 100% error (1/1) |
| Actual = 100, Predicted = 101 | 1% error (1/100) |

**Strategy:**
- Check for zeros: Are they genuine (low demand) or closures?
- If genuine zeros exist → Consider `log(OrderVolume + 1)` transformation
- Add small epsilon in evaluation if needed

#### C. The "No External Data" Constraint
- **Cannot use:** Weather data, economic indicators, public holiday APIs, traffic data
- **Must engineer:** ALL features from the 3 provided files only
- **Implication:** Calendar features are already provided - don't recalculate them

---

### 5. MODEL SELECTION VERDICT

| **Model** | **Pros** | **Cons** | **Recommendation** |
|-----------|----------|----------|-------------------|
| **LightGBM** | Fast, handles categoricals natively, handles missing values, dominant for tabular time-series | Needs manual feature engineering | ✅ **PRIMARY MODEL** |
| **XGBoost** | Robust, battle-tested, great performance | Slightly slower than LightGBM | ✅ **SECONDARY MODEL** (for comparison) |
| **LSTM/GRU** | Learns sequential patterns automatically | Struggles with static features, needs large data, slow to tune | ❌ **SKIP** (not enough time/data) |
| **Ensemble** | Can squeeze extra accuracy | Adds complexity, risk of overfitting | ⚠️ **OPTIONAL** (last 15 min if time permits) |

**Final Verdict:** Use **LightGBM** as primary, **XGBoost** as comparison. Skip LSTM entirely.

---

### 6. FEATURE ENGINEERING BLUEPRINT

Since tree-based models don't "see" time natively, we must manually create temporal features.

#### A. Temporal/Calendar Features (from `orders_train.csv`)
- `day_of_week` (already provided)
- `is_weekend` (derive from day_of_week)
- `month` (extract from Date)
- `week_of_year` (extract from Date)
- `quarter` (extract from Date)
- `days_since_last_holiday` (engineer from holiday_indicator)
- `days_to_next_holiday` (engineer from holiday_indicator)

#### B. Lag Features (Historical Demand) - **POWERFUL!**
*Create these PER HUB:*
- `lag_1`: Yesterday's order volume
- `lag_7`: Order volume 7 days ago (weekly seasonality)
- `lag_14`: Order volume 14 days ago
- `lag_21`: Order volume 21 days ago
- `lag_28`: Order volume 28 days ago (monthly cycle)

#### C. Rolling Window Statistics
*Create these PER HUB:*
- `rolling_mean_7`: 7-day moving average
- `rolling_std_7`: 7-day volatility
- `rolling_mean_14`: 14-day moving average
- `rolling_mean_28`: 28-day moving average
- `rolling_max_7`: 7-day maximum (captures spikes)
- `rolling_min_7`: 7-day minimum (captures dips)

⚠️ **CRITICAL:** Shift by 1 day before rolling to prevent data leakage!

#### D. Hub Metadata Features (from `hub_metadata.csv`)
- `hub_age`: (Current Date - launch_date).days → **CRITICAL** (new hubs behave differently)
- `competitor_distance`: Raw or binned (closer competitors suppress demand)
- `competitor_tenure`: How established the competitor is
- `format`: Categorical - one-hot encode or label encode
- `assortment_tier`: Categorical - encode carefully
- `loyalty_program_status`: Binary flag

#### E. Operational Features (Handle With Care!)
- `rider_availability`: **DANGER FEATURE** (train vs test mismatch)
  - Use but monitor feature importance
  - If dominates → consider dropping
- `operational_status`: Categorical - flag for closures
- `promotion_activity`: Binary or intensity
  - Create interaction: `promotion_activity * is_weekend`
  - Create interaction: `promotion_activity * holiday_indicator`

#### F. Interaction Features
- `promotion_activity × holiday_indicator`
- `promotion_activity × is_weekend`
- `format × assortment_tier`

---

### 7. EDA CHECKLIST (Must Do!)

| **Category** | **What to Check** | **Why** |
|--------------|-------------------|---------|
| **Target Variable** | Distribution, skewness, zeros, outliers | RMSPE sensitivity |
| **Time Patterns** | Weekly seasonality, holiday effects, trends | Guide lag/rolling feature creation |
| **Hub Heterogeneity** | Volume by format, tier, age | Hub-specific modeling needs |
| **Train-Test Shift** | Compare `rider_availability` distribution | Critical! |
| **Missing Values** | Per column, per file | Imputation strategy needed |
| **Correlations** | Feature vs target, feature vs feature | Feature selection guidance |

---

### 8. VALIDATION STRATEGY

**Rule:** NO random splits! Must use **time-based validation**.

**Expanding Window Approach:**
```
Train: Days 1 → T
Val:   Days T+1 → T+7
Test:  Days T+8 → End
```

**Sliding Window Approach:**
```
Train: Days 1 → 80
Val:   Days 81 → 85
Train: Days 6 → 85
Val:   Days 86 → 90
(Repeat)
```

**Recommendation:** Expanding window with last 20% of training data as validation.

---

### 9. SUCCESS CRITERIA

| **Metric** | **Target** |
|------------|------------|
| RMSPE on Validation | < 0.30 (good) / < 0.25 (excellent) |
| Feature Importance | Lags + rolling means should dominate |
| Error by Hub Size | Small hubs should NOT have disproportionately high errors |

---

### 10. PROJECT TIMELINE (2 Hours)

| **Phase** | **Time** | **Deliverable** |
|-----------|----------|-----------------|
| **1. EDA** | 20 min | Visualizations, summary statistics, insights |
| **2. Feature Engineering** | 30 min | All features from blueprint above |
| **3. Model Building** | 30 min | LightGBM baseline + XGBoost comparison |
| **4. Validation & Tuning** | 20 min | RMSPE evaluation, hyperparameter tuning |
| **5. Refinement** | 15 min | Feature selection, ensemble (if time) |
| **6. Documentation** | 5 min | Summarize key findings, next steps |

---
