# 🏪 Demand Forecasting – Micro-Fulfillment Hubs

Kaggle competition project: predict daily order volume (`OrderVolume`) per hub using historical data.

## Problem
Multi-series time-series regression. **Metric: RMSPE** (Root Mean Squared Percentage Error).
Target < 0.30 | Excellent < 0.25.

## Dataset
| File | Description |
|---|---|
| `orders_train.csv` | Train: HubID, Weekday, Date, OrderVolume, AppSessions, IsOpen, PromoActive, RegionalHoliday, SchoolClosureFlag |
| `orders_test.csv` | Test: same columns minus OrderVolume and AppSessions, plus Id |
| `hub_metadata.csv` | Static hub attributes: format, assortment tier, competitor info, loyalty program |
| `sample_submission.csv` | Submission format: Id, OrderVolume |

## Quick Start
```bash
pip install -r requirements.txt
jupyter notebook notebooks/main_workflow.ipynb
```

## Project Structure
```
├── data/               Raw CSV files
├── src/                Python modules (eda, features, model, utils)
├── notebooks/          main_workflow.ipynb (full pipeline)
├── outputs/            plots/, model_artifacts/, predictions.csv
├── artifacts/          EDA observations, pipeline docs, LLM context
└── requirements.txt
```

## Models
- **LightGBM** (primary) + **XGBoost** (comparison) + simple ensemble
- ~44 engineered features: lags, rolling stats, calendar, hub metadata, interaction terms
- Time-based validation split (80/20)

## Key Pitfalls
- AppSessions is NOT in test → use lag/rolling of AppSessions
- IsOpen=0 rows → always predict 0
- RMSPE excludes zero-actual rows (division by zero)
- Time-based split only – NO random splits

## Artifacts (for downstream LLMs)
- `artifacts/01_EDA_OBSERVATIONS.md` – full data findings
- `artifacts/02_PROJECT_PIPELINE.md` – pipeline architecture
- `artifacts/03_CONTEXT_FOR_LLM.md` – self-contained context package
