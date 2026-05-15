# Can Google Search Trends Predict Stock Market Direction?
### A Machine Learning Approach to Weekly S&P 500 Forecasting

MSc thesis — Data Science & Business Administration, Copenhagen Business School (2026)

---

## Overview

This repository contains the full code for the empirical analysis in my MSc thesis, 
which investigates whether aggregate Google search volume carries predictive content 
for weekly S&P 500 directional movement using machine learning classifiers.

The study covers 728 weekly observations from January 2004 to December 2017, 
processing Google Trends data for 90 search terms through a four-stage feature 
selection pipeline and evaluating seven classification models across three 
feature specifications.

---

## Key Results

- **Random Forest** achieves **71.0% directional accuracy** and **ROC AUC = 0.600** 
  on selectively predicted weeks (Specification 1, confidence threshold = 0.58)
- All linear models produce near-random AUC ≈ 0.50, confirming the 
  attention-return relationship is **non-linear**
- Adding macroeconomic variables does **not** improve on the Google Trends-only 
  baseline, suggesting search signals capture information **independent** of 
  macroeconomic conditions
- Walk-forward validation produces mean AUC = 0.532 across five chronological 
  folds, with performance strengthening toward the most recent period — 
  consistent with the **Adaptive Markets Hypothesis**

---


## Final Feature Set (Specification 1)

| Feature | Description | Lag |
|---------|-------------|-----|
| holiday_L1 | Search volume: "holiday" | 1 week |
| politics_L1 | Search volume: "politics" | 1 week |
| trader_L2 | Search volume: "trader" | 2 weeks |
| risk_L3 | Search volume: "risk" | 3 weeks |
| dow_jones | Search volume: "dow jones" | Contemporaneous |
| risk_x_dow | risk_L3 × dow_jones (interaction) | — |

---

## Requirements

```bash
pip install pytrends yfinance pandas numpy scikit-learn xgboost 
            lifelines statsmodels matplotlib seaborn scipy
```

---

## Data Sources

- **Google Trends** — collected via [Pytrends]
- **S&P 500** — Yahoo Finance via yfinance
- **Macroeconomic data** — [FRED](https://fred.stlouisfed.org) 
  (GDP, CPI, UNRATE, FEDFUNDS, VIX, DGS10, DGS2, DBAA, DFF)

---

## Theoretical Framework

The study is framed within Lo's (2004) **Adaptive Markets Hypothesis**, 
testing whether attention-based predictability is time-varying and regime-dependent 
rather than a stable exploitable inefficiency.

---

## Citation

> Aleksandra Michalowska (2026). *Can Google Search Trends Predict Stock Market Direction? 
> A Machine Learning Approach.* MSc Thesis, Copenhagen Business School.
