# Project: Kalman_TB

> The accompanying `Comprehensive_code.ipynb` notebook contains exploratory development and trial iterations.  
> `Clean Flow` represents the **final, structured, and presentable implementation** of that workflow.

This repository implements a **machine learning framework** for analyzing and forecasting equity time-series data through signal extraction, feature selection, and explainable model evaluation.  
The framework integrates **Kalman filtering**, **Bayesian optimization**, and **QuantStats** analysis to assess model reliability and risk-adjusted performance across equities.

---

## Overview

The project builds a modular data-to-model pipeline:

- **Data Acquisition & Preprocessing**  
  Fetches multi-year equity data (e.g., NIFTY 50 constituents), aligns timestamps, computes turnover, and applies outlier smoothing.

- **Feature Engineering**  
  - Adds technical indicators (momentum, volatility, trend) using `ta`.
  - Applies **Kalman filters** (`pykalman`) for latent state estimation.
  - Computes higher-order statistical moments (skewness, kurtosis).

- **Feature Selection & Balancing**  
  Reduces multicollinearity via **VIF filtering** and **Lasso regularization**;  
  addresses class imbalance using **SMOTE** synthetic oversampling.

- **Modeling**  
  Trains **XGBoost classifiers**, hyperparameter-tuned via **Optuna (TPE sampler)** under a **time-series cross-validation** regime.

- **Explainability**  
  Uses **SHAP** and permutation importance to identify key predictive factors and regime-sensitive drivers.

- **Evaluation**  
  Applies **QuantStats** to generate detailed benchmark-vs-strategy performance reports, including risk and drawdown analysis.

---

## Example Model Metrics

Example evaluation for a ticker from the model’s buy list (`TATACONSUM.NS`):

| Metric | Benchmark | Strategy |
|--------|------------|----------|
| **Start Period** | 2023-12-06 | 2023-12-06 |
| **End Period** | 2025-06-18 | 2025-06-18 |
| **Risk-Free Rate** | 0.0% | 0.0% |
| **Time in Market** | 99.0% | 96.0% |
| **Cumulative Return** | 14.14% | **24.21%** |
| **CAGR** | 6.13% | **10.25%** |
| **Sharpe Ratio** | 0.48 | **1.04** |
| **Prob. Sharpe Ratio** | 72.21% | 90.0% |
| **Sortino Ratio** | 0.71 | **1.63** |
| **Max Drawdown** | -29.01% | **-17.49%** |
| **Volatility (ann.)** | 24.68% | **15.03%** |
| **R²** | 0.87 | 0.87 |
| **Calmar Ratio** | 0.21 | **0.59** |
| **Skew / Kurtosis** | 0.21 / 3.02 | 0.33 / 1.66 |
| **Expected Yearly %** | 4.51% | 7.49% |
| **Kelly Criterion** | 4.12% | 8.23% |
| **Daily VaR** | -2.51% | -1.50% |
| **Expected Shortfall (cVaR)** | -2.51% | -1.50% |
| **Payoff Ratio** | 1.10 | **1.19** |
| **Profit Factor** | 1.09 | **1.20** |
| **Common Sense Ratio** | 1.22 | **1.51** |
| **Tail Ratio** | 1.12 | **1.26** |
| **3M Return** | 14.10% | 7.07% |
| **6M Return** | 18.59% | 11.90% |
| **YTD Return** | 17.32% | 11.06% |
| **1Y Return** | -3.54% | 0.82% |
| **Avg. Drawdown** | -3.79% | -2.54% |
| **Recovery Factor** | 0.61 | **1.34** |
| **Ulcer Index** | 0.14 | **0.08** |
| **Serenity Index** | 0.08 | **0.20** |
| **Win Days %** | 49.73% | 50.14% |
| **Win Month %** | 52.63% | 47.37% |
| **Win Quarter %** | 85.71% | 71.43% |
| **Win Year %** | 66.67% | 66.67% |
| **Beta / Alpha** | – | 0.57 / 0.09 |
| **Correlation** | – | **93.07%** |
| **Treynor Ratio** | – | **42.72%** |

---

## Key Outputs

- **Buy List**: Predicted equities with positive signal probability.  
- **Performance Reports**: QuantStats benchmark vs. strategy comparison.  
- **Explainability**: SHAP and permutation-based feature importance plots.  
- **Diagnostics**: Model metrics, confusion matrices, and feature correlation maps.

---

## Dependencies

See `requirements.txt` for environment setup.  
Primary packages: `torch`, `numpy`, `matplotlib`, `xgboost`, `optuna`, `imbalanced-learn`, `ta`, `quantstats`, `shap`.

---
