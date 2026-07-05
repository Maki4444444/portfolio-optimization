# Time Series Forecasting for Portfolio Management Optimization

## Overview

This project builds a data-driven portfolio management workflow for **GMF Investments**, a
fictional financial advisory firm that uses time series forecasting to enhance portfolio
strategies. The engagement combines classical statistical forecasting (ARIMA/SARIMA) and
deep learning (LSTM) to predict asset price movements, then feeds those forecasts into
Modern Portfolio Theory optimization and a historical backtest — mirroring the kind of
end-to-end quantitative workflow a real asset management team would run before adjusting
client allocations.

## Business Need

GMF Investments aims to leverage data-driven insights to optimize portfolio management
strategies. As a Financial Analyst at GMF, the objective is to apply time series
forecasting to historical financial data in order to enhance portfolio management
strategies. The role involves predicting market trends, optimizing asset allocation, and
enhancing portfolio performance to ensure the achievement of the company's overarching
goal — maximizing returns while minimizing risks for its clients. GMF manages diversified
portfolios that leverage cutting-edge technology and data-driven insights to provide
clients with optimized investment strategies. By integrating advanced time series
forecasting models, GMF seeks to predict market trends, optimize asset allocation, and
enhance portfolio performance.

## Description

The core of this project is Modern Portfolio Theory (MPT), applied across three assets
chosen for their distinct risk/return roles:

| Ticker | Role | Description |
|---|---|---|
| **TSLA** | High-growth, high-risk | Tesla, Inc. — high potential returns with high volatility exposure to consumer discretionary/automotive sector risk |
| **BND** | Stability | Vanguard Total Bond Market ETF — tracks U.S. investment-grade bonds, providing income and stability, low risk |
| **SPY** | Diversification | S&P 500 ETF — broad U.S. market exposure, moderate, diversified risk |

Using data from **2015-01-01 to 2026-06-30**, the project proceeds through five stages:

1. **Preprocess and explore the data** — extract, clean, and analyze historical price data
   for all three assets, establishing the statistical properties (trend, volatility,
   stationarity, risk metrics) that inform every downstream modeling choice.
2. **Develop time series forecasting models** — build and compare ARIMA/SARIMA and LSTM
   models to forecast TSLA's future stock price.
3. **Forecast future market trends** — use the best-performing model from Task 2 to
   generate a forward-looking price forecast with confidence intervals, and interpret the
   trend and uncertainty it implies.
4. **Optimize portfolio based on forecast** — combine the return forecast for TSLA with
   the historical performance of BND and SPY to construct an efficient frontier and
   recommend an optimal portfolio allocation.
5. **Backtest the strategy** — simulate the recommended portfolio's performance over a
   recent historical period against a benchmark (e.g., a static 60/40 SPY/BND portfolio)
   to assess whether the strategy would have plausibly outperformed.

## Project Structure

```
portfolio-optimization/
├── .github/workflows/
│   └── unittests.yml          # CI: installs deps and runs pytest on every push/PR
├── data/
│   ├── raw/                  # Cached per-ticker OHLCV CSVs (gitignored)
│   └── processed/            # Cleaned panels, model artifacts, summaries (gitignored)
├── notebooks/
│   ├── 1_EDA.ipynb                    # Task 1: data cleaning & exploration
│   └── 2_Time_Series_Modeling.ipynb   # Task 2: ARIMA vs LSTM
├── src/
│   ├── data_loader.py         # yfinance fetch + local CSV caching + error handling
│   ├── task1_eda.py           # Cleaning, EDA, ADF tests, risk metrics (scriptable)
│   ├── task2_models.py        # ARIMA/LSTM training & evaluation (scriptable)
│   └── task3_forecast.py      # Future forecasting with confidence intervals
├── tests/                     # Unit tests (run in CI via unittests.yml)
├── scripts/
│   ├── build_notebook1.py     # Generates notebooks/1_EDA.ipynb programmatically
│   └── build_notebook2.py     # Generates notebooks/2_Time_Series_Modeling.ipynb
├── reports/figures/           # Exported PNG figures
├── requirements.txt
└── README.md
```

**Design notes:**
- The notebooks in `notebooks/` import their core computation directly from `src/`
  (`data_loader.load_prices`, `task1_eda.clean_and_merge/adf_test/compute_risk_metrics/
  detect_outliers`, `task2_models.chrono_split/fit_arima/forecast_arima/make_sequences/
  build_lstm_model/mape`) rather than reimplementing that logic inline. Notebook cells
  focus on narrative, visualization, and table/plot formatting on top of these functions.
- The same `src/` functions are exercised directly by `tests/` and are independently
  runnable as scripts (e.g. `python src/task1_eda.py` reproduces Task 1 outside Jupyter
  entirely, writing the same processed CSVs and a JSON summary).
- `data_loader.py` centralizes all data access behind a single `load_prices()` function
  with an explicit cache → live-fetch resolution order and a custom `DataFetchError` so
  callers (both notebooks and scripts) get an actionable error instead of a silent
  `KeyError` deep in pandas.
- `tests/` exercises the failure paths explicitly (network errors, empty API responses,
  missing columns, corrupt cache files) rather than only the happy path, and runs
  automatically on every push/PR via `.github/workflows/unittests.yml`.

## Data Sources

All price data is fetched live via [`yfinance`](https://pypi.org/project/yfinance/)
(Yahoo Finance), covering:

| Ticker | Date Range | Fields |
|---|---|---|
| TSLA, BND, SPY | 2015-01-01 to 2026-06-30 | Open, High, Low, Close, Volume (`auto_adjust=True`, so Close is already split/dividend-adjusted) |

Fetched data is cached to `data/raw/<TICKER>.csv` on first run to avoid repeated API
calls; cleaned, merged panels are written to `data/processed/` for reuse across notebooks.

## Setup

```bash
git clone <repo-url>
cd portfolio-optimization
python -m venv .venv
source .venv/bin/activate   # .venv\Scripts\activate on Windows
pip install -r requirements.txt
```

Run the notebooks in order:
```bash
jupyter nbconvert --to notebook --execute --inplace notebooks/1_EDA.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/2_Time_Series_Modeling.ipynb
```
or open them directly in Jupyter/VS Code and run all cells.

Run the test suite:
```bash
pytest tests/ -v
```

## Continuous Integration

`.github/workflows/unittests.yml` runs on every push and pull request: it installs
`requirements.txt` and runs `pytest tests/ -v` on Python 3.11.

## Task 1 — Data Preprocessing & Exploration

Fetches TSLA/BND/SPY via `yfinance` (`auto_adjust=True`), cleans and reindexes to a common
business-day calendar, then runs EDA (price trends, daily % change, 30-day rolling
mean/std, return distributions, z-score outlier detection), ADF stationarity tests on
both price and returns, and computes 95% historical VaR and Sharpe ratio per asset.

**Key findings:**
- All three closing-price series are non-stationary (ADF fails to reject the unit-root
  null); all three return series are stationary — directly motivating ARIMA's
  differencing step in Task 2.
- TSLA carries the highest annualized volatility (~56%) and VaR (~5.1%/day) of the three,
  roughly an order of magnitude above BND's, but also posts the highest Sharpe ratio
  (~0.71) — its return premium more than compensates for its risk over this period.
- BND's Sharpe ratio is negative (~-0.40): its annualized return didn't clear the assumed
  4% risk-free rate, a reminder that "safe" assets aren't automatically risk-adjusted
  winners in every rate environment.

## Task 2 — Time Series Forecasting Models

Chronologically splits TSLA closing price into train (2015–2024) and test (2025–2026),
then fits and compares:
- **ARIMA**, order selected via `pmdarima.auto_arima`'s stepwise AIC search
- **LSTM**, a 2-layer (64→32 unit) recurrent network over 60-day input windows, with
  dropout regularization and a dense output head

Both models are evaluated on MAE, RMSE, and MAPE over the test period, with results
compiled into a single comparison table and discussed in terms of the classical/deep
learning trade-off (interpretability and speed vs. capacity to capture nonlinear trend).

## Notes on Data Availability

All data loading goes through real `yfinance` calls — there is no synthetic-data
fallback in this codebase. If Yahoo Finance is unreachable (e.g. restricted network
egress in a sandboxed CI environment) or a ticker/date range is invalid, `load_prices()`
raises a `DataFetchError` describing exactly which tickers failed and why, rather than
silently substituting placeholder data.