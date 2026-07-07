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
│   ├── 1_EDA.ipynb                      # Task 1: data cleaning & exploration
│   ├── 2_Time_Series_Modeling.ipynb     # Task 2: ARIMA vs LSTM
│   ├── 3_Future_Forecast.ipynb          # Task 3: 12-month forecast with CI
│   ├── 4_Portfolio_Optimization.ipynb   # Task 4: MPT efficient frontier
│   └── 5_Backtesting.ipynb              # Task 5: strategy vs. 60/40 benchmark
├── src/
│   ├── data_loader.py         # yfinance fetch + local CSV caching + error handling
│   ├── task1_eda.py           # Cleaning, EDA, ADF tests, risk metrics (scriptable)
│   ├── task2_models.py        # ARIMA/LSTM training & evaluation (scriptable)
│   ├── task3_forecast.py      # Future forecasting with confidence intervals
│   ├── task4_portfolio.py     # Expected returns, covariance, efficient frontier
│   └── task5_backtest.py      # Backtest simulation & performance metrics
├── tests/                     # Unit tests (run in CI via unittests.yml)
├── scripts/
│   ├── build_notebook1.py     # Generates notebooks/1_EDA.ipynb programmatically
│   ├── build_notebook2.py     # Generates notebooks/2_Time_Series_Modeling.ipynb
│   ├── build_notebook3.py     # Generates notebooks/3_Future_Forecast.ipynb
│   ├── build_notebook4.py     # Generates notebooks/4_Portfolio_Optimization.ipynb
│   └── build_notebook5.py     # Generates notebooks/5_Backtesting.ipynb
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
jupyter nbconvert --to notebook --execute --inplace notebooks/3_Future_Forecast.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/4_Portfolio_Optimization.ipynb
jupyter nbconvert --to notebook --execute --inplace notebooks/5_Backtesting.ipynb
```
or open them directly in Jupyter/VS Code and run all cells.

Run the test suite:
```bash
pytest tests/ -v
```

## Continuous Integration

`.github/workflows/unittests.yml` runs on every push and pull request: it installs
`requirements.txt` and runs `pytest tests/ -v` on Python 3.11.

## Task 1 Data Preprocessing & Exploration

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

## Task 2 Time Series Forecasting Models

Chronologically splits TSLA closing price into train (2015–2024) and test (2025–2026),
then fits and compares:
- **ARIMA**, order selected via `pmdarima.auto_arima`'s stepwise AIC search
- **LSTM**, a 2-layer (64→32 unit) recurrent network over 60-day input windows, with
  dropout regularization and a dense output head. The LSTM is trained on **daily log
  returns** rather than raw price levels (mirroring why ARIMA differences the series —
  Task 1's ADF tests show price is non-stationary, returns are stationary); test
  predictions are reconstructed as `true_previous_price * exp(predicted_log_return)`,
  a walk-forward evaluation anchored to true history at every step.

Both models are evaluated on MAE, RMSE, and MAPE over the test period, with results
compiled into a single comparison table and discussed in terms of the classical/deep
learning trade-off (interpretability and speed vs. capacity to capture nonlinear trend).

## Task 3 Forecast Future Market Trends

Refits the Task 2 best-performing model on the **full** historical TSLA series (not
just the train split) and forecasts 252 trading days (~12 months) forward with a 95%
confidence interval, distinguishing historical data, and the future forecast on one
plot.

- **ARIMA path**: native confidence interval from `auto_arima`'s `.predict()`
- **LSTM path**: iterative multi-step forecasting (each prediction feeds back in as
  input for the next step), trained on log returns for stability, with the central
  forecast additionally shrunk toward a historical-drift random-walk baseline as the
  horizon grows (`shrinkage_weights`) — naive baselines are typically more reliable
  than complex models at long horizons. Bootstrap noise is injected in real log-return
  space so the CI grows like a textbook `sqrt(time)` random walk. An ARIMA cross-check
  is generated alongside the selected model specifically to catch any remaining
  divergence at the far end of the horizon.

Includes trend analysis (direction, total return, max upside/downside), a critical
assessment of how CI width and forecast reliability change over the horizon, and a
market opportunities/risks discussion. The forecast is persisted to
`task3_future_forecast.csv` for Task 4.

## Task 4 Optimize Portfolio Based on Forecast

Builds an expected-returns vector combining Task 3's TSLA forecast (annualized via
CAGR) with BND/SPY's historical annualized average returns, computes the annualized
covariance matrix (with a heatmap), and runs an efficient-frontier optimization via
PyPortfolioOpt — solving for both the **Maximum Sharpe Ratio** ("tangency") portfolio
and the **Minimum Volatility** portfolio, plotted together with the swept frontier
curve. Recommends the Max Sharpe portfolio, with a written justification that
explicitly flags the recommendation's dependence on Task 3's forecast uncertainty.
The recommended weights are persisted to `task4_recommended_portfolio.json` for Task 5.

## Task 5 Strategy Backtesting

Simulates Task 4's recommended portfolio (both static-hold and monthly-rebalanced
variants) over a held-out 2025-2026 backtest window — deliberately the same boundary
as Task 2's test split, so this data was never used in any model training — against a
static 60% SPY / 40% BND benchmark simulated the same way. Compares cumulative
returns, total/annualized return, Sharpe ratio, and max drawdown, with an honest
conclusion section (not hardcoded to declare victory) plus explicit backtest
limitations: single time period, no transaction costs, static target weights within
the window, and an asset universe fixed by the assignment rather than systematically
selected.

## Notes on Data Availability

All data loading goes through real `yfinance` calls there is no synthetic-data
fallback in this codebase. If Yahoo Finance is unreachable (e.g. restricted network
egress in a sandboxed CI environment) or a ticker/date range is invalid, `load_prices()`
raises a `DataFetchError` describing exactly which tickers failed and why, rather than
silently substituting placeholder data.