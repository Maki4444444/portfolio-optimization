```markdown
# Time Series Forecasting for Portfolio Management Optimization

## Overview

This project builds a data-driven portfolio management workflow for **GMF Investments**, a
fictional financial advisory firm that uses time series forecasting to enhance portfolio
strategies. The engagement combines classical statistical forecasting (ARIMA/SARIMA) and
deep learning (LSTM) to predict asset price movements, then feeds those forecasts into
Modern Portfolio Theory optimization and a historical backtest, mirroring the kind of
end-to-end quantitative workflow a real asset management team would run before adjusting
client allocations.

## Business Need

GMF Investments aims to leverage data-driven insights to optimize portfolio management
strategies. As a Financial Analyst at GMF, the objective is to apply time series
forecasting to historical financial data in order to enhance portfolio management
strategies. The role involves predicting market trends, optimizing asset allocation, and
enhancing portfolio performance to ensure the achievement of the company's overarching
goal, maximizing returns while minimizing risks for its clients. GMF manages diversified
portfolios that leverage cutting-edge technology and data-driven insights to provide
clients with optimized investment strategies. By integrating advanced time series
forecasting models, GMF seeks to predict market trends, optimize asset allocation, and
enhance portfolio performance.

## Description

The core of this project is Modern Portfolio Theory (MPT), applied across three assets
chosen for their distinct risk/return roles:

| Ticker | Role | Description |
|---|---|---|
| **TSLA** | High-growth, high-risk | Tesla, Inc. high potential returns with high volatility exposure to consumer discretionary/automotive sector risk |
| **BND** | Stability | Vanguard Total Bond Market ETF, tracks U.S. investment-grade bonds, providing income and stability, low risk |
| **SPY** | Diversification | S&P 500 ETF, broad U.S. market exposure, moderate, diversified risk |

Using data from **2015-01-01 to 2026-06-30**, the project proceeds through five stages:

1. **Preprocess and explore the data** extract, clean, and analyze historical price data
   for all three assets, establishing the statistical properties (trend, volatility,
   stationarity, risk metrics) that inform every downstream modeling choice.
2. **Develop time series forecasting models** build and compare ARIMA/SARIMA and LSTM
   models to forecast TSLA's future stock price.
3. **Forecast future market trends** use the best-performing model from Task 2 to
   generate a forward-looking price forecast with confidence intervals, and interpret the
   trend and uncertainty it implies.
4. **Optimize portfolio based on forecast** combine the return forecast for TSLA with
   the historical performance of BND and SPY to construct an efficient frontier and
   recommend an optimal portfolio allocation.
5. **Backtest the strategy** simulate the recommended portfolio's performance over a
   recent historical period against a benchmark (e.g., a static 60/40 SPY/BND portfolio)
   to assess whether the strategy would have plausibly outperformed.
```