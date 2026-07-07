"""
task4_portfolio.py
Task 4: Optimize Portfolio Based on Forecast using Modern Portfolio Theory.

Combines a forward-looking TSLA return view (derived from Task 3's forecast)
with historical average annualized returns for BND/SPY, builds the covariance
matrix from historical daily returns, and runs an efficient-frontier
optimization via PyPortfolioOpt.
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib

# See task1_eda.py for why this is conditional on __main__ rather than
# unconditional at import time -- forcing Agg here would silently break
# inline plotting in any notebook that imports this module.
if __name__ == "__main__":
    matplotlib.use("Agg")
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(__file__))
from data_loader import load_prices, DataFetchError, TICKERS, START_DATE, END_DATE

FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
os.makedirs(FIG_DIR, exist_ok=True)
os.makedirs(PROCESSED_DIR, exist_ok=True)

TRADING_DAYS = 252
ASSET_ORDER = ["TSLA", "BND", "SPY"]


def load_close_and_returns():
    """Load cleaned close prices (from Task 1) and derive daily returns for
    all three assets. Falls back to a fresh fetch via data_loader if Task 1's
    processed CSV isn't present."""
    processed_path = os.path.join(PROCESSED_DIR, "close_prices.csv")
    if os.path.exists(processed_path):
        close = pd.read_csv(processed_path, index_col=0, parse_dates=True)
    else:
        try:
            price_dict = load_prices(TICKERS, START_DATE, END_DATE, use_cache=True)
        except DataFetchError as e:
            print(f"[task4_portfolio] Data fetch failed: {e}")
            raise
        close = pd.DataFrame({t: df["Close"] for t, df in price_dict.items()})

    close.index = pd.to_datetime(close.index)
    returns = close.pct_change().dropna()
    return close, returns


def tsla_expected_return_from_forecast(forecast_df, last_actual_price):
    """
    Derives TSLA's annualized expected return from a Task 3 forecast_df
    (must have a 'forecast' column), by annualizing the total compounded
    return implied by the forecast's endpoint relative to the last actual
    price, over the forecast's horizon in trading days.
    """
    horizon_days = len(forecast_df)
    end_price = float(forecast_df["forecast"].iloc[-1])
    total_return = end_price / last_actual_price - 1
    annualized_return = (1 + total_return) ** (TRADING_DAYS / horizon_days) - 1
    return float(annualized_return)


def historical_annualized_return(returns, ticker):
    """Historical average daily return, annualized via compounding."""
    daily_mean = returns[ticker].mean()
    return float((1 + daily_mean) ** TRADING_DAYS - 1)


def build_expected_returns(returns, tsla_annual_return):
    """
    Returns a pandas Series of annualized expected returns for TSLA, BND, SPY:
    TSLA uses the forecast-derived view (the "analyst has a specific view on
    one asset" simulation called for in the task); BND/SPY use historical
    averages.
    """
    mu = pd.Series({
        "TSLA": tsla_annual_return,
        "BND": historical_annualized_return(returns, "BND"),
        "SPY": historical_annualized_return(returns, "SPY"),
    })
    return mu[ASSET_ORDER]


def annualized_covariance_matrix(returns):
    return returns[ASSET_ORDER].cov() * TRADING_DAYS


def run_efficient_frontier(mu, cov, risk_free_rate=0.04, n_frontier_points=40):
    """
    Returns a dict with the Max Sharpe portfolio, Min Volatility portfolio,
    and a swept set of (volatility, return) points along the efficient
    frontier for plotting, all via PyPortfolioOpt's EfficientFrontier.

    A fresh EfficientFrontier instance is created for each optimization call
    (max_sharpe, min_volatility, each efficient_return sweep point) because
    PyPortfolioOpt's optimizer objects are single-use -- calling a second
    optimization method on an already-solved instance raises an error.
    """
    from pypfopt import EfficientFrontier

    ef_sharpe = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
    ef_sharpe.max_sharpe(risk_free_rate=risk_free_rate)
    max_sharpe_weights = ef_sharpe.clean_weights()
    max_sharpe_perf = ef_sharpe.portfolio_performance(risk_free_rate=risk_free_rate)

    ef_minvol = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
    ef_minvol.min_volatility()
    min_vol_weights = ef_minvol.clean_weights()
    min_vol_perf = ef_minvol.portfolio_performance(risk_free_rate=risk_free_rate)

    ret_min = min_vol_perf[0]
    ret_max = float(mu.max())
    target_returns = np.linspace(ret_min, ret_max * 0.999, n_frontier_points)

    frontier_vols, frontier_rets = [], []
    for target in target_returns:
        try:
            ef_i = EfficientFrontier(mu, cov, weight_bounds=(0, 1))
            ef_i.efficient_return(target)
            ret_i, vol_i, _ = ef_i.portfolio_performance(risk_free_rate=risk_free_rate)
            frontier_rets.append(ret_i)
            frontier_vols.append(vol_i)
        except Exception:
            continue  # infeasible target return, skip

    return {
        "max_sharpe_weights": max_sharpe_weights,
        "max_sharpe_perf": max_sharpe_perf,   # (return, volatility, sharpe)
        "min_vol_weights": min_vol_weights,
        "min_vol_perf": min_vol_perf,
        "frontier_vols": frontier_vols,
        "frontier_rets": frontier_rets,
    }


def plot_efficient_frontier(result, out_path=None):
    fig, ax = plt.subplots(figsize=(9, 6))
    ax.plot(result["frontier_vols"], result["frontier_rets"], color="steelblue",
            linewidth=2, label="Efficient Frontier")

    ms_ret, ms_vol, ms_sharpe = result["max_sharpe_perf"]
    mv_ret, mv_vol, mv_sharpe = result["min_vol_perf"]

    ax.scatter([ms_vol], [ms_ret], color="crimson", marker="*", s=350, zorder=5,
               label=f"Max Sharpe ({ms_sharpe:.2f})")
    ax.scatter([mv_vol], [mv_ret], color="darkgreen", marker="*", s=350, zorder=5,
               label=f"Min Volatility (Sharpe {mv_sharpe:.2f})")

    ax.set_xlabel("Volatility (Annualized Std. Dev.)")
    ax.set_ylabel("Expected Annual Return")
    ax.set_title("Efficient Frontier: TSLA / BND / SPY")
    ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
    return fig


def plot_covariance_heatmap(cov, out_path=None):
    import seaborn as sns
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cov, annot=True, fmt=".4f", cmap="coolwarm", ax=ax, square=True)
    ax.set_title("Annualized Covariance Matrix")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
    return fig


def run():
    close, returns = load_close_and_returns()

    forecast_path = os.path.join(PROCESSED_DIR, "task3_future_forecast.csv")
    if os.path.exists(forecast_path):
        forecast_df = pd.read_csv(forecast_path, index_col=0, parse_dates=True)
        last_actual_price = float(close["TSLA"].iloc[-1])
        tsla_annual_return = tsla_expected_return_from_forecast(forecast_df, last_actual_price)
    else:
        print("[task4_portfolio] No Task 3 forecast found; using TSLA's historical "
              "annualized return as a fallback expected return.")
        tsla_annual_return = historical_annualized_return(returns, "TSLA")

    mu = build_expected_returns(returns, tsla_annual_return)
    cov = annualized_covariance_matrix(returns)

    result = run_efficient_frontier(mu, cov)

    plot_efficient_frontier(result, out_path=os.path.join(FIG_DIR, "07_efficient_frontier.png"))
    plot_covariance_heatmap(cov, out_path=os.path.join(FIG_DIR, "08_covariance_heatmap.png"))

    print("Expected annualized returns:")
    print(mu)
    print("\nMax Sharpe weights:", result["max_sharpe_weights"])
    print("Max Sharpe performance (return, vol, sharpe):", result["max_sharpe_perf"])
    print("\nMin Volatility weights:", result["min_vol_weights"])
    print("Min Volatility performance (return, vol, sharpe):", result["min_vol_perf"])

    return mu, cov, result


if __name__ == "__main__":
    run()