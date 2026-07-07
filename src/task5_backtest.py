"""
task5_backtest.py
Task 5: Strategy Backtesting -- validate the Task 4 optimal portfolio against
a static 60% SPY / 40% BND benchmark over a held-out backtest window.
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

TRADING_DAYS = 252
BACKTEST_START = "2025-01-01"  # held-out period -- matches Task 2's test split,
                                # so no backtest data was used to train any model
BENCHMARK_WEIGHTS = {"TSLA": 0.0, "BND": 0.4, "SPY": 0.6}


def get_backtest_returns(returns, start=BACKTEST_START):
    """Slice out the held-out backtest window (not used in any model training)."""
    return returns[returns.index >= start]


def simulate_static_hold(returns, weights):
    """
    Simulate a portfolio that holds fixed `weights` for the whole period
    (buy-and-hold, no rebalancing -- weights drift with performance but are
    never reset). Returns a Series of daily portfolio returns.
    """
    w = np.array([weights[t] for t in returns.columns])
    return returns.dot(w)


def simulate_monthly_rebalanced(returns, weights):
    """
    Simulate a portfolio rebalanced back to target `weights` at the start of
    each calendar month. Returns a Series of daily portfolio returns.

    Within a month, weights drift with each asset's daily performance (since
    we don't re-buy daily); at the first trading day of each new month,
    weights are reset back to target before that day's return is computed.
    """
    w_target = np.array([weights[t] for t in returns.columns])
    months = returns.index.to_period("M")

    daily_portfolio_returns = []
    current_weights = w_target.copy()
    current_month = None

    for date, month in zip(returns.index, months):
        if month != current_month:
            current_weights = w_target.copy()
            current_month = month

        asset_returns = returns.loc[date].values
        port_return = float(np.dot(current_weights, asset_returns))
        daily_portfolio_returns.append(port_return)

        # drift weights forward with today's performance, ready for tomorrow
        grown = current_weights * (1 + asset_returns)
        current_weights = grown / grown.sum()

    return pd.Series(daily_portfolio_returns, index=returns.index)


def cumulative_returns(daily_returns):
    return (1 + daily_returns).cumprod() - 1


def performance_metrics(daily_returns, risk_free_rate=0.04):
    """Total return, annualized return, annualized volatility, Sharpe ratio,
    and max drawdown for a daily-return series."""
    cumulative = (1 + daily_returns).cumprod()
    total_return = float(cumulative.iloc[-1] - 1)

    n_days = len(daily_returns)
    annualized_return = float((1 + total_return) ** (TRADING_DAYS / n_days) - 1)
    annualized_vol = float(daily_returns.std() * np.sqrt(TRADING_DAYS))
    sharpe = (annualized_return - risk_free_rate) / annualized_vol if annualized_vol > 0 else float("nan")

    running_max = cumulative.cummax()
    drawdown = cumulative / running_max - 1
    max_drawdown = float(drawdown.min())

    return {
        "total_return": total_return,
        "annualized_return": annualized_return,
        "annualized_volatility": annualized_vol,
        "sharpe_ratio": sharpe,
        "max_drawdown": max_drawdown,
    }


def plot_cumulative_comparison(strategy_returns, benchmark_returns,
                                strategy_label="Strategy", out_path=None):
    fig, ax = plt.subplots(figsize=(12, 5.5))
    ax.plot(strategy_returns.index, cumulative_returns(strategy_returns) * 100,
            label=strategy_label, color="tab:blue", linewidth=1.6)
    ax.plot(benchmark_returns.index, cumulative_returns(benchmark_returns) * 100,
            label="Benchmark (60% SPY / 40% BND)", color="gray", linestyle="--", linewidth=1.4)
    ax.axhline(0, color="black", linewidth=0.5)
    ax.set_title("Cumulative Returns: Strategy vs. Benchmark")
    ax.set_xlabel("Date"); ax.set_ylabel("Cumulative Return (%)")
    ax.legend()
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=130)
        plt.close(fig)
    return fig


def run(strategy_weights, rebalance="static"):
    """
    strategy_weights: dict like {"TSLA": w, "BND": w, "SPY": w} -- typically
    the Task 4 Max Sharpe (or Min Volatility) portfolio weights.
    rebalance: "static" (buy-and-hold) or "monthly".
    """
    from task4_portfolio import load_close_and_returns

    _, returns = load_close_and_returns()
    backtest_returns = get_backtest_returns(returns)

    sim_fn = simulate_monthly_rebalanced if rebalance == "monthly" else simulate_static_hold

    strategy_daily = sim_fn(backtest_returns, strategy_weights)
    benchmark_daily = sim_fn(backtest_returns, BENCHMARK_WEIGHTS)

    strategy_metrics = performance_metrics(strategy_daily)
    benchmark_metrics = performance_metrics(benchmark_daily)

    FIG_DIR = os.path.join(os.path.dirname(__file__), "..", "reports", "figures")
    os.makedirs(FIG_DIR, exist_ok=True)
    plot_cumulative_comparison(strategy_daily, benchmark_daily,
                                out_path=os.path.join(FIG_DIR, "09_backtest_comparison.png"))

    print("Strategy performance:", strategy_metrics)
    print("Benchmark performance:", benchmark_metrics)

    return strategy_metrics, benchmark_metrics


if __name__ == "__main__":
    # Example: equal-weight fallback if run standalone without Task 4 output
    run({"TSLA": 1/3, "BND": 1/3, "SPY": 1/3}, rebalance="static")