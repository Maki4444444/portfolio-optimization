"""
task1_eda.py
Task 1: Preprocess and Explore the Data.

Loads TSLA/BND/SPY, cleans, computes returns/rolling stats, runs ADF
stationarity tests, computes VaR & Sharpe, and saves EDA figures + a
processed combined dataset.
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from statsmodels.tsa.stattools import adfuller
import matplotlib

# Only force the non-interactive Agg backend when this module is run as a
# standalone script (e.g. `python src/task1_eda.py` in a headless CI/server
# environment). Calling matplotlib.use() unconditionally at import time would
# silently break inline figure display when this module is imported from a
# Jupyter notebook (matplotlib.use() switches the backend for the whole
# process, overriding Jupyter's inline capture even though no error is
# raised -- plots just stop appearing).
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
RISK_FREE_RATE = 0.04  # annualized, approx


def clean_and_merge(price_dict):
    """Clean each asset's frame, align on a common business-day index, ffill gaps."""
    if not price_dict:
        raise ValueError("clean_and_merge received an empty price_dict; nothing to clean.")

    frames = {}
    for t, df in price_dict.items():
        df = df.copy()
        df.index = pd.to_datetime(df.index)
        df = df[~df.index.duplicated(keep="first")].sort_index()
        # auto_adjust=True fetches (see data_loader) do not return 'Adj Close',
        # since 'Close' is already adjusted -- only coerce columns that exist.
        numeric_cols = [c for c in ["Open", "High", "Low", "Close", "Adj Close", "Volume"] if c in df.columns]
        if "Close" not in numeric_cols:
            raise ValueError(f"'{t}' is missing a required 'Close' column: {list(df.columns)}")
        for col in numeric_cols:
            df[col] = pd.to_numeric(df[col], errors="coerce")
        frames[t] = df

    full_idx = pd.bdate_range(
        start=min(f.index.min() for f in frames.values()),
        end=max(f.index.max() for f in frames.values()),
    )

    merged = {}
    quality_report = {}
    for t, df in frames.items():
        reindexed = df.reindex(full_idx)
        n_missing_before = reindexed["Close"].isna().sum()
        reindexed = reindexed.ffill().bfill()
        n_missing_after = reindexed["Close"].isna().sum()
        quality_report[t] = {
            "rows": len(reindexed),
            "missing_filled": int(n_missing_before),
            "missing_remaining": int(n_missing_after),
        }
        merged[t] = reindexed

    close = pd.DataFrame({t: merged[t]["Close"] for t in merged})
    return merged, close, quality_report


def compute_returns_and_vol(close):
    returns = close.pct_change().dropna()
    rolling_mean_30 = returns.rolling(30).mean()
    rolling_std_30 = returns.rolling(30).std()
    return returns, rolling_mean_30, rolling_std_30


def adf_test(series, name):
    series = series.dropna()
    result = adfuller(series)
    return {
        "series": name,
        "adf_statistic": float(result[0]),
        "p_value": float(result[1]),
        "n_lags": int(result[2]),
        "n_obs": int(result[3]),
        "critical_values": {k: float(v) for k, v in result[4].items()},
        "is_stationary_5pct": bool(result[1] < 0.05),
    }


def compute_risk_metrics(returns, confidence=0.95):
    metrics = {}
    for col in returns.columns:
        r = returns[col].dropna()
        var_hist = -np.percentile(r, (1 - confidence) * 100)
        ann_return = r.mean() * TRADING_DAYS
        ann_vol = r.std() * np.sqrt(TRADING_DAYS)
        sharpe = (ann_return - RISK_FREE_RATE) / ann_vol if ann_vol > 0 else np.nan
        metrics[col] = {
            "daily_VaR_95_hist": float(var_hist),
            "annualized_return": float(ann_return),
            "annualized_volatility": float(ann_vol),
            "sharpe_ratio": float(sharpe),
        }
    return metrics


def detect_outliers(returns, z_thresh=3.0):
    """Returns dict[ticker] -> pd.Series of flagged returns (sorted ascending)."""
    outliers = {}
    for col in returns.columns:
        r = returns[col].dropna()
        z = (r - r.mean()) / r.std()
        flagged = r[np.abs(z) > z_thresh].sort_values()
        outliers[col] = flagged
    return outliers


def make_plots(close, returns, rolling_std_30):
    # 1. Closing price over time
    fig, ax = plt.subplots(figsize=(11, 5))
    for col in close.columns:
        ax.plot(close.index, close[col], label=col, linewidth=1)
    ax.set_title("Closing Price Over Time (2015-2026)")
    ax.set_xlabel("Date"); ax.set_ylabel("Price ($)")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "01_closing_prices.png"), dpi=130)
    plt.close(fig)

    # 2. Daily % change
    fig, axes = plt.subplots(len(returns.columns), 1, figsize=(11, 7), sharex=True)
    for ax, col in zip(axes, returns.columns):
        ax.plot(returns.index, returns[col] * 100, linewidth=0.5, color="steelblue")
        ax.set_ylabel(f"{col} %")
        ax.axhline(0, color="black", linewidth=0.5)
    axes[0].set_title("Daily Percentage Change")
    fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "02_daily_returns.png"), dpi=130)
    plt.close(fig)

    # 3. Rolling volatility (30-day)
    fig, ax = plt.subplots(figsize=(11, 5))
    for col in rolling_std_30.columns:
        ax.plot(rolling_std_30.index, rolling_std_30[col] * 100, label=col, linewidth=1)
    ax.set_title("30-Day Rolling Volatility (Std of Daily Returns)")
    ax.set_xlabel("Date"); ax.set_ylabel("Rolling Std (%)")
    ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "03_rolling_volatility.png"), dpi=130)
    plt.close(fig)

    # 4. Return distribution
    fig, ax = plt.subplots(figsize=(9, 5))
    for col in returns.columns:
        ax.hist(returns[col], bins=100, alpha=0.5, label=col, density=True)
    ax.set_title("Distribution of Daily Returns")
    ax.set_xlabel("Daily Return"); ax.legend(); fig.tight_layout()
    fig.savefig(os.path.join(FIG_DIR, "04_return_distribution.png"), dpi=130)
    plt.close(fig)


def run():
    try:
        price_dict = load_prices(TICKERS, START_DATE, END_DATE)
    except DataFetchError as e:
        print(f"[task1_eda] Data fetch failed: {e}")
        raise

    merged, close, quality_report = clean_and_merge(price_dict)
    returns, rolling_mean_30, rolling_std_30 = compute_returns_and_vol(close)

    adf_results = {
        "TSLA_close": adf_test(close["TSLA"], "TSLA_close"),
        "TSLA_returns": adf_test(returns["TSLA"], "TSLA_returns"),
    }
    risk_metrics = compute_risk_metrics(returns)
    outliers = detect_outliers(returns)

    make_plots(close, returns, rolling_std_30)

    close.to_csv(os.path.join(PROCESSED_DIR, "close_prices.csv"))
    returns.to_csv(os.path.join(PROCESSED_DIR, "daily_returns.csv"))

    summary = {
        "data_quality": quality_report,
        "adf_tests": adf_results,
        "risk_metrics": risk_metrics,
        "n_outliers_flagged": {k: len(v) for k, v in outliers.items()},
    }
    with open(os.path.join(PROCESSED_DIR, "task1_summary.json"), "w") as f:
        json.dump(summary, f, indent=2, default=str)

    print(json.dumps(summary, indent=2, default=str))
    return summary


if __name__ == "__main__":
    run()