"""
Unit tests for src/task5_backtest.py — backtest window slicing, static and
monthly-rebalanced simulation, and performance metric calculations.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from task5_backtest import (
    get_backtest_returns,
    simulate_static_hold,
    simulate_monthly_rebalanced,
    performance_metrics,
    cumulative_returns,
    TRADING_DAYS,
)


def test_get_backtest_returns_filters_by_date():
    idx = pd.bdate_range("2024-06-01", "2025-06-30")
    returns = pd.DataFrame({"TSLA": 0.001, "BND": 0.0001, "SPY": 0.0005}, index=idx)

    backtest = get_backtest_returns(returns, start="2025-01-01")

    assert backtest.index.min() >= pd.Timestamp("2025-01-01")
    assert backtest.index.max() == idx.max()


def test_simulate_static_hold_matches_weighted_sum_on_first_day():
    idx = pd.bdate_range("2025-01-01", periods=5)
    returns = pd.DataFrame({
        "TSLA": [0.01, -0.02, 0.03, 0.0, 0.01],
        "BND": [0.001, 0.001, -0.001, 0.0, 0.002],
        "SPY": [0.005, -0.005, 0.01, 0.002, 0.0],
    }, index=idx)
    weights = {"TSLA": 0.5, "BND": 0.2, "SPY": 0.3}

    result = simulate_static_hold(returns, weights)

    expected_day0 = 0.5 * 0.01 + 0.2 * 0.001 + 0.3 * 0.005
    assert result.iloc[0] == pytest.approx(expected_day0)
    assert len(result) == len(returns)


def test_simulate_monthly_rebalanced_resets_weights_at_new_month():
    """A large TSLA move in January should distort the drifted weights, but
    the first trading day of February should use the ORIGINAL target
    weights again, not the drifted ones."""
    idx = pd.to_datetime(["2025-01-30", "2025-01-31", "2025-02-03", "2025-02-04"])
    returns = pd.DataFrame({
        "TSLA": [0.50, 0.00, 0.20, 0.00],  # big Jan-30 rally distorts weights
        "BND": [0.0, 0.0, 0.0, 0.0],
        "SPY": [0.0, 0.0, 0.0, 0.0],
    }, index=idx)
    weights = {"TSLA": 0.5, "BND": 0.3, "SPY": 0.2}

    result = simulate_monthly_rebalanced(returns, weights)

    # Feb 3rd is a new month -> weights reset to {0.5, 0.3, 0.2} BEFORE that
    # day's return is computed, so day-3 return should be exactly 0.5 * 0.20,
    # not (drifted TSLA weight, which would be ~0.6) * 0.20
    assert result.iloc[2] == pytest.approx(0.5 * 0.20, rel=1e-6)
    assert len(result) == 4


def test_simulate_monthly_rebalanced_matches_static_within_first_month():
    """Before any month boundary is crossed, monthly rebalancing and static
    hold should be identical (both start from and use the target weights on
    day one)."""
    idx = pd.bdate_range("2025-01-01", periods=5)
    returns = pd.DataFrame({
        "TSLA": [0.01, -0.02, 0.03, 0.0, 0.01],
        "BND": [0.001, 0.001, -0.001, 0.0, 0.002],
        "SPY": [0.005, -0.005, 0.01, 0.002, 0.0],
    }, index=idx)
    weights = {"TSLA": 0.5, "BND": 0.2, "SPY": 0.3}

    static = simulate_static_hold(returns, weights)
    monthly = simulate_monthly_rebalanced(returns, weights)

    assert static.iloc[0] == pytest.approx(monthly.iloc[0], rel=1e-6)


def test_performance_metrics_constant_positive_return_has_zero_drawdown():
    idx = pd.bdate_range("2025-01-01", periods=TRADING_DAYS)
    daily_returns = pd.Series([0.001] * TRADING_DAYS, index=idx)

    metrics = performance_metrics(daily_returns, risk_free_rate=0.0)

    expected_total = (1.001) ** TRADING_DAYS - 1
    assert metrics["total_return"] == pytest.approx(expected_total, rel=1e-6)
    assert metrics["annualized_volatility"] == pytest.approx(0.0, abs=1e-9)
    assert metrics["max_drawdown"] == pytest.approx(0.0, abs=1e-9)


def test_performance_metrics_detects_known_drawdown():
    idx = pd.bdate_range("2025-01-01", periods=5)
    # cumulative wealth: 1.10, 0.88, 0.88, 0.88, 0.88 -> drawdown from peak = -20%
    daily_returns = pd.Series([0.10, -0.20, 0.0, 0.0, 0.0], index=idx)

    metrics = performance_metrics(daily_returns)

    assert metrics["max_drawdown"] == pytest.approx(-0.20, rel=1e-6)


def test_cumulative_returns_compounds_correctly():
    idx = pd.bdate_range("2025-01-01", periods=3)
    daily_returns = pd.Series([0.10, 0.10, -0.10], index=idx)

    cum = cumulative_returns(daily_returns)

    expected_final = 1.10 * 1.10 * 0.90 - 1
    assert cum.iloc[-1] == pytest.approx(expected_final, rel=1e-6)


def test_performance_metrics_sharpe_sign_matches_excess_return():
    idx = pd.bdate_range("2025-01-01", periods=252)
    # Return well above a 4% risk-free rate -> Sharpe should be positive
    good_returns = pd.Series([0.002] * 252, index=idx)
    good_metrics = performance_metrics(good_returns, risk_free_rate=0.04)
    assert good_metrics["sharpe_ratio"] > 0

    # Return below the risk-free rate -> Sharpe should be negative
    poor_returns = pd.Series([0.00005] * 252, index=idx)
    poor_metrics = performance_metrics(poor_returns, risk_free_rate=0.04)
    assert poor_metrics["sharpe_ratio"] < 0