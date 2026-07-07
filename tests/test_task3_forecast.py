"""
Unit tests for src/task3_forecast.py — focused on the pure analysis/derivation
logic (trend direction, CI-width growth) which doesn't require training a real
model to test.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from task3_forecast import analyze_forecast, random_walk_baseline, shrinkage_weights


def _tsla_series(last_price=100.0, n=300):
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.Series(np.linspace(last_price * 0.5, last_price, n), index=idx)


def _forecast_df(n, forecast_values, lower_values, upper_values):
    idx = pd.bdate_range("2026-01-01", periods=n)
    return pd.DataFrame(
        {"forecast": forecast_values, "lower_95": lower_values, "upper_95": upper_values},
        index=idx,
    )


def test_analyze_forecast_detects_upward_trend():
    tsla = _tsla_series(last_price=100.0)
    n = 10
    forecast_values = np.linspace(105, 130, n)  # +30% by end
    lower = forecast_values - 5
    upper = forecast_values + 5
    fdf = _forecast_df(n, forecast_values, lower, upper)

    result = analyze_forecast(tsla, fdf, "LSTM")

    assert result["trend_direction"] == "upward"
    assert result["forecast_total_return_pct"] > 5


def test_analyze_forecast_detects_downward_trend():
    tsla = _tsla_series(last_price=100.0)
    n = 10
    forecast_values = np.linspace(95, 70, n)  # -30% by end
    lower = forecast_values - 5
    upper = forecast_values + 5
    fdf = _forecast_df(n, forecast_values, lower, upper)

    result = analyze_forecast(tsla, fdf, "ARIMA(1,1,1)")

    assert result["trend_direction"] == "downward"
    assert result["forecast_total_return_pct"] < -5


def test_analyze_forecast_detects_flat_trend():
    tsla = _tsla_series(last_price=100.0)
    n = 10
    forecast_values = np.full(n, 101.0)  # ~+1%, within the flat band
    lower = forecast_values - 2
    upper = forecast_values + 2
    fdf = _forecast_df(n, forecast_values, lower, upper)

    result = analyze_forecast(tsla, fdf, "LSTM")

    assert result["trend_direction"] == "roughly flat"


def test_analyze_forecast_ci_growth_factor_reflects_widening_interval():
    tsla = _tsla_series(last_price=100.0)
    n = 5
    forecast_values = np.full(n, 100.0)
    # CI widens linearly from +/-1 to +/-10
    half_widths = np.linspace(1, 10, n)
    lower = forecast_values - half_widths
    upper = forecast_values + half_widths
    fdf = _forecast_df(n, forecast_values, lower, upper)

    result = analyze_forecast(tsla, fdf, "LSTM")

    assert result["ci_width_day_final"] > result["ci_width_day_1"]
    assert result["ci_growth_factor"] == pytest.approx(10.0, rel=1e-6)


def test_analyze_forecast_handles_zero_initial_ci_width():
    """A degenerate zero-width starting CI shouldn't raise a ZeroDivisionError."""
    tsla = _tsla_series(last_price=100.0)
    n = 3
    forecast_values = np.full(n, 100.0)
    lower = forecast_values.copy()
    upper = forecast_values.copy()
    fdf = _forecast_df(n, forecast_values, lower, upper)

    result = analyze_forecast(tsla, fdf, "LSTM")
    assert np.isnan(result["ci_growth_factor"])


def test_random_walk_baseline_compounds_historical_drift():
    """A series with a known constant daily log-return should produce a
    baseline that compounds that exact drift forward."""
    n = 100
    daily_log_return = 0.001
    idx = pd.bdate_range("2024-01-01", periods=n)
    log_prices = np.cumsum(np.full(n, daily_log_return))
    prices = pd.Series(100 * np.exp(log_prices), index=idx)

    horizon = 10
    baseline_path, hist_drift, hist_vol = random_walk_baseline(prices, horizon)

    assert hist_drift == pytest.approx(daily_log_return, abs=1e-9)
    assert hist_vol == pytest.approx(0.0, abs=1e-9)
    expected_final = float(prices.iloc[-1]) * np.exp(daily_log_return * horizon)
    assert baseline_path[-1] == pytest.approx(expected_final, rel=1e-6)
    # monotonically increasing since drift is positive
    assert np.all(np.diff(baseline_path) > 0)


def test_shrinkage_weights_bounds_and_shape():
    horizon = 50
    w = shrinkage_weights(horizon, floor=0.2)

    assert len(w) == horizon
    assert w[0] == pytest.approx(1.0)
    assert w[-1] == pytest.approx(0.2)
    # monotonically non-increasing
    assert np.all(np.diff(w) <= 1e-12)
    assert np.all(w >= 0.2)


def test_shrinkage_weights_respects_floor_for_single_step_horizon():
    """horizon=1 shouldn't divide by zero (horizon - 1 == 0 edge case)."""
    w = shrinkage_weights(1, floor=0.3)
    assert len(w) == 1
    assert w[0] == pytest.approx(1.0)