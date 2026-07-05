"""
Unit tests for src/task1_eda.py — cleaning/merging logic, ADF test wrapper,
and risk metric calculations, using small synthetic fixtures (not the
project's real market data).
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from task1_eda import clean_and_merge, adf_test, compute_risk_metrics, detect_outliers


def _price_frame(prices, start="2024-01-01"):
    idx = pd.bdate_range(start, periods=len(prices))
    return pd.DataFrame(
        {"Open": prices, "High": prices, "Low": prices, "Close": prices, "Volume": [1000] * len(prices)},
        index=idx,
    )


def test_clean_and_merge_raises_on_empty_input():
    with pytest.raises(ValueError):
        clean_and_merge({})


def test_clean_and_merge_raises_on_missing_close_column():
    bad_df = pd.DataFrame({"Open": [1, 2, 3]}, index=pd.bdate_range("2024-01-01", periods=3))
    with pytest.raises(ValueError, match="Close"):
        clean_and_merge({"BAD": bad_df})


def test_clean_and_merge_fills_gaps_and_reports_quality():
    a = _price_frame([100, 101, 102, 103, 104])
    b = _price_frame([50, 51, 52, 53, 54]).drop(pd.Timestamp("2024-01-03"))

    merged, close, quality = clean_and_merge({"A": a, "B": b})

    assert quality["B"]["missing_filled"] == 1
    assert quality["B"]["missing_remaining"] == 0
    assert close.isna().sum().sum() == 0
    assert list(close.columns) == ["A", "B"]


def test_clean_and_merge_works_without_adj_close_column():
    """Regression test: auto_adjust=True fetches omit 'Adj Close' entirely;
    cleaning must not assume that column exists."""
    df = _price_frame([10, 11, 12])
    assert "Adj Close" not in df.columns
    merged, close, _ = clean_and_merge({"X": df})
    assert not close.empty


def test_adf_test_flags_trending_series_as_non_stationary():
    trending = pd.Series(np.arange(500, dtype=float))
    result = adf_test(trending, "trend")
    assert result["is_stationary_5pct"] is False
    assert result["p_value"] > 0.05


def test_adf_test_flags_white_noise_as_stationary():
    rng = np.random.default_rng(0)
    noise = pd.Series(rng.normal(0, 1, size=1000))
    result = adf_test(noise, "noise")
    assert result["is_stationary_5pct"] is True
    assert result["p_value"] < 0.05


def test_compute_risk_metrics_known_values():
    # constant positive daily return -> zero volatility, VaR should be
    # the (negative of the) constant return itself, Sharpe undefined-safe
    returns = pd.DataFrame({"FLAT": [0.001] * 300})
    metrics = compute_risk_metrics(returns)
    assert metrics["FLAT"]["annualized_return"] == pytest.approx(0.001 * 252)
    assert metrics["FLAT"]["annualized_volatility"] == pytest.approx(0.0, abs=1e-9)


def test_detect_outliers_flags_extreme_value():
    rng = np.random.default_rng(1)
    values = list(rng.normal(0, 0.01, size=200))
    values[50] = 5.0  # extreme outlier
    returns = pd.DataFrame({"X": values})
    outliers = detect_outliers(returns, z_thresh=3.0)
    assert 5.0 in outliers["X"].values