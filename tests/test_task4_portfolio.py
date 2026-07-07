"""
Unit tests for src/task4_portfolio.py — expected return derivation, covariance
computation, and efficient frontier optimization.
"""

import os
import sys
import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from task4_portfolio import (
    tsla_expected_return_from_forecast,
    historical_annualized_return,
    build_expected_returns,
    annualized_covariance_matrix,
    run_efficient_frontier,
    TRADING_DAYS,
)


def test_tsla_expected_return_from_forecast_matches_cagr():
    idx = pd.bdate_range("2026-01-01", periods=126)  # ~6 month horizon
    forecast_df = pd.DataFrame({"forecast": np.linspace(100, 121, 126)}, index=idx)

    ann_return = tsla_expected_return_from_forecast(forecast_df, last_actual_price=100.0)

    expected = (1.21) ** (TRADING_DAYS / 126) - 1
    assert ann_return == pytest.approx(expected, rel=1e-6)


def test_historical_annualized_return_known_value():
    returns = pd.DataFrame({"BND": [0.001] * 300})
    result = historical_annualized_return(returns, "BND")
    expected = (1.001) ** TRADING_DAYS - 1
    assert result == pytest.approx(expected, rel=1e-6)


def test_build_expected_returns_uses_tsla_override_and_orders_assets():
    returns = pd.DataFrame({
        "TSLA": [0.002] * 300,
        "BND": [0.0002] * 300,
        "SPY": [0.0006] * 300,
    })
    mu = build_expected_returns(returns, tsla_annual_return=0.5)

    assert mu["TSLA"] == pytest.approx(0.5)
    assert mu["BND"] == pytest.approx((1.0002) ** TRADING_DAYS - 1, rel=1e-6)
    assert mu["SPY"] == pytest.approx((1.0006) ** TRADING_DAYS - 1, rel=1e-6)
    assert list(mu.index) == ["TSLA", "BND", "SPY"]


def test_annualized_covariance_matrix_shape_symmetry_and_positive_diagonal():
    rng = np.random.default_rng(0)
    returns = pd.DataFrame({
        "TSLA": rng.normal(0, 0.02, 500),
        "BND": rng.normal(0, 0.005, 500),
        "SPY": rng.normal(0, 0.01, 500),
    })
    cov = annualized_covariance_matrix(returns)

    assert cov.shape == (3, 3)
    assert list(cov.columns) == ["TSLA", "BND", "SPY"]
    assert np.allclose(cov.values, cov.values.T)
    assert (np.diag(cov.values) > 0).all()
    # annualized variance should be ~ daily_var * 252
    assert cov.loc["TSLA", "TSLA"] == pytest.approx(returns["TSLA"].var() * TRADING_DAYS, rel=1e-6)


def _toy_mu_and_cov():
    rng = np.random.default_rng(1)
    returns = pd.DataFrame({
        "TSLA": rng.normal(0.0015, 0.03, 1000),
        "BND": rng.normal(0.0001, 0.005, 1000),
        "SPY": rng.normal(0.0005, 0.012, 1000),
    })
    mu = pd.Series({
        "TSLA": 0.40,
        "BND": historical_annualized_return(returns, "BND"),
        "SPY": historical_annualized_return(returns, "SPY"),
    })[["TSLA", "BND", "SPY"]]
    cov = annualized_covariance_matrix(returns)
    return mu, cov


def test_run_efficient_frontier_weights_sum_to_one():
    mu, cov = _toy_mu_and_cov()
    result = run_efficient_frontier(mu, cov, risk_free_rate=0.04)

    assert sum(result["max_sharpe_weights"].values()) == pytest.approx(1.0, abs=1e-3)
    assert sum(result["min_vol_weights"].values()) == pytest.approx(1.0, abs=1e-3)


def test_run_efficient_frontier_min_vol_has_lower_or_equal_risk_than_max_sharpe():
    mu, cov = _toy_mu_and_cov()
    result = run_efficient_frontier(mu, cov, risk_free_rate=0.04)

    _, ms_vol, _ = result["max_sharpe_perf"]
    _, mv_vol, _ = result["min_vol_perf"]
    assert mv_vol <= ms_vol + 1e-6


def test_run_efficient_frontier_weights_are_nonnegative_and_bounded():
    mu, cov = _toy_mu_and_cov()
    result = run_efficient_frontier(mu, cov, risk_free_rate=0.04)

    for w in result["max_sharpe_weights"].values():
        assert -1e-6 <= w <= 1.0 + 1e-6
    for w in result["min_vol_weights"].values():
        assert -1e-6 <= w <= 1.0 + 1e-6


def test_run_efficient_frontier_produces_frontier_points():
    mu, cov = _toy_mu_and_cov()
    result = run_efficient_frontier(mu, cov, risk_free_rate=0.04, n_frontier_points=15)

    assert len(result["frontier_vols"]) > 0
    assert len(result["frontier_vols"]) == len(result["frontier_rets"])
    # frontier should be roughly increasing in return as volatility increases
    assert result["frontier_rets"][-1] >= result["frontier_rets"][0]