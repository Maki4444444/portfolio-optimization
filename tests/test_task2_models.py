"""
Unit tests for src/task2_models.py — chronological split correctness and
the MAPE metric helper.
"""

import os
import sys
import pandas as pd
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from task2_models import chrono_split, mape, make_sequences
import numpy as np


def test_chrono_split_respects_boundary_and_order():
    idx = pd.bdate_range("2020-01-01", "2020-12-31")
    series = pd.Series(range(len(idx)), index=idx)

    train, test = chrono_split(series, train_end="2020-06-30")

    assert train.index.max() <= pd.Timestamp("2020-06-30")
    assert test.index.min() > pd.Timestamp("2020-06-30")
    assert len(train) + len(test) == len(series)


def test_mape_zero_error_is_zero():
    y = np.array([100.0, 200.0, 300.0])
    assert mape(y, y) == pytest.approx(0.0)


def test_mape_known_value():
    y_true = np.array([100.0])
    y_pred = np.array([110.0])
    assert mape(y_true, y_pred) == pytest.approx(10.0)


def test_make_sequences_shapes():
    values = np.arange(20).reshape(-1, 1).astype(float)
    X, y = make_sequences(values, window=5)
    assert X.shape == (15, 5)
    assert y.shape == (15,)
    # first sequence should be values[0:5], target values[5]
    assert (X[0] == values[0:5, 0]).all()
    assert y[0] == values[5, 0]


def test_fit_lstm_returns_based_prediction_alignment_and_scale():
    """
    fit_lstm predicts next-day LOG RETURNS internally (not raw price), then
    reconstructs price predictions using the true previous close. This test
    checks the plumbing (output length matches test length, predictions land
    in a sane price range) without asserting on point-accuracy, since a
    handful of training epochs on a short synthetic series won't be precise.
    """
    from task2_models import fit_lstm

    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2020-01-01", periods=400)
    # a mildly-trending, noisy synthetic price series -- long enough to
    # exceed the 60-day window with room to spare on both sides
    log_walk = np.cumsum(rng.normal(0.0005, 0.02, size=400))
    prices = pd.Series(100 * np.exp(log_walk), index=idx)

    train, test = prices.iloc[:340], prices.iloc[340:]
    _, _, preds = fit_lstm(train, test, window=60, epochs=2, batch_size=64)

    assert len(preds) == len(test)
    assert np.all(preds > 0)
    # predictions should be in the same broad ballpark as the actual prices,
    # not off by orders of magnitude (which would indicate a reconstruction bug)
    assert preds.min() > test.values.min() * 0.2
    assert preds.max() < test.values.max() * 5