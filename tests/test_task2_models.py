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