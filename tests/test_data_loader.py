"""
Unit tests for src/data_loader.py — focused on the error-handling paths
(API failures, empty responses, missing tickers) called out in the code
quality rubric, plus basic cache round-tripping.
"""

import os
import sys
import pandas as pd
import pytest
from unittest.mock import patch

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from data_loader import load_prices, _fetch_live, DataFetchError


def _make_ohlcv(n=5):
    idx = pd.bdate_range("2024-01-01", periods=n)
    return pd.DataFrame(
        {
            "Open": range(n), "High": range(n), "Low": range(n),
            "Close": range(n), "Volume": range(n),
        },
        index=idx,
    )


def test_fetch_live_handles_yfinance_exception():
    """A raised exception from yf.download must not propagate -- should
    be caught and result in an empty dict so the caller can decide what
    to do next."""
    with patch("data_loader.yf.download", side_effect=ConnectionError("network down")):
        result = _fetch_live(["TSLA"], "2015-01-01", "2020-01-01")
    assert result == {}


def test_fetch_live_handles_empty_response():
    """An empty (but non-exception) response from yfinance should also
    result in an empty dict, not a crash."""
    with patch("data_loader.yf.download", return_value=pd.DataFrame()):
        result = _fetch_live(["TSLA"], "2015-01-01", "2020-01-01")
    assert result == {}


def test_load_prices_raises_when_all_sources_fail(tmp_path, monkeypatch):
    """With no cache and a failing live fetch, load_prices must raise a
    DataFetchError naming the tickers that failed (not silently return
    partial/fake data)."""
    monkeypatch.setattr("data_loader.RAW_DIR", str(tmp_path))
    with patch("data_loader.yf.download", side_effect=ConnectionError("network down")):
        with pytest.raises(DataFetchError, match="TSLA"):
            load_prices(["TSLA"], "2015-01-01", "2020-01-01", use_cache=False)


def test_load_prices_uses_cache_when_present(tmp_path, monkeypatch):
    """If a cached CSV already exists for a ticker, load_prices should use
    it without calling yfinance at all."""
    monkeypatch.setattr("data_loader.RAW_DIR", str(tmp_path))
    cached = _make_ohlcv()
    cached.to_csv(os.path.join(str(tmp_path), "TSLA.csv"))

    with patch("data_loader.yf.download") as mock_download:
        result = load_prices(["TSLA"], "2015-01-01", "2020-01-01", use_cache=True)
        mock_download.assert_not_called()

    assert "TSLA" in result
    assert len(result["TSLA"]) == len(cached)


def test_load_prices_falls_back_to_live_fetch_on_corrupt_cache(tmp_path, monkeypatch):
    """A corrupt/empty cache file should not crash load_prices -- it
    should be treated as missing and re-fetched."""
    monkeypatch.setattr("data_loader.RAW_DIR", str(tmp_path))
    # Write an empty file where a real cache would be
    open(os.path.join(str(tmp_path), "TSLA.csv"), "w").close()

    fresh = _make_ohlcv()
    with patch("data_loader._fetch_live", return_value={"TSLA": fresh}):
        result = load_prices(["TSLA"], "2015-01-01", "2020-01-01", use_cache=True)

    assert "TSLA" in result
    assert len(result["TSLA"]) == len(fresh)