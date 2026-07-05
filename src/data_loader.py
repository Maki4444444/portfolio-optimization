"""
data_loader.py
Fetches historical OHLCV data for TSLA, BND, SPY via yfinance, with local
CSV caching and explicit error handling for common failure modes (network
errors, empty/partial responses, missing tickers).
"""

import os
import logging
import pandas as pd
import yfinance as yf

logger = logging.getLogger(__name__)

TICKERS = ["TSLA", "BND", "SPY"]
START_DATE = "2015-01-01"
END_DATE = "2026-06-30"

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")


class DataFetchError(RuntimeError):
    """Raised when required price data cannot be obtained from any source."""


def _fetch_live(tickers, start, end):
    """
    Attempt a real yfinance download.

    Returns dict[ticker] -> DataFrame for tickers that returned usable rows.
    Never raises: network/API errors are logged and result in an empty dict
    (or a partial one), letting the caller decide how to handle gaps.
    """
    out = {}
    try:
        raw = yf.download(
            tickers, start=start, end=end, group_by="ticker",
            auto_adjust=True, progress=False,
        )
    except Exception as e:
        logger.warning("yfinance download failed for %s: %s", tickers, e)
        return out

    if raw is None or raw.empty:
        logger.warning("yfinance returned an empty response for %s", tickers)
        return out

    for t in tickers:
        try:
            df = raw[t].copy() if len(tickers) > 1 else raw.copy()
            df = df.dropna(how="all")
            if len(df) > 0:
                out[t] = df
            else:
                logger.warning("No usable rows returned for ticker '%s'", t)
        except KeyError:
            logger.warning("Ticker '%s' missing from yfinance response", t)
        except Exception as e:
            logger.warning("Unexpected error extracting ticker '%s': %s", t, e)

    return out


def load_prices(tickers=TICKERS, start=START_DATE, end=END_DATE, use_cache=True):
    """
    Returns dict[ticker] -> DataFrame with OHLCV columns, indexed by Date.

    Resolution order per ticker: 1) local cache (data/raw/<ticker>.csv),
    2) live yfinance fetch. Successfully fetched tickers are cached to disk
    for reuse.

    Raises:
        DataFetchError: if one or more requested tickers could not be
            obtained from either the cache or a live fetch. The exception
            message lists exactly which tickers failed, so the caller can
            decide whether a partial result is acceptable.
    """
    os.makedirs(RAW_DIR, exist_ok=True)
    result = {}
    missing = []

    if use_cache:
        for t in tickers:
            path = os.path.join(RAW_DIR, f"{t}.csv")
            if os.path.exists(path):
                try:
                    df = pd.read_csv(path, index_col=0, parse_dates=True)
                    if df.empty:
                        raise ValueError("cached file is empty")
                    result[t] = df
                except Exception as e:
                    logger.warning("Failed to read cache for '%s' (%s); will re-fetch", t, e)
                    missing.append(t)
            else:
                missing.append(t)
    else:
        missing = list(tickers)

    if missing:
        live = _fetch_live(missing, start, end)
        for t, df in live.items():
            result[t] = df
            df.to_csv(os.path.join(RAW_DIR, f"{t}.csv"))

        still_missing = [t for t in missing if t not in live]
        if still_missing:
            raise DataFetchError(
                f"Could not obtain data for: {still_missing}. "
                "Check network connectivity to Yahoo Finance "
                "(query1/query2.finance.yahoo.com), verify the ticker "
                "symbols, and confirm the date range is valid."
            )

    return {t: result[t] for t in tickers if t in result}


if __name__ == "__main__":
    try:
        data = load_prices(use_cache=False)
    except DataFetchError as e:
        logger.error("Data load failed: %s", e)
        raise
    for t, df in data.items():
        print(t, df.shape, df.index.min(), df.index.max())