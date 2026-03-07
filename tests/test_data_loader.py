"""Unit tests for data_loader.py — windowing logic (no network calls)."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import patch

from neat_trader.data_loader import get_random_window, download_all_stocks


# ── Helpers ───────────────────────────────────────────────────────────────────

def make_df(n: int = 500, seed: int = 0) -> pd.DataFrame:
    rng = np.random.default_rng(seed)
    close = 100 + np.cumsum(rng.normal(0, 0.5, n))
    return pd.DataFrame({
        "Open":   close,
        "High":   close * 1.01,
        "Low":    close * 0.99,
        "Close":  close,
        "Volume": np.ones(n) * 1_000_000.0,
    })


def make_stocks(n: int = 500) -> dict:
    return {"AAPL": make_df(n)}


# ── get_random_window() ───────────────────────────────────────────────────────

def test_window_length_equals_days_plus_warmup():
    stocks = make_stocks(500)
    df, _ = get_random_window(stocks, window_days=90, warmup=40)
    assert df is not None
    assert len(df) == 90 + 40


def test_window_returns_known_ticker():
    stocks = make_stocks(500)
    _, ticker = get_random_window(stocks, window_days=90)
    assert ticker == "AAPL"


def test_window_returns_dataframe():
    stocks = make_stocks(500)
    df, _ = get_random_window(stocks, window_days=90)
    assert isinstance(df, pd.DataFrame)


def test_insufficient_data_returns_none():
    """Stocks shorter than window+warmup+margin must return (None, None)."""
    stocks = {"TINY": make_df(50)}
    df, ticker = get_random_window(stocks, window_days=90, warmup=40)
    assert df is None
    assert ticker is None


def test_window_is_contiguous_slice():
    """Values in the returned window must match a contiguous slice of the source."""
    stocks = make_stocks(500)
    df, ticker = get_random_window(stocks, window_days=90, warmup=40)
    assert df is not None
    source = stocks[ticker]["Close"].values
    window = df["Close"].values
    # Check that the window appears somewhere in the source array
    found = any(
        np.allclose(source[i : i + len(window)], window)
        for i in range(len(source) - len(window) + 1)
    )
    assert found


def test_default_warmup_is_40():
    """Default warmup matches TradingEnvironment.WARMUP."""
    stocks = make_stocks(500)
    df, _ = get_random_window(stocks, window_days=90)
    assert len(df) == 90 + 40


def test_multiple_calls_can_return_different_windows():
    """Random sampling should not always return the same start index."""
    stocks = make_stocks(500)
    starts = set()
    for _ in range(20):
        df, _ = get_random_window(stocks, window_days=90, warmup=40)
        if df is not None:
            starts.add(df.index[0])
    # With 20 draws from a 500-row series we expect at least 2 distinct starts
    assert len(starts) > 1


# ── download_all_stocks() with mocked download ────────────────────────────────

def test_download_all_stocks_returns_dict(tmp_path, monkeypatch):
    """download_all_stocks() must return a dict[str, DataFrame]."""
    monkeypatch.chdir(tmp_path)
    with patch("neat_trader.data_loader.download_stock", return_value=make_df(300)):
        result = download_all_stocks(tickers=["AAPL", "MSFT"])
    assert isinstance(result, dict)
    assert set(result.keys()) == {"AAPL", "MSFT"}


def test_download_all_stocks_skips_failed_tickers(tmp_path, monkeypatch):
    """Tickers that return None (failed download) must be excluded."""
    monkeypatch.chdir(tmp_path)
    with patch("neat_trader.data_loader.download_stock", side_effect=[make_df(300), None]):
        result = download_all_stocks(tickers=["AAPL", "BAD"])
    assert "AAPL" in result
    assert "BAD" not in result
