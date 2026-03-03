"""
Data loader: downloads and caches S&P 500 stock data from Yahoo Finance.
Uses a subset of 50 liquid stocks for manageable training time.
"""
import os
import random
import pickle

import pandas as pd
import yfinance as yf

CACHE_DIR = "data/cache"

# 50 liquid S&P 500 constituents covering diverse sectors
DEFAULT_TICKERS = [
    # Tech
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "TSLA", "INTC", "AMD", "QCOM",
    "ADBE", "CRM", "CSCO", "TXN", "AVGO",
    # Finance
    "JPM", "BAC", "GS", "MS", "V", "MA",
    # Healthcare
    "JNJ", "UNH", "ABBV", "MRK", "TMO", "ABT", "DHR", "LLY",
    # Consumer
    "PG", "KO", "PEP", "MCD", "SBUX", "NKE", "WMT", "COST", "TGT", "DIS",
    # Industrial / Energy
    "HON", "CAT", "BA", "GE", "UPS", "XOM", "CVX",
    # Other
    "HD", "VZ", "NFLX", "ACN",
]


def download_stock(
    ticker: str,
    start: str = "2000-01-01",
    end: str = "2022-11-27",
) -> pd.DataFrame | None:
    """Download OHLCV data for one ticker (with local pickle cache)."""
    os.makedirs(CACHE_DIR, exist_ok=True)
    cache_path = os.path.join(CACHE_DIR, f"{ticker}.pkl")

    if os.path.exists(cache_path):
        with open(cache_path, "rb") as f:
            return pickle.load(f)

    try:
        df = yf.download(ticker, start=start, end=end, progress=False, auto_adjust=True)
        if len(df) < 200:
            return None
        df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
        df.dropna(inplace=True)
        with open(cache_path, "wb") as f:
            pickle.dump(df, f)
        print(f"  Downloaded {ticker}: {len(df)} rows")
        return df
    except Exception as exc:
        print(f"  Warning: could not download {ticker}: {exc}")
        return None


def download_all_stocks(tickers: list[str] | None = None) -> dict[str, pd.DataFrame]:
    """Download (or load from cache) all tickers. Returns {ticker: df}."""
    if tickers is None:
        tickers = DEFAULT_TICKERS
    stocks: dict[str, pd.DataFrame] = {}
    for ticker in tickers:
        df = download_stock(ticker)
        if df is not None:
            stocks[ticker] = df
    return stocks


def get_random_window(
    stocks: dict[str, pd.DataFrame],
    window_days: int,
    warmup: int = 40,
) -> tuple[pd.DataFrame | None, str | None]:
    """
    Randomly pick a stock and a contiguous date window of `window_days` rows.
    `warmup` extra rows are prepended so indicators can warm up, then sliced off
    before returning — the caller gets a clean window ready for trading.
    """
    ticker = random.choice(list(stocks.keys()))
    df = stocks[ticker]

    needed = window_days + warmup
    if len(df) < needed + 10:
        return None, None

    max_start = len(df) - needed - 1
    if max_start < 1:
        return None, None

    start_idx = random.randint(0, max_start)
    window = df.iloc[start_idx : start_idx + needed].copy()
    return window, ticker
