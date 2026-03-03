"""Unit tests for indicators.py — compute_indicators() on synthetic OHLCV data."""
import numpy as np
import pandas as pd
import pytest
from indicators import compute_indicators

EXPECTED_COLS = {"sma5", "sma10", "slow_k", "slow_d", "willr", "macd_diff", "cci", "rsi", "adosc"}


def make_ohlcv(n: int = 120, seed: int = 42) -> pd.DataFrame:
    """Synthetic OHLCV that respects High >= Close >= Low > 0."""
    rng = np.random.default_rng(seed)
    close  = 100.0 + np.cumsum(rng.normal(0, 0.5, n))
    spread = np.abs(rng.normal(0, 0.5, n)) + 0.01
    return pd.DataFrame(
        {
            "Open":   close + rng.normal(0, 0.2, n),
            "High":   close + spread,
            "Low":    close - spread,
            "Close":  close,
            "Volume": rng.integers(1_000_000, 5_000_000, n).astype(float),
        }
    )


# ── Structure ─────────────────────────────────────────────────────────────────

def test_output_columns():
    result = compute_indicators(make_ohlcv())
    assert set(result.columns) == EXPECTED_COLS


def test_output_length_matches_input():
    df = make_ohlcv(150)
    result = compute_indicators(df)
    assert len(result) == len(df)


def test_output_index_matches_input():
    df = make_ohlcv()
    result = compute_indicators(df)
    assert list(result.index) == list(df.index)


# ── Value ranges ──────────────────────────────────────────────────────────────

def test_slow_k_in_unit_interval():
    result = compute_indicators(make_ohlcv())
    valid = result["slow_k"].dropna()
    assert (valid >= 0.0).all() and (valid <= 1.0).all()


def test_slow_d_in_unit_interval():
    result = compute_indicators(make_ohlcv())
    valid = result["slow_d"].dropna()
    assert (valid >= 0.0).all() and (valid <= 1.0).all()


def test_willr_in_unit_interval():
    result = compute_indicators(make_ohlcv())
    valid = result["willr"].dropna()
    assert (valid >= 0.0).all() and (valid <= 1.0).all()


def test_rsi_in_unit_interval():
    result = compute_indicators(make_ohlcv())
    valid = result["rsi"].dropna()
    assert (valid >= 0.0).all() and (valid <= 1.0).all()


def test_cci_clipped_to_three():
    result = compute_indicators(make_ohlcv())
    valid = result["cci"].dropna()
    assert (valid >= -3.0).all() and (valid <= 3.0).all()


def test_adosc_clipped_to_three():
    result = compute_indicators(make_ohlcv())
    valid = result["adosc"].dropna()
    assert (valid >= -3.0).all() and (valid <= 3.0).all()


def test_sma5_near_one_for_flat_price():
    """For a flat price series, Close/SMA(5) should equal 1.0 after warm-up."""
    close  = np.full(50, 100.0)
    spread = np.full(50, 1.0)
    df = pd.DataFrame({
        "Open": close, "High": close + spread,
        "Low":  close - spread, "Close": close,
        "Volume": np.ones(50) * 1e6,
    })
    result = compute_indicators(df)
    # After the 5-period warm-up, sma5 should be exactly 1.0
    assert result["sma5"].iloc[10:].dropna().apply(lambda v: abs(v - 1.0) < 1e-6).all()


# ── Stability ─────────────────────────────────────────────────────────────────

def test_no_inf_values():
    result = compute_indicators(make_ohlcv())
    assert not result.isin([np.inf, -np.inf]).any().any()


def test_short_series_does_not_crash():
    """Function must not raise even on a very short DataFrame."""
    compute_indicators(make_ohlcv(n=5))
