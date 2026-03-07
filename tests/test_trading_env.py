"""Unit tests for trading_env.py — TradingEnvironment and helper functions."""
import numpy as np
import pandas as pd
import pytest
from unittest.mock import MagicMock

from trading_env import TradingEnvironment, _buy, _sell


REQUIRED_METRIC_KEYS = {
    "pnl", "bh_return", "pnl_relative", "max_drawdown",
    "n_trades", "avg_duration", "win_rate", "exposure_time", "equity_curve",
    "window_days",
}


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_ohlcv(n: int = 100, price: float = 100.0) -> pd.DataFrame:
    """Flat-price OHLCV with a small spread so indicators don't divide by zero."""
    prices = np.full(n, price)
    return pd.DataFrame({
        "Open":   prices,
        "High":   prices * 1.01,
        "Low":    prices * 0.99,
        "Close":  prices,
        "Volume": np.ones(n) * 1_000_000.0,
    })


def make_net(buy: float = 0.0, sell: float = 0.0, vol: float = 0.5):
    """Mock NEAT network that returns constant (buy, sell, vol) outputs."""
    net = MagicMock()
    net.activate.return_value = [buy, sell, vol]
    return net


# ── TradingEnvironment.run() ──────────────────────────────────────────────────

def test_metrics_keys_present():
    env = TradingEnvironment(make_ohlcv(100))
    metrics = env.run(make_net(buy=0.0, sell=0.0))
    assert REQUIRED_METRIC_KEYS.issubset(metrics.keys())


def test_hold_strategy_zero_pnl():
    """A strategy that never trades should return exactly 0% PnL."""
    env = TradingEnvironment(make_ohlcv(100))
    metrics = env.run(make_net(buy=0.0, sell=0.0))
    assert metrics["pnl"] == pytest.approx(0.0, abs=1e-6)


def test_hold_strategy_zero_trades():
    env = TradingEnvironment(make_ohlcv(100))
    metrics = env.run(make_net(buy=0.0, sell=0.0))
    assert metrics["n_trades"] == 0


def test_flat_price_buy_and_hold_zero_pnl():
    """Buying on a flat-price series yields slightly negative PnL due to commission."""
    env = TradingEnvironment(make_ohlcv(100))
    metrics = env.run(make_net(buy=1.0, sell=0.0, vol=1.0))
    assert metrics["pnl"] <= 0.0


def test_bh_return_zero_on_flat_price():
    env = TradingEnvironment(make_ohlcv(100))
    metrics = env.run(make_net(buy=0.0, sell=0.0))
    assert metrics["bh_return"] == pytest.approx(0.0, abs=1e-6)


def test_exposure_time_range():
    env = TradingEnvironment(make_ohlcv(100))
    metrics = env.run(make_net(buy=0.5, sell=0.0, vol=0.3))
    assert 0.0 <= metrics["exposure_time"] <= 100.0


def test_equity_curve_starts_at_initial_capital():
    initial = 10_000.0
    env = TradingEnvironment(make_ohlcv(100), initial_capital=initial)
    metrics = env.run(make_net(buy=0.0, sell=0.0))
    assert metrics["equity_curve"][0] == pytest.approx(initial)


def test_equity_curve_length():
    """Equity curve should have one entry per day after warmup (plus the initial)."""
    n = 100
    env = TradingEnvironment(make_ohlcv(n))
    metrics = env.run(make_net(buy=0.0, sell=0.0))
    # At least as long as (n - WARMUP) entries after the initial value
    assert len(metrics["equity_curve"]) >= n - TradingEnvironment.WARMUP


def test_max_drawdown_non_negative():
    env = TradingEnvironment(make_ohlcv(100))
    metrics = env.run(make_net(buy=0.8, sell=0.2, vol=0.5))
    assert metrics["max_drawdown"] >= 0.0


def test_win_rate_in_unit_interval():
    env = TradingEnvironment(make_ohlcv(100))
    metrics = env.run(make_net(buy=0.8, sell=0.8, vol=0.5))
    assert 0.0 <= metrics["win_rate"] <= 1.0


# ── _buy() helper ─────────────────────────────────────────────────────────────

def test_buy_invests_fraction_of_cash():
    # invest = 0.5 * 10_000 = 5_000; commission = 5_000 * 0.001 = 5
    # cash = 10_000 - 5_000 - 5 = 4_995; shares = 5_000 / 100 = 50
    cash, shares, trade = _buy(10_000.0, 0.0, 100.0, 0.5, 10, None)
    assert cash   == pytest.approx(4_995.0)
    assert shares == pytest.approx(50.0)


def test_buy_creates_open_trade_record():
    _, _, trade = _buy(10_000.0, 0.0, 100.0, 0.5, 10, None)
    assert trade == {"entry_day": 10, "entry_price": 100.0}


def test_buy_does_not_overwrite_existing_open_trade():
    existing = {"entry_day": 5, "entry_price": 90.0}
    _, _, trade = _buy(5_000.0, 10.0, 100.0, 0.5, 20, existing)
    assert trade == existing


def test_buy_noop_when_no_cash():
    cash, shares, trade = _buy(0.0, 10.0, 100.0, 1.0, 5, None)
    assert cash == pytest.approx(0.0)
    assert shares == pytest.approx(10.0)
    assert trade is None


# ── _sell() helper ────────────────────────────────────────────────────────────

def test_sell_full_position():
    # proceeds = 100 * 100 = 10_000; commission = 10_000 * 0.001 = 10
    # cash = 0 + 10_000 - 10 = 9_990
    open_trade = {"entry_day": 5, "entry_price": 80.0}
    cash, shares, trades, ot = _sell(0.0, 100.0, 100.0, 1.0, 10, [], open_trade)
    assert cash   == pytest.approx(9_990.0)
    assert shares == pytest.approx(0.0)


def test_sell_records_pnl_and_duration():
    open_trade = {"entry_day": 5, "entry_price": 80.0}
    _, _, trades, _ = _sell(0.0, 100.0, 100.0, 1.0, 10, [], open_trade)
    assert len(trades) == 1
    assert trades[0]["pnl_pct"] == pytest.approx(25.0, rel=1e-4)
    assert trades[0]["duration"] == 5


def test_sell_noop_when_no_shares():
    cash, shares, trades, ot = _sell(5_000.0, 0.0, 100.0, 1.0, 10, [], None)
    assert cash   == pytest.approx(5_000.0)
    assert shares == pytest.approx(0.0)
    assert trades == []
