"""Unit tests for fitness.py — compute_fitness() is a pure function."""
import pytest
from neat_trader.fitness import compute_fitness


def _metrics(
    pnl=10.0,
    pnl_relative=5.0,
    max_drawdown=2.0,
    n_trades=10,
    avg_duration=3.0,
    sharpe=0.0,
) -> dict:
    return {
        "pnl": pnl,
        "pnl_relative": pnl_relative,
        "max_drawdown": max_drawdown,
        "n_trades": n_trades,
        "avg_duration": avg_duration,
        "sharpe": sharpe,
    }


def test_formula_exact():
    """Fitness equals the risk-adjusted formula to floating-point precision."""
    m = _metrics(
        pnl_relative=4.0, max_drawdown=5.0, n_trades=100,
        avg_duration=2.0, sharpe=1.5,
    )
    # window_days defaults to 365; avg_duration_frac = 2.0 / 365
    expected = (
        10.0 * 1.5
        + 1.0 * 4.0
        - 0.5 * 5.0
        + 0.05 * 100
        - 10.0 * (2.0 / 365)
    )
    assert compute_fitness(m) == pytest.approx(expected)


def test_returns_float():
    assert isinstance(compute_fitness(_metrics()), float)


def test_positive_scenario():
    """Good trading (beats B&H, low drawdown, positive Sharpe) → positive fitness."""
    m = _metrics(pnl_relative=8.0, max_drawdown=3.0, n_trades=20,
                 avg_duration=5.0, sharpe=1.2)
    assert compute_fitness(m) > 0.0


def test_negative_scenario():
    """Bad trading (large loss, large drawdown, long holds) → negative fitness."""
    m = _metrics(pnl_relative=-10.0, max_drawdown=30.0, n_trades=2,
                 avg_duration=30.0, sharpe=-1.5)
    assert compute_fitness(m) < 0.0


def test_higher_sharpe_is_better():
    low  = _metrics(sharpe=0.5)
    high = _metrics(sharpe=2.0)
    assert compute_fitness(high) > compute_fitness(low)


def test_higher_relative_pnl_is_better():
    low  = _metrics(pnl_relative=0.0)
    high = _metrics(pnl_relative=10.0)
    assert compute_fitness(high) > compute_fitness(low)


def test_higher_drawdown_is_worse():
    small_dd = _metrics(max_drawdown=2.0)
    large_dd = _metrics(max_drawdown=20.0)
    assert compute_fitness(small_dd) > compute_fitness(large_dd)


def test_longer_duration_is_worse():
    short = _metrics(avg_duration=2.0)
    long_ = _metrics(avg_duration=20.0)
    assert compute_fitness(short) > compute_fitness(long_)


def test_more_trades_slightly_better():
    few  = _metrics(n_trades=0)
    many = _metrics(n_trades=1000)
    assert compute_fitness(many) > compute_fitness(few)


def test_zero_metrics_returns_zero():
    m = _metrics(pnl=0.0, pnl_relative=0.0, max_drawdown=0.0,
                 n_trades=0, avg_duration=0.0, sharpe=0.0)
    assert compute_fitness(m) == pytest.approx(0.0)
