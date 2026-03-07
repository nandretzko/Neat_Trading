"""
Backtesting environment for NEAT trading agents.

Implements long-only trading with fractional position sizing.
The paper's model showed a strong preference for long trades,
so this implementation focuses on the long side.

Portfolio state inputs fed to the NEAT network each step:
    long_position  = (shares * price) / total_assets   ∈ [0, 1]
    short_position = 0  (long-only)

Transaction costs:
    COMMISSION = 0.1 % per side (buy or sell).  Applied to the traded value
    so that overtrading is penalised realistically.  Without costs the agent
    can learn to churn the portfolio for free, inflating n_trades and
    producing fitness scores that do not generalise to live trading.
"""

from __future__ import annotations

COMMISSION = 0.001  # 0.1 % per side

import numpy as np
import pandas as pd

from indicators import compute_indicators


class TradingEnvironment:
    WARMUP = 40  # rows consumed by indicator look-backs before trading starts

    def __init__(self, df: pd.DataFrame, initial_capital: float = 10_000.0) -> None:
        self.df = df.reset_index(drop=True)
        self.initial_capital = initial_capital
        self._indicators = compute_indicators(self.df)

    # ------------------------------------------------------------------
    def run(self, net) -> dict:
        """
        Feed the entire DataFrame through the NEAT network and return
        a dict of performance metrics needed by the fitness function.
        """
        close = self.df["Close"].values.astype(float)
        n = len(close)

        # Portfolio state
        cash: float = self.initial_capital
        shares: float = 0.0

        # Trade tracking
        trades: list[dict] = []
        open_trade: dict | None = None

        equity_curve: list[float] = [self.initial_capital]
        days_in_market: int = 0

        for i in range(self.WARMUP, n):
            price = close[i]
            total_assets = cash + shares * price

            # Portfolio ratios (paper inputs 0 and 1)
            long_pos = (shares * price) / (total_assets + 1e-8)
            short_pos = 0.0

            # Technical indicator inputs (9 values)
            row = self._indicators.iloc[i]
            if row.isnull().any():
                equity_curve.append(total_assets)
                continue

            inputs = [
                long_pos,
                short_pos,
                float(row["sma5"]),
                float(row["sma10"]),
                float(row["slow_k"]),
                float(row["slow_d"]),
                float(row["willr"]),
                float(row["macd_diff"]),
                float(row["cci"]),
                float(row["rsi"]),
                float(row["adosc"]),
            ]

            output = net.activate(inputs)
            buy_sig  = float(output[0])
            sell_sig = float(output[1])
            volume   = float(np.clip(output[2], 0.0, 1.0))

            # --- Execute action (paper §2.2.2) ---
            if buy_sig > 0.5 and sell_sig > 0.5:
                if buy_sig >= sell_sig:
                    cash, shares, open_trade = _buy(
                        cash, shares, price, volume, i, open_trade
                    )
                else:
                    cash, shares, trades, open_trade = _sell(
                        cash, shares, price, volume, i, trades, open_trade
                    )
            elif buy_sig > 0.5:
                cash, shares, open_trade = _buy(
                    cash, shares, price, volume, i, open_trade
                )
            elif sell_sig > 0.5:
                cash, shares, trades, open_trade = _sell(
                    cash, shares, price, volume, i, trades, open_trade
                )

            total_assets = cash + shares * price
            equity_curve.append(total_assets)
            if shares > 0:
                days_in_market += 1

        # Close any still-open position at end of window
        if shares > 0:
            price = close[n - 1]
            cash, shares, trades, open_trade = _sell(
                cash, shares, price, 1.0, n - 1, trades, open_trade
            )

        final_assets = cash + shares * close[n - 1]
        return _compute_metrics(
            equity_curve=np.array(equity_curve),
            trades=trades,
            initial_capital=self.initial_capital,
            final_assets=final_assets,
            close=close,
            warmup=self.WARMUP,
            days_in_market=days_in_market,
        )


# -----------------------------------------------------------------------
# Helper functions (module-level for clarity)
# -----------------------------------------------------------------------

def _buy(
    cash: float,
    shares: float,
    price: float,
    volume: float,
    day: int,
    open_trade: dict | None,
) -> tuple[float, float, dict | None]:
    if cash < 1e-4:
        return cash, shares, open_trade
    invest = volume * cash
    commission = invest * COMMISSION
    new_shares = invest / price
    shares += new_shares
    cash -= invest + commission
    if open_trade is None:
        open_trade = {"entry_day": day, "entry_price": price}
    return cash, shares, open_trade


def _sell(
    cash: float,
    shares: float,
    price: float,
    volume: float,
    day: int,
    trades: list[dict],
    open_trade: dict | None,
) -> tuple[float, float, list[dict], dict | None]:
    if shares < 1e-8:
        return cash, shares, trades, open_trade
    sell_shares = volume * shares
    proceeds = sell_shares * price
    cash += proceeds * (1.0 - COMMISSION)
    shares -= sell_shares
    if open_trade is not None and sell_shares > 0:
        ret = (price - open_trade["entry_price"]) / (open_trade["entry_price"] + 1e-8)
        duration = day - open_trade["entry_day"]
        trades.append({"pnl_pct": ret * 100.0, "duration": duration})
        open_trade = None if shares < 1e-8 else open_trade
    return cash, shares, trades, open_trade


def _compute_metrics(
    equity_curve: np.ndarray,
    trades: list[dict],
    initial_capital: float,
    final_assets: float,
    close: np.ndarray,
    warmup: int,
    days_in_market: int,
) -> dict:
    # PnL as percentage
    pnl = (final_assets / initial_capital - 1.0) * 100.0

    # Buy-and-hold benchmark (same window, after warmup)
    bh_return = (close[-1] / close[warmup] - 1.0) * 100.0
    pnl_relative = pnl - bh_return

    # Maximum drawdown (on equity curve)
    peak = np.maximum.accumulate(equity_curve)
    dd = (peak - equity_curve) / np.where(peak > 1e-8, peak, 1.0) * 100.0
    max_drawdown = float(np.max(dd))

    # Trade statistics
    n_trades = len(trades)
    avg_duration = float(np.mean([t["duration"] for t in trades])) if trades else 0.0
    win_rate = (
        sum(1 for t in trades if t["pnl_pct"] > 0) / n_trades if n_trades > 0 else 0.0
    )

    # Exposure time
    total_days = max(len(equity_curve) - 1, 1)
    exposure_time = days_in_market / total_days * 100.0

    return {
        "pnl": pnl,
        "pnl_relative": pnl_relative,
        "max_drawdown": max_drawdown,
        "n_trades": n_trades,
        "avg_duration": avg_duration,
        "win_rate": win_rate,
        "exposure_time": exposure_time,
        "bh_return": bh_return,
        "equity_curve": equity_curve,
        "window_days": total_days,
    }
