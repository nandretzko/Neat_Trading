"""
Multi-objective fitness function — Option 3 from the paper (Eq. 3):

    Fitness3(R) = PnL
                + 1.5 × PnL_relative
                − 0.5 × max(drawdown)
                + 0.05 × #trades          (was 0.0005 — effectively zero)
                − 10 × avg_duration_frac  (was raw days — dominated fitness)

All terms are now of comparable magnitude (roughly [-50, +50]):
  • PnL / PnL_relative  : typically [-20, +30] %
  • max_drawdown penalty : typically [0, -20]
  • activity bonus       : up to +5 for ~100 trades (0.05 × 100)
  • long-hold penalty    : avg_duration / window_days ∈ [0, 1], scaled by -10

The raw `- avg_duration` (in days) was window-dependent: a 30-day hold
cost -30 in a 90-day window but also -30 in a 365-day window, making
Stage-1 fitness incomparable to Stage-3 fitness.  Normalising by
`window_days` removes this artefact.

`window_days` is supplied by TradingEnvironment via the metrics dict.
"""


def compute_fitness(metrics: dict) -> float:
    pnl          = metrics["pnl"]           # % return of the strategy
    pnl_relative = metrics["pnl_relative"]  # strategy return − B&H return
    max_dd       = metrics["max_drawdown"]  # max drawdown %
    n_trades     = metrics["n_trades"]
    avg_duration = metrics["avg_duration"]  # average hold in days
    window_days  = metrics.get("window_days", 365)  # trading days in window

    # Normalise hold duration to [0, 1]: fraction of the window spent per trade.
    # A hold spanning the full window scores -10; a 1-day trade scores ≈ 0.
    avg_duration_frac = avg_duration / (window_days + 1e-8)

    fitness = (
        pnl                        # absolute return (%)
        + 1.5 * pnl_relative       # alpha vs buy-and-hold (%)
        - 0.5 * max_dd             # risk penalty (%)
        + 0.05 * n_trades          # activity bonus (≈ +0.05 per trade)
        - 10.0 * avg_duration_frac # long-hold penalty, window-normalised
    )
    return float(fitness)
