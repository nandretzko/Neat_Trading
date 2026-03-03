"""
Multi-objective fitness function — Option 3 from the paper (Eq. 3):

    Fitness3(R) = PnL
                + 1.5 × PnL_relative
                − 0.5 × max(drawdown)
                + 0.0005 × #trades
                − avg(duration)

Goal:
  • Maximise absolute profit (PnL)
  • Outperform Buy & Hold (PnL_relative)
  • Minimise risk / drawdown (max_drawdown)
  • Encourage active trading (+ #trades bonus)
  • Penalise long holds → forces swing-trade behaviour (- avg_duration)
"""


def compute_fitness(metrics: dict) -> float:
    pnl          = metrics["pnl"]           # % return of the strategy
    pnl_relative = metrics["pnl_relative"]  # strategy return − B&H return
    max_dd       = metrics["max_drawdown"]  # max drawdown %
    n_trades     = metrics["n_trades"]
    avg_duration = metrics["avg_duration"]  # average hold in days

    fitness = (
        pnl
        + 1.5 * pnl_relative
        - 0.5 * max_dd
        + 0.0005 * n_trades
        - avg_duration
    )
    return float(fitness)
