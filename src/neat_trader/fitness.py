"""
Fitness function — risk-adjusted variant of Eq. 3 (arXiv 2501.14736).

    Fitness = 10 × Sharpe              (primary: risk-adjusted daily returns)
            +  1 × PnL_relative        (alpha vs Buy & Hold, %)
            − 0.5 × max_drawdown       (risk penalty, %)
            + 0.05 × n_trades          (activity bonus)
            − 10 × avg_duration_frac   (long-hold penalty, window-normalised)

Why Sharpe instead of raw PnL:
  Raw PnL rewards lucky volatile runs equally to steady gains.  Sharpe
  (mean_daily_return / std_daily_return × √252) penalises inconsistency,
  driving the agent toward strategies that profit steadily rather than
  in one lucky spike.  A Sharpe of 1.0 (roughly achievable by good quant
  funds) contributes +10, dominating the other terms only when paired with
  positive alpha (PnL_relative > 0).

Term magnitudes (approximate):
  • 10 × Sharpe          : [-30, +30]  (Sharpe ∈ [-3, +3] for realistic strats)
  • 1 × PnL_relative     : [-20, +20]  (secondary alpha signal)
  • −0.5 × max_drawdown  : [0,   −20]  (risk)
  • +0.05 × n_trades     : [0,    +5]  (activity, ~100 trades max)
  • −10 × dur_frac       : [0,   −10]  (hold penalty, normalised)

`window_days` and `sharpe` are supplied by TradingEnvironment via the
metrics dict; both default gracefully if absent.
"""


def compute_fitness(metrics: dict) -> float:
    pnl_relative = metrics["pnl_relative"]  # strategy return − B&H return (%)
    max_dd       = metrics["max_drawdown"]  # max drawdown %
    n_trades     = metrics["n_trades"]
    avg_duration = metrics["avg_duration"]  # average hold in days
    window_days  = metrics.get("window_days", 365)
    sharpe       = metrics.get("sharpe", 0.0)

    avg_duration_frac = avg_duration / (window_days + 1e-8)

    fitness = (
        10.0 * sharpe              # risk-adjusted return (dominant term)
        + 1.0 * pnl_relative       # alpha vs buy-and-hold (%)
        - 0.5 * max_dd             # drawdown penalty (%)
        + 0.05 * n_trades          # activity bonus
        - 10.0 * avg_duration_frac # long-hold penalty, window-normalised
    )
    return float(fitness)
