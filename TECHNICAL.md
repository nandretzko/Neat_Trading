# Technical Reference — NEAT Stock Trading Agent

This document describes the internal architecture of the project: data flow, indicator maths, NEAT topology, fitness function, training schedule, and the backtesting engine.

> **Paper** — arXiv 2501.14736 — *"Evolving Trading Strategies with NEAT"*

---

## Table of contents

1. [System overview](#1-system-overview)
2. [Data pipeline](#2-data-pipeline)
3. [Technical indicators](#3-technical-indicators)
4. [NEAT configuration](#4-neat-configuration)
5. [Trading environment](#5-trading-environment)
6. [Fitness function](#6-fitness-function)
7. [Progressive training schedule](#7-progressive-training-schedule)
8. [Evaluation protocol](#8-evaluation-protocol)
9. [Module map](#9-module-map)

---

## 1. System overview

```
Yahoo Finance
     │
     ▼
data_loader.py          Download & pickle OHLCV per ticker
     │
     ▼
indicators.py           Compute 9 normalised indicators from OHLCV
     │
     ▼
trading_env.py          Simulate portfolio (long-only, fractional sizing)
     │  ─── metrics ──▶
fitness.py              Score each genome (Eq. 3)
     │
     ▼
train.py  ──NEAT──▶  Population evolves across 2 000 generations
                         │
                         ▼
                    best_genome.pkl
                         │
                         ▼
evaluate.py             100 random 1-year backtests → table + plots
```

---

## 2. Data pipeline

**File:** [data_loader.py](data_loader.py)

### Ticker universe

50 liquid S&P 500 constituents are used, spread across Technology, Finance, Healthcare, Consumer, Industrial, and Energy sectors. The full list is defined in `DEFAULT_TICKERS`.

### Download and cache

`download_stock(ticker, start, end)` calls `yfinance.download()` and saves the result as a pickle file under `data/cache/<ticker>.pkl`. Subsequent calls load the pickle directly — no network request is made.

OHLCV columns kept: `Open`, `High`, `Low`, `Close`, `Volume`. Tickers with fewer than 200 rows are discarded.

### Random window sampling

`get_random_window(stocks, window_days, warmup=40)` picks:
1. A random ticker from the available stocks.
2. A random contiguous slice of `window_days + warmup` rows.

The first `warmup` rows are included to let indicators warm up their look-back windows (longest look-back is 26 periods for the MACD EMA). The caller (training or evaluation code) receives the **full slice** including the warmup prefix; the trading environment itself skips the first 40 rows before executing trades.

---

## 3. Technical indicators

**File:** [indicators.py](indicators.py)

Nine indicators are computed for every row and used as NEAT network inputs 2–10 (inputs 0 and 1 are portfolio state injected by the trading environment):

| # | Name | Formula (simplified) | Output range |
|---|---|---|---|
| 2 | `sma5` | Close / SMA(5) | ≈ [0.8, 1.2] |
| 3 | `sma10` | Close / SMA(10) | ≈ [0.8, 1.2] |
| 4 | `slow_k` | (Close − LL₁₄) / (HH₁₄ − LL₁₄) | [0, 1] |
| 5 | `slow_d` | SMA(3) of slow_k | [0, 1] |
| 6 | `willr` | (HH₁₄ − Close) / (HH₁₄ − LL₁₄) | [0, 1] |
| 7 | `macd_diff` | ΔMACD-histogram / Close | small floats |
| 8 | `cci` | CCI(20) / 200, clipped to [−3, 3] | [−3, 3] |
| 9 | `rsi` | 1 − 1/(1 + AvgUp₁₄/AvgDown₁₄) | [0, 1] |
| 10 | `adosc` | (EMA3(AD) − EMA10(AD)) / rollstd(20) | [−3, 3] |

All indicators are normalised so that the NEAT network receives consistently scaled values regardless of the stock's price level.

---

## 4. NEAT configuration

**File:** [neat_config.cfg](neat_config.cfg)

### Topology

| Parameter | Value | Rationale |
|---|---|---|
| `num_inputs` | 11 | 2 portfolio state + 9 indicators |
| `num_outputs` | 3 | `buy_signal`, `sell_signal`, `volume` |
| `num_hidden` | 0 (initial) | Grown by mutation |
| `feed_forward` | False | **Recurrent** — network has memory across timesteps |
| `initial_connection` | `full_direct` | All inputs connected to all outputs at generation 0 |

### Population and reproduction

| Parameter | Value |
|---|---|
| `pop_size` | 150 |
| `elitism` | 2 (best 2 genomes always survive) |
| `survival_threshold` | 0.20 (top 20 % can reproduce) |
| `max_stagnation` | 20 (species removed after 20 stagnant generations) |

### Mutation rates

| Parameter | Value |
|---|---|
| `weight_mutate_rate` | 0.80 |
| `bias_mutate_rate` | 0.70 |
| `conn_add_prob` | 0.50 |
| `conn_delete_prob` | 0.50 |
| `node_add_prob` | 0.20 |
| `node_delete_prob` | 0.20 |
| `activation_mutate_rate` | 0.05 |
| `activation_options` | `sigmoid`, `tanh`, `relu` |

### Speciation

Genomes are grouped into species using the NEAT compatibility distance. Two genomes belong to the same species if their distance is below `compatibility_threshold = 3.0`. Distance is computed as:

```
d = c1 × disjoint_genes + c2 × Σ|weight_diff|
```

with `c1 = 1.0` (disjoint coefficient) and `c2 = 0.5` (weight coefficient).

---

## 5. Trading environment

**File:** [trading_env.py](trading_env.py)

### Portfolio model

- **Long-only** (no short selling).
- **Fractional position sizing** — the network's `volume` output (clipped to [0, 1]) determines what fraction of available cash to invest (buy) or what fraction of held shares to sell.
- Starting capital: **$10 000**.
- **Transaction costs** — `COMMISSION = 0.001` (0.1 % per side). On a buy, `volume × cash × 0.001` is deducted from cash in addition to the invested amount. On a sell, the agent receives `proceeds × (1 − 0.001)`. This prevents the agent from learning to churn the portfolio for free.

### Step logic

At each trading day `i` (after the 40-row warmup):

1. Compute portfolio ratios:
   - `long_pos = (shares × price) / total_assets`
   - `short_pos = 0` (long-only)
2. Build 11-element input vector and call `net.activate(inputs)`.
3. Read outputs: `buy_sig`, `sell_sig`, `volume`.
4. Execute action:
   - `buy_sig > 0.5` **and** `sell_sig > 0.5` → compare magnitudes; execute the larger.
   - `buy_sig > 0.5` only → buy `volume × cash` worth of shares.
   - `sell_sig > 0.5` only → sell `volume × shares` shares.
   - Both ≤ 0.5 → hold.

At end of window, any remaining open position is force-closed at the last closing price.

### Metrics computed

| Key | Meaning |
|---|---|
| `pnl` | Total return % |
| `bh_return` | Buy-and-hold return % over the same window |
| `pnl_relative` | `pnl − bh_return` |
| `max_drawdown` | Maximum peak-to-trough decline % |
| `n_trades` | Number of completed round-trips |
| `avg_duration` | Average hold duration in days |
| `win_rate` | Fraction of trades that were profitable |
| `exposure_time` | % of days with a non-zero position |
| `equity_curve` | numpy array of portfolio value each day |
| `window_days` | Number of trading days in the window (used by fitness to normalise avg_duration) |

---

## 6. Fitness function

**File:** [fitness.py](fitness.py)

Implements **Option 3** from the paper (Equation 3), with two calibrated deviations to ensure all terms contribute equally:

```
Fitness(R) =   PnL
             + 1.5 × PnL_relative
             − 0.5 × max_drawdown
             + 0.05  × n_trades
             − 10.0  × (avg_duration / window_days)
```

| Term | Effect | Typical range |
|---|---|---|
| `PnL` | Reward absolute profit | [−20, +30] % |
| `1.5 × PnL_relative` | Strongly reward outperforming Buy & Hold | [−30, +45] % |
| `−0.5 × max_drawdown` | Penalise large drawdowns (risk control) | [0, −20] |
| `+0.05 × n_trades` | Bonus for active trading (prevents inactivity) | [0, +5] for ~100 trades |
| `−10 × (avg_duration / window_days)` | Penalise long holds, window-normalised | [0, −10] |

**Why the coefficients differ from the paper:**

- `+0.0005 × n_trades` → `+0.05 × n_trades`: the original coefficient contributed ~0.05 for 100 trades — effectively zero. Increased 100× to a meaningful but still secondary signal.
- `−avg_duration` → `−10 × (avg_duration / window_days)`: raw days are window-dependent — a 30-day hold costs −30 in both a 90-day and a 365-day window, making Stage-1 and Stage-3 fitness scores incomparable across training stages. Dividing by `window_days` normalises to [0, 1], then ×10 gives comparable magnitude to the other terms.

`window_days` is supplied by `TradingEnvironment` in the metrics dict (defaults to 365 if absent).

---

## 7. Progressive training schedule

**File:** [train.py](train.py)

Training proceeds in three stages with increasing window complexity:

| Stage | Generations (full) | Generations (--fast) | Window |
|---|---|---|---|
| 1 | 1 500 | 50 | 90 days — learn short-term reversals |
| 2 | 400 | 20 | 150 days — intermediate horizon |
| 3 | 100 | 10 | 365 days — full-year risk management |

At each generation, **every genome is evaluated on a freshly sampled random window** from a random stock. This prevents overfitting to any single ticker or time period.

### Checkpoint and resume

- A `neat.Checkpointer` saves the entire population to `checkpoints/ckpt-<gen>` every 50 generations and to `checkpoints/latest` at the very end.
- On startup, `train.py` checks for `checkpoints/latest`. If found, the population is restored and training continues.

### Outputs

| File | Content |
|---|---|
| `best_genome.pkl` | The winner genome from the final stage |
| `neat_config_used.pkl` | Serialised `neat.Config` object (used by `evaluate.py`) |
| `checkpoints/` | Population snapshots |

---

## 8. Evaluation protocol

**File:** [evaluate.py](evaluate.py)

1. Load `best_genome.pkl` and rebuild the recurrent network.
2. Download (or load from cache) all 50 stocks.
3. Run **100 random 1-year (365-day) windows** and collect metrics.
4. Print a summary table matching **paper Table 3** (avg return, std, win rate, exposure time, trade count, hold duration).
5. Save `comparison_plot.png` — scatter plot of model return vs Buy & Hold return with a linear fit and a 45° parity line.
6. Save `equity_sample.png` — equity curve for one randomly chosen window.

---

## 9. Module map

```
data_loader.py
├── DEFAULT_TICKERS         list[str]   50 S&P 500 tickers
├── download_stock()        str → DataFrame | None
├── download_all_stocks()   → dict[str, DataFrame]
└── get_random_window()     → (DataFrame | None, str | None)

indicators.py
└── compute_indicators()    DataFrame → DataFrame (9 indicator columns)

trading_env.py
├── TradingEnvironment
│   ├── __init__(df, initial_capital)
│   └── run(net) → dict[str, float | ndarray]
├── _buy()
├── _sell()
└── _compute_metrics()

fitness.py
└── compute_fitness()       dict → float

train.py
├── eval_genome()           (genome, config) → float
├── eval_genomes()          NEAT callback
├── StageReporter           neat.reporting.BaseReporter subclass
└── run_training()          (fast=False) → None

evaluate.py
├── load_genome_and_config() → (genome, config)
├── run_evaluation()         → list[dict]
├── print_table()
├── plot_comparison()
└── plot_equity_sample()
```
