# NEAT Stock Trading Agent

A Python implementation of a **NeuroEvolution of Augmenting Topologies (NEAT)** trading agent trained on S&P 500 stocks, based on the research paper *"Evolving Trading Strategies with NEAT"* (arXiv 2501.14736).

---

## Project layout

```
.
├── data_loader.py      # Download & cache stock data (Yahoo Finance)
├── indicators.py       # 9 normalised technical indicators
├── trading_env.py      # Backtesting environment (long-only, fractional sizing)
├── fitness.py          # Multi-objective fitness function (paper Eq. 3)
├── train.py            # NEAT training with 3-stage progressive schedule
├── evaluate.py         # Backtest best genome and produce reports/plots
├── neat_config.cfg     # NEAT hyperparameters (population, mutation rates, …)
└── requirements.txt    # Python dependencies
```

Directories created automatically at runtime:

| Path | Content |
|---|---|
| `data/cache/` | Per-ticker pickle files (avoid re-downloading) |
| `checkpoints/` | NEAT population snapshots every 50 generations |

---

## Requirements

- **Python 3.10+** (uses `X | Y` union type hints)
- Internet access on first run (Yahoo Finance data download)

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Quick start — smoke test

Verify the full pipeline in a few minutes (50 / 20 / 10 generations instead of 2 000):

```bash
python train.py --fast
python evaluate.py --n 20
```

---

## Full training

```bash
python train.py
```

The three progressive stages run automatically:

| Stage | Generations | Window |
|---|---|---|
| 1 | 1 500 | 90 days |
| 2 | 400 | 150 days |
| 3 | 100 | 365 days |

Progress is printed to stdout every generation.
A checkpoint is saved every 50 generations to `checkpoints/`.
On completion, `best_genome.pkl` and `neat_config_used.pkl` are written to the project root.

> **Resume from checkpoint** — if training is interrupted, simply re-run `python train.py`.
> The script detects `checkpoints/latest` and restores the population automatically.

---

## Evaluation

```bash
# Default: 100 random 1-year backtests using best_genome.pkl
python evaluate.py

# Custom options
python evaluate.py --genome best_genome.pkl   # path to genome file
python evaluate.py --config neat_config.cfg   # path to NEAT config
python evaluate.py --n 50                     # number of backtest windows
python evaluate.py --window 252               # trading days per window
```

### Outputs

| File | Description |
|---|---|
| Console table | Avg return, std, win rate, exposure time, trade stats — mirrors paper Table 3 |
| `comparison_plot.png` | Scatter: model return vs Buy & Hold return per window |
| `equity_sample.png` | Equity curve for one randomly selected window |

---

## Command reference

| Command | What it does |
|---|---|
| `python train.py` | Full 2 000-generation training run |
| `python train.py --fast` | 80-generation smoke-test (50 / 20 / 10) |
| `python evaluate.py` | Evaluate `best_genome.pkl` on 100 windows |
| `python evaluate.py --n N` | Use N backtest windows |
| `python evaluate.py --genome FILE` | Evaluate a specific genome pickle |

---

## Reproducibility notes

- Stock data is downloaded once and cached under `data/cache/` as pickle files.
  Delete this directory to force a fresh download.
- NEAT uses internal randomness; exact results vary between runs.
  Fitness and paper metrics are expected to match in the same order of magnitude, not to the decimal.
