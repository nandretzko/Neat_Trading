"""
Evaluate a trained NEAT genome on 100 random 1-year windows and compare
against the Buy & Hold baseline (replicates Table 3 from the paper).

Usage:
    python evaluate.py                          # uses best_genome.pkl
    python evaluate.py --genome my_genome.pkl  # custom genome file
    python evaluate.py --n 50                  # fewer backtests

Outputs:
    • Console table  (matches paper Table 3)
    • comparison_plot.png  (scatter: model return vs B&H return)
    • equity_sample.png    (one randomly chosen equity curve)
"""
from __future__ import annotations

import argparse
import pickle

import matplotlib.pyplot as plt
import neat
import numpy as np

from neat_trader.data_loader import download_all_stocks, get_random_window
from neat_trader.trading_env import TradingEnvironment


def load_genome_and_config(
    genome_file: str = "best_genome.pkl",
    config_file: str = "neat_config.cfg",
) -> tuple:
    with open(genome_file, "rb") as f:
        genome = pickle.load(f)

    # Try pre-saved config object first, fall back to cfg file
    try:
        with open("neat_config_used.pkl", "rb") as f:
            config = pickle.load(f)
    except FileNotFoundError:
        config = neat.Config(
            neat.DefaultGenome,
            neat.DefaultReproduction,
            neat.DefaultSpeciesSet,
            neat.DefaultStagnation,
            config_file,
        )
    return genome, config


def run_evaluation(
    genome_file: str = "best_genome.pkl",
    config_file: str = "neat_config.cfg",
    n_tests: int = 100,
    window_days: int = 365,
) -> list[dict]:
    genome, config = load_genome_and_config(genome_file, config_file)
    net = neat.nn.RecurrentNetwork.create(genome, config)

    print("Loading stock data for evaluation …")
    stocks = download_all_stocks()

    results: list[dict] = []
    for i in range(n_tests):
        df, ticker = get_random_window(stocks, window_days)
        if df is None:
            continue
        env = TradingEnvironment(df)
        m = env.run(net)
        m["ticker"] = ticker
        results.append(m)
        if (i + 1) % 10 == 0:
            print(f"  {i+1}/{n_tests} backtests done …")

    return results


def print_table(results: list[dict]) -> None:
    model_ret  = [r["pnl"]        for r in results]
    bh_ret     = [r["bh_return"]  for r in results]
    exp_time   = [r["exposure_time"] for r in results]
    n_trades   = [r["n_trades"]   for r in results]
    durations  = [r["avg_duration"] for r in results if r["avg_duration"] > 0]

    model_wins    = sum(1 for r in results if r["pnl"] > 0)
    bh_wins       = sum(1 for b in bh_ret if b > 0)
    relative_wins = sum(1 for r in results if r["pnl"] > r["bh_return"])

    print("\n" + "=" * 52)
    print(f"  {'Metric':<28} {'Model':>10} {'B&H':>10}")
    print("-" * 52)
    print(f"  {'Average Return':<28} {np.mean(model_ret):>9.2f}% {np.mean(bh_ret):>9.2f}%")
    print(f"  {'Std of Return':<28} {np.std(model_ret):>9.2f}% {np.std(bh_ret):>9.2f}%")
    print(f"  {'Win Rate':<28} {model_wins}/{len(results):>6}  {bh_wins}/{len(results)}")
    print(f"  {'Relative Win Rate':<28} {relative_wins:>10} {len(results)-relative_wins:>10}")
    print(f"  {'Avg Exposure Time':<28} {np.mean(exp_time):>9.2f}%      100.00%")
    print(f"  {'Avg #Trades per backtest':<28} {np.mean(n_trades):>10.1f}")
    print(f"  {'Avg Hold Duration (days)':<28} {np.mean(durations) if durations else 0:>10.1f}")
    print("=" * 52)


def plot_comparison(results: list[dict], out: str = "comparison_plot.png") -> None:
    model_ret = [r["pnl"]       for r in results]
    bh_ret    = [r["bh_return"] for r in results]

    fig, ax = plt.subplots(figsize=(8, 6))
    ax.scatter(bh_ret, model_ret, alpha=0.55, edgecolors="steelblue",
               facecolors="lightblue", s=40, label="Backtest windows")

    # Best-fit line
    coef = np.polyfit(bh_ret, model_ret, 1)
    xs   = np.linspace(min(bh_ret), max(bh_ret), 200)
    ax.plot(xs, np.poly1d(coef)(xs), "r-", lw=1.5,
            label=f"Fit  y={coef[0]:.3f}x+{coef[1]:.1f}")

    # 45° parity line
    lim = max(abs(min(bh_ret + model_ret)), abs(max(bh_ret + model_ret))) * 1.1
    ax.plot([-lim, lim], [-lim, lim], "k--", lw=0.8, label="Equal return")
    ax.axhline(0, color="grey", lw=0.5)
    ax.axvline(0, color="grey", lw=0.5)

    ax.set_xlabel("Buy & Hold Return (%)")
    ax.set_ylabel("Model Return (%)")
    ax.set_title("NEAT Strategy vs Buy & Hold (random 1-year windows)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"\n  Scatter plot saved ->{out}")


def plot_equity_sample(results: list[dict], out: str = "equity_sample.png") -> None:
    import random
    r = random.choice(results)
    eq = r["equity_curve"]

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.plot(eq, label=f"Strategy  ({r['pnl']:+.1f}%)", color="steelblue")
    ax.axhline(eq[0], color="grey", lw=0.8, linestyle="--")
    ax.set_title(f"Sample equity curve — {r.get('ticker', '')}")
    ax.set_xlabel("Trading day")
    ax.set_ylabel("Portfolio value ($)")
    ax.legend()
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=120)
    plt.close(fig)
    print(f"  Equity curve saved ->{out}")


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate trained NEAT genome")
    parser.add_argument("--genome", default="best_genome.pkl")
    parser.add_argument("--config", default="neat_config.cfg")
    parser.add_argument("--n",      type=int, default=100, help="# backtests")
    parser.add_argument("--window", type=int, default=365, help="Window in days")
    args = parser.parse_args()

    results = run_evaluation(args.genome, args.config, args.n, args.window)
    if not results:
        print("No results — make sure best_genome.pkl exists (run train.py first).")
    else:
        print_table(results)
        plot_comparison(results)
        plot_equity_sample(results)
