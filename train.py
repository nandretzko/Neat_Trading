"""
Main training script — NEAT with progressive training data (§2.5).

Progressive schedule (paper §2.5):
    Stage 1  → 1 500 generations, 90-day windows   (learn trend reversals)
    Stage 2  →   400 generations, 150-day windows  (intermediate horizon)
    Stage 3  →   100 generations, 365-day windows  (capacity & risk mgmt)

Usage:
    python train.py              # full training
    python train.py --fast       # 50/20/10 gens (smoke-test)
"""
from __future__ import annotations

import argparse
import os
import pickle

import neat

from data_loader import download_all_stocks, get_random_window
from fitness import compute_fitness
from trading_env import TradingEnvironment

# ── Training schedule ────────────────────────────────────────────────
STAGE1_GENS = 1500   # 90-day windows
STAGE2_GENS = 400    # 150-day windows
STAGE3_GENS = 100    # 365-day windows

CHECKPOINT_DIR = "checkpoints"
os.makedirs(CHECKPOINT_DIR, exist_ok=True)

# Populated once in main(), then read by workers
_stocks: dict = {}
_window_days: int = 90


# ── Genome evaluation ────────────────────────────────────────────────

def eval_genome(genome, config) -> float:
    """Evaluate a single genome on a randomly sampled stock window."""
    net = neat.nn.RecurrentNetwork.create(genome, config)
    df, _ = get_random_window(_stocks, _window_days)
    if df is None or len(df) < _window_days + TradingEnvironment.WARMUP:
        return -1_000.0
    env = TradingEnvironment(df)
    metrics = env.run(net)
    return compute_fitness(metrics)


def eval_genomes(genomes, config) -> None:
    for _, genome in genomes:
        genome.fitness = eval_genome(genome, config)


# ── Training loop ────────────────────────────────────────────────────

class StageReporter(neat.reporting.BaseReporter):
    """Prints current training stage at each generation.

    Receives the actual generation boundaries so it works correctly in both
    full and --fast modes.
    """

    def __init__(self, s1: int, s2: int) -> None:
        self._s1 = s1          # end of stage 1 (exclusive)
        self._s1s2 = s1 + s2   # end of stage 2 (exclusive)

    def start_generation(self, gen: int) -> None:
        if gen < self._s1:
            stage, w = 1, 90
        elif gen < self._s1s2:
            stage, w = 2, 150
        else:
            stage, w = 3, 365
        print(f"[Gen {gen:5d}] Stage {stage} | window = {w} days", flush=True)


def run_training(fast: bool = False) -> None:
    global _stocks, _window_days

    s1 = 50  if fast else STAGE1_GENS
    s2 = 20  if fast else STAGE2_GENS
    s3 = 10  if fast else STAGE3_GENS

    # ── Data ──────────────────────────────────────────────────────────
    print("Downloading / loading stock data …")
    _stocks = download_all_stocks()
    print(f"  {len(_stocks)} stocks ready.\n")

    # ── NEAT config ───────────────────────────────────────────────────
    config = neat.Config(
        neat.DefaultGenome,
        neat.DefaultReproduction,
        neat.DefaultSpeciesSet,
        neat.DefaultStagnation,
        "neat_config.cfg",
    )

    # ── Population (restore checkpoint if available) ──────────────────
    latest = os.path.join(CHECKPOINT_DIR, "latest")
    if os.path.exists(latest):
        print(f"Restoring checkpoint from {latest} …")
        pop = neat.Checkpointer.restore_checkpoint(latest)
    else:
        pop = neat.Population(config)

    pop.add_reporter(neat.StdOutReporter(True))
    pop.add_reporter(neat.StatisticsReporter())
    pop.add_reporter(StageReporter(s1, s2))
    pop.add_reporter(
        neat.Checkpointer(
            generation_interval=50,
            filename_prefix=os.path.join(CHECKPOINT_DIR, "ckpt-"),
        )
    )

    # ── Stage 1: 90-day windows ───────────────────────────────────────
    print(f"\n=== Stage 1 ({s1} generations, 90-day windows) ===")
    _window_days = 90
    winner = pop.run(eval_genomes, s1)

    # ── Stage 2: 150-day windows ──────────────────────────────────────
    print(f"\n=== Stage 2 ({s2} generations, 150-day windows) ===")
    _window_days = 150
    winner = pop.run(eval_genomes, s2)

    # ── Stage 3: 365-day windows ──────────────────────────────────────
    print(f"\n=== Stage 3 ({s3} generations, 365-day windows) ===")
    _window_days = 365
    winner = pop.run(eval_genomes, s3)

    # ── Save best genome ──────────────────────────────────────────────
    with open("best_genome.pkl", "wb") as f:
        pickle.dump(winner, f)
    # Also save config so evaluate.py can reconstruct the network
    with open("neat_config_used.pkl", "wb") as f:
        pickle.dump(config, f)

    print("\n✓ Training complete.")
    print(f"  Best genome saved  → best_genome.pkl")
    print(f"  Best fitness       = {winner.fitness:.4f}")

    # Copy the most recently written ckpt-* file to checkpoints/latest so the
    # next run can restore it. neat.Checkpointer appends the generation number
    # to the prefix, so we cannot rely on a predictable fixed filename.
    import glob
    import shutil

    ckpt_files = glob.glob(os.path.join(CHECKPOINT_DIR, "ckpt-*"))
    if ckpt_files:
        newest = max(ckpt_files, key=os.path.getmtime)
        shutil.copy2(newest, os.path.join(CHECKPOINT_DIR, "latest"))
        print(f"  Latest checkpoint → checkpoints/latest  (copied from {os.path.basename(newest)})")


# ── Entry point ───────────────────────────────────────────────────────

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train NEAT trading agent")
    parser.add_argument(
        "--fast",
        action="store_true",
        help="Run 50/20/10 generations instead of full schedule (quick test)",
    )
    args = parser.parse_args()
    run_training(fast=args.fast)
