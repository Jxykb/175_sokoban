"""Standalone RL evaluation.

Runs a trained PPO checkpoint across the Boxoban evaluation tiers (or
the curated XSokoban set) and writes per-level rows to the shared
benchmark CSV. The eval respects the proposal's Section 4.1 metrics
and per-tier time budgets: each level gets a fixed step budget,
solved/timeout counts feed into the same ``summarise()`` helper the
classical solvers use.

This is what produces the "standalone RL" rows in Table 3 of the
final report. The hybrid solver (:class:`sokoban.solvers.hybrid.HybridSolver`)
runs through the regular ``sokoban benchmark`` CLI like any other
solver.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Optional

from sokoban.bench.harness import BenchConfig, run_batch
from sokoban.data.boxoban import sample_boxoban
from sokoban.data.xsokoban import load_xsokoban_curated
from sokoban.rl import is_available


def _require_rl() -> None:
    if not is_available():
        sys.stderr.write(
            "error: the [rl] extra is not installed.\n"
            "       pip install -e '.[rl]'\n"
        )
        sys.exit(2)


def evaluate(
    *,
    checkpoint: str,
    tiers: list[str],
    n_per_tier: int,
    grid_size: int,
    max_steps: int,
    csv_out: Optional[str] = None,
    seed: int = 175,
) -> None:
    """Evaluate a PPO checkpoint across the given tiers."""
    _require_rl()
    from sokoban.solvers.ppo import PPOSolver

    triples: list = []
    for tier in tiers:
        if tier == "xsokoban":
            levels = load_xsokoban_curated()
        else:
            levels = sample_boxoban(tier, n_per_tier, seed=seed)
        for board, state in levels:
            triples.append((tier, board, state))

    solver = PPOSolver(
        checkpoint=checkpoint,
        grid_size=grid_size,
        max_steps=max_steps,
    )

    cfg = BenchConfig(track_memory=False, verbose=False, progress=True)
    rows = run_batch(solver, triples, config=cfg, csv_out=csv_out)

    print(f"\nevaluated {len(rows)} levels across tiers={tiers}")
    if csv_out:
        print(f"rows appended to {csv_out}")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", required=True, help="path to PPO .zip")
    parser.add_argument(
        "--tiers",
        nargs="+",
        default=["unfiltered", "medium", "hard"],
        help="dataset tiers to evaluate on",
    )
    parser.add_argument("--n-per-tier", type=int, default=200)
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--csv", default=None)
    parser.add_argument("--seed", type=int, default=175)
    args = parser.parse_args(argv)

    evaluate(
        checkpoint=args.checkpoint,
        tiers=args.tiers,
        n_per_tier=args.n_per_tier,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        csv_out=args.csv,
        seed=args.seed,
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
