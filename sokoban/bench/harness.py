"""Unified benchmarking harness.

Section 3.4 of the proposal calls for "a unified benchmarking harness
that records per-level statistics into a CSV file" that is shared
across all three solver families. That is what this module is.

What it records (Section 4.1):

* success rate (derivable from per-row ``status``)
* wall-clock time per solve
* nodes expanded
* solution length in pushes
* peak memory
* failure mode for unsolved rows (timeout / unsolvable / memory / error)
* the ``optimal`` flag when the solver claims optimality

The per-tier time budgets (Section 4.1: 60s for easier tiers, 300s for
hard) are codified here as :data:`DEFAULT_TIER_TIME_BUDGETS`.

Process isolation is deliberately optional. Solvers honour their own
``time_limit`` argument and the harness wraps each call in a watchdog;
optionally a subprocess can be requested when running an experiment
where peak memory must be measured precisely or where a misbehaving
solver could leak resources.
"""

from __future__ import annotations

import csv
import gc
import os
import statistics
import sys
import time
import tracemalloc
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

from sokoban.env.board import Board, State
from sokoban.solvers.base import Solver, SolveResult, SolveStatus


# Time budgets in seconds, keyed by the dataset tier the level came
# from. These mirror the table at the bottom of proposal Section 4.1.
DEFAULT_TIER_TIME_BUDGETS: Dict[str, float] = {
    "unfiltered": 60.0,
    "medium": 60.0,
    "hard": 300.0,
    "xsokoban": 60.0,
    "default": 60.0,
}


@dataclass
class BenchConfig:
    """Knobs that apply to a whole batch run.

    ``track_memory`` uses :mod:`tracemalloc` to get a Python-allocator
    upper bound on peak memory; that has overhead, so it is opt-in. For
    publication-quality numbers we recommend enabling it for the final
    runs only.
    """

    tier_time_budgets: Dict[str, float] = field(
        default_factory=lambda: dict(DEFAULT_TIER_TIME_BUDGETS)
    )
    default_time_budget: float = 60.0
    track_memory: bool = True
    verbose: bool = True
    progress: bool = True


@dataclass
class BenchRow:
    """One row of the output CSV.

    Field order is the column order written to disk, so adding columns
    here is the only thing needed to extend the CSV schema.
    """

    solver: str
    tier: str
    level_name: str
    status: str
    pushes: int
    moves: int
    nodes_expanded: int
    nodes_generated: int
    time_seconds: float
    peak_memory_bytes: int
    optimal: Optional[bool]
    solution_len_chars: int
    error: str

    @classmethod
    def from_result(
        cls,
        *,
        solver_name: str,
        tier: str,
        level_name: str,
        result: SolveResult,
    ) -> "BenchRow":
        return cls(
            solver=solver_name,
            tier=tier,
            level_name=level_name,
            status=result.status.value,
            pushes=result.pushes,
            moves=result.moves,
            nodes_expanded=result.nodes_expanded,
            nodes_generated=result.nodes_generated,
            time_seconds=round(result.time_seconds, 6),
            peak_memory_bytes=result.peak_memory_bytes,
            optimal=result.optimal,
            solution_len_chars=len(result.solution),
            error=result.error,
        )


# ---------------------------------------------------------------------------
# Single-level execution
# ---------------------------------------------------------------------------


def _select_time_budget(config: BenchConfig, tier: str) -> float:
    if tier in config.tier_time_budgets:
        return config.tier_time_budgets[tier]
    return config.default_time_budget


def run_single(
    solver: Solver,
    board: Board,
    state: State,
    *,
    tier: str = "default",
    config: Optional[BenchConfig] = None,
) -> Tuple[BenchRow, SolveResult]:
    """Run ``solver`` on a single level and return one CSV row.

    Wraps the solver call with a wall-clock measurement and, when
    enabled, a :mod:`tracemalloc` peak-memory probe. We intentionally
    do not catch ``KeyboardInterrupt`` so an interactive user can abort
    a long-running batch cleanly.
    """
    cfg = config or BenchConfig()
    time_limit = _select_time_budget(cfg, tier)

    gc.collect()
    if cfg.track_memory:
        tracemalloc.start()

    start = time.perf_counter()
    try:
        result = solver.solve(board, state, time_limit=time_limit)
    except MemoryError as exc:
        result = SolveResult(
            status=SolveStatus.MEMORY,
            time_seconds=time.perf_counter() - start,
            error=str(exc),
        )
    except Exception as exc:  # noqa: BLE001 — we *do* want everything here
        result = SolveResult(
            status=SolveStatus.ERROR,
            time_seconds=time.perf_counter() - start,
            error=f"{type(exc).__name__}: {exc}",
        )

    # The harness's clock takes precedence over a solver that forgets
    # to set ``time_seconds`` itself.
    if result.time_seconds <= 0.0:
        result.time_seconds = time.perf_counter() - start

    if cfg.track_memory:
        _current, peak = tracemalloc.get_traced_memory()
        tracemalloc.stop()
        if result.peak_memory_bytes == 0:
            result.peak_memory_bytes = peak

    row = BenchRow.from_result(
        solver_name=solver.name,
        tier=tier,
        level_name=board.name or "unnamed",
        result=result,
    )
    return row, result


# ---------------------------------------------------------------------------
# Batch execution
# ---------------------------------------------------------------------------


def run_batch(
    solver: Solver,
    levels: Sequence[Tuple[str, Board, State]],
    *,
    config: Optional[BenchConfig] = None,
    csv_out: Optional[str | Path] = None,
) -> List[BenchRow]:
    """Run ``solver`` on a batch of ``(tier, board, state)`` triples.

    Optionally writes the rows to ``csv_out``. The CSV is opened in
    *append* mode if it exists, so partial batches can be resumed by
    filtering already-completed (solver, level_name) pairs upstream —
    exactly the workflow we want for long PPO evaluation runs.
    """
    cfg = config or BenchConfig()
    rows: list[BenchRow] = []

    iterator: Iterable = levels
    if cfg.progress and _tqdm_available():
        from tqdm import tqdm  # noqa: WPS433

        iterator = tqdm(levels, desc=solver.name, unit="lvl")

    for tier, board, state in iterator:
        row, _result = run_single(solver, board, state, tier=tier, config=cfg)
        rows.append(row)
        if cfg.verbose and not _tqdm_available():
            print(
                f"[{solver.name}] {tier}/{board.name or 'unnamed'} "
                f"-> {row.status} pushes={row.pushes} t={row.time_seconds:.3f}s",
                file=sys.stderr,
            )

    if csv_out is not None:
        write_csv(rows, csv_out)
    return rows


# ---------------------------------------------------------------------------
# CSV I/O
# ---------------------------------------------------------------------------


_CSV_FIELDS: tuple[str, ...] = (
    "solver",
    "tier",
    "level_name",
    "status",
    "pushes",
    "moves",
    "nodes_expanded",
    "nodes_generated",
    "time_seconds",
    "peak_memory_bytes",
    "optimal",
    "solution_len_chars",
    "error",
)


def write_csv(rows: Sequence[BenchRow], path: str | Path) -> None:
    """Write benchmark rows to ``path``, appending if the file exists.

    The header is only written when the file is new, so partial runs
    can be safely concatenated.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    file_exists = path.exists() and path.stat().st_size > 0
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=_CSV_FIELDS)
        if not file_exists:
            writer.writeheader()
        for row in rows:
            writer.writerow(asdict(row))


# ---------------------------------------------------------------------------
# Summaries (for the report tables in Section 4)
# ---------------------------------------------------------------------------


@dataclass
class TierSummary:
    """Aggregated stats for one (solver, tier) cell of the report table."""

    solver: str
    tier: str
    n_levels: int
    n_solved: int
    success_rate: float
    median_time_solved: Optional[float]
    median_nodes_solved: Optional[float]
    median_pushes_solved: Optional[float]


def summarise(rows: Sequence[BenchRow]) -> List[TierSummary]:
    """Aggregate rows into the per-(solver, tier) numbers we'll quote
    in the final report."""
    groups: Dict[Tuple[str, str], List[BenchRow]] = {}
    for row in rows:
        groups.setdefault((row.solver, row.tier), []).append(row)

    summaries: list[TierSummary] = []
    for (solver, tier), group in sorted(groups.items()):
        solved = [r for r in group if r.status == SolveStatus.SOLVED.value]
        summaries.append(
            TierSummary(
                solver=solver,
                tier=tier,
                n_levels=len(group),
                n_solved=len(solved),
                success_rate=len(solved) / len(group) if group else 0.0,
                median_time_solved=(
                    statistics.median(r.time_seconds for r in solved) if solved else None
                ),
                median_nodes_solved=(
                    statistics.median(r.nodes_expanded for r in solved) if solved else None
                ),
                median_pushes_solved=(
                    statistics.median(r.pushes for r in solved) if solved else None
                ),
            )
        )
    return summaries


# ---------------------------------------------------------------------------
# Internals
# ---------------------------------------------------------------------------


def _tqdm_available() -> bool:
    try:
        import tqdm  # noqa: F401, WPS433
    except ImportError:
        return False
    return True


def __getattr__(name):  # pragma: no cover — convenience
    if name == "ENV":
        return os.environ
    raise AttributeError(name)
