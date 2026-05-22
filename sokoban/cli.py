"""Command-line entry point.

Exposes the four sub-commands the rest of the team needs day to day:

* ``info``      — inspect a level file (count levels, dims, box/goal balance)
* ``play``      — read a level, print it as ASCII (sanity check the parser)
* ``solve``     — run a solver on a single level and print metrics
* ``animate``   — replay a solution string and write a GIF/MP4
* ``benchmark`` — run a solver across a level set, write a CSV

Solvers are looked up by name in :data:`SOLVERS`; Lakshya and Jakob
register their implementations there so the CLI doesn't have to know
the algorithmic details. Right now only the reference ``bfs-push``
solver is wired in.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Callable, Dict, List, Tuple

from sokoban.bench.harness import (
    BenchConfig,
    DEFAULT_TIER_TIME_BUDGETS,
    run_batch,
    run_single,
    summarise,
)
from sokoban.data.boxoban import sample_boxoban
from sokoban.data.xsokoban import load_xsokoban_curated, load_xsokoban_file
from sokoban.env.board import Board, State
from sokoban.env.moves import is_solved
from sokoban.env.parser import parse_xsb_file
from sokoban.solvers.base import Solver
from sokoban.solvers.bfs import BFSPushSolver
from sokoban.solvers.astar import (
    astar_baseline,
    astar_dead,
    astar_freeze,
    astar_tunnels,
    astar_full,
)
from sokoban.solvers.idastar import idastar_baseline, idastar_full
from sokoban.viz.animate import animate_solution
from sokoban.viz.render import render_ascii


# Registry of solver factories. Each entry is a no-arg constructor so
# the CLI can spin a fresh solver per invocation; this matters for
# stateful solvers (transposition tables, learned policies) that
# should not leak state across runs.
SOLVERS: Dict[str, Callable[[], Solver]] = {
    "bfs-push": BFSPushSolver,
    "astar": astar_baseline,
    "astar+dead": astar_dead,
    "astar+freeze": astar_freeze,
    "astar+tunnels": astar_tunnels,
    "astar+all": astar_full,
    "idastar": idastar_baseline,
    "idastar+all": idastar_full,
}


# ---------------------------------------------------------------------------
# Level loading helpers
# ---------------------------------------------------------------------------


def _load_level_source(source: str) -> List[Tuple[str, Board, State]]:
    """Resolve a CLI level source into a list of ``(tier, board, state)``.

    Recognises:

    * ``boxoban:<tier>[:<split>][@<n>]`` — sample N (default 200) from a tier
    * ``xsokoban`` or ``xsokoban-curated`` — the 30 curated XSokoban levels
    * any filesystem path to a ``.xsb`` file (single or multi-level)
    """
    if source.startswith("boxoban:"):
        spec = source[len("boxoban:"):]
        n: int = 200
        if "@" in spec:
            spec, n_str = spec.rsplit("@", 1)
            n = int(n_str)
        if ":" in spec:
            tier, split = spec.split(":", 1)
        else:
            tier, split = spec, "valid"
        levels = sample_boxoban(tier, n, split=split)
        return [(tier, b, s) for (b, s) in levels]

    if source in ("xsokoban", "xsokoban-curated"):
        levels = load_xsokoban_curated()
        return [("xsokoban", b, s) for (b, s) in levels]

    path = Path(source)
    if path.exists():
        levels = parse_xsb_file(path)
        return [("custom", b, s) for (b, s) in levels]

    raise SystemExit(f"unrecognised level source: {source!r}")


def _get_solver(name: str) -> Solver:
    if name not in SOLVERS:
        raise SystemExit(
            f"unknown solver {name!r}; available: {sorted(SOLVERS)}"
        )
    return SOLVERS[name]()


# ---------------------------------------------------------------------------
# Sub-commands
# ---------------------------------------------------------------------------


def cmd_info(args: argparse.Namespace) -> int:
    triples = _load_level_source(args.source)
    print(f"source: {args.source}")
    print(f"levels: {len(triples)}")
    if not triples:
        return 0
    sizes = sorted({(b.height, b.width) for _, b, _ in triples})
    print(f"distinct sizes (HxW): {sizes[:8]}{' ...' if len(sizes) > 8 else ''}")
    box_counts = [len(s.boxes) for _, _, s in triples]
    print(f"boxes per level: min={min(box_counts)} max={max(box_counts)}")
    return 0


def cmd_play(args: argparse.Namespace) -> int:
    triples = _load_level_source(args.source)
    if args.index >= len(triples):
        raise SystemExit(f"index {args.index} out of range (have {len(triples)} levels)")
    tier, board, state = triples[args.index]
    print(f"# {board.name} (tier={tier})")
    print(render_ascii(board, state))
    print(f"# solved={is_solved(board, state)} boxes={len(state.boxes)} goals={len(board.goals)}")
    return 0


def cmd_solve(args: argparse.Namespace) -> int:
    solver = _get_solver(args.solver)
    triples = _load_level_source(args.source)
    if args.index >= len(triples):
        raise SystemExit(f"index {args.index} out of range (have {len(triples)} levels)")
    tier, board, state = triples[args.index]
    config = BenchConfig(track_memory=False, verbose=False, progress=False)
    row, result = run_single(solver, board, state, tier=tier, config=config)
    print(f"level:       {board.name}")
    print(f"solver:      {solver.name}")
    print(f"status:      {row.status}")
    print(f"pushes:      {row.pushes}")
    print(f"moves:       {row.moves}")
    print(f"time:        {row.time_seconds:.3f}s")
    print(f"expanded:    {row.nodes_expanded}")
    print(f"generated:   {row.nodes_generated}")
    print(f"optimal:     {row.optimal}")
    if result.solution:
        print(f"solution:    {result.solution}")
    if args.animate:
        out_path = Path(args.animate)
        animate_solution(board, state, result.solution, out_path=out_path)
        print(f"animation written to {out_path}")
    return 0 if row.status == "solved" else 1


def cmd_animate(args: argparse.Namespace) -> int:
    triples = _load_level_source(args.source)
    if args.index >= len(triples):
        raise SystemExit(f"index {args.index} out of range (have {len(triples)} levels)")
    _tier, board, state = triples[args.index]
    out_path = Path(args.out)
    animate_solution(board, state, args.solution, out_path=out_path, fps=args.fps)
    print(f"wrote {out_path}")
    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    solver = _get_solver(args.solver)
    triples: List[Tuple[str, Board, State]] = []
    for source in args.sources:
        triples.extend(_load_level_source(source))
    if not triples:
        raise SystemExit("no levels to benchmark")

    config = BenchConfig(
        tier_time_budgets=dict(DEFAULT_TIER_TIME_BUDGETS),
        default_time_budget=args.time_limit,
        track_memory=not args.no_memory,
        verbose=True,
        progress=not args.no_progress,
    )
    if args.time_limit_unfiltered is not None:
        config.tier_time_budgets["unfiltered"] = args.time_limit_unfiltered
    if args.time_limit_medium is not None:
        config.tier_time_budgets["medium"] = args.time_limit_medium
    if args.time_limit_hard is not None:
        config.tier_time_budgets["hard"] = args.time_limit_hard

    rows = run_batch(solver, triples, config=config, csv_out=args.csv)
    print()
    print(f"{'solver':<14} {'tier':<12} {'solved/total':<14} {'success':<8} "
          f"{'med_time':<10} {'med_nodes':<10} {'med_pushes':<10}")
    for s in summarise(rows):
        med_time = "-" if s.median_time_solved is None else f"{s.median_time_solved:.3f}s"
        med_nodes = "-" if s.median_nodes_solved is None else f"{int(s.median_nodes_solved)}"
        med_pushes = "-" if s.median_pushes_solved is None else f"{int(s.median_pushes_solved)}"
        print(f"{s.solver:<14} {s.tier:<12} {s.n_solved}/{s.n_levels:<10} "
              f"{s.success_rate * 100:>5.1f}%   {med_time:<10} {med_nodes:<10} {med_pushes:<10}")
    if args.csv:
        print(f"\nrows appended to {args.csv}")
    return 0


# ---------------------------------------------------------------------------
# Argument parsing
# ---------------------------------------------------------------------------


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="sokoban", description=__doc__)
    sub = p.add_subparsers(dest="command", required=True)

    info = sub.add_parser("info", help="inspect a level source")
    info.add_argument("source", help="path to .xsb, 'xsokoban', or 'boxoban:<tier>'")
    info.set_defaults(func=cmd_info)

    play = sub.add_parser("play", help="print a level to stdout")
    play.add_argument("source")
    play.add_argument("--index", type=int, default=0)
    play.set_defaults(func=cmd_play)

    solve = sub.add_parser("solve", help="solve a single level")
    solve.add_argument("source")
    solve.add_argument("--index", type=int, default=0)
    solve.add_argument("--solver", default="bfs-push", choices=sorted(SOLVERS))
    solve.add_argument(
        "--animate",
        default=None,
        help="optional .gif/.mp4 path; writes a step-by-step animation",
    )
    solve.set_defaults(func=cmd_solve)

    anim = sub.add_parser("animate", help="render a known solution string")
    anim.add_argument("source")
    anim.add_argument("--index", type=int, default=0)
    anim.add_argument("--solution", required=True)
    anim.add_argument("--out", required=True)
    anim.add_argument("--fps", type=int, default=4)
    anim.set_defaults(func=cmd_animate)

    bench = sub.add_parser("benchmark", help="run a solver across a level set")
    bench.add_argument(
        "sources",
        nargs="+",
        help="one or more level sources (e.g. 'xsokoban', 'boxoban:medium@200')",
    )
    bench.add_argument("--solver", default="bfs-push", choices=sorted(SOLVERS))
    bench.add_argument(
        "--csv",
        default=None,
        help="optional output CSV (appended to if it exists)",
    )
    bench.add_argument(
        "--time-limit",
        type=float,
        default=60.0,
        help="default per-level time budget (seconds)",
    )
    bench.add_argument("--time-limit-unfiltered", type=float, default=None)
    bench.add_argument("--time-limit-medium", type=float, default=None)
    bench.add_argument("--time-limit-hard", type=float, default=None)
    bench.add_argument("--no-memory", action="store_true")
    bench.add_argument("--no-progress", action="store_true")
    bench.set_defaults(func=cmd_benchmark)

    return p


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    return int(args.func(args) or 0)


if __name__ == "__main__":
    sys.exit(main())
