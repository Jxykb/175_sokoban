"""End-to-end smoke test: BFS solves a few tiny levels and we replay
the solution string to verify it actually solves them.

This catches integration bugs across the whole stack — parser, push
generator, BFS, trace serialisation, replay — that the unit tests can
miss in isolation.
"""

from __future__ import annotations

import pytest

from sokoban.bench.harness import BenchConfig, run_single, summarise
from sokoban.env.moves import is_solved, replay_trace
from sokoban.env.parser import parse_xsb
from sokoban.solvers.bfs import BFSPushSolver


TINY_LEVELS = [
    """\
#####
#@$.#
#####
""",
    """\
########
#      #
# $@$. #
#   .  #
########
""",
    """\
#######
#  .  #
# $@$ #
#  .  #
#######
""",
]


@pytest.mark.parametrize("text", TINY_LEVELS)
def test_bfs_solves_and_solution_replays(text):
    board, state = parse_xsb(text)
    solver = BFSPushSolver()
    row, result = run_single(
        solver, board, state, tier="default",
        config=BenchConfig(track_memory=False, verbose=False, progress=False),
    )
    assert row.status == "solved"
    assert result.solution
    states = replay_trace(board, state, result.solution)
    assert is_solved(board, states[-1])
    # Move count and push count must match the encoded solution.
    assert row.pushes == sum(1 for ch in result.solution if ch.isupper())
    assert row.moves == len(result.solution)


def test_bench_summary_groups_by_tier():
    solver = BFSPushSolver()
    cfg = BenchConfig(track_memory=False, verbose=False, progress=False)
    rows = []
    for text in TINY_LEVELS:
        board, state = parse_xsb(text)
        row, _ = run_single(solver, board, state, tier="default", config=cfg)
        rows.append(row)
    summaries = summarise(rows)
    assert len(summaries) == 1
    s = summaries[0]
    assert s.solver == "bfs-push"
    assert s.tier == "default"
    assert s.n_solved == len(TINY_LEVELS)
    assert s.success_rate == 1.0
