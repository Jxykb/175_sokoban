"""Integration tests for the classical-search solvers.

We run A* (both baseline and full-pruning) and IDA* on the same small
levels we use for the BFS smoke test, then verify each solution is a
real solution by replaying it through the environment.

Pushes returned by A*/IDA* must be at least as small as BFS's optimal
push count (they are all optimal in pushes by construction, so the
counts should be exactly equal).
"""

from __future__ import annotations

import pytest

from sokoban.env.moves import is_solved, replay_trace
from sokoban.env.parser import parse_xsb
from sokoban.solvers.astar import astar_baseline, astar_full
from sokoban.solvers.bfs import BFSPushSolver
from sokoban.solvers.idastar import idastar_full


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
def test_astar_full_matches_bfs_push_count(text):
    board, state = parse_xsb(text)
    bfs_result = BFSPushSolver().solve(board, state, time_limit=5.0)
    astar_result = astar_full().solve(board, state, time_limit=5.0)
    assert astar_result.status.value == "solved"
    assert astar_result.pushes == bfs_result.pushes
    states = replay_trace(board, state, astar_result.solution)
    assert is_solved(board, states[-1])


@pytest.mark.parametrize("text", TINY_LEVELS)
def test_astar_baseline_solves_too(text):
    board, state = parse_xsb(text)
    result = astar_baseline().solve(board, state, time_limit=5.0)
    assert result.status.value == "solved"
    states = replay_trace(board, state, result.solution)
    assert is_solved(board, states[-1])


@pytest.mark.parametrize("text", TINY_LEVELS)
def test_idastar_full_solves(text):
    board, state = parse_xsb(text)
    result = idastar_full().solve(board, state, time_limit=5.0)
    assert result.status.value == "solved"
    states = replay_trace(board, state, result.solution)
    assert is_solved(board, states[-1])
