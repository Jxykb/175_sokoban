"""Hungarian-heuristic and push-distance-map tests."""

from __future__ import annotations

import math

from sokoban.env.parser import parse_xsb
from sokoban.solvers.heuristics import (
    closest_goal_heuristic,
    hungarian,
    hungarian_heuristic,
    push_distance_maps,
)


TRIVIAL = """\
#####
#@$.#
#####
"""


def test_push_distance_map_trivial():
    board, _state = parse_xsb(TRIVIAL)
    maps = push_distance_maps(board)
    assert (1, 3) in maps  # goal
    # The box at (1,2) is one push away from the goal at (1,3).
    assert maps[(1, 3)].get((1, 2)) == 1
    assert maps[(1, 3)].get((1, 3)) == 0


def test_hungarian_square_simple():
    cost = [[1, 2], [3, 1]]
    total, assign = hungarian(cost)
    assert total == 2
    assert assign in ([0, 1], [1, 0])


def test_hungarian_infeasibility_returns_inf():
    cost = [[math.inf, math.inf], [1, 2]]
    total, _assign = hungarian(cost)
    assert math.isinf(total)


def test_hungarian_heuristic_at_least_as_tight_as_closest_goal():
    # Hungarian is at least as tight as the trivial closest-goal sum
    # on any state — by construction it solves the same problem
    # without allowing two boxes to share a goal.
    text = """\
#######
#  .  #
#.$ $.#
#  $  #
#  @  #
#######
"""
    board, state = parse_xsb(text)
    h_hungarian = hungarian_heuristic(board, state)
    h_closest = closest_goal_heuristic(board, state)
    assert h_hungarian >= h_closest
    # The true optimum on this layout is 4 pushes (1 + 1 + 2 for the
    # three boxes' shortest single-box paths); Hungarian must not
    # exceed it to remain admissible.
    assert h_hungarian <= 4
