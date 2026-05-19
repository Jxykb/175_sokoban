"""Tests for the move and push generators and the solution-trace codec."""

from __future__ import annotations

import pytest

from sokoban.env.board import Action
from sokoban.env.moves import (
    apply_action,
    apply_push,
    is_solved,
    legal_actions,
    legal_pushes,
    player_reachable,
    push_path_to_moves,
    replay_trace,
    trace_to_string,
)
from sokoban.env.parser import parse_xsb


TRIVIAL = """\
#####
#@$.#
#####
"""

TWO_STEP = """\
######
#@$ .#
######
"""


def test_legal_actions_single_box():
    board, state = parse_xsb(TRIVIAL)
    actions = legal_actions(board, state)
    # The player can only push right; up/down hit walls, left hits wall.
    assert actions == [Action.RIGHT]


def test_apply_action_push_and_solve():
    board, state = parse_xsb(TRIVIAL)
    new_state, pushed = apply_action(board, state, Action.RIGHT)
    assert pushed
    assert new_state.player == (1, 2)
    assert new_state.boxes == frozenset({(1, 3)})
    assert is_solved(board, new_state)


def test_apply_action_blocked_raises():
    board, state = parse_xsb(TRIVIAL)
    with pytest.raises(ValueError):
        apply_action(board, state, Action.LEFT)


def test_trace_codec_round_trip():
    board, state = parse_xsb(TWO_STEP)
    # Right push, right move, right move: solution should be "R" then "rr"
    s1, p1 = apply_action(board, state, Action.RIGHT)
    assert p1
    s2, p2 = apply_action(board, s1, Action.RIGHT)
    # The second step is also a push because the box is right of player.
    assert p2
    steps = [(Action.RIGHT, True), (Action.RIGHT, True)]
    encoded = trace_to_string(steps)
    assert encoded == "RR"
    # Replay should reach a solved state.
    states = replay_trace(board, state, encoded)
    assert is_solved(board, states[-1])


def test_push_space_basic():
    board, state = parse_xsb(TRIVIAL)
    pushes = list(legal_pushes(board, state))
    # Single legal push of the only box, rightwards.
    assert len(pushes) == 1
    box, action, next_state = pushes[0]
    assert box == (1, 2)
    assert action == Action.RIGHT
    after = apply_push(board, state, box, action)
    assert after == next_state


def test_push_path_to_moves_includes_walk_steps():
    # Player is three cells away from the box and must walk before
    # pushing. The push itself takes the box onto the goal.
    text = """\
########
#@   $.#
########
"""
    board, state = parse_xsb(text)
    pushes = [((1, 5), Action.RIGHT)]
    moves = push_path_to_moves(board, state, pushes)
    encoded = trace_to_string(moves)
    # 3 walk steps to reach (1,4), then 1 push that lands the box on (1,6).
    assert encoded == "rrrR"
    states = replay_trace(board, state, encoded)
    assert is_solved(board, states[-1])


def test_player_reachable_excludes_boxes_and_walls():
    board, state = parse_xsb(TWO_STEP)
    reachable = player_reachable(board, state)
    # Player can stand at (1,1); cannot stand on the box at (1,2);
    # everything else in the corridor right of the box is blocked too.
    assert (1, 1) in reachable
    assert (1, 2) not in reachable
    assert (1, 3) not in reachable
