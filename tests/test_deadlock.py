"""Tests for dead-square and freeze-deadlock detection."""

from __future__ import annotations

from sokoban.env.board import State
from sokoban.env.parser import parse_xsb
from sokoban.solvers.deadlock import (
    dead_squares,
    has_box_on_dead_square,
    is_freeze_deadlock,
)


def test_corner_is_dead_square():
    # Single dummy box just so the parser is happy; we only inspect
    # the static dead-square map of the *board*.
    text = """\
######
#@$  #
#   .#
######
"""
    board, _state = parse_xsb(text)
    dead = dead_squares(board)
    # Top-right corner (1,4) is a non-goal corner: any box there
    # could only be pushed back into the corner — it is dead.
    assert (1, 4) in dead
    # Bottom-left corner (2,1) is a non-goal corner too — dead.
    assert (2, 1) in dead
    # The goal at (2,4) is never marked dead.
    assert (2, 4) not in dead


def test_box_on_dead_square_detected():
    text = """\
######
#$  .#
#@   #
######
"""
    board, state = parse_xsb(text)
    # The box at (1,1) sits in a top-left corner — dead.
    assert has_box_on_dead_square(board, state)


def test_freeze_two_boxes_corner():
    # Three boxes wedged in the top-left corner mutually block each
    # other. Goals live elsewhere so none of the boxes is on a goal
    # (which would suppress the freeze check).
    text = """\
######
#$$  #
#$  .#
#  @.#
#   .#
######
"""
    board, state = parse_xsb(text)
    assert is_freeze_deadlock(board, state)


def test_freeze_clear_state_is_not_deadlock():
    text = """\
#######
#  .  #
# $@$ #
#  .  #
#######
"""
    board, state = parse_xsb(text)
    assert not is_freeze_deadlock(board, state)
