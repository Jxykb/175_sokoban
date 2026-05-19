"""Parser tests.

These cover the XSB character map (proposal Section 2), error handling
for malformed inputs, multi-level files (DeepMind Boxoban-style), and
round-trip rendering.
"""

from __future__ import annotations

import pytest

from sokoban.env.parser import (
    ParseError,
    board_to_xsb,
    parse_xsb,
    parse_xsb_collection,
)


TRIVIAL = """\
#####
#@$.#
#####
"""


def test_parse_trivial():
    board, state = parse_xsb(TRIVIAL, name="trivial")
    assert board.height == 3
    assert board.width == 5
    assert state.player == (1, 1)
    assert state.boxes == frozenset({(1, 2)})
    assert board.goals == frozenset({(1, 3)})
    assert board.name == "trivial"


def test_parse_player_on_goal_and_box_on_goal():
    # Box-on-goal contributes to BOTH counts, so we add a plain box and
    # a plain goal to keep the boxes==goals invariant satisfied while
    # still exercising '*' and '+' tiles.
    text = """\
######
#$ *.#
# +$ #
######
"""
    board, state = parse_xsb(text)
    assert (1, 4) in board.goals       # plain '.'
    assert (1, 3) in board.goals       # '*' contributes a goal here
    assert (1, 3) in state.boxes       # '*' also contributes a box here
    assert (2, 2) in board.goals       # '+' = player on a goal
    assert state.player == (2, 2)
    assert (1, 1) in state.boxes       # plain '$'
    assert (2, 3) in state.boxes       # plain '$'


def test_parse_rejects_unbalanced_boxes_and_goals():
    text = """\
#####
#@$ #
#####
"""
    with pytest.raises(ParseError):
        parse_xsb(text)


def test_parse_rejects_missing_player():
    text = """\
#####
# $.#
#####
"""
    with pytest.raises(ParseError):
        parse_xsb(text)


def test_parse_rejects_double_player():
    text = """\
######
#@$.@#
######
"""
    with pytest.raises(ParseError):
        parse_xsb(text)


def test_collection_separated_by_blank_lines():
    text = """\
#####
#@$.#
#####

#####
#@$.#
#####
"""
    levels = parse_xsb_collection(text, name_prefix="t")
    assert len(levels) == 2
    assert levels[0][0].name.startswith("t_")
    assert levels[1][0].name.startswith("t_")


def test_collection_with_boxoban_style_headers():
    text = """\
; 0
#####
#@$.#
#####
; 1
######
#@ $.#
######
"""
    levels = parse_xsb_collection(text)
    assert len(levels) == 2
    # The leading-';' line preserves the id as the level name.
    assert levels[0][0].name == "0"
    assert levels[1][0].name == "1"


def test_round_trip_render():
    board, state = parse_xsb(TRIVIAL)
    rendered = board_to_xsb(board, state)
    # Rebuilding from rendered text should produce identical static layout.
    rb, rs = parse_xsb(rendered)
    assert rb.walls == board.walls
    assert rb.goals == board.goals
    assert rs.player == state.player
    assert rs.boxes == state.boxes
