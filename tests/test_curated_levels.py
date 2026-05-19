"""Sanity check: all 30 curated XSokoban levels load cleanly.

This guards the Section 4.2 deliverable ("30 hand-selected XSokoban
levels") against accidental edits that would break the benchmark set.
"""

from __future__ import annotations

from sokoban.data.xsokoban import load_xsokoban_curated


def test_curated_count():
    levels = load_xsokoban_curated()
    assert len(levels) == 30


def test_curated_levels_are_well_formed():
    for board, state in load_xsokoban_curated():
        assert len(state.boxes) == len(board.goals), (
            f"box/goal mismatch in {board.name}: "
            f"{len(state.boxes)} boxes vs {len(board.goals)} goals"
        )
        # Player must stand on a floor cell.
        assert state.player in board.floor, f"player off-floor in {board.name}"
        # Every box must stand on a floor cell.
        for b in state.boxes:
            assert b in board.floor, f"box {b} off-floor in {board.name}"
