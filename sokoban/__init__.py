"""CS175 Sokoban solver package.

Public API re-exports the most common classes and helpers so that
downstream code (solvers, visualizer, benchmarks) can ``from sokoban import ...``
without reaching into submodules.
"""

from sokoban.env.board import Board, State, Action, ACTIONS, DXDY
from sokoban.env.parser import parse_xsb, parse_xsb_collection, board_to_xsb
from sokoban.env.moves import (
    legal_actions,
    legal_pushes,
    apply_action,
    apply_push,
    is_solved,
    is_simple_deadlock,
)

__all__ = [
    "Board",
    "State",
    "Action",
    "ACTIONS",
    "DXDY",
    "parse_xsb",
    "parse_xsb_collection",
    "board_to_xsb",
    "legal_actions",
    "legal_pushes",
    "apply_action",
    "apply_push",
    "is_solved",
    "is_simple_deadlock",
]
