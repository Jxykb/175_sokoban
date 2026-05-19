"""Core Sokoban data structures.

The environment cleanly separates *static* layout (walls, goals, level
metadata) from *dynamic* state (player position, box positions). This
matters for search: hashing only the dynamic state keeps the transposition
table small, and lets all reachable states share one ``Board`` object.

Coordinates use ``(row, col)`` throughout (matching numpy row-major
indexing). Action codes follow the standard Sokoban convention used by
gym-sokoban so downstream RL code can interoperate without remapping.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from typing import FrozenSet, Tuple


Pos = Tuple[int, int]


class Action(IntEnum):
    """Single-step player actions (4-direction action space).

    A directional action automatically becomes a *push* when there is a
    box adjacent to the player in that direction and the cell behind the
    box is free. The proposal's solution-string convention encodes this
    distinction via case (lowercase = move, UPPERCASE = push), see
    :func:`sokoban.env.moves.trace_to_string`.

    The order matches the natural ``[up, down, left, right]`` ordering
    used in most Sokoban literature and is the action layout we expose
    to the learning-based solver.
    """

    UP = 0
    DOWN = 1
    LEFT = 2
    RIGHT = 3


ACTIONS: Tuple[Action, ...] = (Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT)

DXDY: dict[Action, Pos] = {
    Action.UP: (-1, 0),
    Action.DOWN: (1, 0),
    Action.LEFT: (0, -1),
    Action.RIGHT: (0, 1),
}

ACTION_NAMES: dict[Action, str] = {
    Action.UP: "U",
    Action.DOWN: "D",
    Action.LEFT: "L",
    Action.RIGHT: "R",
}


@dataclass(frozen=True)
class Board:
    """Static level layout.

    Attributes
    ----------
    height, width: dimensions of the bounding box.
    walls:        frozenset of wall cells (``#``).
    goals:        frozenset of goal cells (``.``, ``*``, ``+``).
    floor:        frozenset of all walkable cells (anything that is not
                  a wall and lies inside the bounding box).
    name:         optional human-readable level id (e.g. ``"xsokoban_01"``).
    """

    height: int
    width: int
    walls: FrozenSet[Pos]
    goals: FrozenSet[Pos]
    floor: FrozenSet[Pos]
    name: str = ""

    def in_bounds(self, pos: Pos) -> bool:
        r, c = pos
        return 0 <= r < self.height and 0 <= c < self.width

    def is_wall(self, pos: Pos) -> bool:
        return pos in self.walls

    def is_floor(self, pos: Pos) -> bool:
        return pos in self.floor


@dataclass(frozen=True)
class State:
    """Dynamic puzzle state. Frozen + hashable so it can be used as a
    transposition-table key."""

    player: Pos
    boxes: FrozenSet[Pos] = field(default_factory=frozenset)

    def with_(self, *, player: Pos | None = None, boxes: FrozenSet[Pos] | None = None) -> "State":
        return State(
            player=self.player if player is None else player,
            boxes=self.boxes if boxes is None else boxes,
        )
