"""Transposition table for the classical search solvers.

Proposal Section 3.1 calls the canonical state "the tuple of box
positions plus the connected component containing the player". We
already build that key in :mod:`sokoban.env.moves`
(``canonical_player`` + ``player_reachable``); this module is a thin
wrapper that lets A* / IDA* track the best known ``g`` (number of
pushes) per canonical state and reject re-expansions cheaply.

Keeping the abstraction explicit lets us swap in a more memory-frugal
implementation later (e.g. Zobrist-hashed bitmasks for Boxoban's
fixed 10x10 grid) without touching the search loops.
"""

from __future__ import annotations

import math
from typing import Dict, FrozenSet, Tuple

from sokoban.env.board import Board, Pos, State
from sokoban.env.moves import canonical_player, player_reachable


StateKey = Tuple[FrozenSet[Pos], Pos]


def state_key(board: Board, state: State) -> StateKey:
    """Canonical push-state key.

    Two states with identical box sets but different player positions
    are equivalent if the player sits in the same reachable component.
    We use the lexicographic-minimum cell of that component as the
    canonical representative so equivalent states collide in the
    transposition table.
    """
    component = player_reachable(board, state)
    return state.boxes, canonical_player(component)


class TranspositionTable:
    """Best-``g`` cache keyed by :func:`state_key`.

    ``best_g(state)`` returns the lowest ``g`` we have ever seen reach
    that state, or ``+inf`` if never. ``record(state, g)`` updates if
    the new ``g`` is strictly better and returns whether the update
    happened — which is exactly the "should we expand this successor?"
    signal that A* needs.
    """

    __slots__ = ("_table",)

    def __init__(self) -> None:
        self._table: Dict[StateKey, float] = {}

    def __len__(self) -> int:
        return len(self._table)

    def __contains__(self, key: StateKey) -> bool:
        return key in self._table

    def best_g(self, key: StateKey) -> float:
        return self._table.get(key, math.inf)

    def record(self, key: StateKey, g: float) -> bool:
        prev = self._table.get(key, math.inf)
        if g < prev:
            self._table[key] = g
            return True
        return False

    def clear(self) -> None:
        self._table.clear()
