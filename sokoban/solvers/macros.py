"""Tunnel macro collapsing.

A *tunnel cell* is a non-goal floor cell whose only two non-wall
neighbours are colinear (i.e. the cell sits on a 1-wide corridor with
walls on the perpendicular axis). When a box is pushed *into* a tunnel
from one end, the player has no choice but to keep pushing it through
to the other end, because (a) the box cannot be pushed sideways out of
the tunnel — there is a wall on each side — and (b) leaving the box
in the middle of a tunnel and walking around to push it back means
making the level harder, not easier.

Collapsing those forced sequences into a single search step shrinks
the branching factor by a factor proportional to tunnel length. This
implementation follows the description in Junghanns & Schaeffer
(2001) and the sokobano.de wiki referenced by Section 3.2 of the
proposal.

The public API is :func:`expand_tunnel_macro`: given a push that
*enters* a tunnel, return the state obtained by pushing the box all
the way through. The A* / IDA* solvers consult this when generating
successors and use the macro state in place of the single-step state.
"""

from __future__ import annotations

from typing import Dict, FrozenSet, Optional, Tuple

from sokoban.env.board import ACTIONS, Action, Board, DXDY, Pos, State


_TUNNEL_CACHE: Dict[Board, FrozenSet[Pos]] = {}


def tunnel_cells(board: Board) -> FrozenSet[Pos]:
    """Return the set of cells that lie inside a 1-wide tunnel.

    A floor cell is a tunnel cell if it is not a goal *and* both
    cells perpendicular to the corridor direction are walls. We detect
    horizontal tunnels (walls above and below) and vertical tunnels
    (walls left and right) separately; a cell can be in both (a
    crossing) but in that case we omit it because pushing through it
    is no longer forced.
    """
    cached = _TUNNEL_CACHE.get(board)
    if cached is not None:
        return cached
    out: set[Pos] = set()
    for cell in board.floor:
        if cell in board.goals:
            continue
        r, c = cell
        north = (r - 1, c) in board.walls
        south = (r + 1, c) in board.walls
        west = (r, c - 1) in board.walls
        east = (r, c + 1) in board.walls
        horizontal_tunnel = north and south and not (west and east)
        vertical_tunnel = west and east and not (north and south)
        # Strictly *one* axis of walls — pure tunnel, not a dead-end
        # or a wide corridor.
        if horizontal_tunnel ^ vertical_tunnel:
            out.add(cell)
    result = frozenset(out)
    _TUNNEL_CACHE[board] = result
    return result


def expand_tunnel_macro(
    board: Board,
    state_after_push: State,
    box_destination: Pos,
    action: Action,
) -> State:
    """Push a box all the way through a tunnel.

    ``state_after_push`` is the state right after the first push that
    landed the box at ``box_destination``. If that destination is a
    tunnel cell in the direction ``action``, we keep pushing along
    ``action`` until either:

    * the cell ahead is no longer a tunnel cell (we exit), or
    * the cell ahead is blocked by a wall or another box, or
    * the cell ahead is a goal (we stop there because parking on a
      goal is always at least as good).

    Returns the resulting state, which is identical to
    ``state_after_push`` when no extension applies.
    """
    tunnels = tunnel_cells(board)
    if box_destination not in tunnels:
        return state_after_push
    dr, dc = DXDY[action]
    current = box_destination
    boxes = set(state_after_push.boxes)
    player = state_after_push.player
    while True:
        nxt = (current[0] + dr, current[1] + dc)
        if nxt not in board.floor or nxt in boxes:
            break
        # Move one cell further along the tunnel.
        boxes.discard(current)
        boxes.add(nxt)
        player = current
        if nxt in board.goals:
            current = nxt
            break
        if nxt not in tunnels:
            current = nxt
            break
        current = nxt
    return State(player=player, boxes=frozenset(boxes))
