"""Sokoban-specific deadlock detection.

This module implements the three pruning techniques from Section 3.2 of
the proposal that operate on *board state* rather than on the search
fringe directly:

1. :func:`dead_squares` — a refined precomputed dead-square map that
   strengthens the cheap version in :mod:`sokoban.env.moves` by also
   tagging 2x2 wall-pockets and corner cells which a single box could
   *technically* reach via the pull-BFS but never actually escape from.
2. :func:`is_freeze_deadlock` — the recursive immovability check from
   Junghanns & Schaeffer (2001). A box is *frozen* when both its
   horizontal and vertical axes are blocked, where another box on the
   board counts as blocking *only if it would itself be frozen*. The
   check is run after every push and rejects states where any
   non-goal box is frozen.
3. :func:`pi_corral_pushes` — the PI-corral pruning preselector. Given
   a state, identifies the player-isolated corral (if one exists) and
   restricts the pushes to those that operate on boxes belonging to
   that corral. A corral is "pi" — the proposal's shorthand for
   "player-isolated" — when every box on its boundary can only be
   pushed *into* the corral, never *out*.

Each of these helpers is exposed as a pure function so the A*/IDA*
solvers (and Jakob's hybrid solver) can opt into individual techniques
and measure them as deltas against the baseline (proposal Section 4.2).
"""

from __future__ import annotations

from collections import deque
from typing import FrozenSet, Iterable, List, Set, Tuple

from sokoban.env.board import ACTIONS, Action, Board, DXDY, Pos, State
from sokoban.env.moves import legal_pushes, player_reachable


# ---------------------------------------------------------------------------
# Refined static dead-square map
# ---------------------------------------------------------------------------


def _pull_reachable_from_goals(board: Board) -> FrozenSet[Pos]:
    """Cells from which a single box could in principle reach some goal.

    Identical algorithm to :func:`sokoban.env.moves._simple_dead_squares`
    but factored out so :func:`dead_squares` can layer additional
    rules on top.
    """
    reachable: set[Pos] = set(board.goals)
    frontier = deque(board.goals)
    while frontier:
        cell = frontier.popleft()
        for a in ACTIONS:
            dr, dc = DXDY[a]
            prev = (cell[0] - dr, cell[1] - dc)
            player_side = (prev[0] - dr, prev[1] - dc)
            if prev in reachable:
                continue
            if prev not in board.floor:
                continue
            if player_side not in board.floor:
                continue
            reachable.add(prev)
            frontier.append(prev)
    return frozenset(reachable)


def _two_by_two_dead(board: Board, cell: Pos) -> bool:
    """True if ``cell`` participates in any 2x2 wall-pocket that
    contains no goal cell.

    A 2x2 block of cells where every cell is a wall or a box and none
    is a goal is an instant freeze-pocket: any box pushed into one of
    its non-wall corners can never come out. Since the dead-square
    test is static, we only check the four 2x2 squares anchored at
    ``cell``'s top-left, top-right, bottom-left, and bottom-right
    corners, treating other boxes optimistically as floor (we want a
    static *necessary* condition).
    """
    r, c = cell
    blocks = board.walls  # boxes treated as floor here (static check)
    offsets = [(-1, -1), (-1, 0), (0, -1), (0, 0)]
    for anchor_r, anchor_c in offsets:
        cells = [
            (r + anchor_r, c + anchor_c),
            (r + anchor_r, c + anchor_c + 1),
            (r + anchor_r + 1, c + anchor_c),
            (r + anchor_r + 1, c + anchor_c + 1),
        ]
        # Every cell in the 2x2 must be a wall *except* possibly ``cell``
        # itself; if so and none is a goal, the pocket is dead.
        others = [x for x in cells if x != cell]
        if all(x in blocks for x in others) and cell not in board.goals:
            return True
    return False


_DEAD_CACHE: dict[Board, FrozenSet[Pos]] = {}


def dead_squares(board: Board) -> FrozenSet[Pos]:
    """Refined dead-square map.

    Cells judged dead by either the pull-BFS or the 2x2-pocket
    pattern. Cached by the ``Board`` itself (a frozen dataclass and
    therefore hashable); we avoid id-based caches because Python
    recycles ids across short-lived ``Board`` objects.
    """
    cached = _DEAD_CACHE.get(board)
    if cached is not None:
        return cached
    pull_reachable = _pull_reachable_from_goals(board)
    dead: set[Pos] = set()
    for cell in board.floor:
        if cell in board.goals:
            continue
        if cell not in pull_reachable:
            dead.add(cell)
            continue
        if _two_by_two_dead(board, cell):
            dead.add(cell)
    result = frozenset(dead)
    _DEAD_CACHE[board] = result
    return result


def has_box_on_dead_square(board: Board, state: State) -> bool:
    """Cheap per-state check: any non-goal box on a precomputed dead
    square?"""
    dead = dead_squares(board)
    return any(b in dead and b not in board.goals for b in state.boxes)


# ---------------------------------------------------------------------------
# Freeze-deadlock detection (recursive immovability)
# ---------------------------------------------------------------------------


_H_DELTAS: Tuple[Tuple[int, int], Tuple[int, int]] = ((0, -1), (0, 1))
_V_DELTAS: Tuple[Tuple[int, int], Tuple[int, int]] = ((-1, 0), (1, 0))


def _side_blocked(
    board: Board,
    boxes: FrozenSet[Pos],
    side: Pos,
    tentatively_frozen: Set[Pos],
    box_being_tested: Pos,
) -> bool:
    """Return ``True`` if ``side`` cannot host a usable push destination.

    Following Junghanns & Schaeffer (2001), a side is *blocked* if any
    of the following hold:

    * it is a wall;
    * it is a precomputed dead square (a box pushed there can never
      reach a goal anyway, so it is effectively immobile);
    * it is another box that — assuming ``box_being_tested`` is itself
      frozen — would also be frozen.

    The ``tentatively_frozen`` set carries the mutual-recursion trick:
    we add the box currently under test before recursing so cycles
    terminate.
    """
    if side in board.walls:
        return True
    if side in dead_squares(board):
        return True
    if side in tentatively_frozen:
        return True
    if side in boxes:
        extended = tentatively_frozen | {box_being_tested}
        return _is_frozen(board, boxes, side, extended)
    return False


def _axis_blocked(
    board: Board,
    boxes: FrozenSet[Pos],
    cell: Pos,
    deltas: Tuple[Tuple[int, int], Tuple[int, int]],
    tentatively_frozen: Set[Pos],
) -> bool:
    """Both sides along ``deltas`` are blocked for the box at ``cell``."""
    for dr, dc in deltas:
        side = (cell[0] + dr, cell[1] + dc)
        if not _side_blocked(board, boxes, side, tentatively_frozen, cell):
            return False
    return True


def _is_frozen(
    board: Board,
    boxes: FrozenSet[Pos],
    cell: Pos,
    tentatively_frozen: Set[Pos],
) -> bool:
    """A box at ``cell`` is frozen iff both axes are blocked."""
    if cell not in boxes:
        return False
    return _axis_blocked(
        board, boxes, cell, _H_DELTAS, tentatively_frozen
    ) and _axis_blocked(
        board, boxes, cell, _V_DELTAS, tentatively_frozen
    )


def is_freeze_deadlock(board: Board, state: State) -> bool:
    """Return ``True`` if any box is frozen on a non-goal cell.

    Run this immediately after a push to reject states that have just
    created a frozen non-solved box.
    """
    boxes = state.boxes
    for box in boxes:
        if box in board.goals:
            continue
        if _is_frozen(board, boxes, box, set()):
            return True
    return False


# ---------------------------------------------------------------------------
# PI-corral pruning
# ---------------------------------------------------------------------------


def _flood_outside_player(
    board: Board, boxes: FrozenSet[Pos], player: Pos
) -> FrozenSet[Pos]:
    """Floor cells the player can reach without pushing any box."""
    seen = {player}
    queue = deque([player])
    while queue:
        p = queue.popleft()
        for a in ACTIONS:
            dr, dc = DXDY[a]
            nxt = (p[0] + dr, p[1] + dc)
            if nxt in seen:
                continue
            if nxt not in board.floor:
                continue
            if nxt in boxes:
                continue
            seen.add(nxt)
            queue.append(nxt)
    return frozenset(seen)


def find_pi_corral(
    board: Board, state: State
) -> Tuple[FrozenSet[Pos], FrozenSet[Pos]] | None:
    """Find a player-isolated corral (PI-corral) if one exists.

    A corral is the floor minus the player-reachable region minus the
    walls. It is *player-isolated* if every box on its boundary can
    only be pushed deeper into the corral, not pulled out. When such
    a corral exists, all useful pushes operate on its boundary boxes —
    so any push that doesn't move a corral box can be safely deferred.

    Returns a pair ``(corral_cells, corral_boxes)`` or ``None`` when no
    PI-corral applies.
    """
    reachable = _flood_outside_player(board, state.boxes, state.player)
    # Corral candidate: floor cells not reachable by the player.
    corral_cells = frozenset(c for c in board.floor if c not in reachable)
    if not corral_cells:
        return None

    # Boxes on the corral boundary: those adjacent to a player-reachable cell.
    boundary_boxes: set[Pos] = set()
    for box in state.boxes:
        for a in ACTIONS:
            dr, dc = DXDY[a]
            if (box[0] + dr, box[1] + dc) in reachable:
                boundary_boxes.add(box)
                break

    if not boundary_boxes:
        return None

    # PI condition: every boundary box can only be pushed *into* the
    # corral (i.e. away from the player), never the other way around.
    for box in boundary_boxes:
        for a in ACTIONS:
            dr, dc = DXDY[a]
            player_side = (box[0] - dr, box[1] - dc)
            push_dest = (box[0] + dr, box[1] + dc)
            if player_side not in reachable:
                continue
            if push_dest in board.floor and push_dest not in state.boxes:
                # This push direction has the box leaving the corral
                # towards the player side — corral is not PI.
                if push_dest in reachable:
                    return None
    return corral_cells, frozenset(state.boxes & corral_cells | boundary_boxes)


def pi_corral_pushes(
    board: Board,
    state: State,
) -> Iterable[Tuple[Pos, Action, State]] | None:
    """Restrict push generation to a PI-corral if one is found.

    Returns ``None`` when no corral applies, in which case callers
    should fall back to :func:`sokoban.env.moves.legal_pushes`. When a
    corral is found, yields the subset of pushes that operate on its
    boxes; the rest can be safely skipped at the current state because
    the corral acts as a forced subproblem.
    """
    found = find_pi_corral(board, state)
    if found is None:
        return None
    _corral_cells, corral_boxes = found
    out = []
    for box, action, next_state in legal_pushes(board, state):
        if box in corral_boxes:
            out.append((box, action, next_state))
    return out


# ---------------------------------------------------------------------------
# Aggregate check
# ---------------------------------------------------------------------------


def is_deadlock(
    board: Board,
    state: State,
    *,
    use_dead_squares: bool = True,
    use_freeze: bool = True,
) -> bool:
    """Aggregate per-state deadlock check.

    Each technique can be toggled independently so the report can
    quote the marginal contribution of each (proposal Section 4.2).
    """
    if use_dead_squares and has_box_on_dead_square(board, state):
        return True
    if use_freeze and is_freeze_deadlock(board, state):
        return True
    return False
