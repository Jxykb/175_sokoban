"""Admissible heuristics for Sokoban A* / IDA*.

The proposal (Section 3.1) calls for the *Hungarian-assignment lower
bound*: each box is matched to a goal under a minimum-cost matching
where the cost is the box-only push distance from box to goal. This
matching ignores the other boxes, which is what keeps the bound
admissible: in the real puzzle the boxes can interfere with each other,
making any actual solution at least as expensive as the matching sum.

Two pieces are needed:

1. Per-goal push-distance maps (:func:`push_distance_maps`), computed
   once per board by reverse-BFS from each goal. The push direction is
   reversed because we are asking "how many pushes does it take to
   move a single box from cell C to this goal?" — the answer is the
   length of a shortest path in the *pull* graph rooted at the goal.

2. A minimum-cost bipartite matching (:func:`hungarian`). We use a
   compact Kuhn-Munkres implementation rather than pulling in
   ``scipy`` for one function — Boxoban levels have at most a handful
   of boxes, so even an O(n^3) algorithm is irrelevant in practice
   but keeps the dependency footprint small (proposal Section 3.4
   lists NumPy + PyTorch + gym-sokoban only).
"""

from __future__ import annotations

import math
from collections import deque
from typing import Dict, FrozenSet, List, Optional, Sequence, Tuple

from sokoban.env.board import ACTIONS, Board, DXDY, Pos, State


INF = math.inf

# Per-board caches keyed by the (frozen) Board itself. We deliberately
# do *not* use ``id(board)`` because Python recycles ids when a Board
# goes out of scope, and a benchmark run that constructs many Boards
# in quick succession would otherwise hit a different level's cached
# maps. ``Board`` is a frozen dataclass and therefore hashable.
_DIST_MAP_CACHE: Dict[Board, Dict[Pos, Dict[Pos, int]]] = {}


# ---------------------------------------------------------------------------
# Single-box push distance maps
# ---------------------------------------------------------------------------


def _push_distance_map_to_goal(board: Board, goal: Pos) -> Dict[Pos, int]:
    """Distance (in pushes) from every floor cell to ``goal`` for a
    *single box* assuming the player can always reach the push-from
    cell. Unreachable cells are omitted.

    We BFS in the pull graph: from ``goal``, consider every direction
    ``d``. A box at ``prev = goal - d`` could have been pushed to
    ``goal`` if the player can stand at ``prev - d`` (i.e. both cells
    are floor in the static layout). We expand outwards in unit-cost
    BFS, yielding shortest push counts.
    """
    distances: dict[Pos, int] = {goal: 0}
    queue: deque[Pos] = deque([goal])
    while queue:
        cell = queue.popleft()
        d = distances[cell]
        for a in ACTIONS:
            dr, dc = DXDY[a]
            prev = (cell[0] - dr, cell[1] - dc)
            player_side = (prev[0] - dr, prev[1] - dc)
            if prev in distances:
                continue
            if prev not in board.floor:
                continue
            if player_side not in board.floor:
                continue
            distances[prev] = d + 1
            queue.append(prev)
    return distances


def push_distance_maps(board: Board) -> Dict[Pos, Dict[Pos, int]]:
    """Return ``{goal: {cell: push_distance}}`` for every goal."""
    cached = _DIST_MAP_CACHE.get(board)
    if cached is not None:
        return cached
    maps = {g: _push_distance_map_to_goal(board, g) for g in board.goals}
    _DIST_MAP_CACHE[board] = maps
    return maps


def reachable_cells_for_some_goal(board: Board) -> FrozenSet[Pos]:
    """Floor cells from which a single box can reach at least one
    goal. Useful as a finer dead-square diagnostic and as a sanity
    check on level construction (proposal Section 4.4 calls for
    qualitative checks on small hand-crafted levels)."""
    maps = push_distance_maps(board)
    out: set[Pos] = set()
    for dist in maps.values():
        out |= dist.keys()
    return frozenset(out)


# ---------------------------------------------------------------------------
# Hungarian-assignment heuristic
# ---------------------------------------------------------------------------


def _build_cost_matrix(
    board: Board, boxes: Sequence[Pos]
) -> Tuple[List[List[float]], List[Pos]]:
    """Build ``cost[i][j]`` = push distance from ``boxes[i]`` to
    ``goals[j]``. Returns the matrix and the goal ordering used.
    """
    maps = push_distance_maps(board)
    goals = list(board.goals)
    cost: list[list[float]] = []
    for box in boxes:
        if box in board.goals:
            row = [0.0 if g == box else float(maps[g].get(box, INF)) for g in goals]
        else:
            row = [float(maps[g].get(box, INF)) for g in goals]
        cost.append(row)
    return cost, goals


def hungarian(cost: Sequence[Sequence[float]]) -> Tuple[float, List[int]]:
    """Minimum-cost perfect matching for a square cost matrix.

    Returns ``(total_cost, assignment)`` where ``assignment[i]`` is the
    column matched to row ``i``. If any feasible matching contains
    ``INF`` we return ``(INF, [])`` to signal the problem is
    infeasible — that is the search signal that some box has no
    reachable goal and the state is dead.

    Implementation: Kuhn-Munkres with the standard four-phase update
    on potentials. O(n^3); for Sokoban n is small (≤ 4 on Boxoban
    standard, ≤ 8 on our XSokoban set) so the constant factor
    dominates.
    """
    n = len(cost)
    if n == 0:
        return 0.0, []
    for row in cost:
        if len(row) != n:
            raise ValueError("hungarian requires a square cost matrix")

    # Quick infeasibility check.
    for i in range(n):
        if all(math.isinf(cost[i][j]) for j in range(n)):
            return INF, []

    # Potentials u (rows) and v (cols), with one extra slot used by
    # the augmenting-path algorithm below.
    u = [0.0] * (n + 1)
    v = [0.0] * (n + 1)
    p = [0] * (n + 1)  # column -> matched row (p[0] = current row)
    way = [0] * (n + 1)

    for i in range(1, n + 1):
        p[0] = i
        j0 = 0
        minv = [INF] * (n + 1)
        used = [False] * (n + 1)
        while True:
            used[j0] = True
            i0 = p[j0]
            delta = INF
            j1 = 0
            for j in range(1, n + 1):
                if used[j]:
                    continue
                cur = cost[i0 - 1][j - 1] - u[i0] - v[j]
                if cur < minv[j]:
                    minv[j] = cur
                    way[j] = j0
                if minv[j] < delta:
                    delta = minv[j]
                    j1 = j
            if math.isinf(delta):
                return INF, []
            for j in range(n + 1):
                if used[j]:
                    u[p[j]] += delta
                    v[j] -= delta
                else:
                    minv[j] -= delta
            j0 = j1
            if p[j0] == 0:
                break
        while j0 != 0:
            j1 = way[j0]
            p[j0] = p[j1]
            j0 = j1

    assignment = [0] * n
    for j in range(1, n + 1):
        if p[j] != 0:
            assignment[p[j] - 1] = j - 1
    total = sum(cost[i][assignment[i]] for i in range(n))
    return total, assignment


def hungarian_heuristic(board: Board, state: State) -> float:
    """Admissible lower bound on remaining pushes for ``state``.

    Returns :data:`math.inf` if any box has no reachable goal, which
    callers should treat as a deadlock signal.
    """
    boxes = sorted(state.boxes)
    if not boxes:
        return 0.0
    cost, _goals = _build_cost_matrix(board, boxes)
    total, _assign = hungarian(cost)
    return total


# ---------------------------------------------------------------------------
# Cheap fallback heuristic
# ---------------------------------------------------------------------------


def closest_goal_heuristic(board: Board, state: State) -> float:
    """Sum-of-minimum-push-distance heuristic.

    Loose but admissible: each box is independently matched to its
    cheapest goal. Useful as a sanity check that ``hungarian_heuristic``
    is at least as tight as the trivial bound (it is by construction,
    but we have a property test on that).
    """
    maps = push_distance_maps(board)
    total = 0.0
    for box in state.boxes:
        best = INF
        for g in board.goals:
            d = maps[g].get(box, INF)
            if d < best:
                best = d
        if math.isinf(best):
            return INF
        total += best
    return total
