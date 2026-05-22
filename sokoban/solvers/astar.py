"""A* solver in the push-state space.

Implements the proposal's Section 3.1 baseline plus all four
Sokoban-specific prunings from Section 3.2:

* dead-square detection (precomputed at level load)
* freeze-deadlock detection (recursive immovability)
* tunnel macro collapsing
* PI-corral pruning

Each technique is gated by a flag on :class:`AStarSolver` so the
benchmark harness can register multiple configurations of the same
algorithm — ``astar``, ``astar+dead``, ``astar+freeze``,
``astar+tunnels``, ``astar+corral``, ``astar+all`` — and the report
can quote the marginal contribution of each (proposal Section 4.2).

The heuristic defaults to Hungarian-assignment; ``closest_goal`` is
also wired in mainly as a regression check on heuristic admissibility.
"""

from __future__ import annotations

import heapq
import math
import time
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from sokoban.env.board import Action, Board, Pos, State
from sokoban.env.moves import (
    is_solved,
    legal_pushes,
    push_path_to_moves,
    trace_to_string,
)
from sokoban.solvers.base import SolveResult, SolveStatus
from sokoban.solvers.deadlock import (
    is_deadlock,
    pi_corral_pushes,
)
from sokoban.solvers.heuristics import (
    closest_goal_heuristic,
    hungarian_heuristic,
)
from sokoban.solvers.macros import expand_tunnel_macro
from sokoban.solvers.transposition import StateKey, TranspositionTable, state_key


HeuristicFn = Callable[[Board, State], float]


# Tie-breaker counter so heapq does not try to compare State objects.
_TIE = 0


def _next_tie() -> int:
    global _TIE
    _TIE += 1
    return _TIE


@dataclass(order=True)
class _HeapItem:
    f: float
    h: float
    tie: int
    g: int = field(compare=False)
    key: StateKey = field(compare=False)


class AStarSolver:
    """Configurable A* solver in the push-state space.

    Parameters
    ----------
    use_dead_squares, use_freeze, use_tunnels, use_corral:
        Pruning toggles. Each defaults to ``True``; flip individually
        to measure deltas. ``astar_baseline()`` and ``astar_full()``
        below are pre-baked configurations.
    heuristic:
        Function ``(board, state) -> float``. Defaults to
        :func:`sokoban.solvers.heuristics.hungarian_heuristic`.
    name:
        Solver name recorded in the benchmark CSV.
    """

    def __init__(
        self,
        *,
        use_dead_squares: bool = True,
        use_freeze: bool = True,
        use_tunnels: bool = True,
        use_corral: bool = True,
        heuristic: HeuristicFn | None = None,
        name: str = "astar",
    ) -> None:
        self.use_dead_squares = use_dead_squares
        self.use_freeze = use_freeze
        self.use_tunnels = use_tunnels
        self.use_corral = use_corral
        self.heuristic: HeuristicFn = heuristic or hungarian_heuristic
        self.name = name

    # ------------------------------------------------------------------

    def _generate_pushes(self, board: Board, state: State):
        if self.use_corral:
            restricted = pi_corral_pushes(board, state)
            if restricted is not None:
                yield from restricted
                return
        yield from legal_pushes(board, state)

    def _apply_macros(
        self, board: Board, next_state: State, box_destination: Pos, action: Action
    ) -> State:
        if not self.use_tunnels:
            return next_state
        return expand_tunnel_macro(board, next_state, box_destination, action)

    def _is_pruned(self, board: Board, next_state: State) -> bool:
        return is_deadlock(
            board,
            next_state,
            use_dead_squares=self.use_dead_squares,
            use_freeze=self.use_freeze,
        )

    # ------------------------------------------------------------------

    def solve(
        self,
        board: Board,
        state: State,
        *,
        time_limit: float = 60.0,
    ) -> SolveResult:
        start_time = time.perf_counter()

        if is_solved(board, state):
            return SolveResult(
                status=SolveStatus.SOLVED,
                solution="",
                pushes=0,
                moves=0,
                nodes_expanded=0,
                nodes_generated=1,
                optimal=True,
            )

        h0 = self.heuristic(board, state)
        if math.isinf(h0):
            return SolveResult(
                status=SolveStatus.UNSOLVABLE,
                nodes_expanded=0,
                nodes_generated=1,
                time_seconds=time.perf_counter() - start_time,
            )

        tt = TranspositionTable()
        start_key = state_key(board, state)
        tt.record(start_key, 0)
        state_by_key: Dict[StateKey, State] = {start_key: state}
        parents: Dict[
            StateKey, Tuple[StateKey, Pos, Action, State]
        ] = {}

        open_heap: List[_HeapItem] = []
        heapq.heappush(
            open_heap, _HeapItem(f=h0, h=h0, tie=_next_tie(), g=0, key=start_key)
        )

        nodes_expanded = 0
        nodes_generated = 1
        goal_key: Optional[StateKey] = None

        while open_heap:
            if time.perf_counter() - start_time > time_limit:
                return SolveResult(
                    status=SolveStatus.TIMEOUT,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    time_seconds=time.perf_counter() - start_time,
                )

            item = heapq.heappop(open_heap)
            cur_key = item.key
            cur_g = item.g
            # Stale heap entry (we already found a shorter path here).
            if cur_g > tt.best_g(cur_key):
                continue

            cur_state = state_by_key[cur_key]
            if is_solved(board, cur_state):
                goal_key = cur_key
                break

            nodes_expanded += 1

            for box, action, next_state in self._generate_pushes(board, cur_state):
                box_dest = _box_destination(box, action)
                expanded_state = self._apply_macros(
                    board, next_state, box_dest, action
                )
                if self._is_pruned(board, expanded_state):
                    continue
                nodes_generated += 1
                k = state_key(board, expanded_state)
                # Tunnel macros condense several primitive pushes into
                # one search edge; count them so ``g`` stays equal to
                # the true number of pushes performed.
                new_g = cur_g + _macro_push_count(next_state, expanded_state)
                if not tt.record(k, new_g):
                    continue
                state_by_key[k] = expanded_state
                parents[k] = (cur_key, box, action, cur_state)
                h_val = self.heuristic(board, expanded_state)
                if math.isinf(h_val):
                    continue
                heapq.heappush(
                    open_heap,
                    _HeapItem(
                        f=new_g + h_val,
                        h=h_val,
                        tie=_next_tie(),
                        g=new_g,
                        key=k,
                    ),
                )

        elapsed = time.perf_counter() - start_time
        if goal_key is None:
            return SolveResult(
                status=SolveStatus.UNSOLVABLE,
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                time_seconds=elapsed,
            )

        # Recover push sequence by walking parents. Note: with tunnel
        # macros enabled, a single recorded parent edge may represent
        # *multiple* underlying pushes. We re-expand them on the way
        # out so the returned solution is still a legal move trace.
        recorded: List[Tuple[Pos, Action]] = []
        cur = goal_key
        while cur in parents:
            prev_key, box, action, _prev_state = parents[cur]
            recorded.append((box, action))
            cur = prev_key
        recorded.reverse()

        # Replay through the env to fold tunnel macros back into
        # primitive pushes — the move-trace generator handles walking
        # cells between pushes automatically.
        primitive_pushes = _expand_recorded_to_primitive(
            board, state, recorded, use_tunnels=self.use_tunnels
        )
        move_trace = push_path_to_moves(board, state, primitive_pushes)
        solution = trace_to_string(move_trace)

        return SolveResult(
            status=SolveStatus.SOLVED,
            solution=solution,
            pushes=len(primitive_pushes),
            moves=len(move_trace),
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            time_seconds=elapsed,
            optimal=True,
        )


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------


def _box_destination(box: Pos, action: Action) -> Pos:
    from sokoban.env.board import DXDY

    dr, dc = DXDY[action]
    return (box[0] + dr, box[1] + dc)


def _macro_push_count(
    state_before_macro: State, state_after_macro: State
) -> int:
    """How many primitive pushes does this macro represent?

    Equals the number of boxes that moved between the two states
    divided by one — since each push moves exactly one box by one
    cell, and the macro is a chain of single-box pushes, the count is
    the chain length.

    Implementation: find the box that exists in the "after" state but
    not in the "before" state — call its position P_after — and find
    the corresponding "before" position P_before by symmetric diff.
    Manhattan distance between them is the chain length (the macro
    only ever walks along one axis).
    """
    moved_to = state_after_macro.boxes - state_before_macro.boxes
    moved_from = state_before_macro.boxes - state_after_macro.boxes
    if not moved_to or not moved_from:
        return 1
    a = next(iter(moved_to))
    b = next(iter(moved_from))
    return abs(a[0] - b[0]) + abs(a[1] - b[1])


def _expand_recorded_to_primitive(
    board: Board,
    start: State,
    recorded: List[Tuple[Pos, Action]],
    *,
    use_tunnels: bool,
) -> List[Tuple[Pos, Action]]:
    """Re-expand a recorded push list into primitive pushes.

    Each recorded ``(box, action)`` was the *first* push of a possibly
    larger macro. We replay each push against the env, applying the
    same tunnel expansion the solver did, and emit the chain of
    primitive pushes that make up the macro.
    """
    from sokoban.env.board import DXDY

    primitive: List[Tuple[Pos, Action]] = []
    cur = start
    for box, action in recorded:
        dr, dc = DXDY[action]
        # First primitive push.
        primitive.append((box, action))
        from sokoban.env.moves import apply_push  # local to avoid cycles

        cur = apply_push(board, cur, box, action)
        if not use_tunnels:
            continue
        # If the just-pushed box landed in a tunnel cell, keep walking.
        from sokoban.solvers.macros import tunnel_cells

        tunnels = tunnel_cells(board)
        while True:
            box_now = (box[0] + dr, box[1] + dc)
            if box_now not in tunnels:
                break
            ahead = (box_now[0] + dr, box_now[1] + dc)
            if ahead not in board.floor or ahead in cur.boxes:
                break
            primitive.append((box_now, action))
            cur = apply_push(board, cur, box_now, action)
            if ahead in board.goals:
                break
            box = box_now
    return primitive


# ---------------------------------------------------------------------------
# Pre-baked configurations
# ---------------------------------------------------------------------------


def astar_baseline() -> AStarSolver:
    """Plain A* with the Hungarian heuristic and no pruning."""
    return AStarSolver(
        use_dead_squares=False,
        use_freeze=False,
        use_tunnels=False,
        use_corral=False,
        name="astar",
    )


def astar_dead() -> AStarSolver:
    return AStarSolver(
        use_dead_squares=True,
        use_freeze=False,
        use_tunnels=False,
        use_corral=False,
        name="astar+dead",
    )


def astar_freeze() -> AStarSolver:
    return AStarSolver(
        use_dead_squares=True,
        use_freeze=True,
        use_tunnels=False,
        use_corral=False,
        name="astar+freeze",
    )


def astar_tunnels() -> AStarSolver:
    return AStarSolver(
        use_dead_squares=True,
        use_freeze=True,
        use_tunnels=True,
        use_corral=False,
        name="astar+tunnels",
    )


def astar_full() -> AStarSolver:
    return AStarSolver(
        use_dead_squares=True,
        use_freeze=True,
        use_tunnels=True,
        use_corral=True,
        name="astar+all",
    )
