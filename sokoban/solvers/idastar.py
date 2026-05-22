"""Iterated Deepening A* (IDA*) for Sokoban.

A* is fast when memory is plentiful but blows up on the Boxoban hard
tier because the open-list is too big to fit in RAM. IDA* trades a
small amount of recomputation for O(depth) memory: we run a sequence
of depth-first searches with an ever-tightening f-bound, where each
iteration's bound is the minimum f-value pruned in the previous one.

Shares the same heuristic, transposition-table key, and pruning
toggles as :class:`AStarSolver`. The transposition table here is a
per-iteration cycle check rather than a cost-minimisation cache; for
the cost-bounded variant we use the standard "remember the lowest f
that exceeded the bound" technique to advance the iteration bound
optimally.
"""

from __future__ import annotations

import math
import time
from dataclasses import dataclass, field
from typing import Dict, List, Optional, Tuple

from sokoban.env.board import Action, Board, Pos, State
from sokoban.env.moves import (
    apply_push,
    is_solved,
    legal_pushes,
    push_path_to_moves,
    trace_to_string,
)
from sokoban.solvers.base import SolveResult, SolveStatus
from sokoban.solvers.deadlock import is_deadlock, pi_corral_pushes
from sokoban.solvers.heuristics import hungarian_heuristic
from sokoban.solvers.macros import expand_tunnel_macro
from sokoban.solvers.transposition import StateKey, state_key
from sokoban.solvers.astar import (
    HeuristicFn,
    _box_destination,
    _expand_recorded_to_primitive,
    _macro_push_count,
)


class IDAStarSolver:
    """IDA* in push-state space, sharing pruning toggles with A*."""

    def __init__(
        self,
        *,
        use_dead_squares: bool = True,
        use_freeze: bool = True,
        use_tunnels: bool = True,
        use_corral: bool = True,
        heuristic: HeuristicFn | None = None,
        name: str = "idastar",
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

        bound = h0
        path: List[Tuple[Pos, Action, State]] = []
        path_keys = {state_key(board, state)}
        nodes_expanded = 0
        nodes_generated = 1
        deadline = start_time + time_limit

        while True:
            self._next_bound = math.inf
            self._timed_out = False
            self._goal_found: Optional[List[Tuple[Pos, Action, State]]] = None
            self._nodes_expanded = nodes_expanded
            self._nodes_generated = nodes_generated

            self._search(
                board,
                state,
                cur_state=state,
                g=0,
                bound=bound,
                path=path,
                path_keys=path_keys,
                deadline=deadline,
            )

            nodes_expanded = self._nodes_expanded
            nodes_generated = self._nodes_generated

            if self._goal_found is not None:
                recorded = [(b, a) for (b, a, _s) in self._goal_found]
                primitive_pushes = _expand_recorded_to_primitive(
                    board, state, recorded, use_tunnels=self.use_tunnels
                )
                move_trace = push_path_to_moves(board, state, primitive_pushes)
                return SolveResult(
                    status=SolveStatus.SOLVED,
                    solution=trace_to_string(move_trace),
                    pushes=len(primitive_pushes),
                    moves=len(move_trace),
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    time_seconds=time.perf_counter() - start_time,
                    optimal=True,
                )

            if self._timed_out:
                return SolveResult(
                    status=SolveStatus.TIMEOUT,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    time_seconds=time.perf_counter() - start_time,
                )

            if math.isinf(self._next_bound):
                return SolveResult(
                    status=SolveStatus.UNSOLVABLE,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    time_seconds=time.perf_counter() - start_time,
                )

            bound = self._next_bound

    # ------------------------------------------------------------------

    def _search(
        self,
        board: Board,
        start_state: State,
        cur_state: State,
        g: int,
        bound: float,
        path: List[Tuple[Pos, Action, State]],
        path_keys: set,
        deadline: float,
    ) -> None:
        """Bounded DFS used inside each IDA* iteration.

        Updates instance variables ``_next_bound``, ``_goal_found``,
        ``_timed_out`` and the node counters. Returning a value would
        have to thread state through several layers of the
        successor-generation generator; mutating ``self`` is cleaner
        here.
        """
        if time.perf_counter() > deadline:
            self._timed_out = True
            return

        h = self.heuristic(board, cur_state)
        if math.isinf(h):
            return
        f = g + h
        if f > bound:
            if f < self._next_bound:
                self._next_bound = f
            return

        if is_solved(board, cur_state):
            # Capture a copy because ``path`` is mutated during DFS.
            self._goal_found = list(path)
            return

        self._nodes_expanded += 1

        for box, action, next_state in self._generate_pushes(board, cur_state):
            box_dest = _box_destination(box, action)
            expanded_state = self._apply_macros(
                board, next_state, box_dest, action
            )
            if self._is_pruned(board, expanded_state):
                continue
            k = state_key(board, expanded_state)
            if k in path_keys:
                # Cycle on this path: skip.
                continue
            self._nodes_generated += 1
            macro_pushes = _macro_push_count(next_state, expanded_state)
            path.append((box, action, expanded_state))
            path_keys.add(k)
            self._search(
                board,
                start_state,
                cur_state=expanded_state,
                g=g + macro_pushes,
                bound=bound,
                path=path,
                path_keys=path_keys,
                deadline=deadline,
            )
            path.pop()
            path_keys.discard(k)
            if self._goal_found is not None:
                return
            if self._timed_out:
                return


def idastar_full() -> IDAStarSolver:
    return IDAStarSolver(name="idastar+all")


def idastar_baseline() -> IDAStarSolver:
    return IDAStarSolver(
        use_dead_squares=False,
        use_freeze=False,
        use_tunnels=False,
        use_corral=False,
        name="idastar",
    )
