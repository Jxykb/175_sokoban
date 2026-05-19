"""Reference BFS solver over the push-state space.

This is *not* one of the three solver families described in the
proposal — those are Lakshya's responsibility (A*, IDA*) and Jakob's
(PPO, hybrid). It exists for two reasons:

1. End-to-end smoke testing: lets us validate the environment, the
   benchmark harness, and the visualiser against a solver that is
   trivially correct and optimal in pushes on small levels.
2. Reproducibility floor: gives the report a baseline that requires no
   heuristic or training, so any classical-search delta Lakshya
   measures has a meaningful "minimum complexity" reference.

It will be slow on anything bigger than ~5 boxes — that is expected.
"""

from __future__ import annotations

import time
from collections import deque
from typing import Dict, FrozenSet, List, Optional, Tuple

from sokoban.env.board import Action, Board, Pos, State
from sokoban.env.moves import (
    canonical_player,
    is_simple_deadlock,
    is_solved,
    legal_pushes,
    player_reachable,
    push_path_to_moves,
    trace_to_string,
)
from sokoban.solvers.base import SolveResult, SolveStatus


_PushKey = Tuple[FrozenSet[Pos], Pos]


def _key(board: Board, state: State) -> _PushKey:
    """Canonical key for the push-state space.

    Two states are equivalent for search if they have the same set of
    box positions and the player sits in the same reachable component
    (Section 3.1 of the proposal). We use the minimum-cell
    representative of that component as the canonical player.
    """
    reachable = player_reachable(board, state)
    return state.boxes, canonical_player(reachable)


class BFSPushSolver:
    """Breadth-first search in the push-state space.

    BFS minimises the *number of pushes* in the solution, which matches
    Section 4.1's "solution length in pushes" metric. The full move
    trace is reconstructed at the end via
    :func:`sokoban.env.moves.push_path_to_moves`.
    """

    name = "bfs-push"

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
                time_seconds=0.0,
                optimal=True,
            )

        parents: Dict[_PushKey, Tuple[_PushKey, Pos, Action, State]] = {}
        start_key = _key(board, state)
        start_state_for_key: Dict[_PushKey, State] = {start_key: state}
        queue: deque[_PushKey] = deque([start_key])
        seen: set[_PushKey] = {start_key}

        nodes_expanded = 0
        nodes_generated = 1
        goal_key: Optional[_PushKey] = None

        while queue:
            if time.perf_counter() - start_time > time_limit:
                return SolveResult(
                    status=SolveStatus.TIMEOUT,
                    nodes_expanded=nodes_expanded,
                    nodes_generated=nodes_generated,
                    time_seconds=time.perf_counter() - start_time,
                )
            current_key = queue.popleft()
            current_state = start_state_for_key[current_key]
            nodes_expanded += 1
            for box, action, next_state in legal_pushes(board, current_state):
                if is_simple_deadlock(board, next_state):
                    continue
                k = _key(board, next_state)
                if k in seen:
                    continue
                seen.add(k)
                parents[k] = (current_key, box, action, current_state)
                start_state_for_key[k] = next_state
                nodes_generated += 1
                if is_solved(board, next_state):
                    goal_key = k
                    break
                queue.append(k)
            if goal_key is not None:
                break

        elapsed = time.perf_counter() - start_time
        if goal_key is None:
            return SolveResult(
                status=SolveStatus.UNSOLVABLE,
                nodes_expanded=nodes_expanded,
                nodes_generated=nodes_generated,
                time_seconds=elapsed,
            )

        # Recover push sequence by walking parents backwards.
        pushes: List[Tuple[Pos, Action]] = []
        cur = goal_key
        while cur in parents:
            prev_key, box, action, _prev_state = parents[cur]
            pushes.append((box, action))
            cur = prev_key
        pushes.reverse()

        move_trace = push_path_to_moves(board, state, pushes)
        solution = trace_to_string(move_trace)

        return SolveResult(
            status=SolveStatus.SOLVED,
            solution=solution,
            pushes=len(pushes),
            moves=len(move_trace),
            nodes_expanded=nodes_expanded,
            nodes_generated=nodes_generated,
            time_seconds=elapsed,
            optimal=True,
        )
