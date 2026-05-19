"""Shared solver interface.

Every solver — Lakshya's A*/IDA* (with and without pruning), Jakob's
PPO policy, and the policy-guided hybrid — implements
:class:`Solver`. The benchmark harness consumes only this interface,
which is what lets Section 4 of the proposal ("each subsequent technique
will be measured as a delta against this baseline") work in practice.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional, Protocol

from sokoban.env.board import Board, State


class SolveStatus(str, Enum):
    """Outcome category for a single solve attempt.

    The categories match the proposal's "failure mode" reporting
    requirement (Section 2): the solver either succeeds, exhausts its
    time budget, runs out of memory, or proves the level is unsolvable
    in its search space.
    """

    SOLVED = "solved"
    TIMEOUT = "timeout"
    MEMORY = "memory"
    UNSOLVABLE = "unsolvable"
    ERROR = "error"


@dataclass
class SolveResult:
    """Per-level solver output.

    The fields cover all metrics required by Section 4.1 of the
    proposal: success rate, wall-clock time, states expanded, solution
    length in pushes, and peak memory. Optimality is a self-reported
    flag (search-based solvers can guarantee it; RL solvers cannot).

    The ``solution`` string follows Section 2's convention:
    lowercase ``u/d/l/r`` for plain moves, UPPERCASE for box pushes.
    """

    status: SolveStatus
    solution: str = ""
    pushes: int = 0
    moves: int = 0
    nodes_expanded: int = 0
    nodes_generated: int = 0
    time_seconds: float = 0.0
    peak_memory_bytes: int = 0
    optimal: Optional[bool] = None
    error: str = ""
    extra: dict = field(default_factory=dict)

    @property
    def solved(self) -> bool:
        return self.status is SolveStatus.SOLVED


class Solver(Protocol):
    """Minimum surface every solver must expose.

    ``name`` is recorded into the CSV so different configurations of
    the same algorithm (e.g. ``"astar"`` vs ``"astar+freeze"``) can be
    compared side by side. ``time_limit`` is in seconds and is honored
    by the implementation, not the harness — solvers may short-circuit
    internally (e.g. an A* checking the clock between node expansions)
    which is cheaper than wrapping each call in a separate process.
    """

    name: str

    def solve(
        self,
        board: Board,
        state: State,
        *,
        time_limit: float = 60.0,
    ) -> SolveResult:
        ...
