"""Solver implementations and the common interface they share.

The harness only ever talks to :class:`sokoban.solvers.base.Solver`, so
all three solver families (classical, RL, hybrid) become drop-in
swappable from the benchmark side.
"""

from sokoban.solvers.base import Solver, SolveResult, SolveStatus
from sokoban.solvers.bfs import BFSPushSolver

__all__ = ["Solver", "SolveResult", "SolveStatus", "BFSPushSolver"]
