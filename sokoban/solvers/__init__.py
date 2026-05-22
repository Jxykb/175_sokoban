"""Solver implementations and the common interface they share.

The harness only ever talks to :class:`sokoban.solvers.base.Solver`, so
all three solver families (classical, RL, hybrid) become drop-in
swappable from the benchmark side.
"""

from sokoban.solvers.base import Solver, SolveResult, SolveStatus
from sokoban.solvers.bfs import BFSPushSolver
from sokoban.solvers.astar import (
    AStarSolver,
    astar_baseline,
    astar_dead,
    astar_freeze,
    astar_tunnels,
    astar_full,
)
from sokoban.solvers.idastar import (
    IDAStarSolver,
    idastar_baseline,
    idastar_full,
)

__all__ = [
    "Solver",
    "SolveResult",
    "SolveStatus",
    "BFSPushSolver",
    "AStarSolver",
    "IDAStarSolver",
    "astar_baseline",
    "astar_dead",
    "astar_freeze",
    "astar_tunnels",
    "astar_full",
    "idastar_baseline",
    "idastar_full",
]
