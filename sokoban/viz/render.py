"""Static renderers: ASCII for terminals and matplotlib for figures.

Both renderers consume the same (Board, State) pair the solver does,
which means the visualiser cannot drift out of sync with the
environment — there is no second source of truth.
"""

from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

from sokoban.env.board import Board, State


# Numeric tile codes used by the matplotlib renderer. Kept as ints
# rather than strings so the per-frame array stays a cheap numpy view.
TILE_WALL = 0
TILE_FLOOR = 1
TILE_GOAL = 2
TILE_BOX = 3
TILE_BOX_ON_GOAL = 4
TILE_PLAYER = 5
TILE_PLAYER_ON_GOAL = 6
TILE_EXTERIOR = 7


def render_ascii(board: Board, state: State) -> str:
    """Return an ASCII rendering of the current state in XSB format.

    This is what gets printed by ``sokoban play`` and dumped into the
    benchmark logs when a level fails, since the textual form survives
    any terminal.
    """

    rows: list[str] = []
    for r in range(board.height):
        cells: list[str] = []
        for c in range(board.width):
            pos = (r, c)
            on_goal = pos in board.goals
            has_box = pos in state.boxes
            is_player = pos == state.player
            if pos in board.walls:
                cells.append("#")
            elif is_player and on_goal:
                cells.append("+")
            elif is_player:
                cells.append("@")
            elif has_box and on_goal:
                cells.append("*")
            elif has_box:
                cells.append("$")
            elif on_goal:
                cells.append(".")
            elif pos in board.floor:
                cells.append(" ")
            else:
                cells.append("#")
        rows.append("".join(cells).rstrip())
    return "\n".join(rows)


def state_to_grid(board: Board, state: State) -> np.ndarray:
    """Convert (Board, State) into an int8 grid of TILE_* codes."""
    grid = np.full((board.height, board.width), TILE_EXTERIOR, dtype=np.int8)
    for r in range(board.height):
        for c in range(board.width):
            pos = (r, c)
            if pos in board.walls:
                grid[r, c] = TILE_WALL
            elif pos in board.floor:
                grid[r, c] = TILE_FLOOR
    for g in board.goals:
        if grid[g] == TILE_FLOOR:
            grid[g] = TILE_GOAL
    for b in state.boxes:
        grid[b] = TILE_BOX_ON_GOAL if b in board.goals else TILE_BOX
    p = state.player
    grid[p] = TILE_PLAYER_ON_GOAL if p in board.goals else TILE_PLAYER
    return grid


# Palette chosen for high contrast and colorblind-safety; reused by
# both static renders and the animation so they look identical.
_PALETTE: dict[int, Tuple[float, float, float]] = {
    TILE_EXTERIOR: (0.10, 0.10, 0.12),
    TILE_WALL: (0.25, 0.25, 0.28),
    TILE_FLOOR: (0.93, 0.93, 0.93),
    TILE_GOAL: (1.00, 0.85, 0.40),
    TILE_BOX: (0.65, 0.45, 0.20),
    TILE_BOX_ON_GOAL: (0.20, 0.65, 0.30),
    TILE_PLAYER: (0.20, 0.45, 0.90),
    TILE_PLAYER_ON_GOAL: (0.10, 0.35, 0.80),
}


def render_matplotlib(
    board: Board,
    state: State,
    *,
    ax=None,
    title: Optional[str] = None,
    show_grid: bool = True,
):
    """Render a single state on a matplotlib Axes and return it.

    If ``ax`` is None a new figure is created. The caller is responsible
    for ``plt.show()``/``savefig`` — keeping I/O out of the renderer
    makes it composable with the animation module.
    """
    # Lazy import so the env can be used in headless test environments
    # that have no display backend installed.
    import matplotlib.pyplot as plt  # noqa: WPS433 (intentional lazy import)
    from matplotlib.colors import ListedColormap

    grid = state_to_grid(board, state)
    codes_in_order = [
        TILE_WALL,
        TILE_FLOOR,
        TILE_GOAL,
        TILE_BOX,
        TILE_BOX_ON_GOAL,
        TILE_PLAYER,
        TILE_PLAYER_ON_GOAL,
        TILE_EXTERIOR,
    ]
    cmap = ListedColormap([_PALETTE[c] for c in codes_in_order])
    # Map each TILE_* value to its index in ``codes_in_order``.
    index_grid = np.zeros_like(grid)
    for idx, code in enumerate(codes_in_order):
        index_grid[grid == code] = idx

    if ax is None:
        _fig, ax = plt.subplots(figsize=(board.width * 0.4 + 1, board.height * 0.4 + 1))
    ax.imshow(index_grid, cmap=cmap, vmin=0, vmax=len(codes_in_order) - 1)
    ax.set_xticks([])
    ax.set_yticks([])
    if show_grid:
        ax.set_xticks(np.arange(-0.5, board.width, 1), minor=True)
        ax.set_yticks(np.arange(-0.5, board.height, 1), minor=True)
        ax.grid(which="minor", color=(0, 0, 0, 0.15), linewidth=0.5)
        ax.tick_params(which="minor", length=0)
    if title:
        ax.set_title(title, fontsize=10)
    return ax
