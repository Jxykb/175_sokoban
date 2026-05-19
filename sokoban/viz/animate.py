"""Step-by-step solution animation overlaid on the level.

Replays a solution string (proposal's ``u/d/l/r`` + UPPERCASE-on-push
convention) and produces either an interactive animation or a saved
GIF/MP4. The animation is what backs the per-level "qualitative
analysis" deliverable (Section 4.4 of the proposal) and the final demo
video.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from sokoban.env.board import Board, State
from sokoban.env.moves import replay_trace
from sokoban.viz.render import render_matplotlib


def animate_solution(
    board: Board,
    state: State,
    solution: str,
    *,
    out_path: Optional[str | Path] = None,
    fps: int = 4,
    title: Optional[str] = None,
):
    """Animate ``solution`` from ``state``.

    Parameters
    ----------
    out_path:
        If provided, the animation is written to this path. ``.gif``
        and ``.mp4`` are inferred from the suffix; ``.gif`` uses
        Pillow which is always available, while ``.mp4`` requires
        ``ffmpeg`` on PATH.
    fps:
        Frames per second. The default of 4 is comfortable for showing
        each move on a presentation slide.
    title:
        Title shown above the frame. The current step index and
        cumulative push count are appended automatically.

    Returns the matplotlib :class:`FuncAnimation` so callers can pass
    it to ``plt.show()`` from notebooks.
    """

    # Lazy imports keep the env importable without matplotlib.
    import matplotlib.pyplot as plt  # noqa: WPS433
    from matplotlib.animation import FuncAnimation, PillowWriter, FFMpegWriter

    frames = replay_trace(board, state, solution)
    pushes_per_step = _cumulative_pushes(solution)
    fig, ax = plt.subplots(figsize=(board.width * 0.4 + 1, board.height * 0.4 + 1))

    def _draw(i: int):
        ax.clear()
        step_title = title or board.name or "sokoban"
        suffix = f"step {i}/{len(frames) - 1} · pushes {pushes_per_step[i]}"
        render_matplotlib(board, frames[i], ax=ax, title=f"{step_title}\n{suffix}")
        return [ax]

    anim = FuncAnimation(
        fig,
        _draw,
        frames=len(frames),
        interval=1000 / max(fps, 1),
        blit=False,
        repeat=False,
    )

    if out_path is not None:
        out_path = Path(out_path)
        out_path.parent.mkdir(parents=True, exist_ok=True)
        suffix = out_path.suffix.lower()
        if suffix == ".gif":
            anim.save(out_path, writer=PillowWriter(fps=fps))
        elif suffix == ".mp4":
            anim.save(out_path, writer=FFMpegWriter(fps=fps))
        else:
            raise ValueError(
                f"unsupported animation suffix {suffix!r} (use .gif or .mp4)"
            )
        plt.close(fig)
    return anim


def _cumulative_pushes(solution: str) -> list[int]:
    """Per-frame cumulative push counter, including the start frame.

    Returned list has length ``len(solution) + 1`` so it indexes the
    same frame list :func:`replay_trace` returns.
    """
    counts = [0]
    running = 0
    for ch in solution:
        if ch.isupper():
            running += 1
        counts.append(running)
    return counts
