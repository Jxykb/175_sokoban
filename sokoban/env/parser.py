"""XSB level format parser.

The XSB format is the de-facto interchange format for Sokoban levels
and is what both the XSokoban set and DeepMind's Boxoban release use.
Per the proposal (Section 2), the character map is:

  =====  =================================
  char   meaning
  =====  =================================
  ``#``  wall
  ` `    floor (also ``-`` per XSB spec)
  ``$``  box
  ``.``  goal
  ``*``  box on a goal
  ``@``  player
  ``+``  player on a goal
  =====  =================================

This module also parses *collections*: text files containing many
levels separated by blank lines or by ``;``-prefixed metadata lines
(the convention DeepMind uses for Boxoban — one level per numbered
block prefixed with ``; <id>``).
"""

from __future__ import annotations

from pathlib import Path
from typing import Iterable, List, Tuple

from sokoban.env.board import Board, State


WALL = "#"
FLOOR_CHARS = (" ", "-")
BOX = "$"
GOAL = "."
BOX_ON_GOAL = "*"
PLAYER = "@"
PLAYER_ON_GOAL = "+"

VALID_CHARS = frozenset({WALL, BOX, GOAL, BOX_ON_GOAL, PLAYER, PLAYER_ON_GOAL, *FLOOR_CHARS})


class ParseError(ValueError):
    """Raised when input text is not a valid XSB level."""


def _looks_like_level_line(line: str) -> bool:
    """A line is part of a level if it contains at least one XSB tile."""
    stripped = line.rstrip("\n\r")
    if not stripped:
        return False
    # Reject comment / metadata lines.
    if stripped.lstrip().startswith(";"):
        return False
    return any(ch in VALID_CHARS for ch in stripped) and all(
        ch in VALID_CHARS for ch in stripped
    )


def parse_xsb(text: str, name: str = "") -> Tuple[Board, State]:
    """Parse a single XSB level into a (Board, State) pair.

    The bounding box is the smallest rectangle enclosing the non-empty
    lines, padded with walls so that all interior floor cells are
    well-defined. Cells outside the level outline are treated as walls.
    """

    raw_lines = [ln.rstrip("\n\r") for ln in text.splitlines()]
    lines = [ln for ln in raw_lines if _looks_like_level_line(ln)]
    if not lines:
        raise ParseError("no level lines found")

    height = len(lines)
    width = max(len(ln) for ln in lines)

    walls: set[tuple[int, int]] = set()
    goals: set[tuple[int, int]] = set()
    boxes: set[tuple[int, int]] = set()
    floor: set[tuple[int, int]] = set()
    player: tuple[int, int] | None = None

    for r, line in enumerate(lines):
        # Right-pad short lines with walls so the grid is rectangular.
        padded = line.ljust(width, WALL)
        for c, ch in enumerate(padded):
            pos = (r, c)
            if ch == WALL:
                walls.add(pos)
            elif ch in FLOOR_CHARS:
                floor.add(pos)
            elif ch == BOX:
                floor.add(pos)
                boxes.add(pos)
            elif ch == GOAL:
                floor.add(pos)
                goals.add(pos)
            elif ch == BOX_ON_GOAL:
                floor.add(pos)
                boxes.add(pos)
                goals.add(pos)
            elif ch == PLAYER:
                floor.add(pos)
                if player is not None:
                    raise ParseError("level has more than one player")
                player = pos
            elif ch == PLAYER_ON_GOAL:
                floor.add(pos)
                goals.add(pos)
                if player is not None:
                    raise ParseError("level has more than one player")
                player = pos
            else:
                raise ParseError(f"unexpected character {ch!r} at {pos}")

    if player is None:
        raise ParseError("level has no player")
    if not boxes:
        raise ParseError("level has no boxes")
    if not goals:
        raise ParseError("level has no goals")
    if len(boxes) != len(goals):
        raise ParseError(
            f"box/goal count mismatch: {len(boxes)} boxes vs {len(goals)} goals"
        )

    board = Board(
        height=height,
        width=width,
        walls=frozenset(walls),
        goals=frozenset(goals),
        floor=frozenset(floor),
        name=name,
    )
    state = State(player=player, boxes=frozenset(boxes))
    return board, state


def parse_xsb_collection(
    text: str, name_prefix: str = "level"
) -> List[Tuple[Board, State]]:
    """Parse a multi-level XSB file.

    Levels are separated either by blank lines or by ``;``-prefixed
    metadata/comment lines, which is the convention used by DeepMind's
    Boxoban level files (one level per block, prefixed with
    ``; <level_id>``). The level id from a leading ``; ID`` line, if
    present, is preserved as the level name; otherwise levels are named
    ``<name_prefix>_<index>``.
    """

    blocks: list[tuple[str, list[str]]] = []
    current_id: str | None = None
    current_lines: list[str] = []

    def _flush() -> None:
        nonlocal current_id, current_lines
        if any(_looks_like_level_line(ln) for ln in current_lines):
            level_lines = [ln for ln in current_lines if _looks_like_level_line(ln)]
            blocks.append((current_id or "", level_lines))
        current_id = None
        current_lines = []

    for raw in text.splitlines():
        stripped = raw.rstrip()
        if stripped.lstrip().startswith(";"):
            # Comment / metadata. Use it as a block separator and capture
            # the id (text after the leading ';' and any spaces). We
            # retain the first ``;`` line of the block as the canonical
            # id so that subsequent ``;`` lines can act as free-form
            # descriptions without clobbering the identifier.
            if current_lines:
                _flush()
            id_text = stripped.lstrip().lstrip(";").strip() or None
            if current_id is None:
                current_id = id_text
        elif not stripped:
            if current_lines:
                _flush()
        else:
            current_lines.append(raw)

    if current_lines:
        _flush()

    levels: list[tuple[Board, State]] = []
    for idx, (lid, lines) in enumerate(blocks):
        name = lid if lid else f"{name_prefix}_{idx:04d}"
        # Sanitize for filesystems: replace whitespace with underscores.
        name = "_".join(name.split())
        try:
            board, state = parse_xsb("\n".join(lines), name=name)
        except ParseError as exc:
            raise ParseError(f"in level block {idx} ({lid!r}): {exc}") from exc
        levels.append((board, state))
    return levels


def parse_xsb_file(path: str | Path) -> List[Tuple[Board, State]]:
    """Parse a file as a collection. A single-level file returns a
    one-element list, so callers do not have to special-case it."""

    p = Path(path)
    text = p.read_text(encoding="utf-8")
    name_prefix = p.stem
    levels = parse_xsb_collection(text, name_prefix=name_prefix)
    if not levels:
        raise ParseError(f"no levels parsed from {path}")
    return levels


def iter_xsb_files(root: str | Path) -> Iterable[Path]:
    """Yield ``.xsb`` files under ``root`` in deterministic order."""
    return sorted(Path(root).rglob("*.xsb"))


# ---------------------------------------------------------------------------
# Serialisation
# ---------------------------------------------------------------------------


def board_to_xsb(board: Board, state: State) -> str:
    """Render a (Board, State) pair back to XSB text.

    Useful for snapshotting intermediate states in the visualiser and
    for round-tripping levels in tests.
    """

    rows: list[str] = []
    for r in range(board.height):
        row_chars: list[str] = []
        for c in range(board.width):
            pos = (r, c)
            on_goal = pos in board.goals
            has_box = pos in state.boxes
            is_player = pos == state.player
            if pos in board.walls:
                row_chars.append(WALL)
            elif is_player and on_goal:
                row_chars.append(PLAYER_ON_GOAL)
            elif is_player:
                row_chars.append(PLAYER)
            elif has_box and on_goal:
                row_chars.append(BOX_ON_GOAL)
            elif has_box:
                row_chars.append(BOX)
            elif on_goal:
                row_chars.append(GOAL)
            elif pos in board.floor:
                row_chars.append(" ")
            else:
                row_chars.append(WALL)
        rows.append("".join(row_chars).rstrip())
    return "\n".join(rows)
