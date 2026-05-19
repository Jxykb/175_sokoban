"""Curated XSokoban level loader.

Section 4.2 of the proposal specifies "30 hand-selected XSokoban levels
of varying size to test generalization beyond the fixed Boxoban grid".
Those 30 levels are versioned in-repo as ``levels/xsokoban_curated.xsb``
so the benchmark set is reproducible without external downloads — the
RL evaluation can therefore be re-run from a fresh checkout.
"""

from __future__ import annotations

from importlib import resources
from pathlib import Path
from typing import List, Tuple

from sokoban.env.board import Board, State
from sokoban.env.parser import parse_xsb_collection


XSOKOBAN_LEVELS_FILE = "xsokoban_curated.xsb"


def load_xsokoban_curated() -> List[Tuple[Board, State]]:
    """Return the canonical curated 30-level XSokoban set.

    The level file is shipped inside the ``sokoban.data.levels`` package
    so it is reachable both from a source checkout and from an installed
    wheel — :mod:`importlib.resources` handles both cases.
    """
    text = resources.files("sokoban.data.levels").joinpath(
        XSOKOBAN_LEVELS_FILE
    ).read_text(encoding="utf-8")
    return parse_xsb_collection(text, name_prefix="xsokoban")


def load_xsokoban_file(path: str | Path) -> List[Tuple[Board, State]]:
    """Load any XSokoban-style ``.xsb`` collection from disk."""
    p = Path(path)
    return parse_xsb_collection(p.read_text(encoding="utf-8"), name_prefix=p.stem)
