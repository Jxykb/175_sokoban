"""DeepMind Boxoban loader.

The Boxoban release (https://github.com/deepmind/boxoban-levels) ships
procedurally-generated 10x10 Sokoban levels split into three difficulty
tiers — **unfiltered**, **medium**, and **hard** — exactly the tiers
referenced in Section 4.2 of the proposal.

Each tier on disk is a directory of text files; each file contains
~1000 levels separated by ``; <id>`` headers in the XSB-collection
format our parser already understands. The loader here is therefore
mostly thin glue plus a sampler with a fixed RNG seed so that the
"200 levels per tier" evaluation set is reproducible across runs.

Networked download is opt-in: ``ensure_boxoban`` will ``git clone`` the
repo into a local cache directory the first time it is called.
Repeated calls are no-ops.
"""

from __future__ import annotations

import os
import random
import shutil
import subprocess
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

from sokoban.env.board import Board, State
from sokoban.env.parser import parse_xsb_collection


BOXOBAN_TIERS: tuple[str, ...] = ("unfiltered", "medium", "hard")
BOXOBAN_SPLITS: tuple[str, ...] = ("train", "valid", "test")
BOXOBAN_REPO_URL = "https://github.com/deepmind/boxoban-levels.git"


def _default_cache_dir() -> Path:
    """Where to cache the cloned Boxoban repository on disk.

    Honours ``SOKOBAN_DATA_DIR`` so CI / cluster runs can point at a
    shared volume instead of re-cloning per worker.
    """
    env = os.environ.get("SOKOBAN_DATA_DIR")
    if env:
        return Path(env).expanduser()
    return Path.home() / ".cache" / "sokoban175"


def ensure_boxoban(
    cache_dir: str | Path | None = None,
    *,
    force: bool = False,
) -> Path:
    """Ensure a local clone of the Boxoban repo exists.

    Returns the path to the cloned repo root. Performs a fresh ``git
    clone`` when the repo is missing (or when ``force=True``). Requires
    ``git`` on PATH — we use the binary rather than a Python library so
    the project keeps a tiny dependency footprint.
    """
    base = Path(cache_dir) if cache_dir is not None else _default_cache_dir()
    base.mkdir(parents=True, exist_ok=True)
    repo_dir = base / "boxoban-levels"

    if repo_dir.exists() and force:
        shutil.rmtree(repo_dir)

    if not repo_dir.exists():
        if shutil.which("git") is None:
            raise RuntimeError(
                "git is required to fetch Boxoban; install git or set "
                "SOKOBAN_DATA_DIR to a directory that already contains "
                "a 'boxoban-levels' clone."
            )
        subprocess.run(
            ["git", "clone", "--depth", "1", BOXOBAN_REPO_URL, str(repo_dir)],
            check=True,
        )
    return repo_dir


def _tier_dir(repo_dir: Path, tier: str, split: str) -> Path:
    """Locate the directory of level files for a given tier and split.

    Boxoban's filesystem layout has varied slightly across releases:
    older versions used ``<tier>/<split>``, newer ones use
    ``boxoban-levels/<tier>/<split>``. We try both.
    """
    if tier not in BOXOBAN_TIERS:
        raise ValueError(f"unknown Boxoban tier {tier!r}; pick from {BOXOBAN_TIERS}")
    if split not in BOXOBAN_SPLITS:
        raise ValueError(f"unknown Boxoban split {split!r}; pick from {BOXOBAN_SPLITS}")
    candidates = [
        repo_dir / tier / split,
        repo_dir / "boxoban-levels" / tier / split,
    ]
    for c in candidates:
        if c.is_dir():
            return c
    raise FileNotFoundError(
        f"Boxoban directory for tier={tier!r}, split={split!r} not found under {repo_dir}"
    )


def load_boxoban_tier(
    tier: str,
    *,
    split: str = "valid",
    cache_dir: str | Path | None = None,
    max_levels: int | None = None,
) -> List[Tuple[Board, State]]:
    """Load all (or up to ``max_levels``) levels from a Boxoban tier/split.

    Returns levels in the deterministic order in which they appear in
    Boxoban's text files (which is the original DeepMind ordering).
    Pair with :func:`sample_boxoban` if you want a reproducible random
    subsample.
    """
    repo_dir = ensure_boxoban(cache_dir)
    tier_dir = _tier_dir(repo_dir, tier, split)
    files = sorted(tier_dir.glob("*.txt"))
    if not files:
        raise FileNotFoundError(f"no level files in {tier_dir}")

    out: list[tuple[Board, State]] = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        levels = parse_xsb_collection(text, name_prefix=f"{tier}_{f.stem}")
        for board, state in levels:
            # Prefix the level name with the source file so that
            # individual rows in the CSV trace back to disk uniquely.
            board_with_name = Board(
                height=board.height,
                width=board.width,
                walls=board.walls,
                goals=board.goals,
                floor=board.floor,
                name=f"{tier}/{split}/{f.stem}/{board.name}",
            )
            out.append((board_with_name, state))
            if max_levels is not None and len(out) >= max_levels:
                return out
    return out


def sample_boxoban(
    tier: str,
    n: int,
    *,
    split: str = "valid",
    seed: int = 175,
    cache_dir: str | Path | None = None,
) -> List[Tuple[Board, State]]:
    """Return a deterministic random sample of ``n`` levels from a tier.

    Section 4.2 of the proposal calls for "200 levels sampled from each
    Boxoban tier". This function is the implementation of that sample,
    seeded by ``seed`` so the report's numbers are reproducible.
    """
    pool = load_boxoban_tier(tier, split=split, cache_dir=cache_dir)
    if n > len(pool):
        raise ValueError(
            f"requested {n} levels from {tier}/{split} but only {len(pool)} are available"
        )
    rng = random.Random(seed)
    indices = rng.sample(range(len(pool)), n)
    indices.sort()
    return [pool[i] for i in indices]


def iter_boxoban_files(tier: str, *, split: str = "valid") -> Iterable[Path]:
    """Yield raw Boxoban text files for a tier/split, in sorted order."""
    repo_dir = ensure_boxoban()
    return sorted(_tier_dir(repo_dir, tier, split).glob("*.txt"))


def levels_to_triples(
    levels: Sequence[Tuple[Board, State]], tier: str
) -> List[Tuple[str, Board, State]]:
    """Tag each (Board, State) with a tier label for the benchmark harness."""
    return [(tier, b, s) for (b, s) in levels]
