"""Gymnasium-compatible wrapper around our Sokoban environment.

Proposal Section 3.3 specifies PPO from Stable-Baselines3 with the
``gym-sokoban`` package, but ``gym-sokoban`` has two issues for our
setup:

1. It pins to the old ``gym`` API and is rarely updated for new
   Gymnasium/SB3 releases.
2. It loads its own level set, making it awkward to feed in the
   Boxoban training split that Section 4 of the proposal asks us to
   train on.

Wrapping our own environment is straightforward — we already have a
complete board, transition function, and goal check — and it gives
us full control over the level distribution, the observation
encoding, and the reward shaping. The wrapper is deliberately thin so
that any classical-search or visualisation tooling already in the
repo can be reused at evaluation time.

Observation encoding
--------------------

Each level is rendered as a fixed-shape multi-channel binary tensor of
shape ``(C, H, W)`` with ``C = 5`` channels:

  0. walls
  1. goals (regardless of box contents)
  2. boxes (regardless of goal contents)
  3. player
  4. box-on-goal (= boxes & goals)

We pad smaller levels with walls so the network sees a fixed grid
size — Boxoban is 10x10 throughout, but the XSokoban evaluation set
varies, so we expose a ``grid_size`` parameter for future
generalisation experiments.

Reward shaping
--------------

Sparse + dense terms, following the gym-sokoban convention used in
the DRC paper (Guez et al. 2019):

* -0.1 per step (penalises wandering)
* +1.0 when a previously empty goal becomes occupied
* -1.0 when a previously occupied goal becomes empty
* +10.0 on solve
* Episode ends on solve or after ``max_steps`` steps.

The shaped rewards do not change the optimal policy but speed up
PPO's credit assignment substantially on the Boxoban training split.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable, List, Optional, Sequence, Tuple

import numpy as np

from sokoban.env.board import ACTIONS, Action, Board, Pos, State
from sokoban.env.moves import apply_action, is_solved, legal_actions

# Lazy import: keep gymnasium out of the module import path so the
# core package stays usable without the [rl] extra.
try:
    import gymnasium as gym
    from gymnasium import spaces

    _GYM_AVAILABLE = True
except ImportError:  # pragma: no cover — exercised in CI matrix only
    gym = None  # type: ignore
    spaces = None  # type: ignore
    _GYM_AVAILABLE = False


NUM_CHANNELS = 5
WALL_CHANNEL = 0
GOAL_CHANNEL = 1
BOX_CHANNEL = 2
PLAYER_CHANNEL = 3
BOX_ON_GOAL_CHANNEL = 4

DEFAULT_MAX_STEPS = 120
STEP_PENALTY = -0.1
BOX_ON_GOAL_REWARD = 1.0
BOX_OFF_GOAL_PENALTY = -1.0
SOLVE_BONUS = 10.0


LevelProvider = Callable[[np.random.Generator], Tuple[Board, State]]


def encode_state(board: Board, state: State, grid_size: int) -> np.ndarray:
    """Render ``(board, state)`` to a fixed-shape (C, H, W) tensor.

    The level is centred in the grid; cells outside the level are
    treated as walls. This lets us mix levels of different sizes
    (e.g. XSokoban + Boxoban) in the same training batch.
    """
    obs = np.zeros((NUM_CHANNELS, grid_size, grid_size), dtype=np.float32)
    obs[WALL_CHANNEL, :, :] = 1.0  # default everything to wall
    pad_r = (grid_size - board.height) // 2
    pad_c = (grid_size - board.width) // 2

    for r in range(board.height):
        for c in range(board.width):
            gr, gc = r + pad_r, c + pad_c
            if gr < 0 or gr >= grid_size or gc < 0 or gc >= grid_size:
                continue
            pos = (r, c)
            if pos in board.walls:
                obs[WALL_CHANNEL, gr, gc] = 1.0
            else:
                obs[WALL_CHANNEL, gr, gc] = 0.0
            if pos in board.goals:
                obs[GOAL_CHANNEL, gr, gc] = 1.0
            if pos in state.boxes:
                obs[BOX_CHANNEL, gr, gc] = 1.0
                if pos in board.goals:
                    obs[BOX_ON_GOAL_CHANNEL, gr, gc] = 1.0
            if pos == state.player:
                obs[PLAYER_CHANNEL, gr, gc] = 1.0
    return obs


@dataclass
class EpisodeStats:
    """Per-episode bookkeeping; exposed via ``info`` for SB3 callbacks."""

    pushes: int = 0
    moves: int = 0
    boxes_on_goal: int = 0
    solved: bool = False


def make_level_provider(
    levels: Sequence[Tuple[Board, State]],
    *,
    shuffle: bool = True,
) -> LevelProvider:
    """Build a callable that returns the next training level.

    Provided as a separate factory so the training script can feed in
    any of the dataset sources (Boxoban training split, curated
    XSokoban, custom files) without the env caring.
    """
    pool = list(levels)
    if not pool:
        raise ValueError("level pool is empty")

    def _provide(rng: np.random.Generator) -> Tuple[Board, State]:
        if shuffle:
            return pool[int(rng.integers(0, len(pool)))]
        idx = _provide.counter % len(pool)  # type: ignore[attr-defined]
        _provide.counter += 1  # type: ignore[attr-defined]
        return pool[idx]

    _provide.counter = 0  # type: ignore[attr-defined]
    return _provide


if _GYM_AVAILABLE:

    class SokobanGymEnv(gym.Env):
        """Gymnasium env that draws levels from a :class:`LevelProvider`.

        Parameters
        ----------
        level_provider:
            Function that returns ``(board, state)`` for each new
            episode. Use :func:`make_level_provider` to build one.
        grid_size:
            Fixed grid side length. Levels are zero-padded with walls.
        max_steps:
            Episode horizon; an unsolved episode terminates after this
            many primitive moves. 120 is the Guez et al. (2019)
            default for the Boxoban training split.
        seed:
            Optional RNG seed; the env's own RNG handles level
            selection so PPO's seeding does not bias the dataset.
        """

        metadata = {"render_modes": ["ansi"]}

        def __init__(
            self,
            level_provider: LevelProvider,
            *,
            grid_size: int = 10,
            max_steps: int = DEFAULT_MAX_STEPS,
            seed: Optional[int] = None,
        ) -> None:
            super().__init__()
            self.level_provider = level_provider
            self.grid_size = grid_size
            self.max_steps = max_steps
            self.action_space = spaces.Discrete(len(ACTIONS))
            self.observation_space = spaces.Box(
                low=0.0,
                high=1.0,
                shape=(NUM_CHANNELS, grid_size, grid_size),
                dtype=np.float32,
            )
            self._rng = np.random.default_rng(seed)
            self._board: Board | None = None
            self._state: State | None = None
            self._step_count = 0
            self._stats = EpisodeStats()

        def reset(self, *, seed: Optional[int] = None, options: dict | None = None):
            if seed is not None:
                self._rng = np.random.default_rng(seed)
            self._board, self._state = self.level_provider(self._rng)
            self._step_count = 0
            self._stats = EpisodeStats()
            obs = encode_state(self._board, self._state, self.grid_size)
            info = {"level_name": self._board.name}
            return obs, info

        def step(self, action: int):
            assert self._board is not None and self._state is not None
            self._step_count += 1
            board = self._board
            state = self._state
            act = ACTIONS[int(action)]

            # Track box-on-goal count before / after to issue dense rewards.
            before = sum(1 for b in state.boxes if b in board.goals)
            reward = STEP_PENALTY

            if act not in legal_actions(board, state):
                # Illegal action: treat as a no-op with the step penalty.
                terminated = False
            else:
                next_state, pushed = apply_action(board, state, act)
                self._stats.moves += 1
                if pushed:
                    self._stats.pushes += 1
                after = sum(1 for b in next_state.boxes if b in board.goals)
                if after > before:
                    reward += BOX_ON_GOAL_REWARD * (after - before)
                elif after < before:
                    reward += BOX_OFF_GOAL_PENALTY * (before - after)
                self._state = next_state
                state = next_state
                terminated = is_solved(board, state)
                if terminated:
                    reward += SOLVE_BONUS
                    self._stats.solved = True

            self._stats.boxes_on_goal = sum(1 for b in state.boxes if b in board.goals)
            obs = encode_state(board, state, self.grid_size)
            truncated = self._step_count >= self.max_steps and not terminated
            info = {
                "level_name": board.name,
                "pushes": self._stats.pushes,
                "moves": self._stats.moves,
                "boxes_on_goal": self._stats.boxes_on_goal,
                "solved": self._stats.solved,
            }
            return obs, reward, terminated, truncated, info

        def render(self) -> str:  # type: ignore[override]
            from sokoban.viz.render import render_ascii

            if self._board is None or self._state is None:
                return ""
            return render_ascii(self._board, self._state)


__all__ = [
    "encode_state",
    "make_level_provider",
    "EpisodeStats",
    "NUM_CHANNELS",
]

if _GYM_AVAILABLE:
    __all__.append("SokobanGymEnv")
