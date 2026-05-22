"""Smoke tests for the learning-based stack.

The RL stack is optional, gated behind the ``[rl]`` extra in
``pyproject.toml``. These tests skip cleanly when ``torch`` /
``gymnasium`` / ``stable_baselines3`` are not installed, so the
default ``pytest`` invocation keeps passing on machines that only
have the core dependencies (the env tests, the classical-search
tests, and the curated-level checks).

We deliberately do not start an actual PPO training run here — that
takes GPU hours and lives in the manual benchmark workflow.
"""

from __future__ import annotations

import importlib.util

import pytest

from sokoban.env.parser import parse_xsb


def _rl_available() -> bool:
    return all(
        importlib.util.find_spec(m) is not None
        for m in ("torch", "gymnasium", "stable_baselines3")
    )


pytestmark = pytest.mark.skipif(
    not _rl_available(), reason="RL extra not installed"
)


TRIVIAL = """\
#####
#@$.#
#####
"""


def test_encode_state_channel_shape():
    from sokoban.rl.env_wrapper import NUM_CHANNELS, encode_state

    board, state = parse_xsb(TRIVIAL)
    obs = encode_state(board, state, grid_size=10)
    assert obs.shape == (NUM_CHANNELS, 10, 10)
    # Walls border the level + the padding region.
    assert obs[0].sum() > 0
    # Exactly one player cell.
    assert obs[3].sum() == 1.0


def test_gym_env_reset_and_step():
    from sokoban.rl.env_wrapper import SokobanGymEnv, make_level_provider

    board, state = parse_xsb(TRIVIAL)
    provider = make_level_provider([(board, state)], shuffle=False)
    env = SokobanGymEnv(provider, grid_size=10, max_steps=8, seed=0)
    obs, info = env.reset(seed=0)
    assert obs.shape == (5, 10, 10)
    # Action 3 = RIGHT on the trivial level pushes the box onto the
    # goal and solves the level in one step.
    obs, reward, terminated, truncated, info = env.step(3)
    assert terminated, info
    assert reward > 0
    assert info["solved"]


def test_drc_forward_pass_shapes():
    import torch
    from sokoban.rl.policy import DRCActorCritic

    model = DRCActorCritic(
        num_actions=4, in_channels=5, channels=8, num_layers=1, ticks=2,
    )
    obs = torch.randn(2, 5, 10, 10)
    pi, v = model(obs)
    assert pi.shape == (2, 4)
    assert v.shape == (2,)


def test_hybrid_uniform_equivalent_to_astar_on_tiny_level():
    """Hybrid with no checkpoint should solve the same set of levels
    as plain A*+all, since the policy penalty is identically zero."""
    from sokoban.solvers.astar import astar_full
    from sokoban.solvers.hybrid import hybrid_uniform_prior

    board, state = parse_xsb(TRIVIAL)
    astar_result = astar_full().solve(board, state, time_limit=2.0)
    hybrid_result = hybrid_uniform_prior().solve(board, state, time_limit=2.0)
    assert astar_result.status == hybrid_result.status
    assert astar_result.pushes == hybrid_result.pushes
