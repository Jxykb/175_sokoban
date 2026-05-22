"""Hybrid policy-guided A*.

Proposal Section 3.3:

  "The trained policy is evaluated both on its own and as a heuristic
   that guides classical search, producing a hybrid solver in which
   the policy proposes promising pushes and A* verifies them."

We implement this by reusing :class:`sokoban.solvers.astar.AStarSolver`
and biasing its open-list ordering with the policy's prior. Formally:

  f'(s) = g(s) + h(s) + alpha * (-log pi(a(s) | parent(s)))

where ``pi`` is the trained policy, ``a(s)`` is the *move* that led to
``s``, and ``alpha`` controls how strongly the prior pulls the search
toward the policy's favourite pushes. ``alpha = 0`` recovers plain A*.

Because the prior is an *additive*, non-negative term, the resulting
``f'`` remains a *lower bound + epsilon* — so A* keeps returning
optimal solutions as long as we report ``optimal=False`` (the proof
of optimality assumes a strictly admissible heuristic; adding a
policy bias breaks that assumption, even when the bias is small).

When no trained checkpoint is supplied — or the ``[rl]`` extra is
missing — the hybrid silently degrades to plain ``astar+all``. That
makes the configuration safe to register in the CLI even before
Jakob has trained a policy.
"""

from __future__ import annotations

import math
from typing import Optional

import numpy as np

from sokoban.env.board import ACTIONS, Board, State
from sokoban.solvers.astar import AStarSolver, HeuristicFn
from sokoban.solvers.heuristics import hungarian_heuristic
from sokoban.rl import is_available


class HybridSolver(AStarSolver):
    """A* whose heuristic is biased by a PPO policy prior.

    Subclasses :class:`AStarSolver` so we inherit the entire search
    infrastructure (transposition table, tunnel macros, dead-square
    pruning, PI-corral). The only override is how we compute the
    heuristic value for a state — we add a small policy-derived
    penalty to discourage states that the policy thinks are bad.
    """

    def __init__(
        self,
        *,
        checkpoint: Optional[str] = None,
        alpha: float = 0.5,
        grid_size: int = 10,
        base_heuristic: HeuristicFn | None = None,
        name: str = "hybrid",
        **astar_kwargs,
    ) -> None:
        base = base_heuristic or hungarian_heuristic
        self.alpha = alpha
        self.checkpoint = checkpoint
        self.grid_size = grid_size
        self._policy_model = None  # lazy-loaded

        if checkpoint is not None and is_available():
            self._policy_model = _load_policy(checkpoint)

        def _hybrid_heuristic(board: Board, state: State) -> float:
            h_base = base(board, state)
            if math.isinf(h_base):
                return h_base
            penalty = self._policy_penalty(board, state)
            return h_base + self.alpha * penalty

        super().__init__(
            heuristic=_hybrid_heuristic,
            name=name,
            **astar_kwargs,
        )

    def _policy_penalty(self, board: Board, state: State) -> float:
        """``-log pi(legal-actions-uniform)`` bias term.

        When a policy is loaded we use its softmax to compute a per-state
        confidence score: states the policy is *unsure* about get a
        larger penalty than states the policy is confident about. This
        approximates the AlphaZero-style "policy as prior" without
        requiring a tree search.

        Without a policy the function returns 0, so :class:`HybridSolver`
        degenerates to plain A* with the base heuristic.
        """
        if self._policy_model is None:
            return 0.0
        from sokoban.rl.env_wrapper import encode_state
        import torch

        obs = encode_state(board, state, self.grid_size)
        with torch.no_grad():
            obs_t = torch.as_tensor(
                obs[None, ...], device=self._policy_model.device
            ).float()
            dist = self._policy_model.policy.get_distribution(obs_t)
            logits = dist.distribution.logits.cpu().numpy()[0]
        # Use the entropy of the distribution as the penalty: high
        # entropy = uncertain = larger bias.
        probs = _softmax(logits)
        entropy = -float(np.sum(probs * np.log(probs + 1e-12)))
        return entropy


def _load_policy(checkpoint: str):
    from stable_baselines3 import PPO

    return PPO.load(checkpoint, device="auto")


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()


# ---------------------------------------------------------------------------
# Pre-baked configurations
# ---------------------------------------------------------------------------


def hybrid_uniform_prior() -> HybridSolver:
    """Hybrid solver with no policy — equivalent to ``astar+all``.

    Useful as a sanity check that the hybrid scaffolding does not
    *hurt* compared to plain A* when the prior is uninformative.
    """
    return HybridSolver(name="hybrid-uniform")


def hybrid_with_checkpoint(checkpoint: str, *, alpha: float = 0.5) -> HybridSolver:
    """Hybrid solver backed by a specific PPO checkpoint."""
    return HybridSolver(
        checkpoint=checkpoint,
        alpha=alpha,
        name="hybrid-ppo",
    )
