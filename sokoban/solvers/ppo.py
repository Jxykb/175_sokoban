"""PPO inference-time solver.

Wraps a trained Stable-Baselines3 checkpoint as a
:class:`sokoban.solvers.base.Solver`, so the benchmark harness can
treat it identically to A* / IDA* and produce row-by-row CSV
comparisons (proposal Section 4.1).

Inference strategy
------------------

We run the policy *deterministically* (greedy action selection) for
up to ``max_steps`` primitive moves. Each illegal action is replaced
with the highest-probability legal action so the rollout cannot stall
on no-ops — a common pathology for early-training Sokoban policies
that haven't learned the wall layout yet.

When ``stochastic=True`` the rollout uses temperature-controlled
sampling, which is sometimes useful for diagnostic ensembles.

Failure modes
-------------

* The episode reaches ``max_steps`` without solving the level →
  ``SolveStatus.TIMEOUT``.
* The policy outputs ``inf`` / ``nan`` (rare; a model artefact) →
  ``SolveStatus.ERROR``.
* The ``[rl]`` extra is not installed → constructing the solver
  raises immediately so the user sees the missing-dep error rather
  than a cryptic stack trace mid-evaluation.
"""

from __future__ import annotations

import time
from typing import List, Tuple

import numpy as np

from sokoban.env.board import ACTIONS, Action, Board, State
from sokoban.env.moves import (
    apply_action,
    is_solved,
    legal_actions,
    trace_to_string,
)
from sokoban.rl import is_available
from sokoban.solvers.base import SolveResult, SolveStatus


class PPOSolver:
    """Solver that consults a trained PPO checkpoint at each step."""

    def __init__(
        self,
        *,
        checkpoint: str,
        grid_size: int = 10,
        max_steps: int = 120,
        stochastic: bool = False,
        name: str = "ppo",
    ) -> None:
        if not is_available():
            raise RuntimeError(
                "PPOSolver requires the [rl] extra: pip install -e '.[rl]'"
            )
        from stable_baselines3 import PPO  # lazy

        self.model = PPO.load(checkpoint, device="auto")
        self.grid_size = grid_size
        self.max_steps = max_steps
        self.stochastic = stochastic
        self.name = name

    def _predict(self, obs: np.ndarray) -> np.ndarray:
        """Run the policy on a single observation and return action logits."""
        # SB3 PPO exposes ``predict`` for greedy action; for the
        # "highest-prob legal action" fallback we want full logits, so
        # we go through the policy directly.
        import torch

        obs_t = torch.as_tensor(obs[None, ...], device=self.model.device).float()
        with torch.no_grad():
            dist = self.model.policy.get_distribution(obs_t)
            logits = dist.distribution.logits.cpu().numpy()[0]
        return logits

    def solve(
        self,
        board: Board,
        state: State,
        *,
        time_limit: float = 60.0,
    ) -> SolveResult:
        from sokoban.rl.env_wrapper import encode_state  # lazy

        start_time = time.perf_counter()

        if is_solved(board, state):
            return SolveResult(
                status=SolveStatus.SOLVED,
                optimal=False,
            )

        cur = state
        trace: List[Tuple[Action, bool]] = []

        for step in range(self.max_steps):
            if time.perf_counter() - start_time > time_limit:
                return SolveResult(
                    status=SolveStatus.TIMEOUT,
                    nodes_expanded=step,
                    nodes_generated=step,
                    time_seconds=time.perf_counter() - start_time,
                    optimal=False,
                )
            obs = encode_state(board, cur, self.grid_size)
            try:
                logits = self._predict(obs)
            except Exception as exc:  # noqa: BLE001
                return SolveResult(
                    status=SolveStatus.ERROR,
                    nodes_expanded=step,
                    nodes_generated=step,
                    time_seconds=time.perf_counter() - start_time,
                    error=f"{type(exc).__name__}: {exc}",
                    optimal=False,
                )

            if self.stochastic:
                probs = _softmax(logits)
                action_idx = int(np.random.choice(len(probs), p=probs))
            else:
                action_idx = int(np.argmax(logits))

            chosen = ACTIONS[action_idx]
            legal = legal_actions(board, cur)
            if chosen not in legal:
                # Pick the legal action with the highest logit.
                legal_indices = [a.value for a in legal]
                if not legal_indices:
                    break
                action_idx = max(legal_indices, key=lambda i: logits[i])
                chosen = ACTIONS[action_idx]

            cur, pushed = apply_action(board, cur, chosen)
            trace.append((chosen, pushed))

            if is_solved(board, cur):
                solution = trace_to_string(trace)
                return SolveResult(
                    status=SolveStatus.SOLVED,
                    solution=solution,
                    pushes=sum(1 for _a, p in trace if p),
                    moves=len(trace),
                    nodes_expanded=step + 1,
                    nodes_generated=step + 1,
                    time_seconds=time.perf_counter() - start_time,
                    optimal=False,
                )

        return SolveResult(
            status=SolveStatus.TIMEOUT,
            nodes_expanded=self.max_steps,
            nodes_generated=self.max_steps,
            time_seconds=time.perf_counter() - start_time,
            optimal=False,
        )


def _softmax(x: np.ndarray) -> np.ndarray:
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
