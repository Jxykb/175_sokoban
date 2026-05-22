"""PPO training script.

Wires the DRC backbone into Stable-Baselines3's :class:`PPO` with a
custom CNN policy. Designed for Section 3.3 of the proposal: train on
the Boxoban training split within a single-GPU budget.

Example
-------

::

    python -m sokoban.rl.train \\
        --tier unfiltered \\
        --total-timesteps 5_000_000 \\
        --num-envs 16 \\
        --save runs/ppo_unfiltered

Or via the CLI sub-command::

    sokoban rl-train --tier unfiltered --total-timesteps 5_000_000

Notes
-----

* This script is the *pipeline* Jakob owns; the actual training run
  is a few GPU-hours and is intentionally not executed during CI.
* The script gracefully reports a clear error when the ``[rl]`` extra
  is not installed, so the rest of the project keeps working without
  ``torch`` on the system.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from sokoban.data.boxoban import load_boxoban_tier
from sokoban.data.xsokoban import load_xsokoban_curated
from sokoban.rl import is_available


@dataclass
class TrainConfig:
    """All knobs the training script exposes.

    Defaults are tuned for the proposal's "scaled-down DRC, single-GPU
    budget" target: a 12-hour run on a consumer GPU should land on the
    Boxoban unfiltered tier at high success rate.
    """

    tier: str = "unfiltered"
    split: str = "train"
    grid_size: int = 10
    max_steps: int = 120
    total_timesteps: int = 5_000_000
    num_envs: int = 16
    learning_rate: float = 3e-4
    n_steps: int = 128
    batch_size: int = 256
    n_epochs: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    ent_coef: float = 0.01
    drc_channels: int = 32
    drc_layers: int = 2
    drc_ticks: int = 3
    save_dir: str = "runs/ppo"
    seed: int = 175
    eval_freq: int = 200_000
    levels_cache_dir: Optional[str] = None


def _require_rl() -> None:
    if not is_available():
        sys.stderr.write(
            "error: the [rl] extra is not installed.\n"
            "       pip install -e '.[rl]'\n"
        )
        sys.exit(2)


def _build_envs(cfg: TrainConfig):
    """Construct vectorised training and evaluation environments."""
    from stable_baselines3.common.vec_env import DummyVecEnv, SubprocVecEnv
    from gymnasium.wrappers import TimeLimit

    from sokoban.rl.env_wrapper import SokobanGymEnv, make_level_provider

    if cfg.tier == "xsokoban":
        levels = load_xsokoban_curated()
    else:
        levels = load_boxoban_tier(
            cfg.tier,
            split=cfg.split,
            cache_dir=cfg.levels_cache_dir,
        )
    if not levels:
        raise RuntimeError(f"no levels loaded for tier={cfg.tier!r}")

    def make_env(rank: int):
        def _init():
            provider = make_level_provider(levels, shuffle=True)
            env = SokobanGymEnv(
                provider,
                grid_size=cfg.grid_size,
                max_steps=cfg.max_steps,
                seed=cfg.seed + rank,
            )
            return TimeLimit(env, max_episode_steps=cfg.max_steps)

        return _init

    env_fns = [make_env(i) for i in range(cfg.num_envs)]
    if cfg.num_envs == 1:
        return DummyVecEnv(env_fns)
    return SubprocVecEnv(env_fns)


def train(cfg: TrainConfig) -> Path:
    """Run PPO training and return the checkpoint path."""
    _require_rl()

    import torch
    from stable_baselines3 import PPO
    from stable_baselines3.common.callbacks import CheckpointCallback

    from sokoban.rl.policy import DRCFeaturesExtractor

    save_dir = Path(cfg.save_dir)
    save_dir.mkdir(parents=True, exist_ok=True)
    (save_dir / "config.json").write_text(json.dumps(cfg.__dict__, indent=2))

    vec_env = _build_envs(cfg)
    device = "cuda" if torch.cuda.is_available() else "cpu"

    policy_kwargs = dict(
        features_extractor_class=DRCFeaturesExtractor,
        features_extractor_kwargs=dict(
            channels=cfg.drc_channels,
            num_layers=cfg.drc_layers,
            ticks=cfg.drc_ticks,
            features_dim=64,
        ),
        # Use the default MLP head on top of the extractor.
        net_arch=dict(pi=[64, 64], vf=[64, 64]),
    )

    model = PPO(
        "CnnPolicy",
        vec_env,
        learning_rate=cfg.learning_rate,
        n_steps=cfg.n_steps,
        batch_size=cfg.batch_size,
        n_epochs=cfg.n_epochs,
        gamma=cfg.gamma,
        gae_lambda=cfg.gae_lambda,
        ent_coef=cfg.ent_coef,
        policy_kwargs=policy_kwargs,
        seed=cfg.seed,
        device=device,
        verbose=1,
        tensorboard_log=str(save_dir / "tb"),
    )

    callbacks = [
        CheckpointCallback(
            save_freq=max(cfg.eval_freq // cfg.num_envs, 1000),
            save_path=str(save_dir / "checkpoints"),
            name_prefix="ppo",
        ),
    ]

    model.learn(total_timesteps=cfg.total_timesteps, callback=callbacks)

    final_path = save_dir / "ppo_final.zip"
    model.save(final_path)
    return final_path


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--tier", default="unfiltered",
                        choices=["unfiltered", "medium", "hard", "xsokoban"])
    parser.add_argument("--split", default="train", choices=["train", "valid", "test"])
    parser.add_argument("--grid-size", type=int, default=10)
    parser.add_argument("--max-steps", type=int, default=120)
    parser.add_argument("--total-timesteps", type=int, default=5_000_000)
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--learning-rate", type=float, default=3e-4)
    parser.add_argument("--n-steps", type=int, default=128)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--n-epochs", type=int, default=4)
    parser.add_argument("--gamma", type=float, default=0.99)
    parser.add_argument("--gae-lambda", type=float, default=0.95)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--drc-channels", type=int, default=32)
    parser.add_argument("--drc-layers", type=int, default=2)
    parser.add_argument("--drc-ticks", type=int, default=3)
    parser.add_argument("--save", default="runs/ppo")
    parser.add_argument("--seed", type=int, default=175)
    parser.add_argument("--eval-freq", type=int, default=200_000)
    parser.add_argument("--levels-cache-dir", default=None)
    args = parser.parse_args(argv)

    cfg = TrainConfig(
        tier=args.tier,
        split=args.split,
        grid_size=args.grid_size,
        max_steps=args.max_steps,
        total_timesteps=args.total_timesteps,
        num_envs=args.num_envs,
        learning_rate=args.learning_rate,
        n_steps=args.n_steps,
        batch_size=args.batch_size,
        n_epochs=args.n_epochs,
        gamma=args.gamma,
        gae_lambda=args.gae_lambda,
        ent_coef=args.ent_coef,
        drc_channels=args.drc_channels,
        drc_layers=args.drc_layers,
        drc_ticks=args.drc_ticks,
        save_dir=args.save,
        seed=args.seed,
        eval_freq=args.eval_freq,
        levels_cache_dir=args.levels_cache_dir,
    )
    out = train(cfg)
    print(f"training finished; checkpoint saved to {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
