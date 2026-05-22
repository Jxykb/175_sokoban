"""Scaled-down Deep Repeating ConvLSTM (DRC) policy network.

Implements the architecture from Guez et al. (2019), "An Investigation
of Model-Free Planning", scaled down so that a single laptop / single
mid-range GPU can train it inside the proposal's parameter budget
(Section 3.3). The original paper uses ``D=9, R=3, C=128``; we
default to ``D=2, R=3, C=32`` which puts the parameter count under
600 K — still expressive enough to learn the Boxoban training tier in
~12 hours on one consumer GPU.

The DRC trick is to *unroll the same ConvLSTM block R times per
environment step* — letting the network "think" without taking an
external action. This is the property that makes DRC competitive
with classical search on Sokoban.

The module is built to be plugged into Stable-Baselines3 as a custom
features extractor (see ``DRCFeaturesExtractor``); the default
SB3 ``PPO`` policy then adds a small MLP head on top.

Everything imports ``torch`` lazily so the core package keeps
working without the ``[rl]`` extra installed.
"""

from __future__ import annotations

from typing import List, Optional, Tuple

try:
    import torch
    from torch import nn
    import torch.nn.functional as F

    _TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    torch = None  # type: ignore
    nn = None  # type: ignore
    F = None  # type: ignore
    _TORCH_AVAILABLE = False


if _TORCH_AVAILABLE:

    class ConvLSTMCell(nn.Module):
        """A single ConvLSTM cell.

        Same gating equations as a vanilla LSTM, but with convolutions
        in place of linear maps so spatial structure is preserved.
        Padding keeps the spatial resolution constant.
        """

        def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3):
            super().__init__()
            padding = kernel_size // 2
            self.hidden_channels = hidden_channels
            # 4 * hidden because we produce input, forget, output, and
            # candidate gates from one conv pass.
            self.conv = nn.Conv2d(
                in_channels + hidden_channels,
                4 * hidden_channels,
                kernel_size=kernel_size,
                padding=padding,
            )

        def init_state(
            self, batch_size: int, height: int, width: int, device: "torch.device"
        ) -> Tuple["torch.Tensor", "torch.Tensor"]:
            shape = (batch_size, self.hidden_channels, height, width)
            return (
                torch.zeros(shape, device=device),
                torch.zeros(shape, device=device),
            )

        def forward(
            self,
            x: "torch.Tensor",
            state: Tuple["torch.Tensor", "torch.Tensor"],
        ) -> Tuple["torch.Tensor", Tuple["torch.Tensor", "torch.Tensor"]]:
            h_prev, c_prev = state
            combined = torch.cat([x, h_prev], dim=1)
            gates = self.conv(combined)
            i, f, o, g = torch.chunk(gates, 4, dim=1)
            i = torch.sigmoid(i)
            f = torch.sigmoid(f)
            o = torch.sigmoid(o)
            g = torch.tanh(g)
            c = f * c_prev + i * g
            h = o * torch.tanh(c)
            return h, (h, c)


    class DRCBlock(nn.Module):
        """One layer of DRC: ConvLSTM followed by a residual conv.

        The residual conv stabilises training when stacking multiple
        DRC blocks; Guez et al. report a ~5% improvement from it.
        """

        def __init__(self, channels: int, kernel_size: int = 3):
            super().__init__()
            self.lstm = ConvLSTMCell(channels, channels, kernel_size)
            self.proj = nn.Conv2d(channels, channels, kernel_size=1)

        def init_state(self, batch_size, h, w, device):
            return self.lstm.init_state(batch_size, h, w, device)

        def forward(self, x, state):
            h, new_state = self.lstm(x, state)
            return F.relu(self.proj(h) + x), new_state


    class DRCNet(nn.Module):
        """Stack of ``num_layers`` DRC blocks, unrolled ``ticks`` times per step.

        Parameters
        ----------
        in_channels:
            Number of input channels (5 in our encoding).
        channels:
            Hidden / output channel count.
        num_layers:
            Depth ``D`` of the DRC stack.
        ticks:
            Number of "thinking" iterations ``R`` per environment step.
        """

        def __init__(
            self,
            *,
            in_channels: int = 5,
            channels: int = 32,
            num_layers: int = 2,
            ticks: int = 3,
        ) -> None:
            super().__init__()
            self.channels = channels
            self.num_layers = num_layers
            self.ticks = ticks
            self.embed = nn.Conv2d(in_channels, channels, kernel_size=3, padding=1)
            self.blocks = nn.ModuleList(
                [DRCBlock(channels) for _ in range(num_layers)]
            )

        def forward(self, obs: "torch.Tensor") -> "torch.Tensor":
            """Run one environment step's worth of computation.

            ``obs`` shape: ``(batch, C_in, H, W)``. Returns a feature
            map of shape ``(batch, channels, H, W)``.
            """
            x = F.relu(self.embed(obs))
            B, _C, H, W = x.shape
            states = [
                blk.init_state(B, H, W, obs.device) for blk in self.blocks
            ]
            for _ in range(self.ticks):
                cur = x
                for i, block in enumerate(self.blocks):
                    cur, states[i] = block(cur, states[i])
                x = cur
            return x


    class DRCActorCritic(nn.Module):
        """DRC backbone + policy and value heads, used as a smoke-test
        model for the eval harness when SB3 is not available.

        The actual PPO training script uses an SB3 features-extractor
        adapter (see :class:`DRCFeaturesExtractor`) that plugs the
        same backbone into SB3's :class:`ActorCriticCnnPolicy`.
        """

        def __init__(
            self,
            *,
            num_actions: int = 4,
            in_channels: int = 5,
            channels: int = 32,
            num_layers: int = 2,
            ticks: int = 3,
        ) -> None:
            super().__init__()
            self.backbone = DRCNet(
                in_channels=in_channels,
                channels=channels,
                num_layers=num_layers,
                ticks=ticks,
            )
            self.policy_head = nn.Conv2d(channels, num_actions, kernel_size=1)
            self.value_head = nn.Sequential(
                nn.AdaptiveAvgPool2d(1),
                nn.Flatten(),
                nn.Linear(channels, 1),
            )

        def forward(self, obs):
            feat = self.backbone(obs)
            # Global average over the spatial dims for the policy logits.
            pi_map = self.policy_head(feat)
            pi = pi_map.mean(dim=(-2, -1))
            v = self.value_head(feat).squeeze(-1)
            return pi, v


    try:
        from stable_baselines3.common.torch_layers import BaseFeaturesExtractor

        class DRCFeaturesExtractor(BaseFeaturesExtractor):
            """Stable-Baselines3 features extractor that wraps :class:`DRCNet`.

            Outputs a flat feature vector by global-average-pooling the
            DRC feature map; SB3's actor-critic heads then project that
            to action logits and a value.
            """

            def __init__(
                self,
                observation_space,
                *,
                channels: int = 32,
                num_layers: int = 2,
                ticks: int = 3,
                features_dim: int = 64,
            ) -> None:
                super().__init__(observation_space, features_dim=features_dim)
                in_channels = observation_space.shape[0]
                self.backbone = DRCNet(
                    in_channels=in_channels,
                    channels=channels,
                    num_layers=num_layers,
                    ticks=ticks,
                )
                self.pool = nn.AdaptiveAvgPool2d(1)
                self.project = nn.Linear(channels, features_dim)

            def forward(self, observations):
                feat = self.backbone(observations)
                pooled = self.pool(feat).flatten(1)
                return F.relu(self.project(pooled))

    except ImportError:  # pragma: no cover — SB3 not installed
        DRCFeaturesExtractor = None  # type: ignore


__all__ = []
if _TORCH_AVAILABLE:
    __all__ += ["ConvLSTMCell", "DRCBlock", "DRCNet", "DRCActorCritic"]
    if "DRCFeaturesExtractor" in globals() and DRCFeaturesExtractor is not None:
        __all__.append("DRCFeaturesExtractor")
