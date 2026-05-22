"""Learning-based stack: Gymnasium env wrapper, DRC policy, PPO training.

Imports inside this package are *lazy* with respect to the heavy
optional dependencies (``torch``, ``stable-baselines3``, ``gymnasium``).
We test for availability via :func:`is_available` so the rest of the
package — and the test suite — does not break when the user installs
only the core dependencies.
"""

from __future__ import annotations


def is_available() -> bool:
    """Return ``True`` when the optional RL stack can be imported."""
    try:
        import torch  # noqa: F401
        import gymnasium  # noqa: F401
        import stable_baselines3  # noqa: F401
    except ImportError:
        return False
    return True


__all__ = ["is_available"]
