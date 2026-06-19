"""Small functional utilities for TFNS."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

try:
    import jax
    import jax.numpy as jnp
except ImportError:  # pragma: no cover - exercised only on hosts without JAX.
    jax = None
    jnp = None


@dataclasses.dataclass(frozen=True)
class RunningRobustStat:
    """Immutable robust scalar normalizer state.

    The update is a clipped exponential estimate of location and absolute
    deviation. It is median/MAD-style rather than a true streaming quantile:
    large residuals are clipped before they can move the center or scale, which
    keeps detector and memory-admission terms resistant to one-off spikes.
    """

    center: float = 0.0
    mad: float = 1.0
    decay: float = 0.99
    eps: float = 1e-6
    clip_sigma: float = 5.0
    initialized: bool = False

    def update(self, x: float) -> "RunningRobustStat":
        """Return a new state after observing scalar ``x``."""

        return update(self, x)

    def normalize(self, x: float):
        """Return the robust z-score of ``x`` under this state."""

        return normalize(self, x)


def update(state: RunningRobustStat, x: float) -> RunningRobustStat:
    """Return an updated robust scalar-stat state using NumPy CPU math."""

    x_arr = np.asarray(x, dtype=np.float64)
    x_val = float(np.nanmedian(x_arr))
    if not state.initialized:
        return dataclasses.replace(state, center=x_val, mad=max(state.mad, state.eps), initialized=True)

    alpha = 1.0 - float(state.decay)
    old_scale = max(float(state.mad), float(state.eps))
    delta = x_val - float(state.center)
    clipped_delta = float(np.clip(delta, -state.clip_sigma * old_scale, state.clip_sigma * old_scale))
    center = float(state.center) + alpha * clipped_delta

    abs_dev = abs(x_val - center)
    clipped_dev = float(np.clip(abs_dev, 0.0, state.clip_sigma * old_scale))
    mad = (1.0 - alpha) * old_scale + alpha * clipped_dev
    return dataclasses.replace(state, center=center, mad=max(mad, state.eps), initialized=True)


def normalize(state: RunningRobustStat, x: float):
    """Return a robust z-score using the normal-consistent MAD scale."""

    scale = 1.4826 * max(float(state.mad), float(state.eps)) + float(state.eps)
    return (np.asarray(x, dtype=np.float64) - float(state.center)) / scale


def tree_zeros_like(tree: Any) -> Any:
    """Return a pytree of zeros with the same structure and leaf shapes."""

    if jax is None or jnp is None:
        raise ImportError("JAX is required for tree_zeros_like.")
    return jax.tree_util.tree_map(jnp.zeros_like, tree)


def tree_global_norm(tree: Any) -> jnp.ndarray:
    """Return the global L2 norm across all numeric pytree leaves."""

    if jax is None or jnp is None:
        raise ImportError("JAX is required for tree_global_norm.")
    leaves = jax.tree_util.tree_leaves(tree)
    total = jnp.array(0.0, dtype=jnp.float32)
    for leaf in leaves:
        leaf_arr = jnp.asarray(leaf)
        total = total + jnp.sum(jnp.square(leaf_arr.astype(jnp.float32)))
    return jnp.sqrt(total)


def tree_add_scaled(a: Any, b: Any, s: float) -> Any:
    """Return the leafwise pytree sum ``a + s * b``."""

    if jax is None:
        raise ImportError("JAX is required for tree_add_scaled.")
    return jax.tree_util.tree_map(lambda x, y: x + s * y, a, b)


__all__ = [
    "RunningRobustStat",
    "normalize",
    "tree_add_scaled",
    "tree_global_norm",
    "tree_zeros_like",
    "update",
]
