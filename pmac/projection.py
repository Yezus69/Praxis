"""Tangent-cone gradient projection from PMA-C sections 8 and 25.2."""

from __future__ import annotations

import jax.numpy as jnp

from pmac.tree_utils import tree_add_scaled, tree_dot, tree_norm


def project_conflicts(g_new, guard_grads, eps=1e-8):
    """Sequentially project away components conflicting with guard gradients."""
    g = g_new
    for g_guard in guard_grads:
        dot = tree_dot(g, g_guard)
        norm = tree_dot(g_guard, g_guard) + eps
        coeff = jnp.minimum(dot, 0.0) / norm
        g = tree_add_scaled(g, g_guard, -coeff)
    return g


def plasticity_ratio(g_projected, g_new, eps=1e-8) -> jnp.ndarray:
    """Return ||g_projected|| / (||g_new|| + eps)."""
    return tree_norm(g_projected) / (tree_norm(g_new) + eps)


__all__ = ["project_conflicts", "plasticity_ratio"]
