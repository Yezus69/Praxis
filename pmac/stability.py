"""Synaptic stability from PMA-C sections 9 and 25.3."""

from __future__ import annotations

import jax
import jax.numpy as jnp

from pmac.tree_utils import tree_zeros_like


def scale_by_stability(g, omega, alpha):
    """Scale gradients as g_j / (1 + alpha * Omega_j)."""
    return jax.tree_util.tree_map(lambda gj, oj: gj / (1.0 + alpha * oj), g, omega)


def update_omega(omega, params, guard_grad, rho):
    """Update Omega_j <- rho Omega_j + (1-rho) |theta_j dG/dtheta_j|."""
    return jax.tree_util.tree_map(
        lambda oj, tj, gj: rho * oj + (1.0 - rho) * jnp.abs(tj * gj),
        omega,
        params,
        guard_grad,
    )


def zeros_omega_like(params):
    """Return a zero stability pytree matching params."""
    return tree_zeros_like(params)


__all__ = ["scale_by_stability", "update_omega", "zeros_omega_like"]
