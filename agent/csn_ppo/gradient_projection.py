"""Nullspace-protected gradient projection for CSN-PPO."""

import jax
import jax.numpy as jnp


def tree_dot(a, b):
    return sum(
        jnp.vdot(x, y)
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    )


def tree_add_scaled(a, b, scale):
    return jax.tree_util.tree_map(lambda x, y: x + scale * y, a, b)


def tree_scalar_mul(a, scale):
    return jax.tree_util.tree_map(lambda x: scale * x, a)


def tree_add(a, b):
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def project_conflicting_gradient(g_ppo, memory_grads, eps=1e-8):
    """Removes gradient components that would increase memory losses.

    Args:
        g_ppo: pytree gradient of current PPO loss.
        memory_grads: list[pytree], each gradient of a guard bucket.
        eps: numerical stabilizer.

    Returns:
        safe PPO gradient pytree.
    """
    g = g_ppo

    for g_mem in memory_grads:
        dot = tree_dot(g, g_mem)
        norm = tree_dot(g_mem, g_mem) + eps

        # If dot < 0, the PPO gradient conflicts with the memory gradient.
        coeff = jnp.minimum(dot, 0.0) / norm

        # Remove only the conflicting component.
        g = tree_add_scaled(g, g_mem, -coeff)

    return g


def combine_safe_and_guard_grads(g_safe, memory_grads, memory_coefs):
    g = g_safe
    for g_mem, coef in zip(memory_grads, memory_coefs):
        g = tree_add_scaled(g, g_mem, coef)
    return g
