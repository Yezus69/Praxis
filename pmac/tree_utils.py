"""JAX pytree linear algebra utilities for PMA-C."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def tree_dot(a, b) -> jnp.ndarray:
    """Return sum_j a_j * b_j over matching pytree leaves."""
    products = jax.tree_util.tree_map(
        lambda x, y: jnp.sum(jnp.asarray(x) * jnp.asarray(y)), a, b
    )
    leaves = jax.tree_util.tree_leaves(products)
    total = jnp.array(0.0)
    for leaf in leaves:
        total = total + leaf
    return total


def tree_norm(a) -> jnp.ndarray:
    """Return the L2 norm of a pytree."""
    return jnp.sqrt(tree_dot(a, a))


def tree_add(a, b):
    """Return leafwise a + b."""
    return jax.tree_util.tree_map(lambda x, y: x + y, a, b)


def tree_sub(a, b):
    """Return leafwise a - b."""
    return jax.tree_util.tree_map(lambda x, y: x - y, a, b)


def tree_scale(a, s):
    """Return leafwise s * a."""
    return jax.tree_util.tree_map(lambda x: s * x, a)


def tree_add_scaled(a, b, s):
    """Return leafwise a + s * b."""
    return jax.tree_util.tree_map(lambda x, y: x + s * y, a, b)


def tree_zeros_like(a):
    """Return a zero pytree matching a."""
    return jax.tree_util.tree_map(jnp.zeros_like, a)


def tree_l2sq(a) -> jnp.ndarray:
    """Return the squared L2 norm of a pytree."""
    return tree_dot(a, a)


__all__ = [
    "tree_dot",
    "tree_norm",
    "tree_add",
    "tree_sub",
    "tree_scale",
    "tree_add_scaled",
    "tree_zeros_like",
    "tree_l2sq",
]
