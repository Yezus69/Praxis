"""Hinge conservation loss from PMA-C sections 7 and 25.1."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, NamedTuple

import jax.numpy as jnp


class AnchorBatch(NamedTuple):
    """Anchor data in spec order: x, context, teacher behavior, tolerance, weight."""

    x: Any
    context: Any = None
    teacher: Any = None
    tolerance: Any = 0.0
    weight: Any = 1.0


def _field(batch, name: str, default=None):
    if isinstance(batch, Mapping):
        return batch.get(name, default)
    return getattr(batch, name, default)


def hinge_violation(d, tolerance) -> jnp.ndarray:
    """Elementwise [d - tolerance]_+."""
    return jnp.maximum(jnp.asarray(d) - jnp.asarray(tolerance), 0.0)


def anchor_loss(d, tolerance, weight) -> jnp.ndarray:
    """Per-example w_i [D_i - epsilon_i]_+^2."""
    violation = hinge_violation(d, tolerance)
    return jnp.asarray(weight) * violation * violation


def conservation_loss(behavior_fn, params, batch, distance_fn) -> jnp.ndarray:
    """Mean anchor conservation loss G_s over a batch."""
    x = _field(batch, "x")
    context = _field(batch, "context", None)
    teacher = _field(batch, "teacher")
    tolerance = _field(batch, "tolerance")
    weight = _field(batch, "weight")

    if context is None:
        current = behavior_fn(params, x)
    else:
        current = behavior_fn(params, x, context)
    d = distance_fn(teacher, current)
    return jnp.mean(anchor_loss(d, tolerance, weight))


__all__ = [
    "AnchorBatch",
    "hinge_violation",
    "anchor_loss",
    "conservation_loss",
]
