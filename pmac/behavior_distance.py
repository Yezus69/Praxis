"""Behavior distances from PMA-C specification section 6."""

from __future__ import annotations

import jax.numpy as jnp


def _log_softmax(logits, axis: int = -1):
    logits = jnp.asarray(logits)
    shifted = logits - jnp.max(logits, axis=axis, keepdims=True)
    return shifted - jnp.log(jnp.sum(jnp.exp(shifted), axis=axis, keepdims=True))


def kl_categorical(logits_teacher, logits_student, temperature=1.0) -> jnp.ndarray:
    """Per-example KL(softmax(z*/T) || softmax(z/T)), teacher first."""
    temp = jnp.asarray(temperature)
    log_p = _log_softmax(jnp.asarray(logits_teacher) / temp)
    log_q = _log_softmax(jnp.asarray(logits_student) / temp)
    p = jnp.exp(log_p)
    kl = jnp.sum(p * (log_p - log_q), axis=-1)
    return jnp.maximum(kl, 0.0)


def mse(y, y_star) -> jnp.ndarray:
    """Per-example squared L2 distance over the final axis."""
    diff = jnp.asarray(y) - jnp.asarray(y_star)
    return jnp.sum(diff * diff, axis=-1)


def cosine_distance(e, e_star) -> jnp.ndarray:
    """Per-example 1 - cosine similarity."""
    e = jnp.asarray(e)
    e_star = jnp.asarray(e_star)
    dot = jnp.sum(e * e_star, axis=-1)
    norm_e = jnp.sqrt(jnp.sum(e * e, axis=-1))
    norm_star = jnp.sqrt(jnp.sum(e_star * e_star, axis=-1))
    eps = jnp.asarray(1e-8, dtype=dot.dtype)
    denom = jnp.maximum(norm_e * norm_star, eps)
    cos = jnp.clip(dot / denom, -1.0, 1.0)
    both_zero = (norm_e <= eps) & (norm_star <= eps)
    return jnp.where(both_zero, 0.0, 1.0 - cos)


def value_abs(v, v_star) -> jnp.ndarray:
    """Per-example absolute scalar value drift."""
    diff = jnp.abs(jnp.asarray(v) - jnp.asarray(v_star))
    if diff.ndim <= 1:
        return diff
    return jnp.reshape(diff, (diff.shape[0], -1)).sum(axis=-1)


def huber(x, delta=1.0) -> jnp.ndarray:
    """Elementwise Huber loss on a residual."""
    dtype = jnp.result_type(x, delta, jnp.float32)
    x = jnp.asarray(x, dtype=dtype)
    delta = jnp.asarray(delta, dtype=dtype)
    abs_x = jnp.abs(x)
    return jnp.where(abs_x <= delta, 0.5 * x * x, delta * (abs_x - 0.5 * delta))  # spec §11


def mean_distance(per_example) -> jnp.ndarray:
    """Mean of a per-example distance vector."""
    return jnp.mean(jnp.asarray(per_example))


DISTANCES = {
    "kl_categorical": kl_categorical,
    "mse": mse,
    "cosine": cosine_distance,
    "value_abs": value_abs,
    "huber": huber,
}


__all__ = [
    "kl_categorical",
    "mse",
    "cosine_distance",
    "value_abs",
    "huber",
    "mean_distance",
    "DISTANCES",
]
