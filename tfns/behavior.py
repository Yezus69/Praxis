"""Behavior-distance losses and tube penalties for protected TFNS targets."""

from __future__ import annotations

import math
from typing import Any

import jax
import jax.numpy as jnp


kl_tol = 0.01
value_tol = 0.1
key_cos_tol = 0.02
router_tol = 0.02


def _cfg_value(cfg: Any, name: str, default: float) -> float:
    if cfg is None:
        return float(default)
    if isinstance(cfg, dict):
        return float(cfg.get(name, default))
    return float(getattr(cfg, name, default))


def kl_categorical(
    target_probs: jnp.ndarray,
    logits_q: jnp.ndarray,
    *,
    eps: float = 1.0e-8,
) -> jnp.ndarray:
    """Return ``D_KL(target_probs || softmax(logits_q))`` over the action axis."""

    p = jax.lax.stop_gradient(jnp.asarray(target_probs, dtype=jnp.float32))
    log_p = jnp.log(jnp.maximum(p, jnp.asarray(eps, dtype=p.dtype)))
    log_q = jax.nn.log_softmax(jnp.asarray(logits_q, dtype=jnp.float32), axis=-1)
    return jnp.sum(p * (log_p - log_q), axis=-1)


def huber(x: jnp.ndarray, delta: float = 1.0) -> jnp.ndarray:
    """Elementwise Huber loss with quadratic radius ``delta``."""

    x = jnp.asarray(x, dtype=jnp.float32)
    delta_arr = jnp.asarray(delta, dtype=x.dtype)
    abs_x = jnp.abs(x)
    quadratic = jnp.minimum(abs_x, delta_arr)
    linear = abs_x - quadratic
    return 0.5 * jnp.square(quadratic) + delta_arr * linear


def cosine_distance(a: jnp.ndarray, b: jnp.ndarray, *, eps: float = 1.0e-8) -> jnp.ndarray:
    """Return ``1 - cos(a, b)`` over the final axis with eps-safe normalization."""

    a = jnp.asarray(a, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)
    a_norm = a / jnp.maximum(jnp.linalg.norm(a, axis=-1, keepdims=True), eps)
    b_norm = b / jnp.maximum(jnp.linalg.norm(b, axis=-1, keepdims=True), eps)
    cos = jnp.sum(a_norm * b_norm, axis=-1)
    return 1.0 - jnp.clip(cos, -1.0, 1.0)


def behavior_components(
    teacher_logits: jnp.ndarray,
    teacher_value: jnp.ndarray,
    teacher_key: jnp.ndarray,
    cur_logits: jnp.ndarray,
    cur_value: jnp.ndarray,
    cur_key: jnp.ndarray,
    temp: float = 1.0,
) -> dict[str, jnp.ndarray]:
    """Return per-example KL, value, and context-key behavior distances."""

    temp_arr = jnp.asarray(temp, dtype=jnp.float32)
    target_probs = jax.lax.stop_gradient(
        jax.nn.softmax(jnp.asarray(teacher_logits, dtype=jnp.float32) / temp_arr, axis=-1)
    )
    teacher_value = jax.lax.stop_gradient(jnp.asarray(teacher_value, dtype=jnp.float32))
    teacher_key = jax.lax.stop_gradient(jnp.asarray(teacher_key, dtype=jnp.float32))
    return {
        "kl": kl_categorical(target_probs, cur_logits),
        "value_err": huber(jnp.asarray(cur_value, dtype=jnp.float32) - teacher_value),
        "key_dist": cosine_distance(cur_key, teacher_key),
    }


def behavior_distance(
    components: dict[str, jnp.ndarray],
    lambda_v: float,
    lambda_q: float,
) -> jnp.ndarray:
    """Return section-10 combined behavior distance ``D_t``."""

    return (
        jnp.asarray(components["kl"], dtype=jnp.float32)
        + float(lambda_v) * jnp.asarray(components["value_err"], dtype=jnp.float32)
        + float(lambda_q) * jnp.asarray(components["key_dist"], dtype=jnp.float32)
    )


def tube_loss(
    D: jnp.ndarray,
    tol: float,
    weights: jnp.ndarray | None = None,
    tail_frac: float = 0.10,
) -> dict[str, jnp.ndarray]:
    """Return mean and worst-tail squared violation for a behavior tube."""

    D = jnp.asarray(D, dtype=jnp.float32)
    viol = jnp.square(jnp.maximum(D - jnp.asarray(tol, dtype=D.dtype), 0.0))
    if weights is None:
        weighted = viol
    else:
        weighted = jnp.asarray(weights, dtype=viol.dtype) * viol
    mean = jnp.mean(weighted)

    flat = jnp.reshape(viol, (-1,))
    n = int(flat.shape[0])
    if n == 0:
        tail = jnp.asarray(0.0, dtype=viol.dtype)
    else:
        k = max(1, min(n, int(math.ceil(float(tail_frac) * n))))
        tail = jnp.mean(jax.lax.top_k(flat, k)[0])
    total = mean + tail
    return {"mean": mean, "tail": tail, "total": total}


def combined_tol(cfg: Any = None) -> float:
    """Return ``kl_tol + lambda_v * value_tol + lambda_q * key_cos_tol``."""

    lam_v = _cfg_value(cfg, "lambda_v", 1.0)
    lam_q = _cfg_value(cfg, "lambda_q", 1.0)
    return (
        _cfg_value(cfg, "kl_tol", kl_tol)
        + lam_v * _cfg_value(cfg, "value_tol", value_tol)
        + lam_q * _cfg_value(cfg, "key_cos_tol", key_cos_tol)
    )


__all__ = [
    "behavior_components",
    "behavior_distance",
    "combined_tol",
    "cosine_distance",
    "huber",
    "key_cos_tol",
    "kl_categorical",
    "kl_tol",
    "router_tol",
    "tube_loss",
    "value_tol",
]
