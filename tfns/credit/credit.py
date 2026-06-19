"""Pure causal-credit and potential-shaping math."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np


def causal_decomposition(F_seq: Any, G0: Any) -> dict[str, jnp.ndarray]:
    """Return the telescoping prefix-return contribution decomposition.

    ``F_seq`` has length ``T + 1`` and is indexed as ``F_seq[t] = F(H_t)``.
    ``H_t`` is the history before action ``a_t``; ``H_{t+1}`` is after action
    ``a_t`` and its observed consequence. Therefore ``c[t]`` is attributed to
    the causal action at the same index ``t``.
    """

    F_seq = jnp.asarray(F_seq, dtype=jnp.float32)
    G0 = jnp.asarray(G0, dtype=F_seq.dtype)
    c = F_seq[1:] - F_seq[:-1]
    c_init = F_seq[0]
    c_term = G0 - F_seq[-1]
    return {
        "c_init": c_init,
        "c": c,
        "c_term": c_term,
        "C": jnp.abs(c),
    }


def eligibility_trace(
    C: Any,
    gamma: float,
    lambda_c: float,
    episode_end_mask: Any | None = None,
) -> jnp.ndarray:
    """Backward credit trace ``I[s] = C[s] + gamma * lambda_c * I[s + 1]``.

    If ``episode_end_mask[s]`` is true, the recursion uses zero future carry at
    ``s`` so later episodes cannot propagate credit backward across that
    terminal transition.
    """

    C = jnp.asarray(C, dtype=jnp.float32)
    if episode_end_mask is None:
        episode_end_mask = jnp.zeros(C.shape, dtype=bool)
    else:
        episode_end_mask = jnp.broadcast_to(jnp.asarray(episode_end_mask, dtype=bool), C.shape)

    decay = jnp.asarray(float(gamma) * float(lambda_c), dtype=C.dtype)

    def step(carry, inputs):
        c_t, end_t = inputs
        carry = jnp.where(end_t, jnp.zeros_like(carry), carry)
        value = c_t + decay * carry
        return value, value

    _, trace = jax.lax.scan(
        step,
        jnp.zeros_like(C[0]),
        (C, episode_end_mask),
        reverse=True,
    )
    return trace


def potential_shaping(
    rewards: Any,
    Phi_seq: Any,
    gamma: float,
    eta: float,
    episode_end_mask: Any,
) -> jnp.ndarray:
    """Apply discount-correct potential shaping with zero terminal potential."""

    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    Phi_seq = jnp.asarray(Phi_seq, dtype=rewards.dtype)
    episode_end_mask = jnp.broadcast_to(
        jnp.asarray(episode_end_mask, dtype=bool),
        rewards.shape,
    )
    phi_next = jnp.where(episode_end_mask, jnp.zeros_like(Phi_seq[1:]), Phi_seq[1:])
    shaping = jnp.asarray(eta, dtype=rewards.dtype) * (
        jnp.asarray(gamma, dtype=rewards.dtype) * phi_next - Phi_seq[:-1]
    )
    return rewards + shaping


def shaping_eta(mse_val: Any, var_G: Any, eps: float = 1e-8) -> jnp.ndarray:
    """Return ``clip(1 - mse_val / (var_G + eps), 0, 0.5)`` when validated."""

    mse_val = jnp.asarray(mse_val, dtype=jnp.float32)
    var_G = jnp.asarray(var_G, dtype=jnp.float32)
    raw = 1.0 - mse_val / (var_G + jnp.asarray(eps, dtype=jnp.float32))
    eta = jnp.clip(raw, 0.0, 0.5)
    return jnp.where(mse_val < var_G, eta, jnp.zeros_like(eta))


def shaping_enabled(val_mses: Any, var_Gs: Any, windows: int) -> bool:
    """Return true after ``windows`` consecutive held-out wins over baseline."""

    windows = int(windows)
    if windows <= 0:
        return True

    mses = np.asarray(val_mses, dtype=np.float32).reshape(-1)
    if mses.size < windows:
        return False

    variances = np.asarray(var_Gs, dtype=np.float32)
    if variances.ndim == 0:
        var_tail = np.full((windows,), float(variances), dtype=np.float32)
    else:
        variances = variances.reshape(-1)
        if variances.size < windows:
            return False
        var_tail = variances[-windows:]

    return bool(np.all(mses[-windows:] < var_tail))


def telescoping_residual(Phi_seq: Any, gamma: float, episode_end_mask: Any) -> jnp.ndarray:
    """Return ``sum_t (gamma * Phi_next_t - Phi_t)`` with terminal Phi zero."""

    Phi_seq = jnp.asarray(Phi_seq, dtype=jnp.float32)
    episode_end_mask = jnp.broadcast_to(
        jnp.asarray(episode_end_mask, dtype=bool),
        Phi_seq[:-1].shape,
    )
    phi_next = jnp.where(episode_end_mask, jnp.zeros_like(Phi_seq[1:]), Phi_seq[1:])
    terms = jnp.asarray(gamma, dtype=Phi_seq.dtype) * phi_next - Phi_seq[:-1]
    return jnp.sum(terms)


__all__ = [
    "causal_decomposition",
    "eligibility_trace",
    "potential_shaping",
    "shaping_enabled",
    "shaping_eta",
    "telescoping_residual",
]
