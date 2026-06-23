"""Fast plastic low-rank scratchpad for the active context (spec section 7).

Each contextualized layer has one temporary low-rank scratch delta
``S = B_s A_s`` (plus an addressed bias ``beta_s``). PPO updates the scratch
factors freely while a context address is locked; old contexts do not constrain
it because the scratch is not yet part of long-term memory.

LoRA-style initialization (spec 7): ``A_s`` small Gaussian/orthogonal, ``B_s``
zero. This yields a zero initial delta while preserving a usable gradient path,
so the *current* scratch value is exactly the applied delta to commit (the
segment starts at zero delta and the scratch is reset after each commit).
"""

from __future__ import annotations

from typing import Any

from flax import struct
import jax
import jax.numpy as jnp

from tfns.castm.synaptic import SynapticMemory, append_component


@struct.dataclass
class ScratchDelta:
    """One layer's temporary low-rank scratch factors (a plain optimizable pytree)."""

    A_s: jnp.ndarray   # (rank, in)
    B_s: jnp.ndarray   # (out, rank)
    beta_s: jnp.ndarray  # (out,)


def init_scratch(
    in_dim: int,
    out_dim: int,
    rank: int,
    key: Any,
    *,
    a_scale: float = 0.02,
    orthogonal: bool = False,
    dtype: Any = jnp.float32,
) -> ScratchDelta:
    """Initialize a scratch delta with a zero effective delta (``B_s = 0``)."""

    in_dim, out_dim, rank = int(in_dim), int(out_dim), int(rank)
    if orthogonal:
        A_s = jax.nn.initializers.orthogonal()(key, (rank, in_dim), dtype)
        A_s = A_s * float(a_scale)
    else:
        A_s = float(a_scale) * jax.random.normal(key, (rank, in_dim), dtype=dtype)
    B_s = jnp.zeros((out_dim, rank), dtype=dtype)
    beta_s = jnp.zeros((out_dim,), dtype=dtype)
    return ScratchDelta(A_s=A_s, B_s=B_s, beta_s=beta_s)


def scratch_delta_weight(scratch: ScratchDelta) -> jnp.ndarray:
    """Return the dense applied scratch weight delta ``S = B_s A_s`` (out, in)."""

    return scratch.B_s @ scratch.A_s


def scratch_forward(scratch: ScratchDelta, x: Any) -> jnp.ndarray:
    """Return the scratch contribution ``(B_s (A_s x)) + beta_s`` for input ``x``."""

    x = jnp.asarray(x, dtype=scratch.A_s.dtype)
    ax = x @ scratch.A_s.T          # (b, rank)
    return ax @ scratch.B_s.T + scratch.beta_s  # (b, out)


def scratch_is_effectively_zero(scratch: ScratchDelta, tol: float = 1e-12) -> bool:
    """True if the applied delta is ~0 (used by the spec 23.8 dead-scratch check)."""

    w = scratch_delta_weight(scratch)
    return bool(
        jnp.max(jnp.abs(w)) <= float(tol) and jnp.max(jnp.abs(scratch.beta_s)) <= float(tol)
    )


def reset_scratch(scratch: ScratchDelta, key: Any, *, a_scale: float = 0.02) -> ScratchDelta:
    """Return a fresh zero-delta scratch with the same shapes (spec 7 / 9.5)."""

    rank, in_dim = int(scratch.A_s.shape[0]), int(scratch.A_s.shape[1])
    out_dim = int(scratch.B_s.shape[0])
    return init_scratch(in_dim, out_dim, rank, key, a_scale=a_scale, dtype=scratch.A_s.dtype)


def commit_scratch_to_memory(
    mem: SynapticMemory,
    scratch: ScratchDelta,
    d_i: Any,
    ctx_id: int,
) -> tuple[SynapticMemory, int]:
    """Append the applied scratch delta to memory at address dual ``d_i`` (spec 15.3).

    Returns ``(mem, slot)``; ``slot`` is ``-1`` if the pool is exhausted.
    """

    return append_component(mem, scratch.A_s, scratch.B_s, scratch.beta_s, d_i, int(ctx_id))


__all__ = [
    "ScratchDelta",
    "commit_scratch_to_memory",
    "init_scratch",
    "reset_scratch",
    "scratch_delta_weight",
    "scratch_forward",
    "scratch_is_effectively_zero",
]
