"""Contextualized layer forwards over factorized synaptic memory.

These pure functions evaluate the effective layer ``W_active = W0 + W_memory(k) +
scratch`` (spec section 7) for the contextualized layer types of section 6.5:
dense, convolution, GRU gates, and policy/value heads.

Conventions
-----------
* Convolution memory stores the kernel flattened as a ``(c_out, kh*kw*c_in)``
  matrix (spec 6.2) inside a standard :class:`SynapticMemory`, so every dual
  write / recompression / consolidation / audit operation works unchanged. The
  conv forward reshapes factors into spatial + 1x1 convolutions on the fly, so no
  dense kernel is ever materialized in the hot path.
* The GRU uses the reset-before form with three independent addressed banks
  (z, r, n), each a memory on ``xi = [x; h]`` (spec 6.3). All three gates have
  independent factor banks.
"""

from __future__ import annotations

from typing import Any

from flax import struct
import jax
import jax.numpy as jnp

from tfns.castm import synaptic as syn
from tfns.castm.scratch import ScratchDelta, scratch_forward
from tfns.castm.synaptic import SynapticMemory


# --- Dense / heads (spec 6.1, 6.4) ---------------------------------------------


def addressed_dense_forward(
    mem: SynapticMemory,
    x: Any,
    k: Any,
    scratch: ScratchDelta | None = None,
) -> jnp.ndarray:
    """Effective dense forward ``W0 x + b0 + memory(k) x + scratch x``."""

    y = syn.forward(mem, x, k)
    if scratch is not None:
        y = y + scratch_forward(scratch, x)
    return y


def addressed_dense_forward_sparse(
    mem: SynapticMemory,
    x: Any,
    k: Any,
    ctx_id: int,
    scratch: ScratchDelta | None = None,
    *,
    include_shared: bool = True,
) -> jnp.ndarray:
    """Sparse-gather dense forward for the selected context (spec 11.2)."""

    y = syn.forward_sparse(mem, x, k, int(ctx_id), include_shared=include_shared)
    if scratch is not None:
        y = y + scratch_forward(scratch, x)
    return y


# --- Convolution (spec 6.2) -----------------------------------------------------


def kernel_to_matrix(kernel_hwio: jnp.ndarray) -> jnp.ndarray:
    """Flatten an ``(kh, kw, c_in, c_out)`` kernel to ``(c_out, kh*kw*c_in)``."""

    kh, kw, c_in, c_out = kernel_hwio.shape
    return kernel_hwio.transpose(3, 0, 1, 2).reshape(c_out, kh * kw * c_in)


def matrix_to_kernel(W: jnp.ndarray, kh: int, kw: int, c_in: int) -> jnp.ndarray:
    """Inverse of :func:`kernel_to_matrix`: ``(c_out, kh*kw*c_in) -> (kh,kw,c_in,c_out)``."""

    c_out = W.shape[0]
    return W.reshape(c_out, kh, kw, c_in).transpose(1, 2, 3, 0)


_CONV_DN = jax.lax.conv_dimension_numbers(
    (1, 84, 84, 1), (1, 1, 1, 1), ("NHWC", "HWIO", "NHWC")
)


def _conv(x: jnp.ndarray, kernel_hwio: jnp.ndarray, strides, padding: str) -> jnp.ndarray:
    dn = jax.lax.conv_dimension_numbers(x.shape, kernel_hwio.shape, ("NHWC", "HWIO", "NHWC"))
    return jax.lax.conv_general_dilated(
        x,
        kernel_hwio,
        window_strides=tuple(int(s) for s in strides),
        padding=str(padding),
        dimension_numbers=dn,
    )


def addressed_conv_forward(
    mem: SynapticMemory,
    x: Any,
    k: Any,
    *,
    kh: int,
    kw: int,
    c_in: int,
    strides: tuple[int, int],
    padding: str = "VALID",
    scratch: ScratchDelta | None = None,
) -> jnp.ndarray:
    """Effective convolution with an addressed low-rank kernel delta (spec 6.2).

    ``mem`` stores the flattened kernel (``out=c_out``, ``in=kh*kw*c_in``).
    Output is ``conv(x, W0) + sum_m (c_m.k) [conv1x1(conv_spatial(x, A_m), B_m)]``
    plus the addressed bias, with no dense kernel materialization.
    """

    x = jnp.asarray(x, dtype=mem.W0.dtype)
    c_out = mem.out_dim
    kh, kw, c_in = int(kh), int(kw), int(c_in)
    W0_kernel = matrix_to_kernel(mem.W0, kh, kw, c_in)
    y = _conv(x, W0_kernel, strides, padding)  # (N,H',W',c_out)
    y = y + mem.b0  # broadcast bias over spatial dims

    s = jnp.where(mem.active, mem.c @ jnp.asarray(k, dtype=mem.c.dtype), 0.0)  # (M,)
    y = y + _conv_memory_delta(mem, x, s, kh, kw, c_in, c_out, strides, padding)
    if scratch is not None:
        y = y + _conv_scratch_delta(scratch, x, kh, kw, c_in, c_out, strides, padding)
    return y


def _conv_memory_delta(mem, x, s, kh, kw, c_in, c_out, strides, padding):
    # A factors -> spatial conv kernels (kh,kw,c_in,r) per component; B -> 1x1.
    M, R = mem.n_slots, mem.comp_rank
    # Reshape all components' A into a single grouped spatial conv producing M*R
    # channels, then apply per-component 1x1 with the address scale folded in.
    A_kernels = mem.A.reshape(M, R, kh, kw, c_in).transpose(2, 3, 4, 0, 1).reshape(kh, kw, c_in, M * R)
    spatial = _conv(x, A_kernels, strides, padding)  # (N,H',W',M*R)
    N, Hp, Wp, _ = spatial.shape
    spatial = spatial.reshape(N, Hp, Wp, M, R)
    # 1x1: out[...,o] = sum_{m,r} s_m * B[m,o,r] * spatial[...,m,r]
    delta = jnp.einsum("nhwmr,m,mor->nhwo", spatial, s, mem.B)
    bias = jnp.einsum("m,mo->o", s, mem.beta)
    return delta + bias


def _conv_scratch_delta(scratch: ScratchDelta, x, kh, kw, c_in, c_out, strides, padding):
    r = int(scratch.A_s.shape[0])
    A_kernel = scratch.A_s.reshape(r, kh, kw, c_in).transpose(1, 2, 3, 0)  # (kh,kw,c_in,r)
    spatial = _conv(x, A_kernel, strides, padding)  # (N,H',W',r)
    delta = jnp.einsum("nhwr,or->nhwo", spatial, scratch.B_s)
    return delta + scratch.beta_s


def materialize_conv_kernel(mem: SynapticMemory, k: Any, kh: int, kw: int, c_in: int) -> jnp.ndarray:
    """Decode the full effective conv kernel ``(kh,kw,c_in,c_out)`` at address ``k`` (audit/test)."""

    W = syn.decode_weight(mem, k)
    return matrix_to_kernel(W, kh, kw, c_in)


# --- GRU gates (spec 6.3) -------------------------------------------------------


@struct.dataclass
class GRUMemory:
    """Three independent addressed gate banks ``z, r, n`` on ``xi=[x;h]``."""

    z: SynapticMemory
    r: SynapticMemory
    n: SynapticMemory


@struct.dataclass
class GRUScratch:
    z: ScratchDelta
    r: ScratchDelta
    n: ScratchDelta


def addressed_gru_step(
    gru: GRUMemory,
    x: Any,
    h_prev: Any,
    k: Any,
    reset: Any,
    scratch: GRUScratch | None = None,
) -> jnp.ndarray:
    """One addressed GRU step (reset-before form).

    ``z = sigma(M_z[x;h] )``, ``r = sigma(M_r[x;h])``,
    ``n = tanh(M_n[x; r*h])``, ``h' = (1-z) n + z h``. The incoming hidden state
    is zeroed where ``reset`` is true before any gate computation.
    """

    x = jnp.asarray(x, dtype=gru.z.W0.dtype)
    h_prev = jnp.asarray(h_prev, dtype=gru.z.W0.dtype)
    reset = jnp.asarray(reset, dtype=bool)
    h_in = jnp.where(reset[..., None], jnp.zeros_like(h_prev), h_prev)
    xi = jnp.concatenate([x, h_in], axis=-1)

    sz = None if scratch is None else scratch.z
    sr = None if scratch is None else scratch.r
    sn = None if scratch is None else scratch.n
    z = jax.nn.sigmoid(addressed_dense_forward(gru.z, xi, k, sz))
    r = jax.nn.sigmoid(addressed_dense_forward(gru.r, xi, k, sr))
    xi_n = jnp.concatenate([x, r * h_in], axis=-1)
    n = jnp.tanh(addressed_dense_forward(gru.n, xi_n, k, sn))
    return (1.0 - z) * n + z * h_in


def addressed_gru_step_sparse(
    gru: GRUMemory,
    x: Any,
    h_prev: Any,
    k: Any,
    ctx_id: int,
    reset: Any,
    scratch: GRUScratch | None = None,
) -> jnp.ndarray:
    """Sparse-gather GRU step for the selected context (spec 11.2)."""

    x = jnp.asarray(x, dtype=gru.z.W0.dtype)
    h_prev = jnp.asarray(h_prev, dtype=gru.z.W0.dtype)
    reset = jnp.asarray(reset, dtype=bool)
    h_in = jnp.where(reset[..., None], jnp.zeros_like(h_prev), h_prev)
    xi = jnp.concatenate([x, h_in], axis=-1)
    sz = None if scratch is None else scratch.z
    sr = None if scratch is None else scratch.r
    sn = None if scratch is None else scratch.n
    z = jax.nn.sigmoid(addressed_dense_forward_sparse(gru.z, xi, k, ctx_id, sz))
    r = jax.nn.sigmoid(addressed_dense_forward_sparse(gru.r, xi, k, ctx_id, sr))
    xi_n = jnp.concatenate([x, r * h_in], axis=-1)
    n = jnp.tanh(addressed_dense_forward_sparse(gru.n, xi_n, k, ctx_id, sn))
    return (1.0 - z) * n + z * h_in


__all__ = [
    "GRUMemory",
    "GRUScratch",
    "addressed_conv_forward",
    "addressed_dense_forward",
    "addressed_dense_forward_sparse",
    "addressed_gru_step",
    "addressed_gru_step_sparse",
    "kernel_to_matrix",
    "materialize_conv_kernel",
    "matrix_to_kernel",
]
