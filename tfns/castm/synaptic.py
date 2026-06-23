"""Compact factorized synaptic memory for one contextualized affine operator.

Implements spec sections 5.3, 5.5, 6, 8.2, 15.3, 15.5, 15.6 and the numerical
invariants of section 16.

Representation
-------------
For an affine layer ``y = W0 x + b0`` the effective weight is addressed::

    W(k) = W0 + sum_m (c_m . k) B_m A_m
    b(k) = b0 + sum_m (c_m . k) beta_m

with ``A_m in R^{r_m x in}``, ``B_m in R^{out x r_m}``, ``c_m in R^{d_k}``,
``beta_m in R^{out}``. Each component ``m`` is a low-rank synaptic memory block,
not a game head; components are addressed by content and may be merged, rotated,
shared, or rewritten.

Storage is a fixed-shape preallocated pool (spec 11.1) so the structure jits and
serializes exactly. Per-slot ``rank`` records how many of the ``R`` factor rows
are live; padded rows/cols are kept at zero so the masked contractions stay
exact regardless of declared rank.

The primary no-forgetting guarantee (spec 5.3): writing a delta for context ``i``
with address factor ``c_m = d_i`` (the dual of ``k_i``) leaves the decoded
weights at every other protected address ``k_j`` unchanged, because
``d_i . k_j = delta_ij``.
"""

from __future__ import annotations

from typing import Any

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np


# Reserved context tags for non-context components.
CTX_FREE = -1
CTX_SHARED = -2  # compensation component written by shared consolidation


@struct.dataclass
class SynapticMemory:
    """Fixed-shape factorized addressed memory for one affine operator."""

    W0: jnp.ndarray            # (out, in)  shared weight (live, revisable)
    b0: jnp.ndarray            # (out,)     shared bias
    A: jnp.ndarray             # (M, R, in) input factors
    B: jnp.ndarray             # (M, out, R) output factors
    beta: jnp.ndarray          # (M, out)   addressed bias factors
    c: jnp.ndarray             # (M, d_k)   address factors (= d_i)
    active: jnp.ndarray        # (M,) bool
    ctx: jnp.ndarray           # (M,) int32 component-to-context index
    rank: jnp.ndarray          # (M,) int32 live rank per slot

    @property
    def out_dim(self) -> int:
        return int(self.W0.shape[0])

    @property
    def in_dim(self) -> int:
        return int(self.W0.shape[1])

    @property
    def comp_rank(self) -> int:
        return int(self.A.shape[1])

    @property
    def n_slots(self) -> int:
        return int(self.A.shape[0])

    @property
    def d_k(self) -> int:
        return int(self.c.shape[1])


def empty_synaptic_memory(
    W0: Any,
    b0: Any,
    *,
    comp_rank: int,
    n_slots: int,
    d_k: int,
    dtype: Any = jnp.float32,
) -> SynapticMemory:
    """Allocate an empty memory pool around shared weights ``W0, b0``."""

    W0 = jnp.asarray(W0, dtype=dtype)
    b0 = jnp.asarray(b0, dtype=dtype)
    out_dim, in_dim = int(W0.shape[0]), int(W0.shape[1])
    R, M = int(comp_rank), int(n_slots)
    return SynapticMemory(
        W0=W0,
        b0=b0,
        A=jnp.zeros((M, R, in_dim), dtype=dtype),
        B=jnp.zeros((M, out_dim, R), dtype=dtype),
        beta=jnp.zeros((M, out_dim), dtype=dtype),
        c=jnp.zeros((M, int(d_k)), dtype=dtype),
        active=jnp.zeros((M,), dtype=bool),
        ctx=jnp.full((M,), CTX_FREE, dtype=jnp.int32),
        rank=jnp.zeros((M,), dtype=jnp.int32),
    )


# --- Decoding (dense materialization; for audits and small layers) --------------


def _scales(mem: SynapticMemory, k: jnp.ndarray) -> jnp.ndarray:
    """Return active per-component address scales ``(c_m . k)`` (inactive -> 0)."""

    k = jnp.asarray(k, dtype=mem.c.dtype)
    s = mem.c @ k  # (M,)
    return jnp.where(mem.active, s, 0.0)


def decode_delta(mem: SynapticMemory, k: Any) -> jnp.ndarray:
    """Return the decoded weight delta ``sum_m (c_m.k) B_m A_m`` at address ``k``."""

    s = _scales(mem, jnp.asarray(k))
    # (M,out,R),(M,R,in) -> (M,out,in); weight by s and sum over components.
    return jnp.einsum("m,mor,mri->oi", s, mem.B, mem.A)


def decode_weight(mem: SynapticMemory, k: Any) -> jnp.ndarray:
    """Return the effective weight ``W(k) = W0 + sum_m (c_m.k) B_m A_m``."""

    return mem.W0 + decode_delta(mem, k)


def decode_bias_delta(mem: SynapticMemory, k: Any) -> jnp.ndarray:
    """Return the decoded bias delta ``sum_m (c_m.k) beta_m`` at address ``k``."""

    s = _scales(mem, jnp.asarray(k))
    return jnp.einsum("m,mo->o", s, mem.beta)


def decode_bias(mem: SynapticMemory, k: Any) -> jnp.ndarray:
    """Return the effective bias ``b(k) = b0 + sum_m (c_m.k) beta_m``."""

    return mem.b0 + decode_bias_delta(mem, k)


# --- Forward (factorized; no dense materialization) -----------------------------


def forward(mem: SynapticMemory, x: Any, k: Any) -> jnp.ndarray:
    """Effective affine forward ``y = W(k) x + b(k)`` using the factorization.

    ``x`` has shape ``(batch, in)``. This evaluates all active components (the
    dense-address path used for soft routing and equivalence tests). The sparse
    top-1 path is :func:`forward_sparse`.
    """

    x = jnp.asarray(x, dtype=mem.W0.dtype)
    s = _scales(mem, jnp.asarray(k))
    base = x @ mem.W0.T + mem.b0  # (b,out)
    ax = jnp.einsum("mri,bi->mbr", mem.A, x)        # (M,b,r)
    bax = jnp.einsum("mor,mbr->mbo", mem.B, ax)     # (M,b,out)
    delta = jnp.einsum("m,mbo->bo", s, bax)         # (b,out)
    bias = jnp.einsum("m,mo->o", s, mem.beta)       # (out,)
    return base + delta + bias


def forward_sparse(
    mem: SynapticMemory,
    x: Any,
    k: Any,
    ctx_id: int,
    *,
    include_shared: bool = True,
) -> jnp.ndarray:
    """Sparse forward gathering only the selected context's components.

    Components whose ``ctx`` differs from ``ctx_id`` (and, optionally, the shared
    compensation components) contribute exactly zero, so changing an unselected
    component cannot change the output (spec 11.2, test 17.9). Cost scales with
    the active components of the selected address, not total stored contexts.
    """

    x = jnp.asarray(x, dtype=mem.W0.dtype)
    select = mem.active & (mem.ctx == int(ctx_id))
    if include_shared:
        select = select | (mem.active & (mem.ctx == CTX_SHARED))
    s = jnp.where(select, mem.c @ jnp.asarray(k, dtype=mem.c.dtype), 0.0)
    base = x @ mem.W0.T + mem.b0
    ax = jnp.einsum("mri,bi->mbr", mem.A, x)
    bax = jnp.einsum("mor,mbr->mbo", mem.B, ax)
    delta = jnp.einsum("m,mbo->bo", s, bax)
    bias = jnp.einsum("m,mo->o", s, mem.beta)
    return base + delta + bias


# --- Writes (spec 5.3, 5.5, 6, 15.3) -------------------------------------------


def _pad_factor_rows(A_s: jnp.ndarray, R: int) -> jnp.ndarray:
    s, in_dim = int(A_s.shape[0]), int(A_s.shape[1])
    if s == R:
        return A_s
    pad = jnp.zeros((R - s, in_dim), dtype=A_s.dtype)
    return jnp.concatenate([A_s, pad], axis=0)


def _pad_factor_cols(B_s: jnp.ndarray, R: int) -> jnp.ndarray:
    out_dim, s = int(B_s.shape[0]), int(B_s.shape[1])
    if s == R:
        return B_s
    pad = jnp.zeros((out_dim, R - s), dtype=B_s.dtype)
    return jnp.concatenate([B_s, pad], axis=1)


def free_slot(mem: SynapticMemory) -> int:
    """Index of the first free slot, or ``-1`` if the pool is full."""

    free = np.where(~np.asarray(mem.active))[0]
    return int(free[0]) if free.size else -1


def append_component(
    mem: SynapticMemory,
    A_s: Any,
    B_s: Any,
    beta_s: Any,
    c_vec: Any,
    ctx_id: int,
) -> tuple[SynapticMemory, int]:
    """Append a low-rank component ``(A_s, B_s, beta_s)`` with address factor ``c_vec``.

    This is the cheap commit of spec 6/15.3: no dense materialization or SVD.
    ``A_s`` is ``(s, in)``, ``B_s`` is ``(out, s)`` with ``s <= comp_rank``.
    Returns the updated memory and the written slot index; the slot index is
    ``-1`` (and memory unchanged) when the pool is exhausted (spec 23.9).
    """

    A_s = jnp.asarray(A_s, dtype=mem.A.dtype)
    B_s = jnp.asarray(B_s, dtype=mem.B.dtype)
    beta_s = jnp.asarray(beta_s, dtype=mem.beta.dtype)
    c_vec = jnp.asarray(c_vec, dtype=mem.c.dtype)
    s = int(A_s.shape[0])
    if s > mem.comp_rank:
        raise ValueError(f"component rank {s} exceeds pool comp_rank {mem.comp_rank}")
    slot = free_slot(mem)
    if slot < 0:
        return mem, -1
    A = mem.A.at[slot].set(_pad_factor_rows(A_s, mem.comp_rank))
    B = mem.B.at[slot].set(_pad_factor_cols(B_s, mem.comp_rank))
    beta = mem.beta.at[slot].set(beta_s)
    c = mem.c.at[slot].set(c_vec)
    active = mem.active.at[slot].set(True)
    ctx = mem.ctx.at[slot].set(int(ctx_id))
    rank = mem.rank.at[slot].set(int(s))
    return mem.replace(A=A, B=B, beta=beta, c=c, active=active, ctx=ctx, rank=rank), slot


def context_components(mem: SynapticMemory, ctx_id: int) -> np.ndarray:
    """Return slot indices of active components belonging to ``ctx_id``."""

    mask = np.asarray(mem.active) & (np.asarray(mem.ctx) == int(ctx_id))
    return np.where(mask)[0]


def context_rank(mem: SynapticMemory, ctx_id: int) -> int:
    """Total live rank stored for ``ctx_id`` (sum over its components)."""

    slots = context_components(mem, ctx_id)
    if slots.size == 0:
        return 0
    return int(np.sum(np.asarray(mem.rank)[slots]))


def _free_slots(mem: SynapticMemory, slots: np.ndarray) -> SynapticMemory:
    A, B, beta, c = mem.A, mem.B, mem.beta, mem.c
    active, ctx, rank = mem.active, mem.ctx, mem.rank
    R, in_dim, out_dim = mem.comp_rank, mem.in_dim, mem.out_dim
    for slot in [int(s) for s in slots]:
        A = A.at[slot].set(jnp.zeros((R, in_dim), dtype=A.dtype))
        B = B.at[slot].set(jnp.zeros((out_dim, R), dtype=B.dtype))
        beta = beta.at[slot].set(jnp.zeros((out_dim,), dtype=beta.dtype))
        c = c.at[slot].set(jnp.zeros((mem.d_k,), dtype=c.dtype))
        active = active.at[slot].set(False)
        ctx = ctx.at[slot].set(CTX_FREE)
        rank = rank.at[slot].set(0)
    return mem.replace(A=A, B=B, beta=beta, c=c, active=active, ctx=ctx, rank=rank)


# --- Recompression (spec 15.5, 16.8) -------------------------------------------


def _truncated_rank(s_vals: np.ndarray, energy: float, max_rank: int) -> int:
    total = float(np.sum(s_vals ** 2))
    if total <= 0.0:
        return 1
    csum = np.cumsum(s_vals ** 2) / total
    r = int(np.searchsorted(csum, float(energy)) + 1)
    return int(min(max(r, 1), int(max_rank), int(s_vals.size)))


def recompress_context(
    mem: SynapticMemory,
    ctx_id: int,
    *,
    energy: float = 0.995,
    max_rank: int | None = None,
    max_elem_tol: float | None = None,
) -> tuple[SynapticMemory, dict[str, Any]]:
    """Recompress all components of ``ctx_id`` into a single low-rank component.

    Decodes the combined delta ``Delta = sum_m B_m A_m`` for the context (all its
    components share the same address factor ``c = d_i``), takes a truncated SVD,
    and replaces the components with one slot of the smallest rank satisfying the
    reconstruction tolerance. Compression is never accepted on average error
    alone: the maximum per-element error must also pass (spec 16.8).

    Returns ``(mem, report)``. On failure (``max_elem_tol`` exceeded at full
    allowed rank) the memory is returned unchanged with ``report['accepted']``
    False so the caller can roll back (spec 15.5 commit-only-on-success).
    """

    slots = context_components(mem, ctx_id)
    report: dict[str, Any] = {"ctx": int(ctx_id), "n_before": int(slots.size)}
    if slots.size <= 1:
        report.update(accepted=True, n_after=int(slots.size), rank=context_rank(mem, ctx_id), noop=True)
        return mem, report

    R = mem.comp_rank if max_rank is None else int(max_rank)
    R = min(int(R), mem.comp_rank)
    c_shared = mem.c[int(slots[0])]
    # Combined delta and bias for the context.
    A_sel = mem.A[jnp.asarray(slots)]   # (g,R,in)
    B_sel = mem.B[jnp.asarray(slots)]   # (g,out,R)
    delta = jnp.einsum("gor,gri->oi", B_sel, A_sel)  # (out,in)
    beta_sum = jnp.sum(mem.beta[jnp.asarray(slots)], axis=0)  # (out,)

    delta_np = np.asarray(delta, dtype=np.float64)
    u, s_vals, vt = np.linalg.svd(delta_np, full_matrices=False)
    r_star = _truncated_rank(s_vals, energy, R)
    # Enforce max per-element reconstruction error if requested (spec 16.8).
    if max_elem_tol is not None:
        while r_star <= R:
            approx = (u[:, :r_star] * s_vals[:r_star]) @ vt[:r_star]
            max_err = float(np.max(np.abs(approx - delta_np)))
            if max_err <= float(max_elem_tol) or r_star >= min(R, int(s_vals.size)):
                break
            r_star += 1
        approx = (u[:, :r_star] * s_vals[:r_star]) @ vt[:r_star]
        max_err = float(np.max(np.abs(approx - delta_np)))
        if max_err > float(max_elem_tol):
            report.update(accepted=False, reason="max_elem_tol_exceeded", max_err=max_err, rank=r_star)
            return mem, report

    sqrt_s = np.sqrt(s_vals[:r_star])
    B_new = jnp.asarray((u[:, :r_star] * sqrt_s), dtype=mem.B.dtype)        # (out, r*)
    A_new = jnp.asarray((sqrt_s[:, None] * vt[:r_star]), dtype=mem.A.dtype)  # (r*, in)

    mem2 = _free_slots(mem, slots)
    mem2, slot = append_component(mem2, A_new, B_new, beta_sum, c_shared, int(ctx_id))
    if slot < 0:
        report.update(accepted=False, reason="no_free_slot")
        return mem, report
    report.update(accepted=True, n_after=1, rank=int(r_star), noop=False)
    return mem2, report


# --- Exact compensated shared consolidation (spec 8.2, 15.6) --------------------


def shared_consolidate(
    mem: SynapticMemory,
    A_S: Any,
    B_S: Any,
    g: Any,
    *,
    beta_S: Any | None = None,
) -> tuple[SynapticMemory, int]:
    """Move a common low-rank component ``S = B_S A_S`` into shared weights.

    ``W0 <- W0 + S`` and a compensation component ``-S`` is appended with address
    factor ``g`` satisfying ``K^T g = 1`` (from :func:`address.compensation_vector`).
    Every protected decoded operator is unchanged because ``g . k_i = 1`` so the
    added shared ``S`` is exactly cancelled at each address (spec 8.2).

    Returns ``(mem, slot)``; ``slot`` is ``-1`` if the pool is full.
    """

    A_S = jnp.asarray(A_S, dtype=mem.A.dtype)
    B_S = jnp.asarray(B_S, dtype=mem.B.dtype)
    g = jnp.asarray(g, dtype=mem.c.dtype)
    S = B_S @ A_S
    W0_new = mem.W0 + S
    b0_new = mem.b0
    if beta_S is not None:
        beta_S = jnp.asarray(beta_S, dtype=mem.beta.dtype)
        b0_new = mem.b0 + beta_S
        comp_beta = -beta_S
    else:
        comp_beta = jnp.zeros((mem.out_dim,), dtype=mem.beta.dtype)
    mem2 = mem.replace(W0=W0_new, b0=b0_new)
    mem2, slot = append_component(mem2, A_S, -B_S, comp_beta, g, CTX_SHARED)
    if slot < 0:
        return mem, -1
    return mem2, slot


# --- Accounting and health (spec 11.3, 16.7) -----------------------------------


def nbytes(mem: SynapticMemory) -> dict[str, int]:
    """Return a byte breakdown of this memory's arrays (spec 11.3)."""

    def nb(x: jnp.ndarray) -> int:
        return int(np.asarray(x).nbytes)

    shared = nb(mem.W0) + nb(mem.b0)
    factors = nb(mem.A) + nb(mem.B) + nb(mem.beta)
    address = nb(mem.c)
    meta = nb(mem.active) + nb(mem.ctx) + nb(mem.rank)
    return {
        "shared": shared,
        "factors": factors,
        "address_factors": address,
        "meta": meta,
        "total": shared + factors + address + meta,
    }


def active_factor_bytes(mem: SynapticMemory) -> int:
    """Bytes attributable to *live* factor rank (excludes padded/free slots)."""

    ranks = np.asarray(mem.rank)
    active = np.asarray(mem.active)
    live_rank = int(np.sum(ranks[active]))
    per_rank = (mem.in_dim + mem.out_dim) * int(np.asarray(mem.A).dtype.itemsize)
    return live_rank * per_rank


def synaptic_has_nan(mem: SynapticMemory) -> bool:
    """True if any factor / address / shared array contains NaN or Inf (spec 16.7)."""

    arrays = [mem.W0, mem.b0, mem.A, mem.B, mem.beta, mem.c]
    return bool(any(not jnp.all(jnp.isfinite(a)) for a in arrays))


__all__ = [
    "CTX_FREE",
    "CTX_SHARED",
    "SynapticMemory",
    "active_factor_bytes",
    "append_component",
    "context_components",
    "context_rank",
    "decode_bias",
    "decode_bias_delta",
    "decode_delta",
    "decode_weight",
    "empty_synaptic_memory",
    "forward",
    "forward_sparse",
    "free_slot",
    "nbytes",
    "recompress_context",
    "shared_consolidate",
    "synaptic_has_nan",
]
