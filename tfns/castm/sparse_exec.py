"""Actually-sparse addressed execution (architecture section 5, test 11).

``synaptic.forward_sparse`` is *functionally* sparse — unselected contexts
contribute exactly zero — but it still contracts over **all** ``M`` stored slots
and masks, so its memory-delta work grows with the number of stored contexts. This
module provides a **gather-before-matmul** forward whose work is constant in the
number of stored contexts.

Layout: each internal context ``c`` owns a fixed contiguous block of
``slots_per_ctx`` slots, ``[c*spc : (c+1)*spc]``. The forward gathers ONLY that
block (a fixed-shape ``dynamic_slice``) and contracts over ``spc`` components. The
gathered tensors have shape independent of the total context count, so the compiled
XLA program — and therefore the FLOP count — is identical whether 1 or 57 contexts
are stored. Changing an unselected context's factors (a different block) cannot
change the selected output: it is never gathered.

This is the proof that the content-addressed factorisation supports true O(1)-in-
contexts execution. The pilot trainer uses the functional path (overhead negligible
at <=8 contexts); this blocked path is the constant-cost demonstration + benchmark.
"""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tfns.castm import synaptic as syn


def blocked_memory(W0, b0, *, n_contexts: int, slots_per_ctx: int, comp_rank: int, d_k: int):
    """Allocate a memory whose slot pool is partitioned into per-context blocks."""

    return syn.empty_synaptic_memory(W0, b0, comp_rank=comp_rank,
                                     n_slots=int(n_contexts) * int(slots_per_ctx), d_k=d_k)


def place_blocked(mem: syn.SynapticMemory, A_s, B_s, beta_s, c_vec, ctx_id: int,
                  slots_per_ctx: int, *, offset: int = 0) -> syn.SynapticMemory:
    """Write a component into context ``ctx_id``'s fixed block at ``offset``."""

    spc = int(slots_per_ctx)
    slot = int(ctx_id) * spc + int(offset)
    A = mem.A.at[slot].set(syn._pad_factor_rows(jnp.asarray(A_s, mem.A.dtype), mem.comp_rank))
    B = mem.B.at[slot].set(syn._pad_factor_cols(jnp.asarray(B_s, mem.B.dtype), mem.comp_rank))
    beta = mem.beta.at[slot].set(jnp.asarray(beta_s, mem.beta.dtype))
    c = mem.c.at[slot].set(jnp.asarray(c_vec, mem.c.dtype))
    active = mem.active.at[slot].set(True)
    ctx = mem.ctx.at[slot].set(int(ctx_id))
    rank = mem.rank.at[slot].set(int(jnp.asarray(A_s).shape[0]))
    return mem.replace(A=A, B=B, beta=beta, c=c, active=active, ctx=ctx, rank=rank)


def forward_sparse_blocked(mem: syn.SynapticMemory, x: Any, k: Any, ctx_id: int,
                           slots_per_ctx: int) -> jnp.ndarray:
    """Gather only ``ctx_id``'s block, then matmul — work is O(slots_per_ctx).

    The gathered factor tensors have static shape ``(spc, R, in)`` / ``(spc, out, R)``
    regardless of how many contexts are stored, so the contracted work is constant
    in the number of stored contexts (architecture section 5).
    """

    x = jnp.asarray(x, dtype=mem.W0.dtype)
    spc = int(slots_per_ctx)
    start = int(ctx_id) * spc
    A_blk = jax.lax.dynamic_slice_in_dim(mem.A, start, spc, axis=0)        # (spc,R,in)
    B_blk = jax.lax.dynamic_slice_in_dim(mem.B, start, spc, axis=0)        # (spc,out,R)
    beta_blk = jax.lax.dynamic_slice_in_dim(mem.beta, start, spc, axis=0)  # (spc,out)
    c_blk = jax.lax.dynamic_slice_in_dim(mem.c, start, spc, axis=0)        # (spc,d_k)
    active_blk = jax.lax.dynamic_slice_in_dim(mem.active, start, spc, axis=0)
    k = jnp.asarray(k, dtype=mem.c.dtype)
    s = jnp.where(active_blk, c_blk @ k, 0.0)                              # (spc,)
    base = x @ mem.W0.T + mem.b0
    ax = jnp.einsum("sri,bi->sbr", A_blk, x)
    bax = jnp.einsum("sor,sbr->sbo", B_blk, ax)
    delta = jnp.einsum("s,sbo->bo", s, bax)
    bias = jnp.einsum("s,so->o", s, beta_blk)
    return base + delta + bias


def gathered_shapes(mem: syn.SynapticMemory, ctx_id: int, slots_per_ctx: int) -> dict:
    """Shapes of the tensors actually contracted (for the constant-work proof)."""

    spc = int(slots_per_ctx)
    start = int(ctx_id) * spc
    A_blk = jax.lax.dynamic_slice_in_dim(mem.A, start, spc, axis=0)
    B_blk = jax.lax.dynamic_slice_in_dim(mem.B, start, spc, axis=0)
    return {"A": tuple(A_blk.shape), "B": tuple(B_blk.shape)}


def benchmark(out_dim=512, in_dim=512, comp_rank=64, slots_per_ctx=2, d_k=128,
              batch=256, context_counts=(1, 5, 20, 57), repeats=50, seed=0) -> dict:
    """Time the blocked forward vs the functional forward at growing context counts.

    Returns median wall-times (ms) per call for each path and context count, and the
    blocked-path overhead ratio between the largest and a 5-context reference.
    """

    import time

    rng = np.random.default_rng(seed)
    W0 = jnp.asarray(rng.standard_normal((out_dim, in_dim)).astype(np.float32) * 0.05)
    b0 = jnp.zeros((out_dim,), jnp.float32)
    x = jnp.asarray(rng.standard_normal((batch, in_dim)).astype(np.float32))
    results = {"blocked_ms": {}, "functional_ms": {}}

    blocked_jit = jax.jit(forward_sparse_blocked, static_argnames=("ctx_id", "slots_per_ctx"))
    func_jit = jax.jit(syn.forward_sparse, static_argnames=("ctx_id",))

    for N in context_counts:
        mem = blocked_memory(W0, b0, n_contexts=N, slots_per_ctx=slots_per_ctx, comp_rank=comp_rank, d_k=d_k)
        for c in range(N):
            A = rng.standard_normal((comp_rank, in_dim)).astype(np.float32) * 0.01
            B = rng.standard_normal((out_dim, comp_rank)).astype(np.float32) * 0.01
            beta = np.zeros((out_dim,), np.float32)
            cvec = rng.standard_normal((d_k,)).astype(np.float32)
            cvec /= np.linalg.norm(cvec)
            mem = place_blocked(mem, A, B, beta, cvec, c, slots_per_ctx)
        k = jnp.asarray(mem.c[0])  # address of context 0's block

        def time_fn(fn):
            r = fn().block_until_ready()
            best = []
            for _ in range(repeats):
                t0 = time.perf_counter()
                fn().block_until_ready()
                best.append((time.perf_counter() - t0) * 1e3)
            return float(np.median(best))

        results["blocked_ms"][N] = time_fn(lambda: blocked_jit(mem, x, k, 0, slots_per_ctx))
        results["functional_ms"][N] = time_fn(lambda: func_jit(mem, x, k, 0))

    b5 = results["blocked_ms"].get(5) or next(iter(results["blocked_ms"].values()))
    bmax = results["blocked_ms"][max(context_counts)]
    results["blocked_overhead_5_to_max"] = bmax / b5
    fmax = results["functional_ms"][max(context_counts)]
    f5 = results["functional_ms"].get(5) or next(iter(results["functional_ms"].values()))
    results["functional_overhead_5_to_max"] = fmax / f5
    results["params"] = {"out": out_dim, "in": in_dim, "rank": comp_rank, "spc": slots_per_ctx,
                         "batch": batch, "context_counts": list(context_counts)}
    return results


__all__ = ["benchmark", "blocked_memory", "forward_sparse_blocked", "gathered_shapes", "place_blocked"]
