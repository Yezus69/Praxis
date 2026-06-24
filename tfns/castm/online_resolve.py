"""Online incremental compensated resolve (architecture section 4, core invariant).

The shared weights ``W0`` stay fully plastic across the whole stream. During an
interval in which inferred context ``a`` is active, PPO updates ``W0 -> W0 + ΔW0``.
To keep ``a`` learning while every inactive context is preserved, the contextual
corrections are updated as

    D_c' = Compress_R( D_c - ΔW0 )   and   β_c' = β_c - Δb0   for c != a,
    D_a' = D_a                       (active context unchanged),

so for every inactive context ``W0' + D_c' = W0 + D_c`` (preserved) and for the
active context ``W0' + D_a' = W0 + D_a + ΔW0`` (full update applied).

This generalises the end-of-game ``resolve_memory`` to run *mid-stream* with an
inferred active context that may itself carry a correction (a revisited context):
its components are copied **verbatim** (exact preservation, zero recompression
error), while inactive contexts are compensated and recompressed. The value head
is protected by default (``include_value=True``); excluding it is an explicit
ablation. A newly-allocated context has ``D=0`` at the current ``W0`` and is the
active context for its first interval, so it rides ``W0`` directly; once the stream
leaves it, it is compensated like any other inactive context.

Every inactive correction's reconstruction error is audited (max element +
relative Frobenius) against the configured budget; the report flags any violation
so the caller can shorten the resolve interval, raise the rank, or reject.
"""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import jax.numpy as jnp
import numpy as np

from tfns.castm import address as addr
from tfns.castm import synaptic as syn


def svd_lowrank(residual: Any, rank: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Return ``(A (r,in), B (out,r))`` with ``B @ A ≈ residual`` truncated to rank.

    Finite-safe: non-finite entries are zeroed and a non-converging SVD falls back
    to a jittered retry, then to a zero factor, rather than crashing the run.
    """

    res = np.nan_to_num(np.asarray(residual, dtype=np.float64), nan=0.0, posinf=0.0, neginf=0.0)
    out_dim, in_dim = res.shape
    r = int(min(int(rank), out_dim, in_dim))
    if r <= 0:
        return jnp.zeros((1, in_dim), jnp.float32), jnp.zeros((out_dim, 1), jnp.float32)
    try:
        u, s, vt = np.linalg.svd(res, full_matrices=False)
    except np.linalg.LinAlgError:
        try:
            u, s, vt = np.linalg.svd(res + 1e-6 * np.random.standard_normal(res.shape), full_matrices=False)
        except np.linalg.LinAlgError:
            return jnp.zeros((r, in_dim), jnp.float32), jnp.zeros((out_dim, r), jnp.float32)
    sq = np.sqrt(s[:r])
    B = (u[:, :r] * sq).astype(np.float32)
    A = (sq[:, None] * vt[:r]).astype(np.float32)
    return jnp.asarray(A), jnp.asarray(B)


def _copy_context_components(mem_src: syn.SynapticMemory, mem_dst: syn.SynapticMemory,
                             ctx_id: int) -> syn.SynapticMemory:
    """Append ``ctx_id``'s active components from ``mem_src`` into ``mem_dst`` verbatim."""

    slots = syn.context_components(mem_src, int(ctx_id))
    for slot in [int(s) for s in slots]:
        rk = int(np.asarray(mem_src.rank)[slot])
        if rk <= 0:
            continue
        A_s = mem_src.A[slot, :rk, :]
        B_s = mem_src.B[slot, :, :rk]
        beta_s = mem_src.beta[slot]
        c_vec = mem_src.c[slot]
        mem_dst, written = syn.append_component(mem_dst, A_s, B_s, beta_s, c_vec, int(ctx_id))
        if written < 0:
            raise RuntimeError(f"resolve: pool exhausted copying active ctx {ctx_id}")
    return mem_dst


def online_resolve(
    banks: Mapping[str, syn.SynapticMemory],
    book: addr.AddressBook,
    snap_W0: Mapping[str, Any],
    snap_b0: Mapping[str, Any],
    active_ctx: int,
    ctx_ids: Sequence[int],
    mem_rank: int,
    *,
    include_value: bool = True,
    max_elem_tol: float = 5e-2,
    rel_fro_tol: float = 1e-1,
) -> tuple[dict[str, syn.SynapticMemory], dict[str, Any]]:
    """Compensate every inactive context for the shared drift since the snapshot.

    ``snap_W0``/``snap_b0`` are the shared weights/bias captured at the start of the
    interval. ``ctx_ids`` is every internal context discovered so far. ``active_ctx``
    is the inferred context that was training during the interval (``-1`` if none —
    e.g. a bootstrap interval — in which case nothing is protected). Returns the new
    banks and an audit report.
    """

    dW0 = {name: (np.asarray(banks[name].W0, np.float64) - np.asarray(snap_W0[name], np.float64))
           for name in banks}
    db0 = {name: (np.asarray(banks[name].b0, np.float64) - np.asarray(snap_b0[name], np.float64))
           for name in banks}
    max_dW0 = float(max((np.max(np.abs(dW0[n])) for n in dW0), default=0.0))

    new_banks: dict[str, syn.SynapticMemory] = {}
    per_layer: dict[str, Any] = {}
    worst_residual = 0.0
    worst_relfro = 0.0
    budget_ok = True
    violations: list[dict] = []

    inactive = [int(c) for c in ctx_ids if int(c) != int(active_ctx)]

    for name, mem in banks.items():
        if name == "value" and not include_value:
            # Ablation: drop value protection — value head freely tracks the stream.
            new_banks[name] = syn.empty_synaptic_memory(
                mem.W0, mem.b0, comp_rank=mem.comp_rank, n_slots=mem.n_slots, d_k=mem.d_k)
            per_layer[name] = {"skipped": True}
            continue

        m2 = syn.empty_synaptic_memory(mem.W0, mem.b0, comp_rank=mem.comp_rank,
                                       n_slots=mem.n_slots, d_k=mem.d_k)
        # Active context: copy its correction verbatim (D_a' = D_a, exact).
        if int(active_ctx) >= 0 and int(active_ctx) in [int(c) for c in ctx_ids]:
            m2 = _copy_context_components(mem, m2, int(active_ctx))

        layer_max = 0.0
        layer_relfro = 0.0
        ctx_err = {}
        for c in inactive:
            k_c = addr.code(book, c)
            cur = np.asarray(syn.decode_delta(mem, k_c), dtype=np.float64)
            cur_beta = np.asarray(syn.decode_bias_delta(mem, k_c), dtype=np.float64)
            residual = cur - dW0[name]
            beta_new = (cur_beta - db0[name]).astype(np.float32)
            A, B = svd_lowrank(residual, min(int(mem_rank), mem.comp_rank))
            approx = np.asarray(B @ A, dtype=np.float64)
            err = float(np.max(np.abs(approx - residual))) if residual.size else 0.0
            relfro = (float(np.linalg.norm(approx - residual)) /
                      (float(np.linalg.norm(residual)) + 1e-12)) if residual.size else 0.0
            ctx_err[c] = {"max_elem": err, "rel_fro": relfro}
            layer_max = max(layer_max, err)
            layer_relfro = max(layer_relfro, relfro)
            m2, written = syn.append_component(m2, A, B, jnp.asarray(beta_new),
                                               addr.code(book, c), int(c))
            if written < 0:
                raise RuntimeError(f"resolve: pool exhausted on inactive ctx {c} layer {name}")
            if err > max_elem_tol or relfro > rel_fro_tol:
                budget_ok = False
                violations.append({"layer": name, "ctx": c, "max_elem": err, "rel_fro": relfro})

        new_banks[name] = m2
        per_layer[name] = {"max_elem": layer_max, "rel_fro": layer_relfro, "ctx": ctx_err}
        worst_residual = max(worst_residual, layer_max)
        worst_relfro = max(worst_relfro, layer_relfro)

    report = {
        "active_ctx": int(active_ctx),
        "n_inactive": len(inactive),
        "max_dW0": max_dW0,
        "max_residual": worst_residual,
        "max_rel_fro": worst_relfro,
        "budget_ok": bool(budget_ok),
        "violations": violations,
        "per_layer": per_layer,
        "include_value": bool(include_value),
    }
    return new_banks, report


def snapshot_shared(banks: Mapping[str, syn.SynapticMemory]) -> tuple[dict, dict]:
    """Capture the current shared ``W0``/``b0`` per layer (a transient global snapshot)."""

    snap_W0 = {name: np.asarray(banks[name].W0) for name in banks}
    snap_b0 = {name: np.asarray(banks[name].b0) for name in banks}
    return snap_W0, snap_b0


__all__ = ["online_resolve", "snapshot_shared", "svd_lowrank"]
