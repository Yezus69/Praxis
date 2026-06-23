"""Exact compensated common-structure consolidation (spec 8.3, 15.6).

Reusable structure shared by several contexts is migrated from context-specific
memory into the shared substrate ``W0`` without forgetting: a common low-rank
component ``S`` is found from the decoded residuals, moved into ``W0``, and
exactly cancelled at every protected address by a compensation component with
address factor ``g`` satisfying ``K^T g = 1`` (spec 8.2).

Per spec 8.3 / 27, shared consolidation is enabled only after addressed writes
and routing pass independently; it becomes required before the full five-game run.
"""

from __future__ import annotations

from typing import Any, Mapping

import numpy as np

from tfns.castm import address as addr
from tfns.castm import audit
from tfns.castm import synaptic as syn


def find_common_component(
    mem: syn.SynapticMemory,
    book: addr.AddressBook,
    *,
    rank: int,
    robust: bool = True,
) -> tuple[np.ndarray, np.ndarray]:
    """Find a common low-rank component ``S = B_S A_S`` of the context residuals.

    Decodes each used context's memory residual ``W(k_i) - W0`` and takes either a
    robust (median) or mean low-rank component via truncated SVD. Returns
    ``(A_S (rank,in), B_S (out,rank))`` as numpy arrays.
    """

    used = np.where(np.asarray(book.used))[0]
    residuals = []
    for i in [int(x) for x in used]:
        delta = np.asarray(syn.decode_delta(mem, addr.code(book, i)), dtype=np.float64)
        residuals.append(delta)
    if not residuals:
        out_dim, in_dim = mem.out_dim, mem.in_dim
        return np.zeros((int(rank), in_dim)), np.zeros((out_dim, int(rank)))
    stack = np.stack(residuals, axis=0)  # (n, out, in)
    common = np.median(stack, axis=0) if robust else np.mean(stack, axis=0)
    u, s, vt = np.linalg.svd(common, full_matrices=False)
    r = int(min(int(rank), s.size))
    sqrt_s = np.sqrt(s[:r])
    B_S = (u[:, :r] * sqrt_s)
    A_S = (sqrt_s[:, None] * vt[:r])
    return A_S, B_S


def consolidate_layer(
    mem: syn.SynapticMemory,
    book: addr.AddressBook,
    *,
    rank: int,
    eps_write: float = 1e-4,
    orthonormal: bool = True,
) -> tuple[syn.SynapticMemory, dict[str, Any]]:
    """Consolidate one layer's common structure into ``W0`` with exact compensation.

    Returns ``(mem, report)``; commits only if every protected decoded operator is
    unchanged within ``eps_write`` (spec 15.6 verify-then-commit).
    """

    A_S, B_S = find_common_component(mem, book, rank=rank)
    g = addr.compensation_vector(book, orthonormal=orthonormal)
    before = {i: np.asarray(syn.decode_weight(mem, addr.code(book, i)))
              for i in [int(x) for x in np.where(np.asarray(book.used))[0]]}
    mem2, slot = syn.shared_consolidate(mem, A_S, B_S, g)
    if slot < 0:
        return mem, {"accepted": False, "reason": "no_free_slot"}
    max_drift = 0.0
    for i, w0 in before.items():
        after = np.asarray(syn.decode_weight(mem2, addr.code(book, i)))
        denom = float(np.linalg.norm(w0)) + 1e-12
        drift = float(np.linalg.norm(after - w0)) / denom
        max_drift = max(max_drift, drift)
    if max_drift > eps_write:
        return mem, {"accepted": False, "reason": "drift", "max_drift": max_drift}
    s_norm = float(np.linalg.norm(B_S @ A_S))
    return mem2, {"accepted": True, "max_drift": max_drift, "s_norm": s_norm, "slot": int(slot)}


def consolidate_bank(
    banks: Mapping[str, syn.SynapticMemory],
    book: addr.AddressBook,
    *,
    rank: int = 4,
    eps_write: float = 1e-4,
    orthonormal: bool = True,
) -> tuple[dict[str, syn.SynapticMemory], dict[str, Any]]:
    """Consolidate every layer atomically; roll back the whole bank on any failure."""

    proposed: dict[str, syn.SynapticMemory] = {}
    layer_reports: dict[str, Any] = {}
    ok = True
    reason = None
    for name, mem in banks.items():
        mem2, rep = consolidate_layer(mem, book, rank=rank, eps_write=eps_write, orthonormal=orthonormal)
        layer_reports[name] = rep
        if not rep["accepted"]:
            ok = False
            reason = f"{name}:{rep.get('reason')}"
            break
        proposed[name] = mem2
    report = {"accepted": ok, "reason": reason, "layers": layer_reports}
    if not ok:
        return dict(banks), report
    return proposed, report


__all__ = ["consolidate_bank", "consolidate_layer", "find_common_component"]
