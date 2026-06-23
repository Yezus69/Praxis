"""Transactional commit / recompression / consolidation across all layers.

Implements the multi-layer orchestration of spec 15.3 (commit scratch to a known
or novel address with exact decoded-weight audit and atomic rollback), 15.5
(per-address recompression), and the capacity handling of 23.9 (recompress on a
full pool before failing).

A "bank" is a ``dict[str, SynapticMemory]`` over the contextualized layers
(spec 6.5): conv1, conv2, conv3, encoder dense, three GRU gates, policy head,
value head. A "scratch bank" is the matching ``dict[str, ScratchDelta]``.

The decoded-weight invariants are the hard no-forgetting guarantee. An optional
``sentinel_fn`` adds functional sentinels (spec 15.3 step 5); when it returns
False the whole transaction rolls back.
"""

from __future__ import annotations

from typing import Any, Callable, Mapping

from tfns.castm import address as addr
from tfns.castm import audit
from tfns.castm import scratch as scr
from tfns.castm import synaptic as syn


# Default tolerances (spec 16.4-16.5). Mixed-precision integration uses 1e-4.
EPS_WRITE = 1e-4
EPS_INTENDED = 1e-3


def _make_room(mem: syn.SynapticMemory, ctx_id, *, energy, max_elem_tol):
    """Ensure a free slot for ``ctx_id``, recompressing the context if full (23.9).

    Returns ``(mem_base, note)`` where ``mem_base`` has a free slot, or
    ``(None, reason)`` if room cannot be made. Recompression preserves every other
    protected context exactly and the recompressed context within ``max_elem_tol``,
    so ``mem_base`` is the correct baseline for auditing the subsequent append.
    """

    if syn.free_slot(mem) >= 0:
        return mem, None
    mem_rc, report = syn.recompress_context(
        mem, ctx_id, energy=energy, max_rank=mem.comp_rank, max_elem_tol=max_elem_tol
    )
    if not report.get("accepted", False):
        return None, "pool_full_recompress_failed"
    if syn.free_slot(mem_rc) < 0:
        return None, "pool_full_after_recompress"
    return mem_rc, "recompressed"


def commit_scratch_bank(
    banks: Mapping[str, syn.SynapticMemory],
    scratch_bank: Mapping[str, scr.ScratchDelta],
    book: addr.AddressBook,
    ctx_id: int,
    *,
    eps_write: float = EPS_WRITE,
    eps_intended: float = EPS_INTENDED,
    energy: float = 0.999,
    max_elem_tol: float | None = 1e-3,
    sentinel_fn: Callable[[dict, dict], bool] | None = None,
    orthonormal: bool = True,
) -> tuple[dict[str, syn.SynapticMemory], dict[str, Any]]:
    """Commit the scratch bank to address ``ctx_id`` atomically (spec 15.3).

    For every contextualized layer the scratch factors are appended with the
    address dual ``d_i``. The transaction then verifies, across every protected
    canonical address ``j != i``, that decoded weights are unchanged within
    ``eps_write`` and that the intended write landed at ``i`` within
    ``eps_intended``. If any layer fails, or a sentinel rejects, nothing is
    committed and the original banks are returned with ``report['accepted']``
    False (spec 15.3 rollback / 25 stop-on-drift).
    """

    d_i = addr.dual(book, int(ctx_id), orthonormal=orthonormal)
    proposed: dict[str, syn.SynapticMemory] = {}
    layer_reports: dict[str, Any] = {}
    max_ni = 0.0
    max_iw = 0.0
    ok = True
    reason = None

    for name, mem in banks.items():
        s = scratch_bank.get(name)
        if s is None:
            proposed[name] = mem
            continue
        expected = scr.scratch_delta_weight(s)
        mem_base, note = _make_room(mem, int(ctx_id), energy=energy, max_elem_tol=max_elem_tol)
        if mem_base is None:
            ok = False
            reason = f"{name}:{note}"
            break
        mem2, slot = syn.append_component(mem_base, s.A_s, s.B_s, s.beta_s, d_i, int(ctx_id))
        if slot < 0:
            ok = False
            reason = f"{name}:append_failed"
            break
        # End-to-end noninterference (original -> final) at every j != i; the
        # intended write is measured against the post-recompression baseline so
        # in-tolerance recompression drift is not charged to the new write.
        ni = audit.max_noninterference_error(mem, mem2, book, int(ctx_id))
        iw = audit.intended_write_error(mem_base, mem2, book, int(ctx_id), expected, bias_expected=s.beta_s)
        layer_reports[name] = {"noninterference": ni, "intended": iw, "slot": int(slot), "note": note}
        max_ni = max(max_ni, ni)
        max_iw = max(max_iw, iw)
        if ni > eps_write:
            ok = False
            reason = f"{name}:noninterference {ni:.2e} > {eps_write:.0e}"
            break
        if iw > eps_intended:
            ok = False
            reason = f"{name}:intended {iw:.2e} > {eps_intended:.0e}"
            break
        proposed[name] = mem2

    if ok and sentinel_fn is not None:
        if not sentinel_fn(dict(banks), proposed):
            ok = False
            reason = "sentinel_rejected"

    report = {
        "accepted": ok,
        "ctx": int(ctx_id),
        "max_noninterference": float(max_ni),
        "max_intended": float(max_iw),
        "reason": reason,
        "layers": layer_reports,
    }
    if not ok:
        return dict(banks), report
    return proposed, report


def commit_scratch_to_novel(
    banks: Mapping[str, syn.SynapticMemory],
    scratch_bank: Mapping[str, scr.ScratchDelta],
    book: addr.AddressBook,
    **kwargs,
) -> tuple[dict[str, syn.SynapticMemory], addr.AddressBook, int, dict[str, Any]]:
    """Allocate a fresh canonical address and commit the scratch to it (spec 15.4)."""

    book2, ctx_id = addr.allocate_canonical(book)
    banks2, report = commit_scratch_bank(banks, scratch_bank, book2, ctx_id, **kwargs)
    if not report["accepted"]:
        # Roll back the allocation too.
        return dict(banks), book, -1, report
    return banks2, book2, ctx_id, report


def recompress_bank(
    banks: Mapping[str, syn.SynapticMemory],
    book: addr.AddressBook,
    ctx_id: int,
    *,
    energy: float = 0.995,
    max_elem_tol: float = 1e-3,
    rel_tol: float = 1e-3,
) -> tuple[dict[str, syn.SynapticMemory], dict[str, Any]]:
    """Recompress one context across all layers, verifying decoded weights (15.5).

    Commits only if every layer's per-context decoded-weight reconstruction stays
    within ``max_elem_tol`` (absolute) and ``rel_tol`` (relative Frobenius), and
    every *other* protected context is exactly unchanged.
    """

    proposed: dict[str, syn.SynapticMemory] = {}
    layer_reports: dict[str, Any] = {}
    ok = True
    reason = None
    for name, mem in banks.items():
        mem2, rep = syn.recompress_context(
            mem, int(ctx_id), energy=energy, max_rank=mem.comp_rank, max_elem_tol=max_elem_tol
        )
        if not rep.get("accepted", False):
            ok = False
            reason = f"{name}:{rep.get('reason', 'recompress_failed')}"
            break
        rec = audit.reconstruction_error(mem, mem2, book, int(ctx_id))
        ni = audit.max_noninterference_error(mem, mem2, book, int(ctx_id))
        layer_reports[name] = {**rep, **rec, "noninterference": ni}
        if rec["max_abs"] > max_elem_tol or rec["rel_fro"] > rel_tol or ni > max_elem_tol:
            ok = False
            reason = f"{name}:reconstruction {rec}"
            break
        proposed[name] = mem2
    report = {"accepted": ok, "ctx": int(ctx_id), "reason": reason, "layers": layer_reports}
    if not ok:
        return dict(banks), report
    return proposed, report


def reset_scratch_bank(scratch_bank: Mapping[str, scr.ScratchDelta], key, *, a_scale: float = 0.02):
    """Return a fresh zero-delta scratch bank (spec 9.5 step 2 / 7)."""

    import jax

    names = list(scratch_bank.keys())
    keys = jax.random.split(key, len(names))
    return {
        name: scr.reset_scratch(scratch_bank[name], keys[i], a_scale=a_scale)
        for i, name in enumerate(names)
    }


__all__ = [
    "EPS_INTENDED",
    "EPS_WRITE",
    "commit_scratch_bank",
    "commit_scratch_to_novel",
    "recompress_bank",
    "reset_scratch_bank",
]
