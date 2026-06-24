"""Online compensated resolve: the central stability/plasticity invariant.

Covers required tests 1 (repeated shared drift), 2 (active plasticity / inactive
preserved), 3 (revisit/reconsolidation), 12 (rank handling), 16 (numerical health).
"""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tfns.castm import address as addr
from tfns.castm import online_resolve as orx
from tfns.castm import synaptic as syn


def _make_bank(out_dim, in_dim, *, comp_rank, n_slots, d_k, seed):
    rng = np.random.default_rng(seed)
    W0 = jnp.asarray(rng.standard_normal((out_dim, in_dim)).astype(np.float32) * 0.1)
    b0 = jnp.asarray(rng.standard_normal((out_dim,)).astype(np.float32) * 0.1)
    return syn.empty_synaptic_memory(W0, b0, comp_rank=comp_rank, n_slots=n_slots, d_k=d_k)


def _store_target(mem, book, ctx, target_delta, target_bias):
    """Store a context's correction as one component at its canonical address."""
    A, B = orx.svd_lowrank(target_delta, mem.comp_rank)
    mem, slot = syn.append_component(mem, A, B, jnp.asarray(target_bias), addr.code(book, ctx), int(ctx))
    assert slot >= 0
    return mem


def _decode_w(mem, book, ctx):
    return np.asarray(syn.decode_weight(mem, addr.code(book, int(ctx))), np.float64)


def _setup(n_ctx=32, out_dim=12, in_dim=16, d_k=64, seed=0):
    """A 2-layer bank (policy-like + value) with n_ctx conflicting stored targets.

    out_dim <= comp_rank so each rank-`comp_rank` correction is stored exactly,
    isolating the resolve invariant from truncation error.
    """
    comp_rank = max(out_dim, in_dim)  # full rank -> exact storage
    n_slots = n_ctx + 4
    book = addr.empty_address_book(d_k=d_k, n_max=n_ctx + 4, seed=seed)
    ctx_ids = []
    for _ in range(n_ctx):
        book, c = addr.allocate_canonical(book)
        ctx_ids.append(c)
    rng = np.random.default_rng(seed + 1)
    banks = {
        "enc_dense": _make_bank(out_dim, in_dim, comp_rank=comp_rank, n_slots=n_slots, d_k=d_k, seed=seed + 2),
        "value": _make_bank(1, in_dim, comp_rank=in_dim, n_slots=n_slots, d_k=d_k, seed=seed + 3),
    }
    targets = {}  # ctx -> {name: (delta, bias)}
    for c in ctx_ids:
        targets[c] = {}
        for name, mem in banks.items():
            od, idim = mem.out_dim, mem.in_dim
            delta = rng.standard_normal((od, idim)).astype(np.float32) * 0.3
            bias = rng.standard_normal((od,)).astype(np.float32) * 0.1
            targets[c][name] = (delta, bias)
            banks[name] = _store_target(banks[name], book, c, delta, bias)
    # Record the absolute target decoded weights (must stay constant for inactive ctx).
    target_w = {c: {name: _decode_w(banks[name], book, c) for name in banks} for c in ctx_ids}
    return banks, book, ctx_ids, targets, target_w


def _drift(banks, scale, rng):
    snap_W0, snap_b0 = orx.snapshot_shared(banks)
    new = {}
    for name, mem in banks.items():
        dW = (rng.standard_normal(mem.W0.shape) * scale).astype(np.float32)
        db = (rng.standard_normal(mem.b0.shape) * scale).astype(np.float32)
        new[name] = mem.replace(W0=jnp.asarray(np.asarray(mem.W0) + dW),
                                b0=jnp.asarray(np.asarray(mem.b0) + db))
    return new, snap_W0, snap_b0


def test_repeated_drift_preserves_inactive():
    """Required test 1: 32 conflicting contexts, repeated drift, inactive preserved."""
    banks, book, ctx_ids, _, target_w = _setup(n_ctx=32, seed=11)
    rng = np.random.default_rng(123)
    active = ctx_ids[0]
    for it in range(6):
        banks, snap_W0, snap_b0 = _drift(banks, 0.25, rng)
        banks, rep = orx.online_resolve(banks, book, snap_W0, snap_b0, active, ctx_ids,
                                        mem_rank=banks["enc_dense"].comp_rank)
        assert not any(syn.synaptic_has_nan(m) for m in banks.values())  # test 16
        # Every inactive context's decoded weights are unchanged from the target.
        for c in ctx_ids:
            if c == active:
                continue
            for name in banks:
                w = _decode_w(banks[name], book, c)
                err = float(np.max(np.abs(w - target_w[c][name])))
                assert err < 1e-4, f"it={it} ctx={c} layer={name} drift_err={err}"


def test_active_gets_full_update_inactive_frozen():
    """Required test 2: active context absorbs ΔW0; inactive decoded ops unchanged."""
    banks, book, ctx_ids, _, target_w = _setup(n_ctx=8, seed=7)
    active = ctx_ids[3]
    w_active_before = _decode_w(banks["enc_dense"], book, active)
    rng = np.random.default_rng(5)
    banks, snap_W0, snap_b0 = _drift(banks, 0.4, rng)
    dW = np.asarray(banks["enc_dense"].W0, np.float64) - np.asarray(snap_W0["enc_dense"], np.float64)
    banks, rep = orx.online_resolve(banks, book, snap_W0, snap_b0, active, ctx_ids,
                                    mem_rank=banks["enc_dense"].comp_rank)
    # Active absorbed the full shared update.
    w_active_after = _decode_w(banks["enc_dense"], book, active)
    assert np.max(np.abs(w_active_after - (w_active_before + dW))) < 1e-4
    # Inactive frozen.
    for c in ctx_ids:
        if c == active:
            continue
        w = _decode_w(banks["enc_dense"], book, c)
        assert np.max(np.abs(w - target_w[c]["enc_dense"])) < 1e-4


def test_revisit_reconsolidation_leaves_other_unchanged():
    """Required test 3: learn A, learn B, revisit A & drift, B stays put."""
    banks, book, ctx_ids, _, _ = _setup(n_ctx=2, seed=21)
    A, B = ctx_ids[0], ctx_ids[1]
    rng = np.random.default_rng(99)
    # Phase 1: A active.
    banks, sW, sb = _drift(banks, 0.2, rng)
    banks, _ = orx.online_resolve(banks, book, sW, sb, A, ctx_ids, mem_rank=banks["enc_dense"].comp_rank)
    # Phase 2: B active.
    banks, sW, sb = _drift(banks, 0.2, rng)
    banks, _ = orx.online_resolve(banks, book, sW, sb, B, ctx_ids, mem_rank=banks["enc_dense"].comp_rank)
    w_B_after_phase2 = {name: _decode_w(banks[name], book, B) for name in banks}
    # Phase 3: revisit A, drift again.
    banks, sW, sb = _drift(banks, 0.2, rng)
    banks, _ = orx.online_resolve(banks, book, sW, sb, A, ctx_ids, mem_rank=banks["enc_dense"].comp_rank)
    for name in banks:
        w_B_now = _decode_w(banks[name], book, B)
        assert np.max(np.abs(w_B_now - w_B_after_phase2[name])) < 1e-4, f"B drifted in {name}"


def test_value_inclusion_flag():
    """Required test (value protection, architecture §4): value compensated iff included."""
    banks, book, ctx_ids, _, target_w = _setup(n_ctx=4, seed=33)
    active = ctx_ids[0]
    rng = np.random.default_rng(1)
    banks_inc, sW, sb = _drift({k: v for k, v in banks.items()}, 0.3, rng)
    out_inc, rep_inc = orx.online_resolve(banks_inc, book, sW, sb, active, ctx_ids,
                                          mem_rank=banks["enc_dense"].comp_rank, include_value=True)
    out_exc, rep_exc = orx.online_resolve(banks_inc, book, sW, sb, active, ctx_ids,
                                          mem_rank=banks["enc_dense"].comp_rank, include_value=False)
    assert rep_inc["include_value"] and not rep_exc["include_value"]
    # Included: an inactive context's value op is preserved; excluded: it is not.
    c = ctx_ids[1]
    w_inc = _decode_w(out_inc["value"], book, c)
    w_exc = _decode_w(out_exc["value"], book, c)
    assert np.max(np.abs(w_inc - target_w[c]["value"])) < 1e-4
    assert np.max(np.abs(w_exc - target_w[c]["value"])) > 1e-3  # value freely tracks the stream


def test_rank_overflow_flags_budget_without_corruption():
    """Required test 12: insufficient rank is flagged, never silently corrupts."""
    # comp_rank smaller than out_dim => residual cannot be captured exactly.
    out_dim, in_dim, d_k = 24, 24, 64
    rank = 2
    book = addr.empty_address_book(d_k=d_k, n_max=8, seed=4)
    ctx_ids = []
    for _ in range(3):
        book, c = addr.allocate_canonical(book)
        ctx_ids.append(c)
    rng = np.random.default_rng(8)
    mem = _make_bank(out_dim, in_dim, comp_rank=rank, n_slots=8, d_k=d_k, seed=4)
    banks = {"enc_dense": mem}
    for c in ctx_ids:
        delta = rng.standard_normal((out_dim, in_dim)).astype(np.float32) * 0.5
        banks["enc_dense"] = _store_target(banks["enc_dense"], book, c, delta, np.zeros((out_dim,), np.float32))
    banks, sW, sb = _drift(banks, 0.5, rng)
    out, rep = orx.online_resolve(banks, book, sW, sb, ctx_ids[0], ctx_ids,
                                  mem_rank=rank, max_elem_tol=1e-3)
    assert not rep["budget_ok"]            # honest: error is reported, not hidden
    assert len(rep["violations"]) >= 1
    assert not syn.synaptic_has_nan(out["enc_dense"])  # no corruption


def test_bootstrap_no_active_protects_nothing():
    """active=-1 (bootstrap interval) is well-defined and corrupts nothing."""
    banks, book, ctx_ids, _, _ = _setup(n_ctx=3, seed=2)
    rng = np.random.default_rng(0)
    banks, sW, sb = _drift(banks, 0.1, rng)
    out, rep = orx.online_resolve(banks, book, sW, sb, -1, ctx_ids,
                                  mem_rank=banks["enc_dense"].comp_rank)
    assert rep["n_inactive"] == len(ctx_ids)
    assert not any(syn.synaptic_has_nan(m) for m in out.values())
