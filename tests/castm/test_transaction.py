"""Multi-layer commit / recompress / consolidate transaction tests (15.3-15.6, 9.5)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tfns.castm import address as addr
from tfns.castm import consolidate as cons
from tfns.castm import scratch as scr
from tfns.castm import synaptic as syn
from tfns.castm import transaction as tx


def _book(n, d_k=64, seed=0):
    book = addr.empty_address_book(d_k=d_k, n_max=max(n, 1), seed=seed)
    for _ in range(n):
        book, _ = addr.allocate_canonical(book)
    return book


def _bank(seed=0, n_slots=16, R=4):
    rng = np.random.default_rng(seed)
    layers = {
        "dense": (12, 10),
        "head": (18, 12),
    }
    banks = {}
    for name, (out_dim, in_dim) in layers.items():
        W0 = jnp.asarray(rng.standard_normal((out_dim, in_dim)).astype(np.float32))
        b0 = jnp.asarray(rng.standard_normal((out_dim,)).astype(np.float32))
        banks[name] = syn.empty_synaptic_memory(W0, b0, comp_rank=R, n_slots=n_slots, d_k=64)
    return banks


def _scratch_bank(banks, seed, R=4, scale=0.3):
    rng = np.random.default_rng(seed)
    out = {}
    for name, mem in banks.items():
        s = scr.init_scratch(mem.in_dim, mem.out_dim, R, jax.random.PRNGKey(seed))
        B = jnp.asarray(rng.standard_normal((mem.out_dim, R)).astype(np.float32) * scale)
        beta = jnp.asarray(rng.standard_normal((mem.out_dim,)).astype(np.float32) * scale)
        out[name] = s.replace(B_s=B, beta_s=beta)
    return out


def test_commit_bank_preserves_all_other_contexts():
    banks = _bank(seed=1)
    book = _book(3, seed=2)
    # Seed contexts 0 and 1.
    for ctx in (0, 1):
        sb = _scratch_bank(banks, seed=10 + ctx)
        banks, report = tx.commit_scratch_bank(banks, sb, book, ctx)
        assert report["accepted"], report

    snap = {name: {c: np.asarray(syn.decode_weight(banks[name], addr.code(book, c)))
                   for c in (0, 1)} for name in banks}

    # Commit to context 2.
    sb2 = _scratch_bank(banks, seed=99)
    banks2, report = tx.commit_scratch_bank(banks, sb2, book, 2)
    assert report["accepted"], report
    assert report["max_noninterference"] < tx.EPS_WRITE
    # Contexts 0 and 1 unchanged across every layer.
    for name in banks2:
        for c in (0, 1):
            after = np.asarray(syn.decode_weight(banks2[name], addr.code(book, c)))
            assert np.max(np.abs(after - snap[name][c])) < 1e-4


def test_commit_rolls_back_on_sentinel_rejection():
    banks = _bank(seed=3)
    book = _book(2, seed=4)
    sb = _scratch_bank(banks, seed=5)
    banks2, report = tx.commit_scratch_bank(banks, sb, book, 0, sentinel_fn=lambda b, a: False)
    assert not report["accepted"]
    assert report["reason"] == "sentinel_rejected"
    # Banks are unchanged (rollback): no active components.
    for name in banks2:
        assert int(np.sum(np.asarray(banks2[name].active))) == 0


def test_commit_to_novel_allocates_and_commits():
    banks = _bank(seed=6)
    book = _book(0, seed=7)  # nothing allocated yet
    sb = _scratch_bank(banks, seed=8)
    banks2, book2, ctx, report = tx.commit_scratch_to_novel(banks, sb, book)
    assert report["accepted"]
    assert ctx == 0
    assert addr.num_used(book2) == 1


def test_commit_triggers_recompression_when_pool_full():
    # Small pool but ample comp_rank: repeated commits to the same context fill
    # the pool, then recompress near-losslessly to fit (spec 23.9). comp_rank must
    # be >= the layer's representable rank (min(out,in)) so the accumulated delta
    # recompresses without loss; in the real config this is max_active_rank (64).
    banks = _bank(seed=11, n_slots=4, R=12)
    book = _book(1, seed=12)
    recompressed = False
    for i in range(8):
        sb = _scratch_bank(banks, seed=20 + i, R=2, scale=0.1)
        banks, report = tx.commit_scratch_bank(
            banks, sb, book, 0, energy=0.9999, max_elem_tol=1e-3
        )
        assert report["accepted"], report
        for name, lr in report["layers"].items():
            if lr.get("note") == "recompressed":
                recompressed = True
    assert recompressed
    # Pool never overflowed.
    for name in banks:
        assert int(np.sum(np.asarray(banks[name].active))) <= 4


def test_recompress_bank_preserves_decoded_weights():
    banks = _bank(seed=13, n_slots=32, R=6)
    book = _book(3, seed=14)
    # Append several redundant rank-1 components per context per layer.
    rng = np.random.default_rng(15)
    for ctx in range(3):
        for _ in range(4):
            sbank = {}
            for name, mem in banks.items():
                A = jnp.asarray(rng.standard_normal((1, mem.in_dim)).astype(np.float32) * 0.2)
                B = jnp.asarray(rng.standard_normal((mem.out_dim, 1)).astype(np.float32) * 0.2)
                beta = jnp.zeros((mem.out_dim,), jnp.float32)
                sbank[name] = scr.ScratchDelta(A_s=A, B_s=B, beta_s=beta)
            banks, rep = tx.commit_scratch_bank(banks, sbank, book, ctx, energy=0.999, max_elem_tol=1.0)
            assert rep["accepted"]
    before = {name: {c: np.asarray(syn.decode_weight(banks[name], addr.code(book, c)))
                     for c in range(3)} for name in banks}
    banks2, report = tx.recompress_bank(banks, book, 1, energy=0.999, max_elem_tol=1e-2, rel_tol=1e-2)
    assert report["accepted"], report
    for name in banks2:
        for c in range(3):
            after = np.asarray(syn.decode_weight(banks2[name], addr.code(book, c)))
            err = np.max(np.abs(after - before[name][c]))
            if c == 1:
                assert err <= 1e-2
            else:
                assert err < 1e-4


def test_consolidate_bank_preserves_contexts_changes_shared():
    banks = _bank(seed=16, n_slots=32, R=4)
    book = _book(4, seed=17)
    # Give every context a shared common component plus idiosyncratic noise.
    rng = np.random.default_rng(18)
    common = {name: (rng.standard_normal((banks[name].out_dim, 2)).astype(np.float32) * 0.3,
                     rng.standard_normal((2, banks[name].in_dim)).astype(np.float32) * 0.3)
              for name in banks}
    for ctx in range(4):
        sbank = {}
        for name, mem in banks.items():
            Bc, Ac = common[name]
            A = jnp.asarray(Ac + 0.02 * rng.standard_normal(Ac.shape).astype(np.float32))
            B = jnp.asarray(Bc + 0.02 * rng.standard_normal(Bc.shape).astype(np.float32))
            sbank[name] = scr.ScratchDelta(A_s=A, B_s=B, beta_s=jnp.zeros((mem.out_dim,), jnp.float32))
        banks, rep = tx.commit_scratch_bank(banks, sbank, book, ctx)
        assert rep["accepted"]

    before = {name: {c: np.asarray(syn.decode_weight(banks[name], addr.code(book, c)))
                     for c in range(4)} for name in banks}
    W0_before = {name: np.asarray(banks[name].W0) for name in banks}
    banks2, report = cons.consolidate_bank(banks, book, rank=2, eps_write=1e-4)
    assert report["accepted"], report
    for name in banks2:
        # Shared substrate changed...
        assert np.max(np.abs(np.asarray(banks2[name].W0) - W0_before[name])) > 1e-4
        # ...but every protected decoded operator is unchanged.
        for c in range(4):
            after = np.asarray(syn.decode_weight(banks2[name], addr.code(book, c)))
            assert np.max(np.abs(after - before[name][c])) < 1e-4
