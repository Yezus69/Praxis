"""Factorized synaptic memory tests (spec 5.3, 5.5, 6, 8.2, 15.5, 16.4-16.5, 17.1)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tfns.castm import address as addr
from tfns.castm import audit
from tfns.castm import synaptic as syn
from tfns.castm import scratch as scr


def _book_with(n: int, d_k: int = 128, seed: int = 0) -> addr.AddressBook:
    book = addr.empty_address_book(d_k=d_k, n_max=max(n, 1), seed=seed)
    for _ in range(n):
        book, _ = addr.allocate_canonical(book)
    return book


def _rand_lowrank(rng, out_dim, in_dim, rank, scale=0.3):
    B = (scale * rng.standard_normal((out_dim, rank))).astype(np.float32)
    A = (scale * rng.standard_normal((rank, in_dim))).astype(np.float32)
    beta = (scale * rng.standard_normal((out_dim,))).astype(np.float32)
    return jnp.asarray(A), jnp.asarray(B), jnp.asarray(beta)


def test_decode_matches_forward_dense():
    rng = np.random.default_rng(0)
    out_dim, in_dim, R = 7, 5, 4
    W0 = jnp.asarray(rng.standard_normal((out_dim, in_dim)).astype(np.float32))
    b0 = jnp.asarray(rng.standard_normal((out_dim,)).astype(np.float32))
    mem = syn.empty_synaptic_memory(W0, b0, comp_rank=R, n_slots=8, d_k=16)
    book = _book_with(3, d_k=16, seed=1)
    for i in range(3):
        A, B, beta = _rand_lowrank(rng, out_dim, in_dim, R)
        mem, slot = syn.append_component(mem, A, B, beta, addr.code(book, i), i)
        assert slot >= 0
    x = jnp.asarray(rng.standard_normal((11, in_dim)).astype(np.float32))
    for i in range(3):
        k = addr.code(book, i)
        # Factorized forward must equal explicit effective-weight forward.
        y_fac = syn.forward(mem, x, k)
        W = syn.decode_weight(mem, k)
        b = syn.decode_bias(mem, k)
        y_exp = x @ W.T + b
        np.testing.assert_allclose(np.asarray(y_fac), np.asarray(y_exp), atol=1e-4, rtol=1e-4)


def test_exact_context_specific_write_dense():
    # Spec 5.3 / 16.4 / 16.5: write to context i; other contexts unchanged.
    rng = np.random.default_rng(2)
    out_dim, in_dim, R = 6, 4, 3
    W0 = jnp.asarray(rng.standard_normal((out_dim, in_dim)).astype(np.float32))
    b0 = jnp.asarray(rng.standard_normal((out_dim,)).astype(np.float32))
    mem = syn.empty_synaptic_memory(W0, b0, comp_rank=R, n_slots=16, d_k=64)
    book = _book_with(5, d_k=64, seed=3)

    for i in range(5):
        A, B, beta = _rand_lowrank(rng, out_dim, in_dim, R)
        delta_expected = B @ A
        mem_before = mem
        mem, slot = syn.append_component(mem, A, B, beta, addr.code(book, i), i)
        assert slot >= 0
        ni = audit.max_noninterference_error(mem_before, mem, book, i)
        iw = audit.intended_write_error(mem_before, mem, book, i, delta_expected, bias_expected=beta)
        assert ni < 1e-6, f"noninterference {ni}"
        assert iw < 1e-5, f"intended write {iw}"


@pytest.mark.parametrize("n_ctx", [32])
def test_32_conflicting_contexts_exact_noninterference(n_ctx):
    # Spec 21.1 mathematical memory gate: >=32 conflicting contexts, exact
    # noninterference. Identical input distribution, mutually incompatible deltas.
    rng = np.random.default_rng(42)
    out_dim, in_dim, R = 16, 12, 4
    W0 = jnp.asarray(rng.standard_normal((out_dim, in_dim)).astype(np.float32))
    b0 = jnp.asarray(rng.standard_normal((out_dim,)).astype(np.float32))
    mem = syn.empty_synaptic_memory(W0, b0, comp_rank=R, n_slots=n_ctx + 4, d_k=128)
    book = _book_with(n_ctx, d_k=128, seed=57)

    expected = {}
    for i in range(n_ctx):
        A, B, beta = _rand_lowrank(rng, out_dim, in_dim, R, scale=0.5)
        delta = B @ A
        mem_before = mem
        mem, slot = syn.append_component(mem, A, B, beta, addr.code(book, i), i)
        assert slot >= 0
        # Noninterference against ALL previously written contexts.
        ni = audit.max_noninterference_error(mem_before, mem, book, i)
        assert ni < 1e-6, f"context {i}: noninterference {ni}"
        expected[i] = np.asarray(delta + np.asarray(W0))

    # After all writes, every protected context decodes to its own target.
    for i in range(n_ctx):
        W = np.asarray(syn.decode_weight(mem, addr.code(book, i)))
        err = np.max(np.abs(W - expected[i]))
        assert err < 1e-4, f"context {i}: decoded drift {err}"


def test_reconsolidation_changes_only_recalled_context():
    # Spec 5.5 / 17.1: alternate updates among contexts; reconsolidation of one
    # context must not change the others.
    rng = np.random.default_rng(9)
    out_dim, in_dim, R = 8, 6, 3
    W0 = jnp.asarray(np.zeros((out_dim, in_dim), np.float32))
    b0 = jnp.asarray(np.zeros((out_dim,), np.float32))
    mem = syn.empty_synaptic_memory(W0, b0, comp_rank=R, n_slots=64, d_k=64)
    book = _book_with(4, d_k=64, seed=19)

    # Seed each context once.
    for i in range(4):
        A, B, beta = _rand_lowrank(rng, out_dim, in_dim, R)
        mem, _ = syn.append_component(mem, A, B, beta, addr.code(book, i), i)

    # Reconsolidate context 1 many times; contexts 0,2,3 must stay fixed.
    fixed = {j: np.asarray(syn.decode_weight(mem, addr.code(book, j))) for j in (0, 2, 3)}
    for _ in range(20):
        A, B, beta = _rand_lowrank(rng, out_dim, in_dim, R, scale=0.1)
        mem_before = mem
        mem, slot = syn.append_component(mem, A, B, beta, addr.code(book, 1), 1)
        assert slot >= 0
        assert audit.max_noninterference_error(mem_before, mem, book, 1) < 1e-6
    for j in (0, 2, 3):
        W = np.asarray(syn.decode_weight(mem, addr.code(book, j)))
        assert np.max(np.abs(W - fixed[j])) < 1e-5


def test_recompression_preserves_decoded_weights():
    # Spec 15.5 / 17.5: append redundant components, recompress, verify decoded
    # weights for all protected contexts stay within tolerance.
    rng = np.random.default_rng(21)
    out_dim, in_dim, R = 10, 8, 6
    W0 = jnp.asarray(rng.standard_normal((out_dim, in_dim)).astype(np.float32))
    b0 = jnp.asarray(rng.standard_normal((out_dim,)).astype(np.float32))
    mem = syn.empty_synaptic_memory(W0, b0, comp_rank=R, n_slots=64, d_k=64)
    book = _book_with(3, d_k=64, seed=23)

    # Each context gets several redundant rank-1 components; the combined delta
    # per context is low rank (<= 4 <= R) so recompression is near-lossless.
    for i in range(3):
        for _ in range(4):
            A, B, beta = _rand_lowrank(rng, out_dim, in_dim, 1, scale=0.2)
            mem, _ = syn.append_component(mem, A, B, beta, addr.code(book, i), i)

    before = {i: np.asarray(syn.decode_weight(mem, addr.code(book, i))) for i in range(3)}
    n_before = int(np.sum(np.asarray(mem.active)))
    mem2, report = syn.recompress_context(mem, 1, energy=0.999, max_rank=R, max_elem_tol=1e-2)
    assert report["accepted"]
    assert report["n_after"] == 1
    # Context 1 reconstructed within tolerance; 0 and 2 exactly unchanged.
    for i in range(3):
        after = np.asarray(syn.decode_weight(mem2, addr.code(book, i)))
        rec = audit.reconstruction_error(mem, mem2, book, i)
        if i == 1:
            assert rec["max_abs"] <= 1e-2
        else:
            assert rec["max_abs"] < 1e-5
    # Pool shrank (4 components -> 1 for ctx 1).
    assert int(np.sum(np.asarray(mem2.active))) == n_before - 3


def test_shared_consolidation_preserves_all_contexts():
    # Spec 8.2 / 17.6: move a common component into W0 with exact compensation;
    # every protected decoded operator unchanged though W0 and memory both change.
    rng = np.random.default_rng(31)
    out_dim, in_dim, R = 9, 7, 4
    W0 = jnp.asarray(rng.standard_normal((out_dim, in_dim)).astype(np.float32))
    b0 = jnp.asarray(rng.standard_normal((out_dim,)).astype(np.float32))
    mem = syn.empty_synaptic_memory(W0, b0, comp_rank=R, n_slots=64, d_k=64)
    book = _book_with(5, d_k=64, seed=37)
    for i in range(5):
        A, B, beta = _rand_lowrank(rng, out_dim, in_dim, R)
        mem, _ = syn.append_component(mem, A, B, beta, addr.code(book, i), i)

    before = {i: np.asarray(syn.decode_weight(mem, addr.code(book, i))) for i in range(5)}
    g = addr.compensation_vector(book, orthonormal=True)
    A_S, B_S, beta_S = _rand_lowrank(rng, out_dim, in_dim, 3, scale=0.4)
    W0_before = np.asarray(mem.W0)
    mem2, slot = syn.shared_consolidate(mem, A_S, B_S, g, beta_S=beta_S)
    assert slot >= 0
    # Shared substrate changed...
    assert np.max(np.abs(np.asarray(mem2.W0) - W0_before)) > 1e-3
    # ...but every protected decoded operator is unchanged.
    for i in range(5):
        after = np.asarray(syn.decode_weight(mem2, addr.code(book, i)))
        assert np.max(np.abs(after - before[i])) < 1e-5


def test_sparse_execution_ignores_unselected_components():
    # Spec 11.2 / 17.9: changing an unselected component must not change output.
    rng = np.random.default_rng(41)
    out_dim, in_dim, R = 6, 5, 3
    W0 = jnp.asarray(rng.standard_normal((out_dim, in_dim)).astype(np.float32))
    b0 = jnp.asarray(rng.standard_normal((out_dim,)).astype(np.float32))
    mem = syn.empty_synaptic_memory(W0, b0, comp_rank=R, n_slots=16, d_k=64)
    book = _book_with(4, d_k=64, seed=43)
    for i in range(4):
        A, B, beta = _rand_lowrank(rng, out_dim, in_dim, R)
        mem, _ = syn.append_component(mem, A, B, beta, addr.code(book, i), i)

    x = jnp.asarray(rng.standard_normal((5, in_dim)).astype(np.float32))
    k0 = addr.code(book, 0)
    y0 = syn.forward_sparse(mem, x, k0, ctx_id=0)
    # Corrupt context 2's component arbitrarily.
    slot2 = int(syn.context_components(mem, 2)[0])
    mem_corrupt = mem.replace(B=mem.B.at[slot2].set(mem.B[slot2] + 100.0))
    y0_corrupt = syn.forward_sparse(mem_corrupt, x, k0, ctx_id=0)
    # Output at context 0 is bit-identical (exact sparse gather).
    np.testing.assert_array_equal(np.asarray(y0), np.asarray(y0_corrupt))


def test_scratch_commit_roundtrip_and_gradients():
    # Spec 7: LoRA scratch (B=0 -> zero initial delta); gradients flow; commit.
    out_dim, in_dim, R = 6, 4, 3
    key = jax.random.PRNGKey(0)
    scratch = scr.init_scratch(in_dim, out_dim, R, key)
    assert scr.scratch_is_effectively_zero(scratch)

    x = jnp.asarray(np.random.default_rng(1).standard_normal((4, in_dim)).astype(np.float32))
    target = jnp.asarray(np.random.default_rng(2).standard_normal((4, out_dim)).astype(np.float32))

    def loss(s):
        return jnp.sum((scr.scratch_forward(s, x) - target) ** 2)

    grads = jax.grad(loss)(scratch)
    # With a nonzero target the LoRA gradient path through B_s is usable even
    # though B_s started at zero (spec 7: zero delta, usable gradient path).
    assert float(jnp.linalg.norm(grads.B_s)) > 0.0

    # Make a nonzero scratch, commit to memory, verify exact placement.
    scratch = scratch.replace(B_s=scratch.B_s + 0.5, beta_s=scratch.beta_s + 0.1)
    rng = np.random.default_rng(7)
    W0 = jnp.asarray(rng.standard_normal((out_dim, in_dim)).astype(np.float32))
    b0 = jnp.asarray(rng.standard_normal((out_dim,)).astype(np.float32))
    mem = syn.empty_synaptic_memory(W0, b0, comp_rank=R, n_slots=8, d_k=32)
    book = _book_with(2, d_k=32, seed=5)
    mem_before = mem
    mem, slot = scr.commit_scratch_to_memory(mem, scratch, addr.code(book, 1), 1)
    assert slot >= 0
    delta = scr.scratch_delta_weight(scratch)
    iw = audit.intended_write_error(mem_before, mem, book, 1, delta, bias_expected=scratch.beta_s)
    assert iw < 1e-5
    assert audit.max_noninterference_error(mem_before, mem, book, 1) < 1e-6
