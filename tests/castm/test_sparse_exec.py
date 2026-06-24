"""True (computational) sparse execution — architecture section 5, required test 11.

Verifies: (a) the blocked gather forward equals the functional forward for the
selected context; (b) editing an unselected context's factors leaves the selected
output bit-identical; (c) the contracted work is constant in the number of stored
contexts (gathered shapes are context-count independent).
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tfns.castm import sparse_exec as sx
from tfns.castm import synaptic as syn


def _build(n_ctx, out=24, in_=32, rank=8, spc=2, d_k=48, seed=0):
    rng = np.random.default_rng(seed)
    W0 = jnp.asarray(rng.standard_normal((out, in_)).astype(np.float32) * 0.05)
    b0 = jnp.asarray(rng.standard_normal((out,)).astype(np.float32) * 0.05)
    mem = sx.blocked_memory(W0, b0, n_contexts=n_ctx, slots_per_ctx=spc, comp_rank=rank, d_k=d_k)
    cvecs = {}
    for c in range(n_ctx):
        A = rng.standard_normal((rank, in_)).astype(np.float32) * 0.1
        B = rng.standard_normal((out, rank)).astype(np.float32) * 0.1
        beta = rng.standard_normal((out,)).astype(np.float32) * 0.05
        cvec = rng.standard_normal((d_k,)).astype(np.float32)
        cvec /= np.linalg.norm(cvec)
        cvecs[c] = jnp.asarray(cvec)
        mem = sx.place_blocked(mem, A, B, beta, cvec, c, spc)
    x = jnp.asarray(rng.standard_normal((16, in_)).astype(np.float32))
    return mem, cvecs, x, spc


def test_blocked_matches_functional_for_selected_ctx():
    mem, cvecs, x, spc = _build(5, seed=1)
    for c in range(5):
        k = cvecs[c]  # decode at this context's address
        out_blocked = sx.forward_sparse_blocked(mem, x, k, c, spc)
        out_func = syn.forward_sparse(mem, x, k, c, include_shared=False)
        assert np.allclose(np.asarray(out_blocked), np.asarray(out_func), atol=1e-5), c


def test_unselected_edit_is_bit_identical():
    """Required test 11: changing an unselected block cannot change the output."""
    mem, cvecs, x, spc = _build(6, seed=2)
    sel = 2
    k = cvecs[sel]
    out0 = np.asarray(sx.forward_sparse_blocked(mem, x, k, sel, spc))
    # Arbitrarily corrupt every OTHER context's block.
    rng = np.random.default_rng(99)
    A2 = mem.A
    B2 = mem.B
    for c in range(6):
        if c == sel:
            continue
        for off in range(spc):
            slot = c * spc + off
            A2 = A2.at[slot].set(jnp.asarray(rng.standard_normal(A2.shape[1:]).astype(np.float32)))
            B2 = B2.at[slot].set(jnp.asarray(rng.standard_normal(B2.shape[1:]).astype(np.float32)))
    mem2 = mem.replace(A=A2, B=B2)
    out1 = np.asarray(sx.forward_sparse_blocked(mem2, x, k, sel, spc))
    assert np.array_equal(out0, out1), "selected output changed when an unselected block was edited"


def test_constant_work_in_context_count():
    """Gathered tensors have context-count-independent shape -> constant FLOPs."""
    shapes = []
    for n in (1, 5, 20, 57):
        mem, cvecs, x, spc = _build(n, seed=3)
        shapes.append(sx.gathered_shapes(mem, 0, spc))
    assert all(s == shapes[0] for s in shapes), f"gathered shapes vary with context count: {shapes}"
    # And the slice depends only on spc, not n.
    assert shapes[0]["A"][0] == 2 and shapes[0]["B"][0] == 2


def test_jit_blocked_forward_runs():
    mem, cvecs, x, spc = _build(8, seed=4)
    fn = jax.jit(sx.forward_sparse_blocked, static_argnames=("ctx_id", "slots_per_ctx"))
    out = np.asarray(fn(mem, x, cvecs[3], 3, spc))
    assert out.shape == (16, mem.out_dim) and np.all(np.isfinite(out))
