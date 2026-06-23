"""Contextualized layer tests (spec 6.2, 6.3, 6.4, 17.4).

For each contextualized layer type: factorized forward equals explicit
materialized-weight forward; old-address invariance after a write; intended
current-address update; gradients flow through scratch factors.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tfns.castm import address as addr
from tfns.castm import audit
from tfns.castm import layers
from tfns.castm import scratch as scr
from tfns.castm import synaptic as syn


def _book_with(n, d_k=64, seed=0):
    book = addr.empty_address_book(d_k=d_k, n_max=max(n, 1), seed=seed)
    for _ in range(n):
        book, _ = addr.allocate_canonical(book)
    return book


# --- Convolution ---------------------------------------------------------------


def test_conv_factorized_equals_materialized():
    rng = np.random.default_rng(0)
    kh, kw, c_in, c_out = 3, 3, 4, 6
    in_dim = kh * kw * c_in
    R = 4
    W0 = jnp.asarray(rng.standard_normal((c_out, in_dim)).astype(np.float32) * 0.1)
    b0 = jnp.asarray(rng.standard_normal((c_out,)).astype(np.float32) * 0.1)
    mem = syn.empty_synaptic_memory(W0, b0, comp_rank=R, n_slots=16, d_k=64)
    book = _book_with(3, d_k=64, seed=1)
    for i in range(3):
        A = jnp.asarray(rng.standard_normal((R, in_dim)).astype(np.float32) * 0.1)
        B = jnp.asarray(rng.standard_normal((c_out, R)).astype(np.float32) * 0.1)
        beta = jnp.asarray(rng.standard_normal((c_out,)).astype(np.float32) * 0.1)
        mem, _ = syn.append_component(mem, A, B, beta, addr.code(book, i), i)

    x = jnp.asarray(rng.standard_normal((2, 8, 8, c_in)).astype(np.float32))
    for i in range(3):
        k = addr.code(book, i)
        y_fac = layers.addressed_conv_forward(
            mem, x, k, kh=kh, kw=kw, c_in=c_in, strides=(1, 1), padding="VALID"
        )
        kernel = layers.materialize_conv_kernel(mem, k, kh, kw, c_in)
        y_mat = layers._conv(x, kernel, (1, 1), "VALID") + syn.decode_bias(mem, k)
        np.testing.assert_allclose(np.asarray(y_fac), np.asarray(y_mat), atol=1e-4, rtol=1e-4)


def test_conv_write_noninterference_and_intended():
    rng = np.random.default_rng(2)
    kh, kw, c_in, c_out = 3, 3, 3, 5
    in_dim = kh * kw * c_in
    R = 3
    W0 = jnp.asarray(rng.standard_normal((c_out, in_dim)).astype(np.float32))
    b0 = jnp.asarray(rng.standard_normal((c_out,)).astype(np.float32))
    mem = syn.empty_synaptic_memory(W0, b0, comp_rank=R, n_slots=16, d_k=64)
    book = _book_with(4, d_k=64, seed=3)
    for i in range(4):
        A = jnp.asarray(rng.standard_normal((R, in_dim)).astype(np.float32) * 0.3)
        B = jnp.asarray(rng.standard_normal((c_out, R)).astype(np.float32) * 0.3)
        beta = jnp.asarray(rng.standard_normal((c_out,)).astype(np.float32) * 0.3)
        delta = B @ A
        before = mem
        mem, _ = syn.append_component(mem, A, B, beta, addr.code(book, i), i)
        assert audit.max_noninterference_error(before, mem, book, i) < 1e-6
        assert audit.intended_write_error(before, mem, book, i, delta, bias_expected=beta) < 1e-5


def test_conv_scratch_gradients_and_equivalence():
    rng = np.random.default_rng(4)
    kh, kw, c_in, c_out = 3, 3, 2, 4
    in_dim = kh * kw * c_in
    W0 = jnp.zeros((c_out, in_dim), jnp.float32)
    b0 = jnp.zeros((c_out,), jnp.float32)
    mem = syn.empty_synaptic_memory(W0, b0, comp_rank=4, n_slots=8, d_k=32)
    book = _book_with(1, d_k=32, seed=5)
    k = addr.code(book, 0)
    scratch = scr.init_scratch(in_dim, c_out, 3, jax.random.PRNGKey(0))
    scratch = scratch.replace(B_s=scratch.B_s + 0.2)
    x = jnp.asarray(rng.standard_normal((2, 6, 6, c_in)).astype(np.float32))

    y = layers.addressed_conv_forward(
        mem, x, k, kh=kh, kw=kw, c_in=c_in, strides=(1, 1), padding="VALID", scratch=scratch
    )
    # Equivalence: conv with (W0 + scratch delta) materialized kernel.
    W_eff = mem.W0 + scr.scratch_delta_weight(scratch)
    kernel = layers.matrix_to_kernel(W_eff, kh, kw, c_in)
    y_mat = layers._conv(x, kernel, (1, 1), "VALID") + scratch.beta_s
    np.testing.assert_allclose(np.asarray(y), np.asarray(y_mat), atol=1e-4, rtol=1e-4)

    target = jnp.asarray(rng.standard_normal(y.shape).astype(np.float32))

    def loss(s):
        out = layers.addressed_conv_forward(
            mem, x, k, kh=kh, kw=kw, c_in=c_in, strides=(1, 1), padding="VALID", scratch=s
        )
        return jnp.sum((out - target) ** 2)

    g = jax.grad(loss)(scratch)
    assert float(jnp.linalg.norm(g.A_s)) > 0.0
    assert float(jnp.linalg.norm(g.B_s)) > 0.0


# --- GRU -----------------------------------------------------------------------


def _make_gru(in_dim, hidden, R, n_slots, d_k, seed):
    rng = np.random.default_rng(seed)
    gate_in = in_dim + hidden

    def bank():
        W0 = jnp.asarray(rng.standard_normal((hidden, gate_in)).astype(np.float32) * 0.1)
        b0 = jnp.asarray(rng.standard_normal((hidden,)).astype(np.float32) * 0.1)
        return syn.empty_synaptic_memory(W0, b0, comp_rank=R, n_slots=n_slots, d_k=d_k)

    return layers.GRUMemory(z=bank(), r=bank(), n=bank())


def test_gru_reset_zeroes_hidden():
    in_dim, hidden = 5, 7
    gru = _make_gru(in_dim, hidden, R=4, n_slots=8, d_k=32, seed=10)
    book = _book_with(1, d_k=32, seed=11)
    k = addr.code(book, 0)
    rng = np.random.default_rng(12)
    x = jnp.asarray(rng.standard_normal((3, in_dim)).astype(np.float32))
    h_prev = jnp.asarray(rng.standard_normal((3, hidden)).astype(np.float32))
    reset = jnp.asarray([True, False, True])
    h_reset = layers.addressed_gru_step(gru, x, h_prev, k, reset)
    h_zero = layers.addressed_gru_step(gru, x, jnp.zeros_like(h_prev), k, jnp.zeros((3,), bool))
    # Reset rows must equal the all-zero-hidden step.
    np.testing.assert_allclose(np.asarray(h_reset[0]), np.asarray(h_zero[0]), atol=1e-5)
    np.testing.assert_allclose(np.asarray(h_reset[2]), np.asarray(h_zero[2]), atol=1e-5)


def test_gru_write_noninterference_all_gates():
    in_dim, hidden, R = 4, 6, 3
    gru = _make_gru(in_dim, hidden, R=R, n_slots=16, d_k=64, seed=20)
    book = _book_with(4, d_k=64, seed=21)
    rng = np.random.default_rng(22)
    gate_in = in_dim + hidden
    for i in range(4):
        for gate in ("z", "r", "n"):
            mem = getattr(gru, gate)
            A = jnp.asarray(rng.standard_normal((R, gate_in)).astype(np.float32) * 0.3)
            B = jnp.asarray(rng.standard_normal((hidden, R)).astype(np.float32) * 0.3)
            beta = jnp.asarray(rng.standard_normal((hidden,)).astype(np.float32) * 0.3)
            before = mem
            mem, _ = syn.append_component(mem, A, B, beta, addr.code(book, i), i)
            assert audit.max_noninterference_error(before, mem, book, i) < 1e-6
            gru = gru.replace(**{gate: mem})


def test_gru_scratch_gradients():
    in_dim, hidden, R = 4, 6, 3
    gru = _make_gru(in_dim, hidden, R=R, n_slots=8, d_k=32, seed=30)
    book = _book_with(1, d_k=32, seed=31)
    k = addr.code(book, 0)
    gate_in = in_dim + hidden
    keys = jax.random.split(jax.random.PRNGKey(0), 3)
    gscr = layers.GRUScratch(
        z=scr.init_scratch(gate_in, hidden, R, keys[0]),
        r=scr.init_scratch(gate_in, hidden, R, keys[1]),
        n=scr.init_scratch(gate_in, hidden, R, keys[2]),
    )
    rng = np.random.default_rng(33)
    x = jnp.asarray(rng.standard_normal((3, in_dim)).astype(np.float32))
    h_prev = jnp.asarray(rng.standard_normal((3, hidden)).astype(np.float32))
    target = jnp.asarray(rng.standard_normal((3, hidden)).astype(np.float32))

    def loss(s):
        h = layers.addressed_gru_step(gru, x, h_prev, k, jnp.zeros((3,), bool), scratch=s)
        return jnp.sum((h - target) ** 2)

    g = jax.grad(loss)(gscr)
    # All three gate scratch banks receive gradient.
    for gate in ("z", "r", "n"):
        gg = getattr(g, gate)
        assert float(jnp.linalg.norm(gg.A_s)) + float(jnp.linalg.norm(gg.B_s)) > 0.0


# --- Heads (dense) -------------------------------------------------------------


def test_policy_value_heads_addressed():
    rng = np.random.default_rng(40)
    hidden = 8
    # Policy head: hidden -> 18; value head: hidden -> 1.
    pol = syn.empty_synaptic_memory(
        jnp.asarray(rng.standard_normal((18, hidden)).astype(np.float32) * 0.1),
        jnp.zeros((18,), jnp.float32),
        comp_rank=4, n_slots=8, d_k=32,
    )
    val = syn.empty_synaptic_memory(
        jnp.asarray(rng.standard_normal((1, hidden)).astype(np.float32) * 0.1),
        jnp.zeros((1,), jnp.float32),
        comp_rank=4, n_slots=8, d_k=32,
    )
    book = _book_with(2, d_k=32, seed=41)
    for i in range(2):
        for mem_name, mem, out in (("pol", pol, 18), ("val", val, 1)):
            A = jnp.asarray(rng.standard_normal((2, hidden)).astype(np.float32) * 0.2)
            B = jnp.asarray(rng.standard_normal((out, 2)).astype(np.float32) * 0.2)
            beta = jnp.asarray(rng.standard_normal((out,)).astype(np.float32) * 0.2)
            before = mem
            mem2, _ = syn.append_component(mem, A, B, beta, addr.code(book, i), i)
            assert audit.max_noninterference_error(before, mem2, book, i) < 1e-6
            if mem_name == "pol":
                pol = mem2
            else:
                val = mem2

    h = jnp.asarray(rng.standard_normal((5, hidden)).astype(np.float32))
    for i in range(2):
        k = addr.code(book, i)
        logits = layers.addressed_dense_forward(pol, h, k)
        value = layers.addressed_dense_forward(val, h, k)
        assert logits.shape == (5, 18)
        assert value.shape == (5, 1)
        # Factorized == materialized.
        np.testing.assert_allclose(
            np.asarray(logits),
            np.asarray(h @ syn.decode_weight(pol, k).T + syn.decode_bias(pol, k)),
            atol=1e-4, rtol=1e-4,
        )
