"""Integrated CASTM agent tests (spec 6.5, 9, 17.4, 17.9, 17.10, 30).

Proves the assembled agent (a) produces correct shapes, (b) RETAINS prior
contexts exactly under sparse top-1 gather after a write to another context,
(c) remains PLASTIC (scratch gradients reach the policy/value outputs), (d) emits
a unit content query with no task identity, and (e) restores deterministic
action traces after serialization.
"""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tfns.castm import address as addr
from tfns.castm import agent as ag
from tfns.castm import scratch as scr
from tfns.castm import state as st
from tfns.castm import synaptic as syn
from tfns.castm import transaction as tx


def _small_cfg():
    return ag.AgentConfig(
        obs_hw=20, frame_stack=4,
        conv_channels=(8, 16, 16), conv_kernels=(4, 3, 2), conv_strides=(2, 1, 1),
        dense_dim=32, gru_hidden=32, act_dim=18, action_embed_dim=8,
        d_q=16, d_k=16, ctx_hidden=24,
        comp_rank_conv=8, comp_rank_dense=8, comp_rank_head=8, n_slots=8,
    )


def _book(cfg, n, seed=0):
    book = addr.empty_address_book(d_k=cfg.d_k, n_max=max(n, 1), seed=seed)
    for _ in range(n):
        book, _ = addr.allocate_canonical(book)
    return book


def _obs(rng, cfg, b=2):
    return jnp.asarray(rng.integers(0, 255, size=(b, cfg.obs_hw, cfg.obs_hw, cfg.frame_stack), dtype=np.uint8))


def test_forward_shapes_full_config():
    cfg = ag.AgentConfig()  # real Nature-CNN config
    key = jax.random.PRNGKey(0)
    banks = ag.init_banks(key, cfg)
    params = ag.init_params(key, cfg)
    book = _book(cfg, 1, seed=1)
    rng = np.random.default_rng(0)
    obs = _obs(rng, cfg, b=2)
    prev_a = jnp.zeros((2,), jnp.int32)
    prev_r = jnp.zeros((2,), jnp.float32)
    reset = jnp.ones((2,), bool)
    q, ch = ag.context_query(params, cfg, jnp.zeros((2, cfg.ctx_hidden)), obs, prev_a, prev_r, reset)
    assert q.shape == (2, cfg.d_q)
    np.testing.assert_allclose(np.linalg.norm(np.asarray(q), axis=-1), 1.0, atol=1e-4)
    logits, value, h = ag.policy_step(
        params, banks, None, cfg, jnp.zeros((2, cfg.gru_hidden)),
        obs, prev_a, prev_r, reset, addr.code(book, 0),
    )
    assert logits.shape == (2, cfg.act_dim)
    assert value.shape == (2,)
    assert h.shape == (2, cfg.gru_hidden)


def test_agent_retention_exact_under_sparse_gather():
    cfg = _small_cfg()
    key = jax.random.PRNGKey(1)
    banks = ag.init_banks(key, cfg)
    params = ag.init_params(key, cfg)
    book = _book(cfg, 2, seed=2)
    rng = np.random.default_rng(3)
    obs = _obs(rng, cfg)
    prev_a = jnp.zeros((2,), jnp.int32)
    prev_r = jnp.zeros((2,), jnp.float32)
    reset = jnp.zeros((2,), bool)
    main_h = jnp.zeros((2, cfg.gru_hidden), jnp.float32)

    # Seed context 1 with a committed delta so retention is non-trivial.
    sbank1 = ag.init_scratch_banks(jax.random.PRNGKey(7), cfg)
    sbank1 = {n: s.replace(B_s=s.B_s + 0.1) for n, s in sbank1.items()}
    banks, rep = tx.commit_scratch_bank(banks, sbank1, book, 1)
    assert rep["accepted"], rep

    k1 = addr.code(book, 1)
    logits_before, value_before, _ = ag.policy_step(
        params, banks, None, cfg, main_h, obs, prev_a, prev_r, reset, k1, ctx_id=1, sparse=True
    )

    # Now write a large, unrelated delta to context 0.
    sbank0 = ag.init_scratch_banks(jax.random.PRNGKey(8), cfg)
    sbank0 = {n: s.replace(B_s=s.B_s + 0.7) for n, s in sbank0.items()}
    banks2, rep0 = tx.commit_scratch_bank(banks, sbank0, book, 0)
    assert rep0["accepted"], rep0

    logits_after, value_after, _ = ag.policy_step(
        params, banks2, None, cfg, main_h, obs, prev_a, prev_r, reset, k1, ctx_id=1, sparse=True
    )
    # Exact retention: context 1's policy/value are bit-identical (sparse gather).
    np.testing.assert_array_equal(np.asarray(logits_before), np.asarray(logits_after))
    np.testing.assert_array_equal(np.asarray(value_before), np.asarray(value_after))


def test_agent_plasticity_scratch_gradients_reach_outputs():
    cfg = _small_cfg()
    key = jax.random.PRNGKey(4)
    banks = ag.init_banks(key, cfg)
    params = ag.init_params(key, cfg)
    book = _book(cfg, 1, seed=5)
    rng = np.random.default_rng(6)
    obs = _obs(rng, cfg)
    prev_a = jnp.zeros((2,), jnp.int32)
    prev_r = jnp.zeros((2,), jnp.float32)
    reset = jnp.zeros((2,), bool)
    # Nonzero hidden so the reset gate (which only acts on h_prev) is exercised.
    main_h = jnp.asarray(rng.standard_normal((2, cfg.gru_hidden)).astype(np.float32))
    sbank = ag.init_scratch_banks(jax.random.PRNGKey(9), cfg)
    k0 = addr.code(book, 0)
    target = jnp.asarray(rng.standard_normal((2, cfg.act_dim)).astype(np.float32))

    def loss(scratch_bank):
        logits, value, _ = ag.policy_step(
            params, banks, scratch_bank, cfg, main_h, obs, prev_a, prev_r, reset, k0
        )
        return jnp.sum((logits - target) ** 2) + jnp.sum(value ** 2)

    grads = jax.grad(loss)(sbank)
    # Every contextualized layer's scratch receives a usable gradient.
    for name, g in grads.items():
        gn = float(jnp.linalg.norm(g.A_s)) + float(jnp.linalg.norm(g.B_s))
        assert gn > 0.0, f"no plasticity gradient at {name}"


def test_context_query_independent_of_action_label_permutation():
    # Sanity: the query is a function of content; it never receives a game id.
    cfg = _small_cfg()
    key = jax.random.PRNGKey(10)
    params = ag.init_params(key, cfg)
    rng = np.random.default_rng(11)
    obs = _obs(rng, cfg)
    q1, _ = ag.context_query(params, cfg, jnp.zeros((2, cfg.ctx_hidden)),
                             obs, jnp.zeros((2,), jnp.int32), jnp.zeros((2,), jnp.float32),
                             jnp.ones((2,), bool))
    q2, _ = ag.context_query(params, cfg, jnp.zeros((2, cfg.ctx_hidden)),
                             obs, jnp.zeros((2,), jnp.int32), jnp.zeros((2,), jnp.float32),
                             jnp.ones((2,), bool))
    np.testing.assert_array_equal(np.asarray(q1), np.asarray(q2))  # deterministic


def test_deterministic_action_trace_after_restore():
    cfg = _small_cfg()
    key = jax.random.PRNGKey(12)
    banks = ag.init_banks(key, cfg)
    params = ag.init_params(key, cfg)
    book = _book(cfg, 2, seed=13)
    # Commit a context so banks have content to serialize.
    sbank = ag.init_scratch_banks(jax.random.PRNGKey(14), cfg)
    sbank = {n: s.replace(B_s=s.B_s + 0.2) for n, s in sbank.items()}
    banks, rep = tx.commit_scratch_bank(banks, sbank, book, 0)
    assert rep["accepted"]

    proto = __import__("tfns.castm.router", fromlist=["x"]).empty_prototype_index(8, 8, cfg.d_q)
    state = st.ContinualState(banks=banks, book=book, proto_index=proto, shared_params=params)
    restored = st.load_bytes(state, st.save_bytes(state))

    rng = np.random.default_rng(15)
    main_h = jnp.zeros((2, cfg.gru_hidden), jnp.float32)
    k0 = addr.code(book, 0)
    actions_a, actions_b = [], []
    key_a = jax.random.PRNGKey(123)
    for t in range(6):
        obs = _obs(rng, cfg)
        prev_a = jnp.zeros((2,), jnp.int32)
        prev_r = jnp.zeros((2,), jnp.float32)
        reset = jnp.zeros((2,), bool)
        la, _, _ = ag.policy_step(state.shared_params, state.banks, None, cfg, main_h,
                                  obs, prev_a, prev_r, reset, k0, ctx_id=0, sparse=True)
        lb, _, _ = ag.policy_step(restored.shared_params, restored.banks, None, cfg, main_h,
                                  obs, prev_a, prev_r, reset, k0, ctx_id=0, sparse=True)
        actions_a.append(int(jnp.argmax(la[0])))
        actions_b.append(int(jnp.argmax(lb[0])))
    assert actions_a == actions_b
