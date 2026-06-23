"""Prototype router tests with a synthetic switching scenario (spec 9, 22 Stage A)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tfns.castm import router as rt


def _unit(v):
    v = np.asarray(v, dtype=np.float32)
    return v / (np.linalg.norm(v) + 1e-8)


def _seed_two_contexts(d_q=16, seed=0):
    rng = np.random.default_rng(seed)
    c0 = _unit(rng.standard_normal(d_q))
    # Make c1 orthogonal to c0 for clean separation.
    c1 = rng.standard_normal(d_q)
    c1 = c1 - (c1 @ c0) * c0
    c1 = _unit(c1)
    cfg = rt.RouterConfig(min_dwell=4, novel_persist=8, confidence_thresh=0.6)
    index = rt.empty_prototype_index(cfg.max_contexts, cfg.prototypes_per_context, d_q)
    # Seed each context from a window of jittered queries near its centroid.
    def window(c):
        w = c[None, :] + 0.02 * rng.standard_normal((32, d_q)).astype(np.float32)
        return _normalize_np(w)
    index, ctx0 = rt.allocate_context(index, window(c0), cfg)
    index, ctx1 = rt.allocate_context(index, window(c1), cfg)
    return index, cfg, c0, c1, ctx0, ctx1


def _normalize_np(w):
    w = np.asarray(w, dtype=np.float32)
    return w / (np.linalg.norm(w, axis=-1, keepdims=True) + 1e-8)


def test_route_locks_correct_context_and_switches():
    index, cfg, c0, c1, ctx0, ctx1 = _seed_two_contexts()
    d_q = c0.shape[0]
    state = rt.init_router_state(batch=1, max_contexts=cfg.max_contexts)
    rng = np.random.default_rng(1)

    def q_near(c):
        return jnp.asarray((_unit(c + 0.02 * rng.standard_normal(d_q)))[None, :])

    low_err = jnp.zeros((1,), jnp.float32)
    # Feed context-0 queries; router should lock ctx0 as KNOWN.
    decisions = []
    for _ in range(20):
        state, info = rt.route_step(state, index, q_near(c0), low_err, cfg)
        decisions.append((int(info["decision"][0]), int(info["selected_ctx"][0])))
    # After dwell, it is locked and KNOWN on ctx0.
    assert int(state.locked_ctx[0]) == ctx0
    last_known = [d for d in decisions if d[0] == rt.KNOWN]
    assert last_known and last_known[-1][1] == ctx0

    # Now switch to context 1; after the dwell window it relocks to ctx1.
    for _ in range(20):
        state, info = rt.route_step(state, index, q_near(c1), low_err, cfg)
    assert int(state.locked_ctx[0]) == ctx1
    assert int(info["decision"][0]) == rt.KNOWN
    assert int(info["selected_ctx"][0]) == ctx1


def test_hysteresis_resists_single_outlier():
    index, cfg, c0, c1, ctx0, ctx1 = _seed_two_contexts(seed=2)
    d_q = c0.shape[0]
    state = rt.init_router_state(batch=1, max_contexts=cfg.max_contexts)
    rng = np.random.default_rng(3)

    def q_near(c):
        return jnp.asarray((_unit(c + 0.02 * rng.standard_normal(d_q)))[None, :])

    low_err = jnp.zeros((1,), jnp.float32)
    for _ in range(20):
        state, _ = rt.route_step(state, index, q_near(c0), low_err, cfg)
    assert int(state.locked_ctx[0]) == ctx0
    # A single context-1 frame must NOT flip the lock (dwell not satisfied).
    state, info = rt.route_step(state, index, q_near(c1), low_err, cfg)
    assert int(state.locked_ctx[0]) == ctx0


def test_novelty_detection_persists():
    index, cfg, c0, c1, ctx0, ctx1 = _seed_two_contexts(seed=4)
    d_q = c0.shape[0]
    state = rt.init_router_state(batch=1, max_contexts=cfg.max_contexts)
    # Build a robust low dyn-error baseline first with in-distribution queries.
    rng = np.random.default_rng(5)
    for _ in range(50):
        q = jnp.asarray((_unit(c0 + 0.02 * rng.standard_normal(d_q)))[None, :])
        state, _ = rt.route_step(state, index, q, jnp.array([0.1], jnp.float32), cfg)

    # A novel query orthogonal to both contexts with elevated dynamics error,
    # persisting for the window, must declare NOVEL.
    novel_dir = rng.standard_normal(d_q)
    novel_dir = novel_dir - (novel_dir @ c0) * c0
    novel_dir = novel_dir - (novel_dir @ c1) * c1
    novel_dir = _unit(novel_dir)
    saw_novel = False
    for _ in range(cfg.novel_persist + 4):
        q = jnp.asarray(novel_dir[None, :])
        state, info = rt.route_step(state, index, q, jnp.array([5.0], jnp.float32), cfg)
        if int(info["decision"][0]) == rt.NOVEL:
            saw_novel = True
    assert saw_novel


def test_route_step_is_jittable_and_batched():
    index, cfg, c0, c1, ctx0, ctx1 = _seed_two_contexts(seed=6)
    d_q = c0.shape[0]
    B = 4
    state = rt.init_router_state(batch=B, max_contexts=cfg.max_contexts)
    q = jnp.asarray(np.stack([_unit(c0), _unit(c1), _unit(c0), _unit(c1)]).astype(np.float32))
    err = jnp.zeros((B,), jnp.float32)
    jitted = jax.jit(lambda s, q, e: rt.route_step(s, index, q, e, cfg))
    state2, info = jitted(state, q, err)
    assert info["decision"].shape == (B,)
    assert info["posterior"].shape == (B, cfg.max_contexts)


def test_refresh_prototypes_keeps_slot():
    index, cfg, c0, c1, ctx0, ctx1 = _seed_two_contexts(seed=7)
    d_q = c0.shape[0]
    rng = np.random.default_rng(8)
    # Re-encode anchors (drifted) and refresh ctx0's prototypes in place.
    window = _normalize_np(c0[None, :] + 0.05 * rng.standard_normal((32, d_q)).astype(np.float32))
    before_used = np.asarray(index.used).copy()
    index2 = rt.refresh_prototypes(index, ctx0, window, cfg)
    assert bool(index2.used[ctx0])
    # No extra context slot was consumed.
    assert int(np.sum(np.asarray(index2.used))) == int(np.sum(before_used))
