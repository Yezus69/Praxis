"""Online context manager: task-free discovery/recall from content alone.

Covers required tests 4 (three-context alternation), 5 (online novelty
allocation), 6 (no identity leakage), 8 (prototype refresh under drift),
9 (false-split protection), 10 (per-stream routing).
"""

from __future__ import annotations

import inspect

import numpy as np
import pytest

from tfns.castm import context_manager as cm


D_Q = 64


def _orthonormal_centers(k, d_q, seed):
    rng = np.random.default_rng(seed)
    g = rng.standard_normal((d_q, k))
    q, _ = np.linalg.qr(g)
    return q[:, :k].T.astype(np.float32)  # (k, d_q) orthonormal rows


def _make_world(k, d_q=D_Q, seed=0):
    centers = _orthonormal_centers(k, d_q, seed)

    def gen_frames(cluster, b, noise=0.03, rng=None):
        rng = rng or np.random.default_rng()
        vec = centers[cluster][None, :] + noise * rng.standard_normal((b, d_q)).astype(np.float32)
        # Encode as recoverable uint8 "frames".
        u = np.clip((vec * 0.4 + 0.5) * 255.0, 0, 255).astype(np.uint8)
        return u

    def query_fn(frames, R=None):
        f = np.asarray(frames, np.float32) / 255.0
        v = (f - 0.5) / 0.4
        if R is not None:
            v = v @ R
        n = v / (np.linalg.norm(v, axis=-1, keepdims=True) + 1e-6)
        return n

    return centers, gen_frames, query_fn


def _cfg(**kw):
    # Adaptive routing params; small warmup so short synthetic segments still exercise
    # discovery (production uses warmup_rollouts=6 for ~122-rollout game segments).
    base = dict(d_q=D_Q, d_k=64, n_max=40, max_contexts=40, proto_per_ctx=8,
                switch_margin=0.03, novel_persist=3, min_dwell=3, warmup_rollouts=2,
                known_k=4.0, novel_k=6.0, anchor_cap=512, seed=0)
    base.update(kw)
    return cm.ManagerConfig(**base)


def test_online_novelty_allocation_discovers_K():
    """Required test 5: an unlabeled stream of K regimes discovers exactly K contexts."""
    K = 4
    _, gen, qfn = _make_world(K, seed=1)
    mgr = cm.OnlineContextManager(_cfg(seed=1))
    rng = np.random.default_rng(2)
    for cluster in range(K):
        for _ in range(6):  # > novel_persist + min_dwell
            frames = gen(cluster, 16, rng=rng)
            q = qfn(frames)
            mgr.update(q, frames)
    assert mgr.num_contexts == K, f"discovered {mgr.num_contexts} != {K}"
    # The last segment's active context matches the cluster discovered for it.
    assert mgr.active_ctx == mgr.ctx_ids[-1]


def test_three_context_alternation_no_excess():
    """Required test 4: A→B→C→A→B with no supplied ids; revisits recall, no proliferation."""
    _, gen, qfn = _make_world(3, seed=3)
    mgr = cm.OnlineContextManager(_cfg(seed=3))
    rng = np.random.default_rng(4)
    order = [0, 1, 2, 0, 1]
    seg_active = []
    cluster_to_ctx = {}
    for cluster in order:
        for _ in range(6):
            frames = gen(cluster, 16, rng=rng)
            mgr.update(qfn(frames), frames)
        seg_active.append(mgr.active_ctx)
        cluster_to_ctx.setdefault(cluster, mgr.active_ctx)
    assert mgr.num_contexts == 3, f"excess contexts: {mgr.num_contexts}"
    # Revisits route back to the SAME internal context, not a fresh one.
    assert seg_active[3] == cluster_to_ctx[0], "A-revisit did not recall A"
    assert seg_active[4] == cluster_to_ctx[1], "B-revisit did not recall B"


def test_per_stream_routing_rows_differ():
    """Required test 10: different rows in one batch select different contexts."""
    _, gen, qfn = _make_world(3, seed=5)
    mgr = cm.OnlineContextManager(_cfg(seed=5))
    rng = np.random.default_rng(6)
    # Discover all three.
    for cluster in range(3):
        for _ in range(6):
            frames = gen(cluster, 16, rng=rng)
            mgr.update(qfn(frames), frames)
    # Build a mixed batch: rows 0..4 cluster0, 5..9 cluster1, 10..14 cluster2.
    f0, f1, f2 = gen(0, 5, rng=rng), gen(1, 5, rng=rng), gen(2, 5, rng=rng)
    frames = np.concatenate([f0, f1, f2], axis=0)
    routed = mgr.per_stream_route(qfn(frames))
    assert len(set(routed.tolist())) == 3, f"rows did not diverge: {routed}"
    ctx0, ctx1, ctx2 = mgr.ctx_ids[0], mgr.ctx_ids[1], mgr.ctx_ids[2]
    assert np.all(routed[:5] == ctx0) and np.all(routed[5:10] == ctx1) and np.all(routed[10:] == ctx2)


def test_false_split_protection_transient_anomaly():
    """Required test 9 (split half): a single anomalous rollout must not allocate."""
    _, gen, qfn = _make_world(2, seed=7)
    mgr = cm.OnlineContextManager(_cfg(seed=7))
    rng = np.random.default_rng(8)
    # Establish context 0.
    for _ in range(6):
        frames = gen(0, 16, rng=rng)
        mgr.update(qfn(frames), frames)
    assert mgr.num_contexts == 1
    # ONE anomalous rollout of cluster 1, then back to 0.
    mgr.update(qfn(gen(1, 16, rng=rng)), gen(1, 16, rng=rng))
    assert mgr.num_contexts == 1, "transient anomaly wrongly allocated a context"
    for _ in range(3):
        frames = gen(0, 16, rng=rng)
        mgr.update(qfn(frames), frames)
    assert mgr.num_contexts == 1
    # SUSTAINED cluster 1 now does allocate (novelty persists).
    for _ in range(5):
        frames = gen(1, 16, rng=rng)
        mgr.update(qfn(frames), frames)
    assert mgr.num_contexts == 2, "sustained novelty failed to allocate"


def test_prototype_refresh_restores_routing_under_drift():
    """Required test 8: encoder drift breaks routing; refresh from raw anchors restores it."""
    K = 3
    _, gen, qfn = _make_world(K, seed=9)
    mgr = cm.OnlineContextManager(_cfg(seed=9))
    rng = np.random.default_rng(10)
    for cluster in range(K):
        for _ in range(6):
            frames = gen(cluster, 16, rng=rng)
            mgr.update(qfn(frames), frames)
    # Simulate encoder drift: a random rotation of query space.
    g = np.random.default_rng(11).standard_normal((D_Q, D_Q))
    R, _ = np.linalg.qr(g)
    R = R.astype(np.float32)
    drift_qfn = lambda fr: qfn(fr, R=R)
    # Under drift, the stale prototypes mis-route held-out anchors.
    before = mgr.audit_routing(drift_qfn)["overall"]
    # Refresh prototypes by re-encoding raw anchors with the drifted encoder.
    after = mgr.refresh_prototypes(drift_qfn)["overall"]
    assert after > 0.95, f"refresh failed to restore routing ({after})"
    assert after > before + 0.25, f"refresh did not materially improve routing ({before}->{after})"


def test_no_identity_leakage_api_and_source():
    """Required test 6: the manager API and source carry no game/task/curriculum identity."""
    sig = inspect.signature(cm.OnlineContextManager.update)
    params = list(sig.parameters)[1:]  # drop self
    assert params == ["queries", "frames"], f"update() leaks non-content inputs: {params}"
    import pathlib
    src = pathlib.Path(cm.__file__).read_text(encoding="utf-8").lower()
    # Concrete game names and label-derived identifiers must never appear as data/code.
    # (Prose like "no game id" in the docstring is allowed; we ban actual identifiers.)
    for banned in ("breakout", "pong", "seaquest", "beamrider", "spaceinvaders",
                   "-v5", "game_id", "task_id", "game_name", "curriculum_index"):
        assert banned not in src, f"identity leakage token in source: {banned}"


def test_state_dict_roundtrip():
    """Serialization of manager state (supports test 13 resume)."""
    _, gen, qfn = _make_world(2, seed=12)
    mgr = cm.OnlineContextManager(_cfg(seed=12))
    rng = np.random.default_rng(13)
    for cluster in range(2):
        for _ in range(6):
            frames = gen(cluster, 16, rng=rng)
            mgr.update(qfn(frames), frames)
    sd = mgr.state_dict()
    mgr2 = cm.OnlineContextManager(_cfg(seed=999))
    mgr2.load_state_dict(sd)
    assert mgr2.ctx_ids == mgr.ctx_ids
    assert mgr2.active_ctx == mgr.active_ctx
    # Routing is identical after restore.
    frames = gen(1, 12, rng=rng)
    assert np.array_equal(mgr.per_stream_route(qfn(frames)), mgr2.per_stream_route(qfn(frames)))
