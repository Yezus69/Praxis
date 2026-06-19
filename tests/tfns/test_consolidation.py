from __future__ import annotations

import dataclasses
import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import optax

from tfns.config import (
    AdapterConfig,
    ConsolidateConfig,
    MemoryConfig,
    ModelConfig,
    ProtectConfig,
    ReplayConfig,
    TFNSConfig,
)
from tfns.consolidate.lifecycle import consolidate
from tfns.consolidate.state import ContinualState, Snapshot, snapshot
from tfns.memory.bank import SequenceMemoryBank
from tfns.memory.record import EpisodeSequence, frames_from_obs, make_record, nbytes, reconstruct_obs
from tfns.model.agent import RecurrentAgent
from tfns.protect.bases import empty_basis
from tfns.protect.projection import build_protected_modules, collect_conv_basis_columns


MODEL_CFG = ModelConfig(
    act_dim=4,
    conv_channels=(2, 2, 2),
    dense_dim=8,
    action_embed_dim=4,
    gru_hidden=8,
    key_dim=8,
)
ADAPTER_CFG = AdapterConfig(num_adapters=2, rank=2, top_k=1)
CFG = TFNSConfig(
    model=MODEL_CFG,
    adapter=ADAPTER_CFG,
    protect=ProtectConfig(residual_energy=1.0, max_rank_frac=1.0),
    consolidate=ConsolidateConfig(
        learned_threshold=0.9,
        stable_windows=2,
        retention_accept=0.9,
    ),
    replay=ReplayConfig(burn_in=0),
)


def _obs(t: int, value: int) -> np.ndarray:
    obs = np.zeros((t, 84, 84, 4), dtype=np.uint8)
    obs[:, :8, :8, :] = np.uint8(value)
    return obs


def _record(t: int = 3, value: int = 0, episode_id: int = 0) -> EpisodeSequence:
    init_stack, new_frames = frames_from_obs(_obs(t, value))
    key_anchor = np.zeros((t, 128), dtype=np.float32)
    key_anchor[:, 0] = 1.0
    return make_record(
        init_stack=init_stack,
        new_frames=new_frames,
        actions=np.arange(t, dtype=np.int32) % MODEL_CFG.act_dim,
        rewards_clipped=np.zeros((t,), dtype=np.float32),
        rewards_raw=np.zeros((t,), dtype=np.float32),
        ppo_mask=np.ones((t,), dtype=np.bool_),
        reset_mask=np.array([True] + [False] * (t - 1), dtype=np.bool_),
        teacher_logits=np.zeros((t, 18), dtype=np.float32),
        teacher_value=np.zeros((t,), dtype=np.float32),
        key_anchor=key_anchor,
        causal_contrib=np.ones((t,), dtype=np.float32),
        credit_trace=np.ones((t,), dtype=np.float32),
        adv_mag=np.ones((t,), dtype=np.float32),
        td_mag=np.ones((t,), dtype=np.float32),
        surprise=np.zeros((t,), dtype=np.float32),
        teacher_entropy=np.zeros((t,), dtype=np.float32),
        episode_id=episode_id,
        chunk_index=0,
    )


def _agent_state():
    agent = RecurrentAgent(model_config=MODEL_CFG, adapter_config=ADAPTER_CFG)
    obs = jnp.zeros((1, 84, 84, 4), dtype=jnp.float32)
    prev_action = jnp.zeros((1,), dtype=jnp.int32)
    prev_reward = jnp.zeros((1,), dtype=jnp.float32)
    reset = jnp.ones((1,), dtype=bool)
    hidden = agent.init_hidden(1)
    params = agent.init(jax.random.PRNGKey(0), obs, prev_action, prev_reward, reset, hidden)[
        "params"
    ]
    tx = optax.adam(1.0e-3)
    modules = build_protected_modules(params, MODEL_CFG)

    rec0 = _record(value=0, episode_id=0)
    rec1 = _record(value=7, episode_id=1)
    bank = SequenceMemoryBank(
        MemoryConfig(byte_budget=10 * (nbytes(rec0) + nbytes(rec1)), min_per_cluster=0)
    )
    assert bank.add(rec0)
    assert bank.add(rec1)

    state = ContinualState(
        params=params,
        opt_state=tx.init(params),
        ema_params=params,
        bases={name: empty_basis(module.d_aug) for name, module in modules.items()},
        memory=bank,
        predictor_params={"w": jnp.array([1.0], dtype=jnp.float32)},
        predictor_opt_state={"m": np.array([0.5], dtype=np.float32)},
        detector_state={"page_hinkley": np.array([0.25], dtype=np.float32)},
        adapter_dormant=np.array([True, True], dtype=np.bool_),
        robust_stats={"cluster_risk": {0: 0.1}, "last": np.array([2.0], dtype=np.float32)},
        protected_clusters=[{"cluster_id": 99, "note": "pre"}],
        rng=jnp.array([0, 1], dtype=jnp.uint32),
        block_index=5,
        skills={"old": {"protected": True}},
        rollout_carry={"h": np.array([3.0], dtype=np.float32)},
    )
    return agent, tx, state, modules


def _assert_tree_equal(actual, expected):
    leaves_a = jax.tree_util.tree_leaves(actual)
    leaves_b = jax.tree_util.tree_leaves(expected)
    assert len(leaves_a) == len(leaves_b)
    for a, b in zip(leaves_a, leaves_b, strict=True):
        np.testing.assert_array_equal(np.asarray(a), np.asarray(b))


def _assert_nested_equal(actual, expected):
    if isinstance(expected, dict):
        assert set(actual) == set(expected)
        for key in expected:
            _assert_nested_equal(actual[key], expected[key])
        return
    if isinstance(expected, (list, tuple)):
        assert len(actual) == len(expected)
        for a, b in zip(actual, expected, strict=True):
            _assert_nested_equal(a, b)
        return
    if isinstance(expected, np.ndarray) or hasattr(expected, "__array__"):
        np.testing.assert_array_equal(np.asarray(actual), np.asarray(expected))
        return
    assert actual == expected


def _assert_record_equal(actual: EpisodeSequence, expected: EpisodeSequence):
    for field in dataclasses.fields(EpisodeSequence):
        a = getattr(actual, field.name)
        b = getattr(expected, field.name)
        if isinstance(b, np.ndarray):
            np.testing.assert_array_equal(a, b)
        else:
            assert a == b


def _assert_memory_equal(actual: SequenceMemoryBank, expected: SequenceMemoryBank):
    assert len(actual) == len(expected)
    assert actual.bytes_used() == expected.bytes_used()
    assert actual.clusters() == expected.clusters()
    for rec_a, rec_b in zip(actual.records(), expected.records(), strict=True):
        _assert_record_equal(rec_a, rec_b)


def _assert_state_matches_snapshot(state: ContinualState, snap: Snapshot):
    _assert_tree_equal(state.params, snap.params)
    _assert_tree_equal(state.opt_state, snap.opt_state)
    _assert_tree_equal(state.ema_params, snap.ema_params)
    assert set(state.bases) == set(snap.bases)
    for name in state.bases:
        np.testing.assert_array_equal(np.asarray(state.bases[name]), np.asarray(snap.bases[name]))
    _assert_memory_equal(state.memory, snap.memory)
    _assert_tree_equal(state.predictor_params, snap.predictor_params)
    _assert_nested_equal(state.predictor_opt_state, snap.predictor_opt_state)
    _assert_nested_equal(state.detector_state, snap.detector_state)
    np.testing.assert_array_equal(state.adapter_dormant, snap.adapter_dormant)
    _assert_nested_equal(state.robust_stats, snap.robust_stats)
    _assert_nested_equal(state.protected_clusters, snap.protected_clusters)
    np.testing.assert_array_equal(np.asarray(state.rng), np.asarray(snap.rng))
    assert state.block_index == snap.block_index
    _assert_nested_equal(state.skills, snap.skills)
    _assert_nested_equal(state.rollout_carry, snap.rollout_carry)


def test_consolidate_rejects_closed_loop_regression_and_restores_complete_snapshot():
    agent, tx, state, _ = _agent_state()
    pre = snapshot(state)

    state, accepted, report = consolidate(
        state,
        agent,
        tx,
        eval_fn=lambda _params: {"old_game": 0.75},
        candidate_records=list(state.memory.records()),
        cfg=CFG,
        S_random=0.0,
        S_single=10.0,
        score_windows=[9.3, 9.2],
        learned_game_keys=["old_game"],
    )

    assert accepted is False
    assert report["reason"] == "closed_loop_regression"
    assert report["gate"]["per_game"]["old_game"]["pass"] is False
    _assert_state_matches_snapshot(state, pre)


def test_consolidate_accepts_protects_records_and_grows_conv_basis():
    agent, tx, state, modules = _agent_state()
    pre_ranks = {name: int(basis.shape[1]) for name, basis in state.bases.items()}

    first = state.memory.records()[0]
    obs_seq = jnp.asarray(reconstruct_obs(first), dtype=jnp.float32)[:, None, ...]
    act_seq = jnp.asarray(first.actions, dtype=jnp.int32)[:, None]
    rew_seq = jnp.asarray(first.rewards_clipped, dtype=jnp.float32)[:, None]
    reset_seq = jnp.asarray(first.reset_mask, dtype=bool)[:, None]
    outputs, _ = agent.unroll(
        state.params,
        obs_seq,
        act_seq,
        rew_seq,
        reset_seq,
        agent.init_hidden(1),
        collect_presyn=True,
    )
    assert {"conv1_in", "conv2_in", "conv3_in", "encoder_dense"} <= set(outputs.presyn)
    conv1 = modules["encoder_conv1"]
    conv_cols = collect_conv_basis_columns(
        outputs.presyn["conv1_in"].reshape((-1, 84, 84, 4)),
        conv1.kh,
        conv1.kw,
        conv1.stride,
        conv1.c_in,
    )
    assert conv_cols.shape[0] == conv1.d_aug

    state, accepted, report = consolidate(
        state,
        agent,
        tx,
        eval_fn=lambda _params: {"old_game": 0.95},
        candidate_records=list(state.memory.records()),
        cfg=CFG,
        S_random=0.0,
        S_single=10.0,
        score_windows=[9.3, 9.2],
        learned_game_keys=["old_game"],
    )

    assert accepted is True
    assert report["reason"] == "accepted"
    assert int(state.bases["encoder_conv1"].shape[1]) > pre_ranks["encoder_conv1"]
    assert any(int(state.bases[name].shape[1]) > pre_ranks[name] for name in state.bases)
    assert any(rec.status == "protected" for rec in state.memory.records())
    assert any(rec.is_sentinel for rec in state.memory.records())
    assert state.protected_clusters
