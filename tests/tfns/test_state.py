from __future__ import annotations

import os
import pickle

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from tfns.config import MemoryConfig
from tfns.consolidate.state import ContinualState, deserialize, restore, serialize, snapshot
from tfns.detect.change import PageHinkleyDetector
from tfns.memory.bank import SequenceMemoryBank
from tfns.memory.record import frames_from_obs, make_record, nbytes, reconstruct_obs


def _overlap_obs(t: int) -> np.ndarray:
    frames = np.arange((t + 4) * 84 * 84, dtype=np.uint8).reshape((t + 4, 84, 84))
    return np.stack([np.moveaxis(frames[i + 1 : i + 5], 0, -1) for i in range(t)], axis=0)


def _record(t: int = 3):
    obs = _overlap_obs(t)
    init_stack, new_frames = frames_from_obs(obs)
    key = np.zeros((t, 128), dtype=np.float32)
    key[:, 0] = 1.0
    return make_record(
        init_stack=init_stack,
        new_frames=new_frames,
        actions=np.arange(t, dtype=np.int32),
        rewards_clipped=np.zeros((t,), dtype=np.float32),
        rewards_raw=np.zeros((t,), dtype=np.float32),
        ppo_mask=np.zeros((t,), dtype=np.bool_),
        reset_mask=np.zeros((t,), dtype=np.bool_),
        teacher_logits=np.zeros((t, 18), dtype=np.float32),
        teacher_value=np.zeros((t,), dtype=np.float32),
        key_anchor=key,
        causal_contrib=np.zeros((t,), dtype=np.float32),
        credit_trace=np.zeros((t,), dtype=np.float32),
        adv_mag=np.zeros((t,), dtype=np.float32),
        td_mag=np.zeros((t,), dtype=np.float32),
        surprise=np.zeros((t,), dtype=np.float32),
        teacher_entropy=np.zeros((t,), dtype=np.float32),
        episode_id=1,
        chunk_index=0,
    )


def test_frames_from_obs_round_trips_reconstruct_obs_exactly():
    obs = _overlap_obs(6)
    init_stack, new_frames = frames_from_obs(obs)
    rec = make_record(
        init_stack=init_stack,
        new_frames=new_frames,
        actions=np.zeros((6,), dtype=np.int32),
        rewards_clipped=np.zeros((6,), dtype=np.float32),
        rewards_raw=np.zeros((6,), dtype=np.float32),
        ppo_mask=np.zeros((6,), dtype=np.bool_),
        reset_mask=np.zeros((6,), dtype=np.bool_),
        teacher_logits=np.zeros((6, 18), dtype=np.float32),
        teacher_value=np.zeros((6,), dtype=np.float32),
        key_anchor=np.zeros((6, 128), dtype=np.float32),
        causal_contrib=np.zeros((6,), dtype=np.float32),
        credit_trace=np.zeros((6,), dtype=np.float32),
        adv_mag=np.zeros((6,), dtype=np.float32),
        td_mag=np.zeros((6,), dtype=np.float32),
        surprise=np.zeros((6,), dtype=np.float32),
        teacher_entropy=np.zeros((6,), dtype=np.float32),
        episode_id=7,
        chunk_index=0,
    )

    np.testing.assert_array_equal(init_stack[0], obs[0, ..., 0])
    np.testing.assert_array_equal(init_stack[1], obs[0, ..., 0])
    np.testing.assert_array_equal(new_frames, obs[..., -1])
    np.testing.assert_array_equal(reconstruct_obs(rec), obs)


def test_continual_state_snapshot_restore_reproduces_all_mutable_fields():
    rec = _record()
    bank = SequenceMemoryBank(MemoryConfig(byte_budget=nbytes(rec) * 4, min_per_cluster=0))
    assert bank.add(rec)
    detector = PageHinkleyDetector()
    detector_state = detector.init()
    detector_state, _ = detector.update(detector_state, 0.5)

    state = ContinualState(
        params={"w": jnp.array([1.0, 2.0], dtype=jnp.float32)},
        opt_state={"m": np.array([0.25], dtype=np.float32)},
        ema_params={"w": jnp.array([0.5, 1.5], dtype=jnp.float32)},
        bases={"dense": np.eye(2, 1, dtype=np.float32)},
        memory=bank,
        predictor_params={"p": jnp.array([3.0], dtype=jnp.float32)},
        predictor_opt_state={"pm": np.array([4.0], dtype=np.float32)},
        detector_state=detector_state,
        adapter_dormant=np.array([True, False], dtype=np.bool_),
        robust_stats={"predictor_val_mses": [1.0], "last_signature": np.ones((128,), dtype=np.float32)},
        protected_clusters=[{"cluster": 1}],
        rng=jnp.array([0, 1], dtype=jnp.uint32),
        block_index=7,
        skills={"note": {"protected": False}},
    )
    snap = snapshot(state)

    state.params = {"w": jnp.array([-1.0, -2.0], dtype=jnp.float32)}
    state.opt_state["m"][0] = 9.0
    state.ema_params = {"w": jnp.array([9.0, 9.0], dtype=jnp.float32)}
    state.bases["dense"][0, 0] = 0.0
    state.memory.records()[0].actions[0] = 99
    state.predictor_params = {"p": jnp.array([-3.0], dtype=jnp.float32)}
    state.predictor_opt_state["pm"][0] = -4.0
    state.detector_state, _ = detector.update(state.detector_state, 100.0)
    state.adapter_dormant[0] = False
    state.robust_stats["predictor_val_mses"].append(2.0)
    state.protected_clusters.append({"cluster": 2})
    state.rng = jnp.array([9, 9], dtype=jnp.uint32)
    state.block_index = 99
    state.skills["note"]["protected"] = True

    restore(state, snap)

    np.testing.assert_allclose(np.asarray(state.params["w"]), np.array([1.0, 2.0], dtype=np.float32))
    np.testing.assert_allclose(state.opt_state["m"], np.array([0.25], dtype=np.float32))
    np.testing.assert_allclose(np.asarray(state.ema_params["w"]), np.array([0.5, 1.5], dtype=np.float32))
    np.testing.assert_allclose(state.bases["dense"], np.eye(2, 1, dtype=np.float32))
    np.testing.assert_array_equal(state.memory.records()[0].actions, np.arange(3, dtype=np.int32))
    assert state.detector_state == detector_state
    np.testing.assert_array_equal(state.adapter_dormant, np.array([True, False], dtype=np.bool_))
    assert state.block_index == 7
    assert state.protected_clusters == [{"cluster": 1}]
    assert state.skills == {"note": {"protected": False}}

    payload = serialize(state)
    pickle.dumps(payload)
    restored = deserialize(payload)
    assert restored.block_index == state.block_index
