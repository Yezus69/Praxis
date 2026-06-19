from __future__ import annotations

import inspect
import os
import re

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tfns.envs import clip_reward
from tfns.model.agent import RecurrentAgent, protected_param_paths
from tfns.model.encoder import Encoder
from tfns.model.gru import ExplicitGRU


BATCH = 2


def _inputs(batch: int = BATCH, key=jax.random.PRNGKey(0)):
    obs_key, = jax.random.split(key, 1)
    obs = jax.random.uniform(obs_key, (batch, 84, 84, 4), dtype=jnp.float32)
    prev_action = jnp.arange(batch, dtype=jnp.int32) % 18
    prev_reward = jnp.linspace(-1.0, 1.0, batch, dtype=jnp.float32)
    reset = jnp.zeros((batch,), dtype=bool)
    return obs, prev_action, prev_reward, reset


@pytest.fixture(scope="module")
def initialized_agent():
    agent = RecurrentAgent()
    obs, prev_action, prev_reward, reset = _inputs()
    hidden = agent.init_hidden(BATCH)
    variables = agent.init(
        jax.random.PRNGKey(7),
        obs,
        prev_action,
        prev_reward,
        reset,
        hidden,
    )
    return agent, variables["params"], (obs, prev_action, prev_reward, reset, hidden)


def test_forward_shapes_single_step_and_unroll(initialized_agent):
    agent, params, inputs = initialized_agent
    obs, prev_action, prev_reward, reset, hidden = inputs

    out = agent.apply({"params": params}, obs, prev_action, prev_reward, reset, hidden)
    assert out.logits.shape == (BATCH, 18)
    assert out.value.shape == (BATCH,)
    assert out.q_key.shape == (BATCH, 128)
    np.testing.assert_allclose(
        np.asarray(jnp.linalg.norm(out.q_key, axis=-1)),
        np.ones((BATCH,), dtype=np.float32),
        rtol=1e-4,
        atol=1e-4,
    )
    assert out.h_next.shape == (BATCH, 512)

    t = 3
    obs_seq = jnp.stack([obs + float(i) * 0.001 for i in range(t)], axis=0)
    act_seq = jnp.stack([(prev_action + i) % 18 for i in range(t)], axis=0)
    rew_seq = jnp.stack([prev_reward for _ in range(t)], axis=0)
    reset_seq = jnp.zeros((t, BATCH), dtype=bool)
    seq_out, h_final = agent.unroll(params, obs_seq, act_seq, rew_seq, reset_seq, hidden)

    assert seq_out.logits.shape == (t, BATCH, 18)
    assert seq_out.value.shape == (t, BATCH)
    assert seq_out.q_key.shape == (t, BATCH, 128)
    assert seq_out.h_next.shape == (t, BATCH, 512)
    assert h_final.shape == (BATCH, 512)


def test_recurrent_agent_signature_has_no_task_argument():
    sig = inspect.signature(RecurrentAgent.__call__)
    forbidden = re.compile(r"game|task|onehot|(^|_)id($|_)", re.IGNORECASE)
    for name in sig.parameters:
        if name == "self":
            continue
        assert forbidden.search(name) is None


def test_determinism_same_params_inputs(initialized_agent):
    agent, params, inputs = initialized_agent
    out_a = agent.apply({"params": params}, *inputs)
    out_b = agent.apply({"params": params}, *inputs)

    leaves_a = jax.tree_util.tree_leaves(out_a)
    leaves_b = jax.tree_util.tree_leaves(out_b)
    assert len(leaves_a) == len(leaves_b)
    for a, b in zip(leaves_a, leaves_b, strict=True):
        np.testing.assert_allclose(np.asarray(a), np.asarray(b), rtol=0.0, atol=0.0)


def test_adapter_zero_init_no_op_for_both_sites(initialized_agent):
    agent, params, inputs = initialized_agent
    disabled = jnp.ones((8,), dtype=bool)
    enabled = jnp.zeros((8,), dtype=bool)

    out_disabled = agent.apply({"params": params}, *inputs, adapter_dormant=disabled)
    out_enabled = agent.apply({"params": params}, *inputs, adapter_dormant=enabled)

    np.testing.assert_allclose(
        np.asarray(out_disabled.logits), np.asarray(out_enabled.logits), rtol=0.0, atol=1e-6
    )
    np.testing.assert_allclose(
        np.asarray(out_disabled.value), np.asarray(out_enabled.value), rtol=0.0, atol=1e-6
    )


def test_explicit_gru_matches_hand_computation():
    cell = ExplicitGRU(hidden=2)
    x = np.array([[0.2, -0.1, 0.4]], dtype=np.float32)
    h = np.array([[0.3, -0.2]], dtype=np.float32)
    params = {
        "W_z": jnp.array([[0.1, -0.2], [0.3, 0.4], [-0.5, 0.2]], dtype=jnp.float32),
        "U_z": jnp.array([[0.2, -0.1], [0.0, 0.3]], dtype=jnp.float32),
        "b_z": jnp.array([0.05, -0.02], dtype=jnp.float32),
        "W_r": jnp.array([[-0.2, 0.1], [0.4, -0.3], [0.2, 0.5]], dtype=jnp.float32),
        "U_r": jnp.array([[0.1, 0.2], [-0.4, 0.1]], dtype=jnp.float32),
        "b_r": jnp.array([0.03, 0.07], dtype=jnp.float32),
        "W_n": jnp.array([[0.3, -0.2], [-0.1, 0.2], [0.4, -0.5]], dtype=jnp.float32),
        "U_n": jnp.array([[0.2, 0.1], [0.3, -0.2]], dtype=jnp.float32),
        "b_n": jnp.array([-0.04, 0.06], dtype=jnp.float32),
    }

    def sigmoid(y):
        return 1.0 / (1.0 + np.exp(-y))

    p = {k: np.asarray(v) for k, v in params.items()}
    z = sigmoid(x @ p["W_z"] + h @ p["U_z"] + p["b_z"])
    r = sigmoid(x @ p["W_r"] + h @ p["U_r"] + p["b_r"])
    n = np.tanh(x @ p["W_n"] + r * (h @ p["U_n"]) + p["b_n"])
    expected = (1.0 - z) * n + z * h

    actual = cell.apply(
        {"params": params},
        jnp.asarray(x),
        jnp.asarray(h),
        jnp.array([False]),
    )
    np.testing.assert_allclose(np.asarray(actual), expected, rtol=1e-5, atol=1e-5)


def test_unroll_reset_uses_zero_incoming_hidden(initialized_agent):
    agent, params, inputs = initialized_agent
    obs, prev_action, prev_reward, _, hidden = inputs
    h0 = hidden + 0.7
    obs_seq = jnp.stack([obs, obs * 0.5, obs * 0.25], axis=0)
    act_seq = jnp.stack([prev_action, (prev_action + 1) % 18, prev_action], axis=0)
    rew_seq = jnp.stack([prev_reward, -prev_reward, prev_reward], axis=0)
    reset_seq = jnp.array([[False, False], [True, True], [False, False]])

    seq_out, _ = agent.unroll(params, obs_seq, act_seq, rew_seq, reset_seq, h0)
    single = agent.apply(
        {"params": params},
        obs_seq[1],
        act_seq[1],
        rew_seq[1],
        reset_seq[1],
        jnp.zeros_like(h0),
    )

    np.testing.assert_allclose(np.asarray(seq_out.h_next[1]), np.asarray(single.h_next), atol=1e-6)


def test_crelu_encoder_preserves_dense_width():
    encoder = Encoder(dense_dim=64, activation="crelu")
    obs = jnp.ones((1, 84, 84, 4), dtype=jnp.float32)
    variables = encoder.init(jax.random.PRNGKey(11), obs)
    out = encoder.apply(variables, obs)
    assert out.shape == (1, 64)


def test_reward_clip_transform_is_context_free():
    rewards = jnp.array([-3.0, -1.0, -0.2, 0.0, 0.8, 2.5], dtype=jnp.float32)
    expected = jnp.array([-1.0, -1.0, -0.2, 0.0, 0.8, 1.0], dtype=jnp.float32)
    np.testing.assert_allclose(np.asarray(clip_reward(rewards)), np.asarray(expected))
    np.testing.assert_allclose(np.asarray(clip_reward(rewards)), np.asarray(clip_reward(rewards)))


def test_protected_param_paths_include_core_and_exclude_aux(initialized_agent):
    _, params, _ = initialized_agent
    paths = protected_param_paths(params)
    assert paths

    path_set = set(paths)
    for gate in ("z", "r", "n"):
        assert ("gru", f"W_{gate}") in path_set
        assert ("gru", f"U_{gate}") in path_set
        assert ("gru", f"b_{gate}") in path_set

    assert any(path[0] == "policy_head" for path in paths)
    assert any(path[0] == "value_head" for path in paths)
    assert any(path[0] == "key_head" for path in paths)
    assert ("visual_adapter", "V") in path_set
    assert ("visual_adapter", "U") in path_set
    assert ("post_adapter", "V") in path_set
    assert ("post_adapter", "U") in path_set

    aux_prefixes = {"next_feat_head", "reward_cat_head", "terminal_head"}
    assert not any(path[0] in aux_prefixes for path in paths)
