from __future__ import annotations

import dataclasses
import inspect
import os
import re

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from flax.traverse_util import flatten_dict
import jax
import jax.numpy as jnp
import numpy as np
import pytest

from tfns.config import AdapterConfig, MemoryConfig, ModelConfig
from tfns.memory.bank import SequenceMemoryBank
from tfns.memory.record import EpisodeSequence, make_record, nbytes
from tfns.memory.sampling import sample_sequences
from tfns.model.agent import RecurrentAgent


MODEL_CFG = ModelConfig(
    act_dim=4,
    conv_channels=(2, 2, 2),
    dense_dim=8,
    action_embed_dim=4,
    gru_hidden=8,
    key_dim=8,
)
ADAPTER_CFG = AdapterConfig(num_adapters=2, rank=2, top_k=1)
FORBIDDEN_ARG = re.compile(r"game|task|onehot|curriculum|(^|_)id($|_)", re.IGNORECASE)


@pytest.fixture(scope="module")
def initialized_agent():
    agent = RecurrentAgent(model_config=MODEL_CFG, adapter_config=ADAPTER_CFG)
    obs = jnp.zeros((2, 84, 84, 4), dtype=jnp.uint8)
    prev_action = jnp.zeros((2,), dtype=jnp.int32)
    prev_reward = jnp.zeros((2,), dtype=jnp.float32)
    reset = jnp.ones((2,), dtype=bool)
    hidden = agent.init_hidden(2)
    params = agent.init(
        jax.random.PRNGKey(0),
        obs,
        prev_action,
        prev_reward,
        reset,
        hidden,
    )["params"]
    return agent, params


def _trajectory(t: int = 4, batch: int = 2):
    obs = jax.random.randint(
        jax.random.PRNGKey(11),
        (t, batch, 84, 84, 4),
        0,
        256,
        dtype=jnp.uint8,
    )
    actions = (jnp.arange(t * batch, dtype=jnp.int32).reshape(t, batch) % MODEL_CFG.act_dim)
    rewards = jnp.linspace(-1.0, 1.0, t * batch, dtype=jnp.float32).reshape(t, batch)
    resets = jnp.zeros((t, batch), dtype=bool).at[0].set(True)
    h0 = jnp.zeros((batch, MODEL_CFG.gru_hidden), dtype=jnp.float32)
    return obs, actions, rewards, resets, h0


def _record(episode_id: int, signature_index: int, *, corrupt_policy: bool = False) -> EpisodeSequence:
    t = 4
    init_stack = np.zeros((4, 84, 84), dtype=np.uint8)
    new_frames = np.full((t, 84, 84), signature_index, dtype=np.uint8)
    key_anchor = np.zeros((t, 128), dtype=np.float32)
    key_anchor[:, signature_index % 128] = 1.0
    teacher_logits = np.zeros((t, 18), dtype=np.float32)
    if corrupt_policy:
        teacher_logits[:, ::2] = 1000.0
        teacher_logits[:, 1::2] = -1000.0
    return make_record(
        init_stack=init_stack,
        new_frames=new_frames,
        actions=np.arange(t, dtype=np.int32) % MODEL_CFG.act_dim,
        rewards_clipped=np.zeros((t,), dtype=np.float32),
        rewards_raw=np.zeros((t,), dtype=np.float32),
        ppo_mask=np.zeros((t,), dtype=np.bool_),
        reset_mask=np.array([True] + [False] * (t - 1), dtype=np.bool_),
        teacher_logits=teacher_logits,
        teacher_value=np.zeros((t,), dtype=np.float32),
        key_anchor=key_anchor,
        causal_contrib=np.ones((t,), dtype=np.float32),
        credit_trace=np.ones((t,), dtype=np.float32),
        adv_mag=np.zeros((t,), dtype=np.float32),
        td_mag=np.zeros((t,), dtype=np.float32),
        surprise=np.zeros((t,), dtype=np.float32),
        teacher_entropy=np.zeros((t,), dtype=np.float32),
        episode_id=episode_id,
        chunk_index=0,
    )


def _bank(records: list[EpisodeSequence]) -> SequenceMemoryBank:
    budget = max(1, sum(nbytes(rec) for rec in records) * 4)
    bank = SequenceMemoryBank(MemoryConfig(byte_budget=budget, min_per_cluster=0))
    for rec in records:
        assert bank.add(rec)
    return bank


def test_no_task_argument_on_forward_or_unroll():
    for fn in (RecurrentAgent.__call__, RecurrentAgent.unroll):
        sig = inspect.signature(fn)
        for name in sig.parameters:
            if name == "self":
                continue
            assert FORBIDDEN_ARG.search(name) is None


def test_label_permutation_invariance_and_label_free_memory(initialized_agent):
    agent, params = initialized_agent
    obs, actions, rewards, resets, h0 = _trajectory()

    external_labels_a = np.array(["SpaceInvaders-v5", "Breakout-v5"])
    external_labels_b = external_labels_a[::-1].copy()
    assert not np.array_equal(external_labels_a, external_labels_b)

    outputs_a, h_final_a = agent.unroll(params, obs, actions, rewards, resets, h0)
    outputs_b, h_final_b = agent.unroll(params, obs, actions, rewards, resets, h0)

    for field in ("logits", "value", "q_key", "h_next"):
        np.testing.assert_array_equal(
            np.asarray(getattr(outputs_a, field)),
            np.asarray(getattr(outputs_b, field)),
        )
    np.testing.assert_array_equal(np.asarray(h_final_a), np.asarray(h_final_b))

    allowed_internal = {"cluster_id", "episode_id"}
    forbidden_field = re.compile(
        r"game|task|env|label|onehot|curriculum|(^|_)id($|_)",
        re.IGNORECASE,
    )
    for field in dataclasses.fields(EpisodeSequence):
        if field.name in allowed_internal:
            continue
        assert forbidden_field.search(field.name) is None

    records_a = [_record(episode_id=0, signature_index=0), _record(episode_id=1, signature_index=1)]
    records_b = [_record(episode_id=0, signature_index=0), _record(episode_id=1, signature_index=1)]
    labels_a = ["game_a", "game_b"]
    labels_b = ["game_b", "game_a"]
    assert labels_a != labels_b

    bank_a = _bank(records_a)
    bank_b = _bank(records_b)
    assert len(bank_a) == len(bank_b)
    assert bank_a.clusters() == bank_b.clusters()

    sample_a = sample_sequences(bank_a, 123, 8)
    sample_b = sample_sequences(bank_b, 123, 8)
    assert [(rec.episode_id, rec.chunk_index) for rec in sample_a] == [
        (rec.episode_id, rec.chunk_index) for rec in sample_b
    ]


def test_no_per_game_parameter_tree(initialized_agent):
    _agent, params = initialized_agent
    flat = flatten_dict(params, sep="/")
    forbidden_key = re.compile(
        r"(^|/|_)(game|task|onehot|curriculum|id)($|/|_)"
        r"|spaceinvaders|breakout|beamrider|asterix|qbert|pong|seaquest|demonattack",
        re.IGNORECASE,
    )
    for key in flat:
        assert forbidden_key.search(str(key)) is None

    top = set(params.keys())
    assert "policy_head" in top
    assert "value_head" in top
    assert "gru" in top
    assert sum(1 for name in top if name == "policy_head") == 1
    assert sum(1 for name in top if name == "value_head") == 1
    assert sum(1 for name in top if name == "gru") == 1

    gru = params["gru"]
    for gate in ("z", "r", "n"):
        assert f"W_{gate}" in gru
        assert f"U_{gate}" in gru
        assert f"b_{gate}" in gru


class SyntheticEvalEnv:
    def __init__(self, num_envs: int = 2, seed: int = 0):
        self.num_envs = int(num_envs)
        self.rng = np.random.default_rng(int(seed))
        self.step_count = 0
        self.obs = self.rng.integers(
            0,
            256,
            size=(self.num_envs, 84, 84, 4),
            dtype=np.uint8,
        )

    def __call__(self, action):
        action = np.asarray(action, dtype=np.int32)
        self.step_count += 1
        new_frame = np.zeros((self.num_envs, 84, 84), dtype=np.uint8)
        for env_index in range(self.num_envs):
            new_frame[env_index] = (
                self.step_count * 17 + env_index * 29 + int(action[env_index]) * 31
            ) % 256
        self.obs = np.concatenate([self.obs[..., 1:], new_frame[..., None]], axis=-1)
        reward = np.zeros((self.num_envs,), dtype=np.float32)
        done = np.zeros((self.num_envs,), dtype=np.bool_)
        reset = np.zeros((self.num_envs,), dtype=np.bool_)
        if self.step_count % 5 == 0:
            reset[:] = True
        return self.obs, reward, done, reset, {}


def _synthetic_action_trace(agent, params, memory_payload, *, seed: int) -> np.ndarray:
    _ = memory_payload
    env = SyntheticEvalEnv(num_envs=2, seed=seed)
    obs = jnp.asarray(env.obs)
    hidden = agent.init_hidden(env.num_envs)
    prev_action = jnp.zeros((env.num_envs,), dtype=jnp.int32)
    prev_reward = jnp.zeros((env.num_envs,), dtype=jnp.float32)
    prev_reset = jnp.ones((env.num_envs,), dtype=bool)
    actions = []
    for _step in range(12):
        out = agent.apply({"params": params}, obs, prev_action, prev_reward, prev_reset, hidden)
        action = jnp.argmax(out.logits, axis=-1).astype(jnp.int32)
        action_np = np.asarray(jax.device_get(action), dtype=np.int32)
        obs_np, reward_np, _done, reset_np, _extra = env(action_np)
        actions.append(action_np)
        reset = jnp.asarray(reset_np, dtype=bool)
        hidden = jnp.where(reset[:, None], jnp.zeros_like(out.h_next), out.h_next)
        obs = jnp.asarray(obs_np)
        prev_action = jnp.asarray(action_np, dtype=jnp.int32)
        prev_reward = jnp.asarray(reward_np, dtype=jnp.float32)
        prev_reset = reset
    return np.stack(actions, axis=0)


def test_no_inference_memory_path_action_trace_identical(initialized_agent):
    agent, params = initialized_agent
    for fn in (RecurrentAgent.__call__, RecurrentAgent.unroll):
        assert "memory" not in inspect.signature(fn).parameters

    empty_bank = _bank([])
    full_bank = _bank([_record(0, 0), _record(1, 1)])
    corrupt_shuffled = [
        _record(1, 1, corrupt_policy=True),
        _record(0, 0, corrupt_policy=True),
    ]

    trace_empty = _synthetic_action_trace(agent, params, empty_bank, seed=19)
    trace_full = _synthetic_action_trace(agent, params, full_bank, seed=19)
    trace_corrupt = _synthetic_action_trace(agent, params, corrupt_shuffled, seed=19)

    np.testing.assert_array_equal(trace_empty, trace_full)
    np.testing.assert_array_equal(trace_empty, trace_corrupt)
