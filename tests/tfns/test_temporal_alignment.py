"""Temporal-correctness tests for episodic replay.

These tests pin the fix for the critical replay bug: a stored record must feed
``(o_t, a_{t-1}, r_{t-1}, reset_t)`` as the recurrent inputs that generated its
teacher output, and only fragments beginning at a true episode reset (so the
recurrent context is reconstructible from a zero hidden state) may be admitted.
"""

from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from tfns.config import AdapterConfig, AuxConfig, MemoryConfig, ModelConfig, PPOConfig, TFNSConfig
from tfns.consolidate.state import ContinualState
from tfns.memory.bank import SequenceMemoryBank
from tfns.memory.record import make_record, reconstruct_obs, seq_len
from tfns.model.agent import RecurrentAgent
from tfns.ppo.losses import ppo_loss
from tfns.ppo.rollout import RolloutCarry, SequenceMinibatch, collect_rollout
from tfns.train.block import _admit_rollout_memories


def _tiny_agent(act_dim: int = 3, hidden: int = 8) -> RecurrentAgent:
    return RecurrentAgent(
        model_config=ModelConfig(
            act_dim=act_dim,
            conv_channels=(2, 2, 2),
            dense_dim=16,
            action_embed_dim=4,
            gru_hidden=hidden,
            key_dim=4,
        ),
        adapter_config=AdapterConfig(num_adapters=2, rank=2, top_k=1),
    )


def _stacked_obs(num_steps: int, B: int, seed: int = 0) -> np.ndarray:
    """Return a proper sliding 4-frame stack so reconstruct_obs is exact."""

    rng = np.random.default_rng(seed)
    frames = rng.integers(0, 256, size=(num_steps + 4, B, 84, 84), dtype=np.uint8)
    obs = np.empty((num_steps, B, 84, 84, 4), dtype=np.uint8)
    for t in range(num_steps):
        obs[t] = np.moveaxis(frames[t : t + 4], 0, -1)
    return obs


def _init_params(agent: RecurrentAgent, B: int):
    obs0 = jnp.asarray(_stacked_obs(1, B)[0])
    prev_action = jnp.zeros((B,), dtype=jnp.int32)
    prev_reward = jnp.zeros((B,), dtype=jnp.float32)
    reset = jnp.ones((B,), dtype=bool)
    h0 = agent.init_hidden(B)
    return agent.init(jax.random.PRNGKey(0), obs0, prev_action, prev_reward, reset, h0)["params"]


class _ScriptedEnv:
    """Deterministic env emitting a fixed obs/reward/reset script."""

    def __init__(self, obs_script, rewards, true_reset):
        self._obs_script = np.asarray(obs_script, dtype=np.uint8)
        self._rewards = np.asarray(rewards, dtype=np.float32)
        self._reset = np.asarray(true_reset, dtype=np.bool_)
        self.obs = self._obs_script[0]
        self.t = 0

    def __call__(self, action):
        t = self.t
        self.t += 1
        self.obs = self._obs_script[t + 1]
        raw = (self._rewards[t] * 5.0).astype(np.float32)  # distinct unclipped reward
        extra = {"reward_raw": raw, "fired": np.zeros_like(np.asarray(action), dtype=np.bool_)}
        return self.obs, self._rewards[t], np.zeros_like(self._reset[t]), self._reset[t], extra


def _run_rollout(agent, params, obs_script, rewards, true_reset, prev_reset0):
    T = int(rewards.shape[0])
    N = int(rewards.shape[1])
    env = _ScriptedEnv(obs_script, rewards, true_reset)
    carry = RolloutCarry(
        hidden=agent.init_hidden(N),
        prev_action=jnp.zeros((N,), dtype=jnp.int32),
        prev_reward_clipped=jnp.zeros((N,), dtype=jnp.float32),
        prev_reset=jnp.asarray(prev_reset0, dtype=bool),
    )
    rollout, _, _ = collect_rollout(env, agent, params, carry, T, jax.random.PRNGKey(7))
    return rollout


def test_replay_reconstructs_live_hidden_logits_value_and_key():
    agent = _tiny_agent()
    T, N = 6, 1
    params = _init_params(agent, N)
    obs_script = _stacked_obs(T + 1, N, seed=3)
    rewards = jnp.asarray(np.linspace(-1.0, 1.0, T, dtype=np.float32)[:, None])
    # A true episode reset occurs so that stored reset_mask has a reset at t=3.
    true_reset = np.zeros((T, N), dtype=np.bool_)
    true_reset[2, 0] = True  # next reset -> stored reset_mask[3] = True
    rollout = _run_rollout(agent, params, obs_script, rewards, true_reset, prev_reset0=[True])

    reset_mask = np.asarray(rollout.reset_mask)
    start = int(np.flatnonzero(reset_mask[:, 0])[1])  # the mid-block reset
    assert start == 3
    end = T

    # The live per-timestep outputs: the exact teacher labels admission stores.
    live_out, _ = agent.unroll(
        params,
        rollout.obs,
        rollout.prev_action,
        rollout.prev_reward_clipped,
        rollout.reset_mask,
        rollout.h0,
    )

    sl = slice(start, end)
    rec = make_record(
        init_stack=np.zeros((4, 84, 84), dtype=np.uint8),  # overwritten below
        new_frames=np.zeros((end - start, 84, 84), dtype=np.uint8),
        actions=np.asarray(rollout.action)[sl, 0],
        prev_action=np.asarray(rollout.prev_action)[sl, 0],
        prev_reward_clipped=np.asarray(rollout.prev_reward_clipped)[sl, 0],
        rewards_clipped=np.asarray(rollout.reward)[sl, 0],
        rewards_raw=np.asarray(rollout.reward_raw)[sl, 0],
        ppo_mask=np.asarray(rollout.ppo_mask)[sl, 0],
        reset_mask=reset_mask[sl, 0],
        teacher_logits=np.zeros((end - start, 18), dtype=np.float32),
        teacher_value=np.zeros((end - start,), dtype=np.float32),
        key_anchor=np.zeros((end - start, 128), dtype=np.float32),
        causal_contrib=np.zeros((end - start,), dtype=np.float32),
        credit_trace=np.zeros((end - start,), dtype=np.float32),
        adv_mag=np.zeros((end - start,), dtype=np.float32),
        td_mag=np.zeros((end - start,), dtype=np.float32),
        surprise=np.zeros((end - start,), dtype=np.float32),
        teacher_entropy=np.zeros((end - start,), dtype=np.float32),
        episode_id=0,
        chunk_index=0,
    )
    # Build the stored frame fields from the true rollout observations.
    from tfns.memory.record import frames_from_obs

    init_stack, new_frames = frames_from_obs(np.asarray(rollout.obs)[sl, 0])
    rec.init_stack = init_stack
    rec.new_frames = new_frames

    # Frame round-trip is exact for a proper sliding stack.
    np.testing.assert_array_equal(reconstruct_obs(rec), np.asarray(rollout.obs)[sl, 0])

    obs_rep = jnp.asarray(reconstruct_obs(rec), dtype=jnp.float32)[:, None, ...]
    replay_out, _ = agent.unroll(
        params,
        obs_rep,
        jnp.asarray(rec.prev_action, dtype=jnp.int32)[:, None],
        jnp.asarray(rec.prev_reward_clipped, dtype=jnp.float32)[:, None],
        jnp.asarray(rec.reset_mask, dtype=bool)[:, None],
        agent.init_hidden(1),
    )

    np.testing.assert_allclose(
        np.asarray(replay_out.logits[:, 0]), np.asarray(live_out.logits[sl, 0]), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(replay_out.value[:, 0]), np.asarray(live_out.value[sl, 0]), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(replay_out.q_key[:, 0]), np.asarray(live_out.q_key[sl, 0]), rtol=1e-4, atol=1e-4
    )
    np.testing.assert_allclose(
        np.asarray(replay_out.h_next[:, 0]), np.asarray(live_out.h_next[sl, 0]), rtol=1e-4, atol=1e-4
    )

    # Negative control: feeding the action/reward at t (the old buggy alignment)
    # as recurrent inputs must NOT reproduce the live trajectory.
    buggy_out, _ = agent.unroll(
        params,
        obs_rep,
        jnp.asarray(rec.actions, dtype=jnp.int32)[:, None],
        jnp.asarray(rec.rewards_clipped, dtype=jnp.float32)[:, None],
        jnp.asarray(rec.reset_mask, dtype=bool)[:, None],
        agent.init_hidden(1),
    )
    assert not np.allclose(
        np.asarray(buggy_out.logits[:, 0]), np.asarray(live_out.logits[sl, 0]), atol=1e-4
    )


def test_raw_reward_is_unclipped_in_record():
    agent = _tiny_agent()
    T, N = 4, 1
    params = _init_params(agent, N)
    obs_script = _stacked_obs(T + 1, N, seed=5)
    rewards = jnp.asarray(np.array([[1.0], [-1.0], [1.0], [0.0]], dtype=np.float32))
    rollout = _run_rollout(agent, params, obs_script, rewards, np.zeros((T, N), bool), [True])
    raw = np.asarray(rollout.reward_raw)[:, 0]
    clipped = np.asarray(rollout.reward)[:, 0]
    # raw reward is 5x the clipped script -> magnitude exceeds the clip range.
    assert np.any(np.abs(raw) > 1.0)
    assert not np.allclose(raw, clipped)


def _minimal_state(agent, params, memory):
    return ContinualState(
        params=params,
        opt_state=None,
        ema_params=params,
        bases={},
        memory=memory,
        predictor_params=None,
        predictor_opt_state=None,
        detector_state=None,
        adapter_dormant=jnp.ones((int(agent.adapter_config.num_adapters),), dtype=bool),
        robust_stats={},
        protected_clusters=[],
        rng=jax.random.PRNGKey(0),
        block_index=0,
    )


def test_admission_drops_mid_episode_leading_fragments():
    agent = _tiny_agent()
    T, N = 6, 2
    params = _init_params(agent, N)
    obs_script = _stacked_obs(T + 1, N, seed=9)
    rewards = jnp.zeros((T, N), dtype=jnp.float32)
    # env 0: no reset all block -> entirely mid-episode, nothing admissible.
    # env 1: reset so reset_mask[3]=True -> fragment [3:T) admissible, [0:3) dropped.
    true_reset = np.zeros((T, N), dtype=np.bool_)
    true_reset[2, 1] = True
    rollout = _run_rollout(
        agent, params, obs_script, rewards, true_reset, prev_reset0=[False, False]
    )

    memory = SequenceMemoryBank(MemoryConfig(byte_budget=1 << 26, min_per_cluster=0, max_admit_per_block=32))
    state = _minimal_state(agent, params, memory)
    zeros = jnp.zeros((T, N), dtype=jnp.float32)
    admitted = _admit_rollout_memories(state, agent, rollout, zeros, zeros, zeros, zeros, zeros, TFNSConfig())

    assert admitted >= 1
    for rec in memory.records():
        # Every admitted fragment must begin at a true episode reset.
        assert bool(rec.reset_mask[0]) is True
    # env 0 contributed nothing (no reset in its block).
    assert all(seq_len(rec) <= T - 3 for rec in memory.records())


def test_ppo_ratio_excludes_forced_transitions():
    agent = _tiny_agent()
    B, L = 1, 4
    params = _init_params(agent, B)
    obs = jnp.asarray(_stacked_obs(L, B, seed=11)).swapaxes(0, 1)  # (B, L, ...)
    prev_action = jnp.zeros((B, L), dtype=jnp.int32)
    prev_reward = jnp.zeros((B, L), dtype=jnp.float32)
    reset = jnp.zeros((B, L), dtype=bool)
    action = jnp.zeros((B, L), dtype=jnp.int32)
    old_logprob = jnp.zeros((B, L), dtype=jnp.float32)
    value = jnp.zeros((B, L), dtype=jnp.float32)
    adv = jnp.asarray(np.array([[1.0, 2.0, 3.0, 4.0]], dtype=np.float32))
    ret = jnp.ones((B, L), dtype=jnp.float32)
    cfg = TFNSConfig(ppo=PPOConfig(), aux=AuxConfig())

    def make_mb(forced):
        return SequenceMinibatch(
            obs=obs,
            prev_action=prev_action,
            prev_reward_clipped=prev_reward,
            action=action,
            old_logprob=old_logprob,
            value=value,
            reward=jnp.zeros((B, L), dtype=jnp.float32),
            adv=adv,
            ret=ret,
            ppo_mask=jnp.zeros((B, L), dtype=bool),
            reset_mask=reset,
            h0_chunk=agent.init_hidden(B),
            valid_mask=jnp.ones((B, L), dtype=bool),
            forced_mask=forced,
        )

    none_forced = make_mb(jnp.zeros((B, L), dtype=bool))
    last_forced = make_mb(jnp.asarray(np.array([[False, False, False, True]], dtype=np.bool_)))

    (_, aux_none) = ppo_loss(params, agent, none_forced, cfg)
    (_, aux_forced) = ppo_loss(params, agent, last_forced, cfg)
    # Forcing the last transition removes it from the policy-gradient statistics,
    # so pg_loss / approx_kl change relative to using all transitions.
    assert not np.isclose(float(aux_none["pg_loss"]), float(aux_forced["pg_loss"]))
