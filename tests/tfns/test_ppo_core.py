from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from tfns.config import AdapterConfig, AuxConfig, ModelConfig, PPOConfig, TFNSConfig
from tfns.model.agent import RecurrentAgent
from tfns.ppo.losses import ppo_loss
from tfns.ppo.rollout import (
    RolloutBatch,
    RolloutCarry,
    SequenceMinibatch,
    categorical_log_prob,
    collect_rollout,
    compute_gae,
    make_sequence_minibatches,
    reconstruct_hidden,
)


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


def _obs(T: int, B: int, offset: int = 0):
    base = jnp.arange(T * B, dtype=jnp.uint8).reshape(T, B, 1, 1, 1)
    return jnp.broadcast_to((base + offset).astype(jnp.uint8), (T, B, 84, 84, 4))


def _init_agent(agent: RecurrentAgent, B: int = 2):
    obs = _obs(1, B)[0]
    prev_action = jnp.zeros((B,), dtype=jnp.int32)
    prev_reward = jnp.zeros((B,), dtype=jnp.float32)
    reset = jnp.zeros((B,), dtype=bool)
    h0 = agent.init_hidden(B)
    variables = agent.init(jax.random.PRNGKey(13), obs, prev_action, prev_reward, reset, h0)
    return variables["params"]


def _manual_gae(reward, value, done, last_value, gamma, lam):
    reward = np.asarray(reward, dtype=np.float32)
    value = np.asarray(value, dtype=np.float32)
    done = np.asarray(done, dtype=bool)
    adv = np.zeros_like(reward, dtype=np.float32)
    gae = np.zeros_like(last_value, dtype=np.float32)
    next_value = np.asarray(last_value, dtype=np.float32)
    for t in reversed(range(reward.shape[0])):
        nt = 1.0 - done[t].astype(np.float32)
        delta = reward[t] + gamma * next_value * nt - value[t]
        gae = delta + gamma * lam * nt * gae
        adv[t] = gae
        next_value = value[t]
    return adv, adv + value


def test_gae_correctness_and_ppo_bootstrap_cut():
    reward = jnp.array([[1.0], [1.0], [1.0]], dtype=jnp.float32)
    value = jnp.array([[0.5], [0.25], [0.75]], dtype=jnp.float32)
    ppo_mask = jnp.array([[False], [True], [False]])
    last_value = jnp.array([10.0], dtype=jnp.float32)
    last_done = jnp.array([False])
    adv, ret = compute_gae(reward, value, ppo_mask, last_value, last_done, 0.9, 0.8)

    expected_adv, expected_ret = _manual_gae(reward, value, ppo_mask, last_value, 0.9, 0.8)
    np.testing.assert_allclose(np.asarray(adv), expected_adv, rtol=1e-5, atol=1e-5)
    np.testing.assert_allclose(np.asarray(ret), expected_ret, rtol=1e-5, atol=1e-5)

    no_cut, _ = compute_gae(
        reward,
        value,
        jnp.array([[False], [False], [False]]),
        last_value,
        last_done,
        0.9,
        0.8,
    )
    assert float(adv[1, 0]) != float(no_cut[1, 0])


def test_terminal_semantics_life_loss_does_not_reset_recurrence():
    agent = _tiny_agent()
    B, T = 1, 5
    params = _init_agent(agent, B)
    obs = _obs(T, B, offset=3)
    prev_action = jnp.array([[0], [1], [2], [0], [1]], dtype=jnp.int32)
    prev_reward = jnp.zeros((T, B), dtype=jnp.float32)
    ppo_mask = jnp.array([[False], [False], [True], [False], [False]])
    reset_mask = jnp.array([[False], [False], [False], [True], [False]])
    values = jnp.zeros((T, B), dtype=jnp.float32)
    reward = jnp.ones((T, B), dtype=jnp.float32)
    adv, _ = compute_gae(
        reward,
        values,
        ppo_mask,
        jnp.zeros((B,), dtype=jnp.float32),
        jnp.zeros((B,), dtype=bool),
        0.99,
        0.95,
    )
    assert np.isclose(float(adv[2, 0]), 1.0, atol=1e-5)

    h0 = agent.init_hidden(B) + 0.25
    out, _ = agent.unroll(params, obs, prev_action, prev_reward, reset_mask, h0)
    h_before_life = out.h_next[1]
    single_life = agent.apply(
        {"params": params},
        obs[2],
        prev_action[2],
        prev_reward[2],
        reset_mask[2],
        h_before_life,
    )
    np.testing.assert_allclose(np.asarray(out.h_next[2]), np.asarray(single_life.h_next), atol=1e-6)

    fresh_reset = agent.apply(
        {"params": params},
        obs[3],
        prev_action[3],
        prev_reward[3],
        reset_mask[3],
        jnp.zeros_like(h0),
    )
    np.testing.assert_allclose(np.asarray(out.h_next[3]), np.asarray(fresh_reset.h_next), atol=1e-6)


def test_burn_in_reconstruction_ignores_corrupt_stored_hidden():
    agent = _tiny_agent()
    B, T, burn_in = 1, 6, 4
    params = _init_agent(agent, B)
    obs = _obs(T, B, offset=11)
    prev_action = (jnp.arange(T, dtype=jnp.int32)[:, None] % 3)
    prev_reward = jnp.linspace(-1.0, 1.0, T, dtype=jnp.float32)[:, None]
    reset = jnp.zeros((T, B), dtype=bool)
    h0 = agent.init_hidden(B)

    full_out, _ = agent.unroll(params, obs, prev_action, prev_reward, reset, h0)
    h_recon = reconstruct_hidden(
        agent,
        params,
        obs[:burn_in],
        prev_action[:burn_in],
        prev_reward[:burn_in],
        reset[:burn_in],
    )
    protected_out, _ = agent.unroll(
        params,
        obs[burn_in:],
        prev_action[burn_in:],
        prev_reward[burn_in:],
        reset[burn_in:],
        h_recon,
    )
    np.testing.assert_allclose(
        np.asarray(protected_out.logits),
        np.asarray(full_out.logits[burn_in:]),
        rtol=1e-3,
        atol=1e-3,
    )
    np.testing.assert_allclose(
        np.asarray(protected_out.value),
        np.asarray(full_out.value[burn_in:]),
        rtol=1e-3,
        atol=1e-3,
    )

    corrupt_stored_hidden = jnp.ones_like(h_recon) * 999.0
    h_recon_again = reconstruct_hidden(
        agent,
        params,
        obs[:burn_in],
        prev_action[:burn_in],
        prev_reward[:burn_in],
        reset[:burn_in],
        h_init=None,
    )
    protected_again, _ = agent.unroll(
        params,
        obs[burn_in:],
        prev_action[burn_in:],
        prev_reward[burn_in:],
        reset[burn_in:],
        h_recon_again,
    )
    assert not np.allclose(np.asarray(corrupt_stored_hidden), np.asarray(h_recon_again))
    np.testing.assert_allclose(np.asarray(h_recon_again), np.asarray(h_recon), atol=0.0)
    np.testing.assert_allclose(
        np.asarray(protected_again.logits),
        np.asarray(protected_out.logits),
        rtol=0.0,
        atol=0.0,
    )


def test_sequence_minibatches_preserve_chunks_and_h0_boundaries():
    T, N, H, seq_chunk = 4, 2, 3, 2
    h0 = jnp.arange(N * H, dtype=jnp.float32).reshape(N, H)
    hidden_after = jnp.arange(T * N * H, dtype=jnp.float32).reshape(T, N, H) + 10.0
    rollout = RolloutBatch(
        obs=_obs(T, N),
        prev_action=jnp.tile(jnp.arange(T, dtype=jnp.int32)[:, None], (1, N)),
        prev_reward_clipped=jnp.zeros((T, N), dtype=jnp.float32),
        action=jnp.tile(jnp.arange(T, dtype=jnp.int32)[:, None], (1, N)),
        logprob=jnp.zeros((T, N), dtype=jnp.float32),
        value=jnp.zeros((T, N), dtype=jnp.float32),
        reward=jnp.ones((T, N), dtype=jnp.float32),
        ppo_mask=jnp.zeros((T, N), dtype=bool),
        reset_mask=jnp.zeros((T, N), dtype=bool),
        last_value=jnp.zeros((N,), dtype=jnp.float32),
        last_ppo_done=jnp.zeros((N,), dtype=bool),
        h0=h0,
        hidden_after=hidden_after,
        last_obs=_obs(1, N, offset=99)[0],
        last_reset=jnp.zeros((N,), dtype=bool),
    )
    adv = jnp.ones((T, N), dtype=jnp.float32)
    ret = jnp.ones((T, N), dtype=jnp.float32) * 2.0
    minibatches = list(make_sequence_minibatches(rollout, adv, ret, seq_chunk, jax.random.PRNGKey(0), minibatch_size=2))

    seen = []
    for mb in minibatches:
        assert mb.obs.shape[1] == seq_chunk
        assert mb.action.ndim == 2
        seen.extend(np.asarray(mb.seq_id).astype(int).tolist())
        for i in range(int(mb.seq_id.shape[0])):
            start = int(mb.chunk_start[i])
            env = int(mb.env_index[i])
            expected_actions = np.arange(start, start + seq_chunk, dtype=np.int32)
            np.testing.assert_array_equal(np.asarray(mb.action[i]), expected_actions)
            expected_h = h0[env] if start == 0 else hidden_after[start - 1, env]
            np.testing.assert_allclose(np.asarray(mb.h0_chunk[i]), np.asarray(expected_h))

    assert sorted(seen) == list(range((T // seq_chunk) * N))


def test_collect_rollout_uses_distinct_shifted_reset_and_ppo_masks():
    agent = _tiny_agent()
    T, N = 3, 2
    params = _init_agent(agent, N)
    obs_script = _obs(T + 1, N, offset=31)
    rewards = jnp.array([[1.0, 0.0], [-1.0, 0.5], [0.0, 1.0]], dtype=jnp.float32)
    ppo_done = jnp.array([[False, True], [True, False], [False, False]])
    true_reset = jnp.array([[False, False], [True, False], [False, True]])

    class ScriptedEnv:
        def __init__(self):
            self.obs = obs_script[0]
            self.t = 0

        def __call__(self, action):
            assert action.shape == (N,)
            t = self.t
            self.t += 1
            self.obs = obs_script[t + 1]
            return self.obs, rewards[t], ppo_done[t], true_reset[t], {"t": t}

    carry = RolloutCarry(
        hidden=agent.init_hidden(N),
        prev_action=jnp.zeros((N,), dtype=jnp.int32),
        prev_reward_clipped=jnp.zeros((N,), dtype=jnp.float32),
        prev_reset=jnp.array([False, True]),
    )
    rollout, new_carry, info = collect_rollout(
        ScriptedEnv(),
        agent,
        params,
        carry,
        T,
        jax.random.PRNGKey(9),
    )

    assert rollout.obs.shape == (T, N, 84, 84, 4)
    np.testing.assert_array_equal(np.asarray(rollout.ppo_mask), np.asarray(ppo_done))
    expected_reset_inputs = jnp.concatenate([carry.prev_reset[None], true_reset[:-1]], axis=0)
    np.testing.assert_array_equal(np.asarray(rollout.reset_mask), np.asarray(expected_reset_inputs))
    np.testing.assert_array_equal(np.asarray(new_carry.prev_reset), np.asarray(true_reset[-1]))
    assert len(info["extras"]) == T


def test_collect_rollout_records_forced_exec_action_and_logprob():
    agent = _tiny_agent()
    T, N = 2, 2
    params = _init_agent(agent, N)
    obs_script = _obs(T + 1, N, offset=51)
    rewards = jnp.zeros((T, N), dtype=jnp.float32)
    ppo_done = jnp.zeros((T, N), dtype=bool)
    true_reset = jnp.zeros((T, N), dtype=bool)
    fired_script = [
        np.array([True, False], dtype=np.bool_),
        np.array([False, True], dtype=np.bool_),
    ]

    class FireEnv:
        def __init__(self):
            self.obs = obs_script[0]
            self.t = 0
            self.exec_actions = []

        def __call__(self, action):
            action = np.asarray(action, dtype=np.int32)
            t = self.t
            self.t += 1
            fired = fired_script[t]
            exec_action = action.copy()
            exec_action[fired] = 1
            self.exec_actions.append(exec_action.copy())
            self.obs = obs_script[t + 1]
            return (
                self.obs,
                rewards[t],
                ppo_done[t],
                true_reset[t],
                {"fired": fired.copy(), "exec_action": exec_action.copy()},
            )

    env = FireEnv()
    carry = RolloutCarry(
        hidden=agent.init_hidden(N),
        prev_action=jnp.zeros((N,), dtype=jnp.int32),
        prev_reward_clipped=jnp.zeros((N,), dtype=jnp.float32),
        prev_reset=jnp.zeros((N,), dtype=bool),
    )
    rollout, new_carry, _info = collect_rollout(
        env,
        agent,
        params,
        carry,
        T,
        jax.random.PRNGKey(17),
    )

    expected_actions = np.stack(env.exec_actions, axis=0)
    np.testing.assert_array_equal(np.asarray(rollout.action), expected_actions)
    np.testing.assert_array_equal(np.asarray(rollout.prev_action[1]), expected_actions[0])
    np.testing.assert_array_equal(np.asarray(new_carry.prev_action), expected_actions[-1])

    # Forced FIRE transitions must be flagged so the PPO importance-ratio loss
    # can exclude them: their behavior probability is 1, not pi_theta(FIRE).
    expected_forced = np.stack(fired_script, axis=0)
    np.testing.assert_array_equal(np.asarray(rollout.forced_mask), expected_forced)

    outputs, _ = agent.unroll(
        params,
        rollout.obs,
        rollout.prev_action,
        rollout.prev_reward_clipped,
        rollout.reset_mask,
        rollout.h0,
    )
    expected_logprob = categorical_log_prob(outputs.logits, rollout.action)
    np.testing.assert_allclose(
        np.asarray(rollout.logprob),
        np.asarray(expected_logprob),
        rtol=1e-6,
        atol=1e-6,
    )


def test_ppo_loss_runs_and_is_differentiable():
    agent = _tiny_agent()
    B, L = 2, 3
    params = _init_agent(agent, B)
    obs = _obs(L, B, offset=21)
    prev_action = jnp.zeros((L, B), dtype=jnp.int32)
    prev_reward = jnp.zeros((L, B), dtype=jnp.float32)
    reset = jnp.zeros((L, B), dtype=bool)
    h0 = agent.init_hidden(B)
    out, _ = agent.unroll(params, obs, prev_action, prev_reward, reset, h0)
    action = jnp.argmax(out.logits, axis=-1).astype(jnp.int32)
    logprob = jax.nn.log_softmax(out.logits, axis=-1)
    old_logprob = jnp.take_along_axis(logprob, action[..., None], axis=-1)[..., 0]

    mb = SequenceMinibatch(
        obs=jnp.swapaxes(obs, 0, 1),
        prev_action=jnp.swapaxes(prev_action, 0, 1),
        prev_reward_clipped=jnp.swapaxes(prev_reward, 0, 1),
        action=jnp.swapaxes(action, 0, 1),
        old_logprob=jnp.swapaxes(old_logprob, 0, 1),
        value=jnp.swapaxes(out.value, 0, 1),
        reward=jnp.zeros((B, L), dtype=jnp.float32),
        adv=jnp.ones((B, L), dtype=jnp.float32),
        ret=jnp.swapaxes(out.value, 0, 1) + 0.5,
        ppo_mask=jnp.zeros((B, L), dtype=bool),
        reset_mask=jnp.swapaxes(reset, 0, 1),
        h0_chunk=h0,
        valid_mask=jnp.ones((B, L), dtype=bool),
    )
    cfg = TFNSConfig(ppo=PPOConfig(), aux=AuxConfig())

    (loss, aux), grads = jax.value_and_grad(lambda p: ppo_loss(p, agent, mb, cfg), has_aux=True)(params)
    assert bool(jnp.isfinite(loss))
    assert bool(jnp.isfinite(aux["clipfrac"]))
    assert bool(jnp.isfinite(aux["approx_kl"]))
    param_leaves = jax.tree_util.tree_leaves(params)
    grad_leaves = jax.tree_util.tree_leaves(grads)
    assert len(param_leaves) == len(grad_leaves)
    assert all(bool(jnp.all(jnp.isfinite(g))) for g in grad_leaves)
