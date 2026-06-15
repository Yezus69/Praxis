"""Bounded CleanRL-style PPO for envpool Atari."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pmac.agents.atari_net import atari_apply, init_atari
from pmac.envs.atari_envpool import ACT_DIM, EpisodeReturnTracker, make_train_env


@dataclass(frozen=True)
class AtariPPOConfig:
    total_timesteps: int = 5_000_000
    num_envs: int = 64
    num_steps: int = 128
    update_epochs: int = 4
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 2.5e-4
    anneal_lr: bool = True


class TrainBatch(NamedTuple):
    obs: jnp.ndarray
    actions: jnp.ndarray
    logprobs: jnp.ndarray
    advantages: jnp.ndarray
    returns: jnp.ndarray
    values: jnp.ndarray


def _validate_config(cfg: AtariPPOConfig) -> int:
    steps_per_update = int(cfg.num_envs) * int(cfg.num_steps)
    if steps_per_update <= 0:
        raise ValueError("num_envs*num_steps must be positive")
    num_updates = int(cfg.total_timesteps) // steps_per_update
    if num_updates <= 0:
        raise ValueError("total_timesteps must cover at least one PPO update")
    if int(cfg.update_epochs) <= 0:
        raise ValueError("update_epochs must be positive")
    if int(cfg.num_minibatches) <= 0:
        raise ValueError("num_minibatches must be positive")
    if steps_per_update % int(cfg.num_minibatches) != 0:
        raise ValueError("num_envs*num_steps must be divisible by num_minibatches")
    return num_updates


def _categorical_log_prob(logits, actions):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    return jnp.take_along_axis(log_probs, actions[..., None], axis=-1)[..., 0]


def _categorical_entropy(logits):
    log_probs = jax.nn.log_softmax(logits, axis=-1)
    probs = jnp.exp(log_probs)
    return -jnp.sum(probs * log_probs, axis=-1)


@jax.jit
def jit_policy(params, obs, game_onehot, rng):
    rng, action_key = jax.random.split(rng)
    logits, value = atari_apply(params, obs, game_onehot)
    actions = jax.random.categorical(action_key, logits, axis=-1).astype(jnp.int32)
    logprobs = _categorical_log_prob(logits, actions)
    return actions, logprobs, value, rng


@jax.jit
def jit_greedy_policy(params, obs, game_onehot):
    logits, value = atari_apply(params, obs, game_onehot)
    actions = jnp.argmax(logits, axis=-1).astype(jnp.int32)
    logprobs = _categorical_log_prob(logits, actions)
    return actions, logprobs, value


@jax.jit
def gae(rewards, dones, values, last_value, gamma: float, gae_lambda: float):
    rewards = jnp.asarray(rewards, dtype=jnp.float32)
    dones = jnp.asarray(dones, dtype=jnp.float32)
    values = jnp.asarray(values, dtype=jnp.float32)
    last_value = jnp.asarray(last_value, dtype=jnp.float32)

    def gae_step(carry, inputs):
        gae_acc, next_value = carry
        reward, done, value = inputs
        not_done = 1.0 - done
        delta = reward + gamma * next_value * not_done - value
        gae_acc = delta + gamma * gae_lambda * not_done * gae_acc
        return (gae_acc, value), gae_acc

    init = (jnp.zeros_like(last_value), last_value)
    _, advantages_rev = jax.lax.scan(
        gae_step,
        init,
        (rewards[::-1], dones[::-1], values[::-1]),
    )
    advantages = advantages_rev[::-1]
    returns = advantages + values
    return jax.lax.stop_gradient(advantages), jax.lax.stop_gradient(returns)


def _flatten_rollout(obs, actions, logprobs, advantages, returns, values, batch_size: int):
    batch = TrainBatch(
        obs=jnp.asarray(obs).reshape((batch_size,) + tuple(obs.shape[2:])),
        actions=jnp.asarray(actions, dtype=jnp.int32).reshape((batch_size,)),
        logprobs=jnp.asarray(logprobs, dtype=jnp.float32).reshape((batch_size,)),
        advantages=jnp.asarray(advantages, dtype=jnp.float32).reshape((batch_size,)),
        returns=jnp.asarray(returns, dtype=jnp.float32).reshape((batch_size,)),
        values=jnp.asarray(values, dtype=jnp.float32).reshape((batch_size,)),
    )
    return jax.tree_util.tree_map(jax.lax.stop_gradient, batch)


def _make_minibatches(batch: TrainBatch, permutation, num_minibatches: int, minibatch_size: int):
    def shuffle(x):
        x = jnp.take(x, permutation, axis=0)
        return x.reshape((num_minibatches, minibatch_size) + x.shape[1:])

    return jax.tree_util.tree_map(shuffle, batch)


def _ppo_loss(params, batch: TrainBatch, game_onehot, clip_coef: float, vf_coef: float, ent_coef: float):
    logits, new_values = atari_apply(params, batch.obs, game_onehot)
    new_logprobs = _categorical_log_prob(logits, batch.actions)
    entropy = _categorical_entropy(logits)
    logratio = new_logprobs - batch.logprobs
    ratio = jnp.exp(logratio)

    advantages = (batch.advantages - jnp.mean(batch.advantages)) / (jnp.std(batch.advantages) + 1.0e-8)
    pg_loss1 = -advantages * ratio
    pg_loss2 = -advantages * jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    pg_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))

    v_loss_unclipped = jnp.square(new_values - batch.returns)
    v_clipped = batch.values + jnp.clip(new_values - batch.values, -clip_coef, clip_coef)
    v_loss_clipped = jnp.square(v_clipped - batch.returns)
    v_loss = 0.5 * jnp.mean(jnp.maximum(v_loss_unclipped, v_loss_clipped))

    entropy_loss = jnp.mean(entropy)
    approx_kl = jnp.mean((ratio - 1.0) - logratio)
    clipfrac = jnp.mean((jnp.abs(ratio - 1.0) > clip_coef).astype(jnp.float32))
    loss = pg_loss + vf_coef * v_loss - ent_coef * entropy_loss
    aux = jnp.asarray([pg_loss, v_loss, entropy_loss, approx_kl, clipfrac], dtype=jnp.float32)
    return loss, aux


@partial(
    jax.jit,
    static_argnames=(
        "update_epochs",
        "num_minibatches",
        "minibatch_size",
        "clip_coef",
        "vf_coef",
        "ent_coef",
        "max_grad_norm",
    ),
)
def ppo_update(
    params,
    opt_state,
    batch: TrainBatch,
    game_onehot,
    rng,
    learning_rate: float,
    update_epochs: int,
    num_minibatches: int,
    minibatch_size: int,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
):
    batch_size = int(num_minibatches) * int(minibatch_size)
    tx = optax.chain(
        optax.clip_by_global_norm(float(max_grad_norm)),
        optax.adam(learning_rate=learning_rate),
    )

    def epoch_step(carry, _):
        params, opt_state, rng = carry
        rng, perm_key = jax.random.split(rng)
        permutation = jax.random.permutation(perm_key, batch_size)
        minibatches = _make_minibatches(batch, permutation, int(num_minibatches), int(minibatch_size))

        def minibatch_step(carry, minibatch):
            params, opt_state = carry
            loss_fn = lambda p: _ppo_loss(p, minibatch, game_onehot, clip_coef, vf_coef, ent_coef)
            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            updates, opt_state = tx.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            metrics = jnp.concatenate([jnp.asarray([loss], dtype=jnp.float32), aux], axis=0)
            return (params, opt_state), metrics

        (params, opt_state), metrics = jax.lax.scan(minibatch_step, (params, opt_state), minibatches)
        return (params, opt_state, rng), jnp.mean(metrics, axis=0)

    (params, opt_state, rng), metrics = jax.lax.scan(
        epoch_step,
        (params, opt_state, rng),
        None,
        length=int(update_epochs),
    )
    return params, opt_state, rng, jnp.mean(metrics, axis=0)


def _learning_rate(cfg: AtariPPOConfig, update: int, num_updates: int) -> float:
    if not bool(cfg.anneal_lr):
        return float(cfg.lr)
    frac = 1.0 - float(update - 1) / float(num_updates)
    return float(cfg.lr) * frac


def _mean_or_previous(recent_returns, previous: float) -> float:
    if len(recent_returns) == 0:
        return float(previous)
    return float(np.mean(np.asarray(recent_returns, dtype=np.float32)))


def train_ppo_atari(game, game_id, n_games, cfg, seed, init_params=None) -> dict:
    """Train one Atari game using bounded host envpool rollouts and jitted updates."""
    cfg = cfg or AtariPPOConfig()
    num_updates = _validate_config(cfg)
    batch_size = int(cfg.num_envs) * int(cfg.num_steps)
    minibatch_size = batch_size // int(cfg.num_minibatches)
    game_onehot = jax.nn.one_hot(int(game_id), int(n_games), dtype=jnp.float32)

    rng = jax.random.PRNGKey(int(seed))
    rng, init_key = jax.random.split(rng)
    params = init_params
    if params is None:
        params = init_atari(init_key, int(n_games))

    tx = optax.chain(
        optax.clip_by_global_norm(float(cfg.max_grad_norm)),
        optax.adam(learning_rate=float(cfg.lr)),
    )
    opt_state = tx.init(params)

    env = make_train_env(str(game), int(cfg.num_envs), int(seed))
    next_obs, _ = env.reset()
    next_obs = np.asarray(next_obs, dtype=np.uint8)
    tracker = EpisodeReturnTracker(int(cfg.num_envs))
    recent_returns = deque(maxlen=100)
    returns_curve: list[float] = []
    last_curve_value = 0.0

    obs_buf = np.zeros((int(cfg.num_steps), int(cfg.num_envs), 4, 84, 84), dtype=np.uint8)
    actions_buf = np.zeros((int(cfg.num_steps), int(cfg.num_envs)), dtype=np.int32)
    logprobs_buf = np.zeros((int(cfg.num_steps), int(cfg.num_envs)), dtype=np.float32)
    rewards_buf = np.zeros((int(cfg.num_steps), int(cfg.num_envs)), dtype=np.float32)
    dones_buf = np.zeros((int(cfg.num_steps), int(cfg.num_envs)), dtype=np.float32)
    values_buf = np.zeros((int(cfg.num_steps), int(cfg.num_envs)), dtype=np.float32)

    for update in range(1, num_updates + 1):
        completed_this_update: list[float] = []
        for step in range(int(cfg.num_steps)):
            obs_buf[step] = next_obs
            actions, logprobs, values, rng = jit_policy(params, next_obs, game_onehot, rng)
            actions_np = np.asarray(jax.device_get(actions), dtype=np.int32)
            logprobs_buf[step] = np.asarray(jax.device_get(logprobs), dtype=np.float32)
            values_buf[step] = np.asarray(jax.device_get(values), dtype=np.float32)
            actions_buf[step] = actions_np

            next_obs, rewards, terminated, truncated, info = env.step(actions_np)
            next_obs = np.asarray(next_obs, dtype=np.uint8)
            rewards = np.asarray(rewards, dtype=np.float32)
            terminated = np.asarray(terminated, dtype=bool)
            truncated = np.asarray(truncated, dtype=bool)
            done = np.logical_or(terminated, truncated)

            rewards_buf[step] = rewards
            dones_buf[step] = done.astype(np.float32)
            completed = tracker.update(rewards, terminated, truncated, info)
            completed_this_update.extend(completed)

        if completed_this_update:
            recent_returns.extend(completed_this_update)
        last_curve_value = _mean_or_previous(recent_returns, last_curve_value)
        returns_curve.append(last_curve_value)

        _, _, last_value = jit_greedy_policy(params, next_obs, game_onehot)
        advantages, returns = gae(
            jnp.asarray(rewards_buf),
            jnp.asarray(dones_buf),
            jnp.asarray(values_buf),
            last_value,
            float(cfg.gamma),
            float(cfg.gae_lambda),
        )
        batch = _flatten_rollout(
            obs_buf,
            actions_buf,
            logprobs_buf,
            advantages,
            returns,
            values_buf,
            batch_size,
        )
        lr = _learning_rate(cfg, update, num_updates)
        params, opt_state, rng, _ = ppo_update(
            params,
            opt_state,
            batch,
            game_onehot,
            rng,
            float(lr),
            int(cfg.update_epochs),
            int(cfg.num_minibatches),
            int(minibatch_size),
            float(cfg.clip_coef),
            float(cfg.vf_coef),
            float(cfg.ent_coef),
            float(cfg.max_grad_norm),
        )

    final_return = float(returns_curve[-1]) if returns_curve else 0.0
    return {
        "params": params,
        "returns_curve": [float(v) for v in returns_curve],
        "final_return": final_return,
        "timesteps": int(num_updates * batch_size),
    }


__all__ = [
    "AtariPPOConfig",
    "TrainBatch",
    "gae",
    "jit_greedy_policy",
    "jit_policy",
    "ppo_update",
    "train_ppo_atari",
]
