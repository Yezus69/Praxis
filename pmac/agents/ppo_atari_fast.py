"""Fast envpool-XLA Atari PPO rollouts."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pmac.agents.atari_net import atari_apply, init_atari
from pmac.agents.ppo_atari import (
    AtariPPOConfig,
    _categorical_log_prob,
    _flatten_rollout,
    _learning_rate,
    _validate_config,
    gae,
    ppo_update,
)
from pmac.envs.atari_envpool import EpisodeReturnTracker, make_train_env_xla


@dataclass(frozen=True)
class FastPPOConfig(AtariPPOConfig):
    """Atari PPO config tuned for envpool-XLA throughput."""

    num_envs: int = 256


class XlaRollout(NamedTuple):
    obs: jnp.ndarray
    actions: jnp.ndarray
    logprobs: jnp.ndarray
    values: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray


def _make_rollout_fn(step_env, num_steps: int):
    """Build the single jitted scan that contains envpool's XLA step."""

    @jax.jit
    def rollout(params, handle, obs, game_onehot, rng):
        def step(carry, _):
            handle, obs, rng = carry
            rng, action_key = jax.random.split(rng)
            logits, value = atari_apply(params, obs, game_onehot)
            action = jax.random.categorical(action_key, logits, axis=-1).astype(jnp.int32)
            logprob = _categorical_log_prob(logits, action)

            handle, ts = step_env(handle, action)
            next_obs, reward, terminated, truncated, info = ts
            done = jnp.logical_or(terminated, truncated)
            _ = info
            row = XlaRollout(
                obs=obs,
                actions=action,
                logprobs=logprob,
                values=value,
                rewards=jnp.asarray(reward, dtype=jnp.float32),
                dones=done.astype(jnp.float32),
            )
            return (handle, next_obs, rng), row

        (handle, obs, rng), traj = jax.lax.scan(
            step,
            (handle, obs, rng),
            None,
            length=int(num_steps),
        )
        return handle, obs, rng, traj

    return rollout


@jax.jit
def _last_value(params, obs, game_onehot):
    return atari_apply(params, obs, game_onehot)[1]


def _completed_returns_from_rollout(traj: XlaRollout, tracker: EpisodeReturnTracker) -> list[float]:
    rewards = np.asarray(jax.device_get(traj.rewards), dtype=np.float32)
    dones = np.asarray(jax.device_get(traj.dones), dtype=bool)
    completed: list[float] = []
    for step in range(int(rewards.shape[0])):
        tracker.returns += rewards[step]
        done = dones[step]
        completed.extend(tracker.returns[done].astype(float).tolist())
        tracker.returns[done] = 0.0
    return completed


def _mean_or_previous(recent_returns: deque[float], previous: float) -> float:
    if not recent_returns:
        return float(previous)
    return float(np.mean(np.asarray(recent_returns, dtype=np.float32)))


def train_ppo_atari_fast(
    game,
    game_id,
    n_games,
    cfg,
    seed,
    init_params=None,
) -> dict:
    """Train one Atari game with a single jitted envpool-XLA rollout scan."""
    cfg = cfg or FastPPOConfig()
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

    env = make_train_env_xla(str(game), int(cfg.num_envs), int(seed))
    handle, recv, send, step_env = env.xla()
    _ = send
    env.async_reset()
    handle, ts0 = recv(handle)
    next_obs, _, _, _, _ = ts0
    rollout = _make_rollout_fn(step_env, int(cfg.num_steps))

    tracker = EpisodeReturnTracker(int(cfg.num_envs))
    recent_returns: deque[float] = deque(maxlen=100)
    returns_curve: list[float] = []
    last_curve_value = 0.0
    warm_steps = 0
    warm_seconds = 0.0
    total_seconds = 0.0

    for update in range(1, num_updates + 1):
        update_start = time.perf_counter()
        handle, next_obs, rng, traj = rollout(params, handle, next_obs, game_onehot, rng)
        completed_this_update = _completed_returns_from_rollout(traj, tracker)
        if completed_this_update:
            recent_returns.extend(completed_this_update)
        last_curve_value = _mean_or_previous(recent_returns, last_curve_value)
        returns_curve.append(last_curve_value)

        last_value = _last_value(params, next_obs, game_onehot)
        advantages, returns = gae(
            traj.rewards,
            traj.dones,
            traj.values,
            last_value,
            float(cfg.gamma),
            float(cfg.gae_lambda),
        )
        batch = _flatten_rollout(
            traj.obs,
            traj.actions,
            traj.logprobs,
            advantages,
            returns,
            traj.values,
            batch_size,
        )
        params, opt_state, rng, metrics = ppo_update(
            params,
            opt_state,
            batch,
            game_onehot,
            rng,
            float(_learning_rate(cfg, update, num_updates)),
            int(cfg.update_epochs),
            int(cfg.num_minibatches),
            int(minibatch_size),
            float(cfg.clip_coef),
            float(cfg.vf_coef),
            float(cfg.ent_coef),
            float(cfg.max_grad_norm),
        )
        jax.block_until_ready(metrics)
        elapsed = time.perf_counter() - update_start
        total_seconds += elapsed
        if update > 1:
            warm_steps += batch_size
            warm_seconds += elapsed

    timesteps = int(num_updates * batch_size)
    if warm_steps > 0 and warm_seconds > 0.0:
        steps_per_sec = float(warm_steps / warm_seconds)
    elif total_seconds > 0.0:
        steps_per_sec = float(timesteps / total_seconds)
    else:
        steps_per_sec = 0.0

    final_return = float(returns_curve[-1]) if returns_curve else 0.0
    return {
        "params": params,
        "returns_curve": [float(v) for v in returns_curve],
        "final_return": final_return,
        "timesteps": timesteps,
        "steps_per_sec": steps_per_sec,
    }


__all__ = [
    "FastPPOConfig",
    "XlaRollout",
    "train_ppo_atari_fast",
]
