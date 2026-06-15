"""Bounded single-game PPO for gymnax MinAtar."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pmac.agents.ac_net import ac_apply, init_ac, set_action_masks
from pmac.envs.minatar_gymnax import GameSpec, pad_obs, vreset, vstep

warnings.filterwarnings("ignore")


@dataclass(frozen=True)
class PPOConfig:
    total_timesteps: int = 3_000_000
    num_envs: int = 64
    num_steps: int = 128
    update_epochs: int = 4
    num_minibatches: int = 8
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_eps: float = 0.2
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 2.5e-4
    anneal_lr: bool = True


class Transition(NamedTuple):
    obs: jnp.ndarray
    actions: jnp.ndarray
    logprobs: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    values: jnp.ndarray


class TrainBatch(NamedTuple):
    obs: jnp.ndarray
    actions: jnp.ndarray
    logprobs: jnp.ndarray
    advantages: jnp.ndarray
    targets: jnp.ndarray


def _validate_config(cfg: PPOConfig) -> int:
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


def _where_done_array(done, reset_value, step_value):
    selector = done.reshape(done.shape + (1,) * (step_value.ndim - done.ndim))
    return jnp.where(selector, reset_value, step_value)


def _where_done_tree(done, reset_tree, step_tree):
    return jax.tree_util.tree_map(
        lambda reset_value, step_value: _where_done_array(done, reset_value, step_value),
        reset_tree,
        step_tree,
    )


def _action_masks_for_game(game_spec: GameSpec, n_games: int, act_max: int):
    masks = jnp.ones((int(n_games), int(act_max)), dtype=jnp.float32)
    return masks.at[int(game_spec.game_id)].set(jnp.asarray(game_spec.mask, dtype=jnp.float32))


def _force_action_masks(params, action_masks):
    new_params = dict(params)
    new_params["action_masks"] = action_masks
    return new_params


def _compute_gae(traj: Transition, last_value, gamma: float, gae_lambda: float):
    def gae_step(carry, inputs):
        gae, next_value = carry
        reward, done, value = inputs
        not_done = 1.0 - done
        delta = reward + float(gamma) * next_value * not_done - value
        gae = delta + float(gamma) * float(gae_lambda) * not_done * gae
        return (gae, value), gae

    init = (jnp.zeros_like(last_value), last_value)
    _, advantages_rev = jax.lax.scan(
        gae_step,
        init,
        (traj.rewards[::-1], traj.dones[::-1], traj.values[::-1]),
    )
    advantages = advantages_rev[::-1]
    targets = advantages + traj.values
    return jax.lax.stop_gradient(advantages), jax.lax.stop_gradient(targets)


def _flatten_batch(traj: Transition, advantages, targets, batch_size: int):
    batch = TrainBatch(
        obs=traj.obs.reshape((batch_size,) + traj.obs.shape[2:]),
        actions=traj.actions.reshape((batch_size,)),
        logprobs=traj.logprobs.reshape((batch_size,)),
        advantages=advantages.reshape((batch_size,)),
        targets=targets.reshape((batch_size,)),
    )
    return jax.tree_util.tree_map(jax.lax.stop_gradient, batch)


def _make_minibatches(batch: TrainBatch, permutation, num_minibatches: int, minibatch_size: int):
    def shuffle(x):
        x = jnp.take(x, permutation, axis=0)
        return x.reshape((num_minibatches, minibatch_size) + x.shape[1:])

    return jax.tree_util.tree_map(shuffle, batch)


def _ppo_loss(params, batch: TrainBatch, game_onehot, clip_eps: float, vf_coef: float, ent_coef: float):
    logits, value = ac_apply(params, batch.obs, game_onehot)
    logprob = _categorical_log_prob(logits, batch.actions)
    ratio = jnp.exp(logprob - batch.logprobs)

    advantages = batch.advantages
    advantages = (advantages - jnp.mean(advantages)) / (jnp.std(advantages) + 1.0e-8)
    loss_actor_unclipped = ratio * advantages
    loss_actor_clipped = jnp.clip(ratio, 1.0 - float(clip_eps), 1.0 + float(clip_eps)) * advantages
    actor_loss = -jnp.mean(jnp.minimum(loss_actor_unclipped, loss_actor_clipped))
    value_loss = 0.5 * jnp.mean(jnp.square(value - batch.targets))
    entropy = jnp.mean(_categorical_entropy(logits))
    total_loss = actor_loss + float(vf_coef) * value_loss - float(ent_coef) * entropy
    return total_loss, jnp.asarray([actor_loss, value_loss, entropy], dtype=jnp.float32)


def train_ppo_single(game_spec, n_games, act_max, cfg, seed, init_params=None) -> dict:
    """Train one MinAtar game with a fixed-length, fully scanned PPO loop."""
    cfg = cfg or PPOConfig()
    num_updates = _validate_config(cfg)
    batch_size = int(cfg.num_envs) * int(cfg.num_steps)
    minibatch_size = batch_size // int(cfg.num_minibatches)
    c_in = int(getattr(game_spec, "c_max", game_spec.channels))
    n_games = int(n_games)
    act_max = int(act_max)
    action_masks = _action_masks_for_game(game_spec, n_games, act_max)
    game_onehot = jax.nn.one_hot(int(game_spec.game_id), n_games, dtype=jnp.float32)

    def lr_schedule(count):
        if not bool(cfg.anneal_lr):
            return float(cfg.lr)
        updates_per_outer = int(cfg.update_epochs) * int(cfg.num_minibatches)
        outer_update = count // updates_per_outer
        frac = 1.0 - outer_update.astype(jnp.float32) / float(num_updates)
        return float(cfg.lr) * frac

    tx = optax.chain(
        optax.clip_by_global_norm(float(cfg.max_grad_norm)),
        optax.adam(learning_rate=lr_schedule if bool(cfg.anneal_lr) else float(cfg.lr)),
    )

    rng = jax.random.PRNGKey(int(seed))
    rng, init_key, reset_key = jax.random.split(rng, 3)
    reset_keys = jax.random.split(reset_key, int(cfg.num_envs))
    raw_obs, env_state = vreset(game_spec.env, game_spec.params, reset_keys)
    obs = pad_obs(raw_obs, c_in)
    params = init_params
    if params is None:
        params = init_ac(init_key, c_in, n_games, act_max)
    params = set_action_masks(params, action_masks)
    opt_state = tx.init(params)

    def rollout(params, env_state, obs, running_return, rng):
        def step(carry, _):
            env_state, obs, running_return, rng = carry
            rng, action_key, step_key, reset_key = jax.random.split(rng, 4)
            logits, value = ac_apply(params, obs, game_onehot)
            actions = jax.random.categorical(action_key, logits, axis=-1).astype(jnp.int32)
            logprobs = _categorical_log_prob(logits, actions)

            step_keys = jax.random.split(step_key, int(cfg.num_envs))
            next_obs_raw, step_state, reward, done, _ = vstep(
                game_spec.env,
                game_spec.params,
                step_keys,
                env_state,
                actions,
            )
            reset_keys = jax.random.split(reset_key, int(cfg.num_envs))
            reset_obs_raw, reset_state = vreset(game_spec.env, game_spec.params, reset_keys)
            next_state = _where_done_tree(done, reset_state, step_state)
            next_obs = _where_done_array(done, pad_obs(reset_obs_raw, c_in), pad_obs(next_obs_raw, c_in))

            reward = reward.astype(jnp.float32)
            done_f = done.astype(jnp.float32)
            finished_return = running_return + reward
            completed_sum = jnp.sum(jnp.where(done, finished_return, 0.0))
            completed_count = jnp.sum(done_f)
            next_running_return = jnp.where(done, 0.0, finished_return)

            transition = Transition(
                obs=obs,
                actions=actions,
                logprobs=logprobs,
                rewards=reward,
                dones=done_f,
                values=value,
            )
            next_carry = (next_state, next_obs, next_running_return, rng)
            return next_carry, (transition, completed_sum, completed_count)

        init_carry = (env_state, obs, running_return, rng)
        (env_state, obs, running_return, rng), (traj, completed_sum, completed_count) = jax.lax.scan(
            step,
            init_carry,
            None,
            length=int(cfg.num_steps),
        )
        return (
            env_state,
            obs,
            running_return,
            rng,
            traj,
            jnp.sum(completed_sum),
            jnp.sum(completed_count),
        )

    def update_policy(params, opt_state, batch, rng):
        def epoch_step(carry, _):
            params, opt_state, rng = carry
            rng, perm_key = jax.random.split(rng)
            permutation = jax.random.permutation(perm_key, batch_size)
            minibatches = _make_minibatches(
                batch,
                permutation,
                int(cfg.num_minibatches),
                minibatch_size,
            )

            def minibatch_step(carry, minibatch):
                params, opt_state = carry
                loss_fn = lambda p: _ppo_loss(
                    p,
                    minibatch,
                    game_onehot,
                    cfg.clip_eps,
                    cfg.vf_coef,
                    cfg.ent_coef,
                )
                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
                updates, opt_state = tx.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
                params = _force_action_masks(params, action_masks)
                metrics = jnp.concatenate([jnp.asarray([loss], dtype=jnp.float32), aux], axis=0)
                return (params, opt_state), metrics

            (params, opt_state), metrics = jax.lax.scan(
                minibatch_step,
                (params, opt_state),
                minibatches,
            )
            return (params, opt_state, rng), jnp.mean(metrics, axis=0)

        (params, opt_state, _), metrics = jax.lax.scan(
            epoch_step,
            (params, opt_state, rng),
            None,
            length=int(cfg.update_epochs),
        )
        return params, opt_state, jnp.mean(metrics, axis=0)

    def train_loop(params, opt_state, env_state, obs, rng):
        def update_step(carry, _):
            params, opt_state, env_state, obs, running_return, return_estimate, seen_return, rng = carry
            rng, rollout_key, update_key = jax.random.split(rng, 3)
            (
                env_state,
                obs,
                running_return,
                rollout_key,
                traj,
                completed_sum,
                completed_count,
            ) = rollout(params, env_state, obs, running_return, rollout_key)
            _, last_value = ac_apply(params, obs, game_onehot)
            advantages, targets = _compute_gae(traj, last_value, cfg.gamma, cfg.gae_lambda)
            batch = _flatten_batch(traj, advantages, targets, batch_size)
            params, opt_state, _ = update_policy(params, opt_state, batch, update_key)

            update_return = jnp.where(
                completed_count > 0.0,
                completed_sum / jnp.maximum(completed_count, 1.0),
                return_estimate,
            )
            new_return_estimate = jnp.where(
                completed_count > 0.0,
                jnp.where(seen_return, 0.9 * return_estimate + 0.1 * update_return, update_return),
                return_estimate,
            )
            new_seen_return = jnp.logical_or(seen_return, completed_count > 0.0)
            next_carry = (
                params,
                opt_state,
                env_state,
                obs,
                running_return,
                new_return_estimate,
                new_seen_return,
                rng,
            )
            return next_carry, new_return_estimate

        init_carry = (
            params,
            opt_state,
            env_state,
            obs,
            jnp.zeros((int(cfg.num_envs),), dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(False),
            rng,
        )
        final_carry, returns_curve = jax.lax.scan(
            update_step,
            init_carry,
            None,
            length=num_updates,
        )
        return final_carry[0], returns_curve

    params, returns_curve = jax.jit(train_loop)(params, opt_state, env_state, obs, rng)
    returns_np = np.asarray(jax.device_get(returns_curve), dtype=np.float32)
    return {
        "params": params,
        "returns_curve": returns_np.astype(float).tolist(),
        "final_return": float(returns_np[-1]),
        "timesteps": int(num_updates * batch_size),
    }


__all__ = ["PPOConfig", "train_ppo_single"]
