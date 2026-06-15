"""Continual MinAtar PPO: matched warm-start baseline versus PMA-C."""

from __future__ import annotations

import argparse
import json
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pmac.agents.ac_net import ac_apply, init_ac, set_action_masks
from pmac.agents.ppo_minatar import (
    PPOConfig,
    Transition,
    _categorical_log_prob,
    _compute_gae,
    _flatten_batch,
    _make_minibatches,
    _ppo_loss,
    _validate_config,
    _where_done_array,
    _where_done_tree,
)
from pmac.anchors import AnchorStore
from pmac.atlas import Atlas
from pmac.behavior_distance import kl_categorical
from pmac.checkpoint import ChampionStore
from pmac.conservation import AnchorBatch, conservation_loss
from pmac.continual import clip_global
from pmac.envs.minatar_gymnax import GAMES, GameSpec, make_games, pad_obs, vreset, vstep
from pmac.sentinels import SentinelStore

warnings.filterwarnings("ignore")


ALLOWED_MINATAR_ABLATIONS = {None, "none", "no_conservation", "no_replay"}


@dataclass(frozen=True)
class ContinualRLConfig:
    per_game_steps: int = 3_000_000
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
    eval_episodes: int = 64
    eval_horizon: int | None = None
    guard_coef: float = 1.0
    guard_batch: int = 512
    anchor_buffer_per_game: int = 4096
    guard_tolerance: float = 0.005
    value_coef: float = 1.0
    ablation: str | None = None


class GuardPool(NamedTuple):
    obs: jnp.ndarray
    game_onehot: jnp.ndarray
    teacher_logits: jnp.ndarray
    teacher_value: jnp.ndarray
    count: jnp.ndarray


class GuardBatch(NamedTuple):
    obs: jnp.ndarray
    game_onehot: jnp.ndarray
    teacher_logits: jnp.ndarray
    teacher_value: jnp.ndarray
    tolerance: jnp.ndarray
    weight: jnp.ndarray
    count: jnp.ndarray


class MinAtarAnchorBuffer(NamedTuple):
    obs: np.ndarray
    game_onehot: np.ndarray
    teacher_logits: np.ndarray
    teacher_value: np.ndarray


def _parse_csv(text):
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_seeds(text):
    return [int(part) for part in _parse_csv(text)]


def _parse_ablations(text):
    return [None if value == "none" else value for value in _parse_csv(text)]


def _to_ppo_config(cfg: ContinualRLConfig) -> PPOConfig:
    return PPOConfig(
        total_timesteps=int(cfg.per_game_steps),
        num_envs=int(cfg.num_envs),
        num_steps=int(cfg.num_steps),
        update_epochs=int(cfg.update_epochs),
        num_minibatches=int(cfg.num_minibatches),
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_eps=float(cfg.clip_eps),
        ent_coef=float(cfg.ent_coef),
        vf_coef=float(cfg.vf_coef),
        max_grad_norm=float(cfg.max_grad_norm),
        lr=float(cfg.lr),
        anneal_lr=bool(cfg.anneal_lr),
    )


def _validate_continual_config(cfg: ContinualRLConfig) -> int:
    if int(cfg.eval_episodes) <= 0:
        raise ValueError("eval_episodes must be positive")
    if int(cfg.guard_batch) <= 0:
        raise ValueError("guard_batch must be positive")
    if int(cfg.anchor_buffer_per_game) <= 0:
        raise ValueError("anchor_buffer_per_game must be positive")
    if cfg.eval_horizon is not None and int(cfg.eval_horizon) <= 0:
        raise ValueError("eval_horizon must be positive when set")
    return _validate_config(_to_ppo_config(cfg))


def _n_games(specs) -> int:
    return max(int(len(specs)), 1 + max(int(spec.game_id) for spec in specs))


def _act_max(specs) -> int:
    return max(int(spec.act_max) for spec in specs)


def _c_max(specs) -> int:
    return max(int(spec.c_max) for spec in specs)


def _action_masks_for_specs(specs, n_games: int, act_max: int):
    masks = jnp.zeros((int(n_games), int(act_max)), dtype=jnp.float32)
    for spec in specs:
        masks = masks.at[int(spec.game_id)].set(jnp.asarray(spec.mask, dtype=jnp.float32))
    return masks


def _force_action_masks(params, action_masks):
    return set_action_masks(params, action_masks)


def _empty_guard_pool(c_in: int, n_games: int, act_max: int) -> GuardPool:
    return GuardPool(
        obs=jnp.zeros((1, 10, 10, int(c_in)), dtype=jnp.float32),
        game_onehot=jnp.zeros((1, int(n_games)), dtype=jnp.float32),
        teacher_logits=jnp.zeros((1, int(act_max)), dtype=jnp.float32),
        teacher_value=jnp.zeros((1,), dtype=jnp.float32),
        count=jnp.asarray(0, dtype=jnp.int32),
    )


def _guard_pool_from_buffers(
    buffers: list[MinAtarAnchorBuffer],
    cfg: ContinualRLConfig,
    c_in: int,
    n_games: int,
    act_max: int,
    ablation,
) -> GuardPool:
    selected = [] if ablation == "no_conservation" else list(buffers)
    if ablation == "no_replay" and selected:
        selected = selected[-1:]
    if not selected:
        return _empty_guard_pool(c_in, n_games, act_max)

    max_rows = max(1, int(cfg.anchor_buffer_per_game) * len(selected))
    obs = np.zeros((max_rows, 10, 10, int(c_in)), dtype=np.float32)
    game_onehot = np.zeros((max_rows, int(n_games)), dtype=np.float32)
    teacher_logits = np.zeros((max_rows, int(act_max)), dtype=np.float32)
    teacher_value = np.zeros((max_rows,), dtype=np.float32)

    obs_cat = np.concatenate([buf.obs for buf in selected], axis=0)
    goh_cat = np.concatenate([buf.game_onehot for buf in selected], axis=0)
    logits_cat = np.concatenate([buf.teacher_logits for buf in selected], axis=0)
    value_cat = np.concatenate([buf.teacher_value for buf in selected], axis=0)
    count = min(int(obs_cat.shape[0]), max_rows)
    obs[:count] = obs_cat[:count]
    game_onehot[:count] = goh_cat[:count]
    teacher_logits[:count] = logits_cat[:count]
    teacher_value[:count] = value_cat[:count]

    return GuardPool(
        obs=jnp.asarray(obs),
        game_onehot=jnp.asarray(game_onehot),
        teacher_logits=jnp.asarray(teacher_logits),
        teacher_value=jnp.asarray(teacher_value),
        count=jnp.asarray(count, dtype=jnp.int32),
    )


def _sample_guard_batch(pool: GuardPool, key, batch_size: int, tolerance: float) -> GuardBatch:
    safe_count = jnp.maximum(pool.count, jnp.asarray(1, dtype=jnp.int32))
    idx = jax.random.randint(key, (int(batch_size),), 0, safe_count, dtype=jnp.int32)
    return GuardBatch(
        obs=jnp.take(pool.obs, idx, axis=0),
        game_onehot=jnp.take(pool.game_onehot, idx, axis=0),
        teacher_logits=jnp.take(pool.teacher_logits, idx, axis=0),
        teacher_value=jnp.take(pool.teacher_value, idx, axis=0),
        tolerance=jnp.full((int(batch_size),), float(tolerance), dtype=jnp.float32),
        weight=jnp.ones((int(batch_size),), dtype=jnp.float32),
        count=pool.count,
    )


def _pack_behavior(logits, value):
    return jnp.concatenate([logits, value[..., None]], axis=-1)


def _guard_distance(teacher, current, value_coef: float):
    teacher_logits = teacher[..., :-1]
    teacher_value = teacher[..., -1]
    current_logits = current[..., :-1]
    current_value = current[..., -1]
    policy_kl = kl_categorical(teacher_logits, current_logits)
    value_drift = jnp.abs(current_value - teacher_value)
    return policy_kl + float(value_coef) * value_drift


def _guard_loss(params, guard_batch: GuardBatch, value_coef: float):
    teacher = _pack_behavior(guard_batch.teacher_logits, guard_batch.teacher_value)
    batch = AnchorBatch(
        x=guard_batch.obs,
        context=guard_batch.game_onehot,
        teacher=teacher,
        tolerance=guard_batch.tolerance,
        weight=guard_batch.weight,
    )

    def behavior_fn(p, obs, game_onehot):
        logits, value = ac_apply(p, obs, game_onehot)
        return _pack_behavior(logits, value)

    distance_fn = lambda teacher_v, current_v: _guard_distance(
        teacher_v, current_v, value_coef
    )
    raw_loss = conservation_loss(behavior_fn, params, batch, distance_fn)
    return jnp.where(guard_batch.count > 0, raw_loss, jnp.asarray(0.0, dtype=jnp.float32))


def _ppo_guard_loss(
    params,
    batch,
    game_onehot,
    guard_batch: GuardBatch,
    cfg: ContinualRLConfig,
    guard_coef: float,
):
    ppo_loss, ppo_aux = _ppo_loss(
        params,
        batch,
        game_onehot,
        cfg.clip_eps,
        cfg.vf_coef,
        cfg.ent_coef,
    )
    guard = _guard_loss(params, guard_batch, cfg.value_coef)
    total = ppo_loss + float(guard_coef) * guard
    aux = jnp.concatenate([ppo_aux, jnp.asarray([guard], dtype=jnp.float32)], axis=0)
    return total, aux


def _train_seed(seed: int, task_i: int) -> int:
    return int(seed) + 100_003 * int(task_i)


def _eval_seed(seed: int, task_i: int) -> int:
    return int(seed) + 200_003 * int(task_i) + 17


def _anchor_seed(seed: int, task_i: int) -> int:
    return int(seed) + 300_007 * int(task_i) + 31


def _train_minatar_game(
    game_spec: GameSpec,
    specs,
    cfg: ContinualRLConfig,
    seed: int,
    init_params,
    guard_pool: GuardPool,
    guard_coef: float,
) -> dict:
    ppo_cfg = _to_ppo_config(cfg)
    num_updates = _validate_config(ppo_cfg)
    batch_size = int(ppo_cfg.num_envs) * int(ppo_cfg.num_steps)
    minibatch_size = batch_size // int(ppo_cfg.num_minibatches)
    n_games = _n_games(specs)
    act_max = _act_max(specs)
    c_in = _c_max(specs)
    action_masks = _action_masks_for_specs(specs, n_games, act_max)
    game_onehot = jax.nn.one_hot(int(game_spec.game_id), n_games, dtype=jnp.float32)

    def lr_schedule(count):
        if not bool(ppo_cfg.anneal_lr):
            return float(ppo_cfg.lr)
        updates_per_outer = int(ppo_cfg.update_epochs) * int(ppo_cfg.num_minibatches)
        outer_update = count // updates_per_outer
        frac = 1.0 - outer_update.astype(jnp.float32) / float(num_updates)
        return float(ppo_cfg.lr) * frac

    tx = optax.adam(learning_rate=lr_schedule if bool(ppo_cfg.anneal_lr) else float(ppo_cfg.lr))

    rng = jax.random.PRNGKey(int(seed))
    rng, init_key, reset_key = jax.random.split(rng, 3)
    reset_keys = jax.random.split(reset_key, int(ppo_cfg.num_envs))
    raw_obs, env_state = vreset(game_spec.env, game_spec.params, reset_keys)
    obs = pad_obs(raw_obs, c_in)
    params = init_params
    if params is None:
        params = init_ac(init_key, c_in, n_games, act_max)
    params = _force_action_masks(params, action_masks)
    opt_state = tx.init(params)

    def rollout(params, env_state, obs, running_return, rng):
        def step(carry, _):
            env_state, obs, running_return, rng = carry
            rng, action_key, step_key, reset_key = jax.random.split(rng, 4)
            logits, value = ac_apply(params, obs, game_onehot)
            actions = jax.random.categorical(action_key, logits, axis=-1).astype(jnp.int32)
            logprobs = _categorical_log_prob(logits, actions)

            step_keys = jax.random.split(step_key, int(ppo_cfg.num_envs))
            next_obs_raw, step_state, reward, done, _ = vstep(
                game_spec.env,
                game_spec.params,
                step_keys,
                env_state,
                actions,
            )
            reset_keys = jax.random.split(reset_key, int(ppo_cfg.num_envs))
            reset_obs_raw, reset_state = vreset(game_spec.env, game_spec.params, reset_keys)
            next_state = _where_done_tree(done, reset_state, step_state)
            next_obs = _where_done_array(done, pad_obs(reset_obs_raw, c_in), pad_obs(next_obs_raw, c_in))

            reward = reward.astype(jnp.float32)
            done_f = done.astype(jnp.float32)
            finished_return = running_return + reward
            completed_sum = jnp.sum(jnp.where(done, finished_return, 0.0))
            completed_count = jnp.sum(done_f)
            next_running_return = jnp.where(done, 0.0, finished_return)

            transition = (
                obs,
                actions,
                logprobs,
                reward,
                done_f,
                value,
            )
            next_carry = (next_state, next_obs, next_running_return, rng)
            return next_carry, (transition, completed_sum, completed_count)

        init_carry = (env_state, obs, running_return, rng)
        (env_state, obs, running_return, rng), (traj_tuple, completed_sum, completed_count) = (
            jax.lax.scan(
                step,
                init_carry,
                None,
                length=int(ppo_cfg.num_steps),
            )
        )
        transition = Transition(*traj_tuple)
        return (
            env_state,
            obs,
            running_return,
            rng,
            transition,
            jnp.sum(completed_sum),
            jnp.sum(completed_count),
        )

    def update_policy(params, opt_state, batch, guard_batch: GuardBatch, rng):
        def epoch_step(carry, _):
            params, opt_state, rng = carry
            rng, perm_key = jax.random.split(rng)
            permutation = jax.random.permutation(perm_key, batch_size)
            minibatches = _make_minibatches(
                batch,
                permutation,
                int(ppo_cfg.num_minibatches),
                minibatch_size,
            )

            def minibatch_step(carry, minibatch):
                params, opt_state = carry

                def loss_fn(p):
                    return _ppo_guard_loss(
                        p,
                        minibatch,
                        game_onehot,
                        guard_batch,
                        cfg,
                        guard_coef,
                    )

                (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
                grads = clip_global(grads, ppo_cfg.max_grad_norm)
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
            length=int(ppo_cfg.update_epochs),
        )
        return params, opt_state, jnp.mean(metrics, axis=0)

    def train_loop(params, opt_state, env_state, obs, rng, guard_pool):
        def update_step(carry, _):
            params, opt_state, env_state, obs, running_return, return_estimate, seen_return, rng = carry
            rng, rollout_key, guard_key, update_key = jax.random.split(rng, 4)
            (
                env_state,
                obs,
                running_return,
                _,
                traj,
                completed_sum,
                completed_count,
            ) = rollout(params, env_state, obs, running_return, rollout_key)
            _, last_value = ac_apply(params, obs, game_onehot)
            advantages, targets = _compute_gae(traj, last_value, ppo_cfg.gamma, ppo_cfg.gae_lambda)
            batch = _flatten_batch(traj, advantages, targets, batch_size)
            guard_batch = _sample_guard_batch(
                guard_pool,
                guard_key,
                int(cfg.guard_batch),
                float(cfg.guard_tolerance),
            )
            params, opt_state, metrics = update_policy(params, opt_state, batch, guard_batch, update_key)

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
            return next_carry, (new_return_estimate, metrics)

        init_carry = (
            params,
            opt_state,
            env_state,
            obs,
            jnp.zeros((int(ppo_cfg.num_envs),), dtype=jnp.float32),
            jnp.asarray(0.0, dtype=jnp.float32),
            jnp.asarray(False),
            rng,
        )
        final_carry, (returns_curve, metrics_curve) = jax.lax.scan(
            update_step,
            init_carry,
            None,
            length=num_updates,
        )
        return final_carry[0], returns_curve, metrics_curve

    params, returns_curve, metrics_curve = jax.jit(train_loop)(
        params, opt_state, env_state, obs, rng, guard_pool
    )
    returns_np = np.asarray(jax.device_get(returns_curve), dtype=np.float32)
    metrics_np = np.asarray(jax.device_get(metrics_curve), dtype=np.float32)
    return {
        "params": params,
        "returns_curve": returns_np.astype(float).tolist(),
        "loss_curve": metrics_np[:, 0].astype(float).tolist(),
        "guard_curve": metrics_np[:, -1].astype(float).tolist(),
        "final_return": float(returns_np[-1]),
        "timesteps": int(num_updates * batch_size),
        "num_updates": int(num_updates),
    }


def _eval_horizon_for_spec(spec: GameSpec, eval_horizon) -> int:
    if eval_horizon is not None:
        return int(eval_horizon)
    params = spec.params
    for name in ("max_steps_in_episode", "max_steps", "time_limit", "episode_length"):
        if hasattr(params, name):
            return int(getattr(params, name))
    return 1000


def evaluate_all_games(
    params,
    specs,
    n_games,
    act_max,
    key,
    eval_episodes,
    eval_horizon=None,
) -> np.ndarray:
    """Greedy fixed-horizon evaluation on every MinAtar game."""
    specs = list(specs)
    if not specs:
        return np.asarray([], dtype=np.float32)
    n_games = int(n_games)
    act_max = int(act_max)
    c_in = _c_max(specs)
    eval_episodes = int(eval_episodes)
    if not hasattr(key, "shape"):
        key = jax.random.PRNGKey(int(key))
    action_masks = _action_masks_for_specs(specs, n_games, act_max)
    params = _force_action_masks(params, action_masks)
    game_keys = jax.random.split(key, len(specs))
    scores = []

    for game_key, spec in zip(game_keys, specs):
        game_onehot = jax.nn.one_hot(int(spec.game_id), n_games, dtype=jnp.float32)
        horizon = _eval_horizon_for_spec(spec, eval_horizon)

        def eval_one(params, key):
            reset_keys = jax.random.split(key, eval_episodes)
            raw_obs, env_state = vreset(spec.env, spec.params, reset_keys)
            obs = pad_obs(raw_obs, c_in)
            done_seen = jnp.zeros((eval_episodes,), dtype=bool)
            returns = jnp.zeros((eval_episodes,), dtype=jnp.float32)

            def step(carry, _):
                env_state, obs, done_seen, returns, rng = carry
                rng, step_key = jax.random.split(rng)
                logits, _ = ac_apply(params, obs, game_onehot)
                actions = jnp.argmax(logits, axis=-1).astype(jnp.int32)
                step_keys = jax.random.split(step_key, eval_episodes)
                next_obs_raw, next_state, reward, done, _ = vstep(
                    spec.env,
                    spec.params,
                    step_keys,
                    env_state,
                    actions,
                )
                active = jnp.logical_not(done_seen)
                returns = returns + jnp.where(active, reward.astype(jnp.float32), 0.0)
                done_seen = jnp.logical_or(done_seen, done)
                next_obs = pad_obs(next_obs_raw, c_in)
                return (next_state, next_obs, done_seen, returns, rng), None

            (_, _, _, returns, _), _ = jax.lax.scan(
                step,
                (env_state, obs, done_seen, returns, key),
                None,
                length=int(horizon),
            )
            return jnp.mean(returns)

        scores.append(float(jax.device_get(jax.jit(eval_one)(params, game_key))))
    return np.asarray(scores, dtype=np.float32)


def _collect_anchor_buffer(
    params,
    game_spec: GameSpec,
    specs,
    cfg: ContinualRLConfig,
    key,
) -> MinAtarAnchorBuffer:
    n = int(cfg.anchor_buffer_per_game)
    n_games = _n_games(specs)
    act_max = _act_max(specs)
    c_in = _c_max(specs)
    action_masks = _action_masks_for_specs(specs, n_games, act_max)
    params = _force_action_masks(params, action_masks)
    anchor_envs = max(1, min(int(cfg.num_envs), n))
    anchor_steps = int(np.ceil(n / float(anchor_envs)))
    game_onehot = jax.nn.one_hot(int(game_spec.game_id), n_games, dtype=jnp.float32)
    game_onehot_env = jnp.broadcast_to(game_onehot, (anchor_envs, n_games))
    if not hasattr(key, "shape"):
        key = jax.random.PRNGKey(int(key))

    def collect(params, key):
        key, reset_key = jax.random.split(key)
        reset_keys = jax.random.split(reset_key, anchor_envs)
        raw_obs, env_state = vreset(game_spec.env, game_spec.params, reset_keys)
        obs = pad_obs(raw_obs, c_in)

        def step(carry, _):
            env_state, obs, rng = carry
            rng, action_key, step_key, reset_key = jax.random.split(rng, 4)
            logits, value = ac_apply(params, obs, game_onehot)
            actions = jax.random.categorical(action_key, logits, axis=-1).astype(jnp.int32)
            step_keys = jax.random.split(step_key, anchor_envs)
            next_obs_raw, step_state, _, done, _ = vstep(
                game_spec.env,
                game_spec.params,
                step_keys,
                env_state,
                actions,
            )
            reset_keys = jax.random.split(reset_key, anchor_envs)
            reset_obs_raw, reset_state = vreset(game_spec.env, game_spec.params, reset_keys)
            next_state = _where_done_tree(done, reset_state, step_state)
            next_obs = _where_done_array(done, pad_obs(reset_obs_raw, c_in), pad_obs(next_obs_raw, c_in))
            sample = (obs, game_onehot_env, logits, value)
            return (next_state, next_obs, rng), sample

        (_, _, _), samples = jax.lax.scan(
            step,
            (env_state, obs, key),
            None,
            length=anchor_steps,
        )
        obs_s, goh_s, logits_s, value_s = samples
        obs_f = obs_s.reshape((anchor_steps * anchor_envs, 10, 10, c_in))[:n]
        goh_f = goh_s.reshape((anchor_steps * anchor_envs, n_games))[:n]
        logits_f = logits_s.reshape((anchor_steps * anchor_envs, act_max))[:n]
        value_f = value_s.reshape((anchor_steps * anchor_envs,))[:n]
        return obs_f, goh_f, logits_f, value_f

    obs, game_onehot_np, teacher_logits, teacher_value = jax.jit(collect)(params, key)
    return MinAtarAnchorBuffer(
        obs=np.asarray(jax.device_get(obs), dtype=np.float32),
        game_onehot=np.asarray(jax.device_get(game_onehot_np), dtype=np.float32),
        teacher_logits=np.asarray(jax.device_get(teacher_logits), dtype=np.float32),
        teacher_value=np.asarray(jax.device_get(teacher_value), dtype=np.float32),
    )


def _certify_game(
    params,
    spec: GameSpec,
    task_i: int,
    eval_score: float,
    buffer: MinAtarAnchorBuffer,
    cfg: ContinualRLConfig,
    atlas: Atlas,
    champions: ChampionStore,
):
    teacher = np.concatenate(
        [buffer.teacher_logits, buffer.teacher_value[:, None]],
        axis=-1,
    )
    confidence = np.max(
        np.asarray(jax.nn.softmax(jnp.asarray(buffer.teacher_logits), axis=-1)),
        axis=-1,
    )
    n = int(buffer.obs.shape[0])
    anchors = AnchorStore(cfg.anchor_buffer_per_game)
    anchors.add(
        buffer.obs,
        teacher,
        np.full((n,), float(cfg.guard_tolerance), dtype=np.float32),
        np.ones((n,), dtype=np.float32),
        confidence.astype(np.float32),
        contexts=buffer.game_onehot,
        skill_ids=[spec.name] * n,
        labels=np.full((n,), int(spec.game_id), dtype=np.int32),
    )
    sent_n = min(n, int(cfg.eval_episodes))
    sentinels = SentinelStore(
        x=buffer.obs[:sent_n],
        y=np.full((sent_n,), int(spec.game_id), dtype=np.int32),
        seeds=np.arange(sent_n, dtype=np.int32),
    )
    champion = champions.freeze(
        params,
        route=spec.name,
        meta={"skill_id": spec.name, "task_index": int(task_i)},
    )
    return atlas.create_or_update_node(
        spec.name,
        context_key=spec.name,
        anchors=anchors,
        sentinels=sentinels,
        status="protected",
        champion_ref=champion,
        best_score=float(eval_score),
        current_score=float(eval_score),
        retention=1.0,
        allowed_regression=0.0,
        last_certified_step=int(task_i),
        guard_lambda=float(cfg.guard_coef),
        certified_impls=[spec.name],
    )


def compute_rl_metrics(return_matrix, single_game_scores=None) -> dict:
    returns = np.asarray(return_matrix, dtype=np.float32)
    if returns.ndim != 2 or returns.shape[0] == 0 or returns.shape[1] == 0:
        raise ValueError("return_matrix must be a non-empty 2D array")
    final = returns[-1]
    peak = np.max(returns, axis=0)
    learned = np.diag(returns)
    eps = np.asarray(1e-9, dtype=np.float32)
    retention = final / np.maximum(peak, eps)
    if returns.shape[1] > 1:
        prior = slice(0, returns.shape[1] - 1)
        forgetting = float(np.mean(peak[prior] - final[prior]))
    else:
        forgetting = 0.0
    metrics = {
        "mean_final": float(np.mean(final)),
        "mean_final_return": float(np.mean(final)),
        "forgetting": float(forgetting),
        "Forgetting": float(forgetting),
        "retention": retention.astype(float).tolist(),
        "mean_retention": float(np.mean(retention)),
        "worst_retention": float(np.min(retention)),
        "learned": learned.astype(float).tolist(),
        "learned_returns": learned.astype(float).tolist(),
        "final": final.astype(float).tolist(),
        "final_returns": final.astype(float).tolist(),
        "peak": peak.astype(float).tolist(),
        "peak_returns": peak.astype(float).tolist(),
    }
    if single_game_scores is not None:
        ref = np.asarray(single_game_scores, dtype=np.float32)
        denom = np.maximum(np.abs(ref), eps)
        norm_final = final / denom
        norm_learned = learned / denom[: learned.shape[0]]
        metrics.update(
            {
                "normalized_final": norm_final.astype(float).tolist(),
                "mean_normalized_final": float(np.mean(norm_final)),
                "normalized_learned": norm_learned.astype(float).tolist(),
            }
        )
    return metrics


def _result(return_matrix, mode: str, cfg: ContinualRLConfig, seed: int, extra=None) -> dict:
    return_matrix = np.asarray(return_matrix, dtype=np.float32)
    metrics = compute_rl_metrics(return_matrix)
    return {
        "mode": mode,
        "return_matrix": return_matrix.astype(float),
        "learned_returns": np.diag(return_matrix).astype(float),
        "final_returns": return_matrix[-1].astype(float),
        "peak_returns": np.max(return_matrix, axis=0).astype(float),
        "metrics": metrics,
        "extra": {
            "seed": int(seed),
            "config": asdict(cfg),
            **dict(extra or {}),
        },
    }


def run_minatar_baseline(specs, cfg: ContinualRLConfig | None = None, seed: int = 0) -> dict:
    """Sequential warm-start PPO over MinAtar games without PMA-C protection."""
    cfg = cfg or ContinualRLConfig()
    _validate_continual_config(cfg)
    specs = list(specs)
    if not specs:
        raise ValueError("at least one game spec is required")
    n_games = _n_games(specs)
    act_max = _act_max(specs)
    c_in = _c_max(specs)
    return_matrix = np.zeros((len(specs), len(specs)), dtype=np.float32)
    curves = {}
    guard_pool = _empty_guard_pool(c_in, n_games, act_max)
    params = None
    total_updates = 0

    for task_i, spec in enumerate(specs):
        train = _train_minatar_game(
            spec,
            specs,
            cfg,
            _train_seed(seed, task_i),
            params,
            guard_pool,
            guard_coef=0.0,
        )
        params = train["params"]
        total_updates += int(train["num_updates"])
        curves[spec.name] = train["returns_curve"]
        return_matrix[task_i] = evaluate_all_games(
            params,
            specs,
            n_games,
            act_max,
            jax.random.PRNGKey(_eval_seed(seed, task_i)),
            cfg.eval_episodes,
            cfg.eval_horizon,
        )

    return _result(
        return_matrix,
        "baseline",
        cfg,
        seed,
        extra={
            "game_order": [spec.name for spec in specs],
            "updates": int(total_updates),
            "returns_curves": curves,
        },
    )


def run_minatar_pmac(
    specs,
    cfg: ContinualRLConfig | None = None,
    seed: int = 0,
    ablation=None,
) -> dict:
    """Sequential warm-start PPO with PMA-C anchor conservation on prior games."""
    cfg = cfg or ContinualRLConfig()
    ablation = cfg.ablation if ablation is None else ablation
    ablation = None if ablation == "none" else ablation
    if ablation not in ALLOWED_MINATAR_ABLATIONS:
        raise ValueError(f"unknown MinAtar PMA-C ablation: {ablation}")
    _validate_continual_config(cfg)
    specs = list(specs)
    if not specs:
        raise ValueError("at least one game spec is required")

    n_games = _n_games(specs)
    act_max = _act_max(specs)
    c_in = _c_max(specs)
    atlas = Atlas()
    champions = ChampionStore()
    buffers: list[MinAtarAnchorBuffer] = []
    return_matrix = np.zeros((len(specs), len(specs)), dtype=np.float32)
    curves = {}
    guard_curves = {}
    params = None
    total_updates = 0
    guard_enabled = ablation != "no_conservation"
    effective_guard_coef = float(cfg.guard_coef) if guard_enabled else 0.0

    for task_i, spec in enumerate(specs):
        guard_pool = _guard_pool_from_buffers(buffers, cfg, c_in, n_games, act_max, ablation)
        train = _train_minatar_game(
            spec,
            specs,
            cfg,
            _train_seed(seed, task_i),
            params,
            guard_pool,
            guard_coef=effective_guard_coef,
        )
        params = train["params"]
        total_updates += int(train["num_updates"])
        curves[spec.name] = train["returns_curve"]
        guard_curves[spec.name] = train["guard_curve"]
        return_matrix[task_i] = evaluate_all_games(
            params,
            specs,
            n_games,
            act_max,
            jax.random.PRNGKey(_eval_seed(seed, task_i)),
            cfg.eval_episodes,
            cfg.eval_horizon,
        )
        buffer = _collect_anchor_buffer(
            params,
            spec,
            specs,
            cfg,
            jax.random.PRNGKey(_anchor_seed(seed, task_i)),
        )
        buffers.append(buffer)
        _certify_game(
            params,
            spec,
            task_i,
            float(return_matrix[task_i, task_i]),
            buffer,
            cfg,
            atlas,
            champions,
        )

    mode = "pmac" if ablation is None else f"pmac_{ablation}"
    return _result(
        return_matrix,
        mode,
        cfg,
        seed,
        extra={
            "ablation": ablation,
            "game_order": [spec.name for spec in specs],
            "updates": int(total_updates),
            "returns_curves": curves,
            "guard_loss_curves": guard_curves,
            "protected_skills": list(atlas.nodes.keys()),
            "anchor_counts": [int(buf.obs.shape[0]) for buf in buffers],
            "guard_enabled": bool(guard_enabled),
            "guard_source": "all_prior" if ablation != "no_replay" else "most_recent_prior",
        },
    )


def _jsonify_result(result: dict) -> dict:
    return {
        "mode": result["mode"],
        "return_matrix": np.asarray(result["return_matrix"]).astype(float).tolist(),
        "learned_returns": np.asarray(result["learned_returns"]).astype(float).tolist(),
        "final_returns": np.asarray(result["final_returns"]).astype(float).tolist(),
        "peak_returns": np.asarray(result["peak_returns"]).astype(float).tolist(),
        "metrics": result["metrics"],
        "extra": result["extra"],
    }


def _aggregate(results_by_mode):
    aggregate = {}
    for mode, results in results_by_mode.items():
        stats = {}
        for key, value in results[0]["metrics"].items():
            if isinstance(value, (list, tuple)):
                continue
            arr = np.asarray([result["metrics"][key] for result in results], dtype=np.float64)
            stats[key] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
        aggregate[mode] = stats
    return aggregate


def _plot_results(first_seed_results, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = list(first_seed_results.keys())
    first = first_seed_results[modes[0]]
    n_games = int(np.asarray(first["return_matrix"]).shape[1])
    x = np.arange(n_games)
    width = 0.8 / max(1, len(modes))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    fig.suptitle("Continual MinAtar: PMA-C vs Baseline")

    for i, mode in enumerate(modes):
        result = first_seed_results[mode]
        offset = (i - (len(modes) - 1) / 2.0) * width
        axes[0].bar(x + offset, np.asarray(result["final_returns"]), width=width, label=mode)
        axes[1].plot(np.asarray(result["return_matrix"])[:, 0], marker="o", label=mode)
    axes[0].set_title("Final Return by Game")
    axes[0].set_xlabel("Game")
    axes[0].set_ylabel("Return")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Game 0 Across Training")
    axes[1].set_xlabel("After Game")
    axes[1].set_ylabel("Return")
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _run_timed(seed: int, run_fn):
    start = time.perf_counter()
    result = run_fn()
    wall_s = time.perf_counter() - start
    result["extra"] = dict(result["extra"])
    result["extra"]["wall_s"] = float(wall_s)
    learned = ", ".join(f"{v:.3f}" for v in np.asarray(result["learned_returns"]))
    final = ", ".join(f"{v:.3f}" for v in np.asarray(result["final_returns"]))
    print(
        f"{result['mode']} seed={int(seed)} wall_s={wall_s:.3f} "
        f"learned=[{learned}] final=[{final}] "
        f"mean_final={result['metrics']['mean_final']:.3f} "
        f"forgetting={result['metrics']['forgetting']:.3f}"
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--games",
        default="Breakout-MinAtar,Asterix-MinAtar,Freeway-MinAtar,SpaceInvaders-MinAtar",
    )
    parser.add_argument("--per-game-steps", type=int, default=3_000_000)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--ablations", default="none")
    parser.add_argument("--guard-coef", type=float, default=1.0)
    parser.add_argument("--out", default="runs/minatar_continual")
    parser.add_argument("--eval-episodes", type=int, default=64)
    parser.add_argument("--eval-horizon", type=int, default=None)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=8)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--anchor-buffer-per-game", type=int, default=4096)
    parser.add_argument("--guard-batch", type=int, default=512)
    parser.add_argument("--guard-tolerance", type=float, default=0.005)
    parser.add_argument("--value-coef", type=float, default=1.0)
    args = parser.parse_args(argv)

    games = _parse_csv(args.games)
    invalid_games = [name for name in games if name not in GAMES]
    if invalid_games:
        parser.error(
            "unknown game(s): "
            + ", ".join(invalid_games)
            + "; valid games are "
            + ",".join(GAMES)
        )
    seeds = _parse_seeds(args.seeds)
    ablations = _parse_ablations(args.ablations)
    invalid_ablations = [value for value in ablations if value not in ALLOWED_MINATAR_ABLATIONS]
    if invalid_ablations:
        parser.error(
            "unknown ablation(s): "
            + ", ".join(str(value) for value in invalid_ablations)
            + "; valid values are none,no_conservation,no_replay"
        )

    cfg = ContinualRLConfig(
        per_game_steps=int(args.per_game_steps),
        num_envs=int(args.num_envs),
        num_steps=int(args.num_steps),
        update_epochs=int(args.update_epochs),
        num_minibatches=int(args.num_minibatches),
        lr=float(args.lr),
        eval_episodes=int(args.eval_episodes),
        eval_horizon=args.eval_horizon,
        guard_coef=float(args.guard_coef),
        guard_batch=int(args.guard_batch),
        anchor_buffer_per_game=int(args.anchor_buffer_per_game),
        guard_tolerance=float(args.guard_tolerance),
        value_coef=float(args.value_coef),
    )
    specs = make_games(games)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = {
        "seeds": seeds,
        "games": games,
        "config": asdict(cfg),
        "runs": {},
    }
    results_by_mode = {}
    first_seed_results = {}

    for seed in seeds:
        seed_results = {}
        baseline = _run_timed(seed, lambda seed=seed: run_minatar_baseline(specs, cfg, seed))
        seed_results[baseline["mode"]] = baseline
        results_by_mode.setdefault(baseline["mode"], []).append(baseline)

        pmac = _run_timed(seed, lambda seed=seed: run_minatar_pmac(specs, cfg, seed, None))
        seed_results[pmac["mode"]] = pmac
        results_by_mode.setdefault(pmac["mode"], []).append(pmac)

        for ablation in ablations:
            if ablation is None:
                continue
            result = _run_timed(
                seed,
                lambda seed=seed, ablation=ablation: run_minatar_pmac(
                    specs,
                    cfg,
                    seed,
                    ablation,
                ),
            )
            seed_results[result["mode"]] = result
            results_by_mode.setdefault(result["mode"], []).append(result)

        if not first_seed_results:
            first_seed_results = dict(seed_results)
        raw["runs"][str(seed)] = {
            "results": {mode: _jsonify_result(result) for mode, result in seed_results.items()}
        }

    raw["aggregate"] = _aggregate(results_by_mode)
    results_path = out_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)
    plot_path = out_dir / "comparison.png"
    _plot_results(first_seed_results, plot_path)
    print(f"wrote {results_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()


__all__ = [
    "ALLOWED_MINATAR_ABLATIONS",
    "ContinualRLConfig",
    "MinAtarAnchorBuffer",
    "compute_rl_metrics",
    "evaluate_all_games",
    "run_minatar_baseline",
    "run_minatar_pmac",
]
