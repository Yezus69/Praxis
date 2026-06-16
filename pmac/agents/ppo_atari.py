"""Bounded CleanRL-style PPO for envpool Atari."""

from __future__ import annotations

from collections import deque
from collections.abc import Mapping
from dataclasses import dataclass
from functools import partial
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pmac.agents.atari_net import atari_apply, init_atari
from pmac.behavior_distance import kl_categorical
from pmac.envs.atari_envpool import ACT_DIM, EpisodeReturnTracker, make_train_env, norm_obs


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


class GuardBatch(NamedTuple):
    obs: jnp.ndarray
    game_onehot: jnp.ndarray
    teacher_logits: jnp.ndarray
    teacher_value: jnp.ndarray


class GuardHost(NamedTuple):
    obs_uint8: np.ndarray
    game_onehot: np.ndarray
    teacher_logits: np.ndarray
    teacher_value: np.ndarray
    prior_offsets: np.ndarray | None
    guard_lambdas: np.ndarray | None
    guard_coef: float
    value_coef: float
    guard_tolerance: float
    guard_batch: int


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


def _guard_get(guard, name: str):
    if isinstance(guard, Mapping):
        return guard[name]
    return getattr(guard, name)


def _guard_get_optional(guard, name: str, default=None):
    if isinstance(guard, Mapping):
        return guard.get(name, default)
    return getattr(guard, name, default)


def _prepare_guard_host(guard, n_games: int) -> GuardHost:
    obs_uint8 = np.asarray(_guard_get(guard, "obs_uint8"), dtype=np.uint8)
    game_onehot = np.asarray(_guard_get(guard, "game_onehot"), dtype=np.float32)
    teacher_logits = np.asarray(_guard_get(guard, "teacher_logits"), dtype=np.float32)
    teacher_value = np.asarray(_guard_get(guard, "teacher_value"), dtype=np.float32).reshape(-1)
    guard_batch = int(_guard_get(guard, "guard_batch"))
    prior_offsets = _guard_get_optional(guard, "prior_offsets")
    guard_lambdas = _guard_get_optional(guard, "guard_lambdas")

    if obs_uint8.ndim != 4 or obs_uint8.shape[1:] != (4, 84, 84):
        raise ValueError(f"guard obs_uint8 must have shape [M,4,84,84], got {obs_uint8.shape}")
    m = int(obs_uint8.shape[0])
    if m <= 0:
        raise ValueError("guard obs_uint8 must contain at least one anchor")
    if game_onehot.shape != (m, int(n_games)):
        raise ValueError(f"guard game_onehot must have shape [{m},{int(n_games)}], got {game_onehot.shape}")
    if teacher_logits.shape != (m, ACT_DIM):
        raise ValueError(f"guard teacher_logits must have shape [{m},{ACT_DIM}], got {teacher_logits.shape}")
    if teacher_value.shape != (m,):
        raise ValueError(f"guard teacher_value must have shape [{m}], got {teacher_value.shape}")
    if guard_batch <= 0:
        raise ValueError("guard_batch must be positive")
    if prior_offsets is not None:
        prior_offsets = np.asarray(prior_offsets, dtype=np.int64).reshape(-1)
        if int(prior_offsets.shape[0]) < 2:
            raise ValueError("guard prior_offsets must contain at least one slice")
        if int(prior_offsets[0]) != 0 or int(prior_offsets[-1]) != m:
            raise ValueError("guard prior_offsets must start at 0 and end at the guard size")
        if np.any(np.diff(prior_offsets) <= 0):
            raise ValueError("guard prior_offsets must define non-empty increasing slices")
    if guard_lambdas is not None:
        guard_lambdas = np.asarray(guard_lambdas, dtype=np.float32).reshape(-1)
        expected = 1 if prior_offsets is None else int(prior_offsets.shape[0] - 1)
        if int(guard_lambdas.shape[0]) != expected:
            raise ValueError(
                f"guard_lambdas must have length {expected}, got {guard_lambdas.shape[0]}"
            )

    return GuardHost(
        obs_uint8=obs_uint8,
        game_onehot=game_onehot,
        teacher_logits=teacher_logits,
        teacher_value=teacher_value,
        prior_offsets=prior_offsets,
        guard_lambdas=guard_lambdas,
        guard_coef=float(_guard_get(guard, "guard_coef")),
        value_coef=float(_guard_get(guard, "value_coef")),
        guard_tolerance=float(_guard_get(guard, "guard_tolerance")),
        guard_batch=guard_batch,
    )


def _sample_balanced_guard_indices(guard: GuardHost, rng: np.random.Generator) -> np.ndarray:
    offsets = np.asarray(guard.prior_offsets, dtype=np.int64)
    num_priors = int(offsets.shape[0] - 1)
    batch_size = int(guard.guard_batch)
    counts = np.full((num_priors,), batch_size // num_priors, dtype=np.int64)
    remainder = int(batch_size % num_priors)
    if remainder:
        order = rng.permutation(num_priors)
        counts[order[:remainder]] += 1

    parts = []
    for prior_i, count in enumerate(counts):
        count = int(count)
        if count <= 0:
            continue
        start = int(offsets[prior_i])
        stop = int(offsets[prior_i + 1])
        parts.append(rng.integers(start, stop, size=count, endpoint=False))
    idx = np.concatenate(parts, axis=0).astype(np.int64, copy=False)
    rng.shuffle(idx)
    return idx


def _sample_guard_batch_host(guard: GuardHost, rng: np.random.Generator) -> GuardBatch:
    if guard.prior_offsets is None:
        idx = rng.integers(
            0,
            int(guard.obs_uint8.shape[0]),
            size=int(guard.guard_batch),
            endpoint=False,
        )
    else:
        idx = _sample_balanced_guard_indices(guard, rng)
    return GuardBatch(
        obs=jnp.asarray(norm_obs(guard.obs_uint8[idx]), dtype=jnp.float32),
        game_onehot=jnp.asarray(guard.game_onehot[idx], dtype=jnp.float32),
        teacher_logits=jnp.asarray(guard.teacher_logits[idx], dtype=jnp.float32),
        teacher_value=jnp.asarray(guard.teacher_value[idx], dtype=jnp.float32),
    )


def _projected_guard_offsets(guard: GuardHost) -> np.ndarray:
    if guard.prior_offsets is None:
        return np.asarray([0, int(guard.obs_uint8.shape[0])], dtype=np.int64)
    return np.asarray(guard.prior_offsets, dtype=np.int64)


def _projected_guard_lambdas(guard: GuardHost) -> jnp.ndarray:
    offsets = _projected_guard_offsets(guard)
    num_priors = int(offsets.shape[0] - 1)
    if guard.guard_lambdas is not None:
        return jnp.asarray(guard.guard_lambdas, dtype=jnp.float32)
    return jnp.full(
        (num_priors,),
        float(guard.guard_coef) / float(max(num_priors, 1)),
        dtype=jnp.float32,
    )


def _sample_stacked_guard_batch_host(guard: GuardHost, rng: np.random.Generator) -> GuardBatch:
    offsets = _projected_guard_offsets(guard)
    num_priors = int(offsets.shape[0] - 1)
    per_skill_batch = int(guard.guard_batch) // max(num_priors, 1)
    if per_skill_batch <= 0:
        raise ValueError("guard_batch must be at least the number of prior skills for pmac_projected")

    obs_parts = []
    game_parts = []
    logits_parts = []
    value_parts = []
    for prior_i in range(num_priors):
        start = int(offsets[prior_i])
        stop = int(offsets[prior_i + 1])
        idx = rng.integers(start, stop, size=per_skill_batch, endpoint=False)
        obs_parts.append(norm_obs(guard.obs_uint8[idx]))
        game_parts.append(guard.game_onehot[idx])
        logits_parts.append(guard.teacher_logits[idx])
        value_parts.append(guard.teacher_value[idx])

    return GuardBatch(
        obs=jnp.asarray(np.stack(obs_parts, axis=0), dtype=jnp.float32),
        game_onehot=jnp.asarray(np.stack(game_parts, axis=0), dtype=jnp.float32),
        teacher_logits=jnp.asarray(np.stack(logits_parts, axis=0), dtype=jnp.float32),
        teacher_value=jnp.asarray(np.stack(value_parts, axis=0), dtype=jnp.float32),
    )


def _guard_loss(params, guard_batch: GuardBatch, value_coef: float, guard_tolerance: float):
    logits, value = atari_apply(params, guard_batch.obs, guard_batch.game_onehot)
    policy_kl = kl_categorical(guard_batch.teacher_logits, logits)
    value_drift = jnp.abs(value - guard_batch.teacher_value)
    violation = jnp.maximum(
        policy_kl + float(value_coef) * value_drift - float(guard_tolerance),
        0.0,
    )
    return jnp.mean(jnp.square(violation))


def _ppo_guard_loss(
    params,
    batch: TrainBatch,
    game_onehot,
    guard_batch: GuardBatch,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    guard_coef: float,
    guard_value_coef: float,
    guard_tolerance: float,
):
    ppo_loss, ppo_aux = _ppo_loss(
        params,
        batch,
        game_onehot,
        clip_coef,
        vf_coef,
        ent_coef,
    )
    guard_loss = _guard_loss(params, guard_batch, guard_value_coef, guard_tolerance)
    total = ppo_loss + float(guard_coef) * guard_loss
    aux = jnp.concatenate([ppo_aux, jnp.asarray([guard_loss], dtype=jnp.float32)], axis=0)
    return total, aux


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
        "guard_coef",
        "guard_value_coef",
        "guard_tolerance",
    ),
)
def ppo_guard_update(
    params,
    opt_state,
    batch: TrainBatch,
    game_onehot,
    guard_batch: GuardBatch,
    rng,
    learning_rate: float,
    update_epochs: int,
    num_minibatches: int,
    minibatch_size: int,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
    guard_coef: float,
    guard_value_coef: float,
    guard_tolerance: float,
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

            def loss_fn(p):
                return _ppo_guard_loss(
                    p,
                    minibatch,
                    game_onehot,
                    guard_batch,
                    clip_coef,
                    vf_coef,
                    ent_coef,
                    guard_coef,
                    guard_value_coef,
                    guard_tolerance,
                )

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


def train_ppo_atari(
    game,
    game_id,
    n_games,
    cfg,
    seed,
    init_params=None,
    *,
    guard=None,
    update_mode: str = "guard_loss",
    pmac_cfg=None,
    omega=None,
) -> dict:
    """Train one Atari game using bounded host envpool rollouts and jitted updates."""
    cfg = cfg or AtariPPOConfig()
    if str(update_mode) not in {"guard_loss", "pmac_projected"}:
        raise ValueError(f"unknown Atari PPO update_mode: {update_mode}")
    num_updates = _validate_config(cfg)
    batch_size = int(cfg.num_envs) * int(cfg.num_steps)
    minibatch_size = batch_size // int(cfg.num_minibatches)
    game_onehot = jax.nn.one_hot(int(game_id), int(n_games), dtype=jnp.float32)
    guard_host = None if guard is None else _prepare_guard_host(guard, int(n_games))

    rng = jax.random.PRNGKey(int(seed))
    rng, init_key = jax.random.split(rng)
    params = init_params
    if params is None:
        params = init_atari(init_key, int(n_games))
    omega_state = omega
    if str(update_mode) == "pmac_projected":
        from pmac.rl_update import PMACUpdateConfig, PPO_PMAC_METRIC_NAMES, ppo_pmac_update
        from pmac.stability import zeros_omega_like

        pmac_cfg = PMACUpdateConfig() if pmac_cfg is None else pmac_cfg
        omega_state = zeros_omega_like(params) if omega_state is None else omega_state
        pmac_metric_index = {name: i for i, name in enumerate(PPO_PMAC_METRIC_NAMES)}
    else:
        pmac_metric_index = {}

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
    guard_curve: list[float] = []
    projection_curve: list[float] = []
    last_curve_value = 0.0
    guard_rng = None
    if guard_host is not None:
        guard_rng = np.random.default_rng(int(seed) + 700_001)

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
        if guard_host is None:
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
        elif str(update_mode) == "guard_loss":
            guard_batch = _sample_guard_batch_host(guard_host, guard_rng)
            params, opt_state, rng, metrics = ppo_guard_update(
                params,
                opt_state,
                batch,
                game_onehot,
                guard_batch,
                rng,
                float(lr),
                int(cfg.update_epochs),
                int(cfg.num_minibatches),
                int(minibatch_size),
                float(cfg.clip_coef),
                float(cfg.vf_coef),
                float(cfg.ent_coef),
                float(cfg.max_grad_norm),
                float(guard_host.guard_coef),
                float(guard_host.value_coef),
                float(guard_host.guard_tolerance),
            )
            metrics_np = np.asarray(jax.device_get(metrics), dtype=np.float32)
            guard_curve.append(float(metrics_np[-1]))
        else:
            guard_batch = _sample_stacked_guard_batch_host(guard_host, guard_rng)
            guard_lambdas = _projected_guard_lambdas(guard_host)
            params, opt_state, rng, metrics = ppo_pmac_update(
                params,
                opt_state,
                batch,
                game_onehot,
                guard_batch.obs,
                guard_batch.game_onehot,
                guard_batch.teacher_logits,
                guard_batch.teacher_value,
                guard_lambdas,
                omega_state,
                rng,
                float(lr),
                int(cfg.update_epochs),
                int(cfg.num_minibatches),
                int(minibatch_size),
                float(cfg.clip_coef),
                float(cfg.vf_coef),
                float(cfg.ent_coef),
                float(cfg.max_grad_norm),
                float(guard_host.value_coef),
                float(guard_host.guard_tolerance),
                pmac_cfg,
            )
            metrics_np = np.asarray(jax.device_get(metrics), dtype=np.float32)
            guard_curve.append(float(metrics_np[pmac_metric_index["guard_loss"]]))
            projection_curve.append(float(metrics_np[pmac_metric_index["projection_ratio"]]))

    final_return = float(returns_curve[-1]) if returns_curve else 0.0
    result = {
        "params": params,
        "returns_curve": [float(v) for v in returns_curve],
        "final_return": final_return,
        "timesteps": int(num_updates * batch_size),
    }
    if guard_host is not None:
        result["guard_curve"] = [float(v) for v in guard_curve]
    if str(update_mode) == "pmac_projected":
        result["projection_curve"] = [float(v) for v in projection_curve]
        result["omega"] = omega_state
    return result


__all__ = [
    "AtariPPOConfig",
    "GuardBatch",
    "TrainBatch",
    "gae",
    "jit_greedy_policy",
    "jit_policy",
    "ppo_guard_update",
    "ppo_update",
    "train_ppo_atari",
]
