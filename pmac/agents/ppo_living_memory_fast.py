"""Fast single-game PPO trainer for the memory-conditioned Atari agent.

This is the M7b sibling of ``ppo_living_memory``: envpool-XLA scan rollouts
plus a fixed-capacity GPU-resident hot bank with jitted writes.
"""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import partial
import math
import time
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pmac.agents.atari_mem_net import mem_apply, mem_apply_key, mem_init
from pmac.agents.ppo_atari import (
    TrainBatch,
    _categorical_entropy,
    _categorical_log_prob,
    _flatten_rollout,
    _learning_rate,
    _make_minibatches,
    gae,
)
from pmac.agents.ppo_living_memory import LMConfig, _lm_ppo_loss, lm_update
from pmac.envs.atari_envpool import ACT_DIM, EpisodeReturnTracker, make_train_env_xla
from pmac.memory.losses import latent_conservation_loss
from pmac.memory.reader import ema_update
from pmac.memory.runtime import RunningValueNorm, default_retrieval_hp, pad_bank
from pmac.memory.write import DEFAULT_WRITE_WEIGHTS
from pmac.projection import plasticity_ratio, project_conflicts
from pmac.stability import scale_by_stability, update_omega, zeros_omega_like
from pmac.tree_utils import tree_add_scaled, tree_dot, tree_norm, tree_scale, tree_zeros_like


EPS = 1.0e-8
DEFAULT_WRITE_SUBSAMPLE_SIZE = 4096
DEFAULT_MAX_WRITES_PER_SEGMENT = 512


@dataclass(frozen=True)
class FastLMConfig(LMConfig):
    """Living-memory config tuned for envpool-XLA throughput."""

    num_envs: int = 256
    write_subsample_size: int = DEFAULT_WRITE_SUBSAMPLE_SIZE
    max_writes_per_segment: int = DEFAULT_MAX_WRITES_PER_SEGMENT
    guard_lambda_total: float = 1.0
    guard_kappa: float = 0.75
    guard_lambda_v: float = 1.0
    guard_stability_alpha: float = 10.0
    guard_rho_omega: float = 0.99
    guard_sample_atoms: int = 256


class FastLMRollout(NamedTuple):
    obs: jnp.ndarray
    actions: jnp.ndarray
    logprobs: jnp.ndarray
    values: jnp.ndarray
    rewards: jnp.ndarray
    dones: jnp.ndarray
    keys: jnp.ndarray


class GuardCombineMetrics(NamedTuple):
    g_task_norm: jnp.ndarray
    g_guard_norm: jnp.ndarray
    g_guard_clipped_norm: jnp.ndarray
    g_safe_norm: jnp.ndarray
    g_total_norm: jnp.ndarray
    projection_ratio: jnp.ndarray
    conflict_dot: jnp.ndarray
    nonfinite: jnp.ndarray


def _cfg_int(cfg, name: str, default: int) -> int:
    return int(getattr(cfg, name, default))


def _cfg_float(cfg, name: str, default: float) -> float:
    return float(getattr(cfg, name, default))


def _validate_fast_config(cfg) -> int:
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
    if int(cfg.hot_capacity) <= 0:
        raise ValueError("hot_capacity must be positive")
    if int(cfg.top_k) <= 0 or int(cfg.top_k) > int(cfg.hot_capacity):
        raise ValueError("top_k must be in [1, hot_capacity]")
    if not (0.0 <= float(cfg.write_top_fraction) <= 1.0):
        raise ValueError("write_top_fraction must be in [0, 1]")
    if float(cfg.teacher_temperature) < 1.0:
        raise ValueError("teacher_temperature must be >= 1")
    if _cfg_int(cfg, "write_subsample_size", DEFAULT_WRITE_SUBSAMPLE_SIZE) <= 0:
        raise ValueError("write_subsample_size must be positive")
    if _cfg_int(cfg, "max_writes_per_segment", DEFAULT_MAX_WRITES_PER_SEGMENT) < 0:
        raise ValueError("max_writes_per_segment must be non-negative")
    return num_updates


@jax.jit
def write_importance(
    abs_adv_hat,
    abs_delta_hat,
    novelty,
    entropy,
    life,
    ret_hat,
    forget_risk,
):
    """JAX version of M3's fixed write-importance rule."""

    weights = DEFAULT_WRITE_WEIGHTS
    return (
        float(weights["adv"]) * jnp.asarray(abs_adv_hat, dtype=jnp.float32)
        + float(weights["delta"]) * jnp.asarray(abs_delta_hat, dtype=jnp.float32)
        + float(weights["novelty"]) * jnp.asarray(novelty, dtype=jnp.float32)
        + float(weights["entropy"]) * jnp.asarray(entropy, dtype=jnp.float32)
        + float(weights["life"]) * jnp.asarray(life, dtype=jnp.float32)
        + float(weights["ret"]) * jnp.asarray(ret_hat, dtype=jnp.float32)
        + float(weights["forget"]) * jnp.asarray(forget_risk, dtype=jnp.float32)
    )


def _tree_all_finite(tree):
    finite = jnp.asarray(True)
    for leaf in jax.tree_util.tree_leaves(tree):
        finite = jnp.logical_and(finite, jnp.all(jnp.isfinite(leaf)))
    return finite


def _select_tree(new_tree, old_tree, predicate):
    return jax.tree_util.tree_map(lambda new, old: jnp.where(predicate, new, old), new_tree, old_tree)


def _zero_if_nonfinite(tree, template, nonfinite):
    zeros = tree_zeros_like(template)
    return jax.tree_util.tree_map(lambda x, z: jnp.where(nonfinite, z, x), tree, zeros)


@partial(jax.jit, static_argnames=("project",))
def _combine_latent_guard_core(
    params,
    g_task,
    g_guard,
    omega,
    lambda_total,
    kappa,
    stability_alpha,
    rho_omega,
    project: bool,
):
    g_task_norm = tree_norm(g_task)
    g_guard_norm = tree_norm(g_guard)
    inputs_finite = jnp.logical_and(_tree_all_finite(g_task), _tree_all_finite(g_guard))
    inputs_finite = jnp.logical_and(inputs_finite, _tree_all_finite(omega))

    clip_scale = jnp.minimum(
        jnp.asarray(1.0, dtype=jnp.float32),
        jnp.asarray(kappa, dtype=jnp.float32) * g_task_norm / (g_guard_norm + EPS),
    )
    g_guard = tree_scale(g_guard, clip_scale)  # spec §15
    conflict_dot = tree_dot(g_task, g_guard)

    if bool(project):
        g_safe = project_conflicts(g_task, [g_guard], EPS)  # spec §15
    else:
        g_safe = g_task  # spec §15

    g_total = tree_add_scaled(
        g_safe,
        g_guard,
        jnp.asarray(lambda_total, dtype=jnp.float32),
    )  # spec §16
    g_total = scale_by_stability(
        g_total,
        omega,
        jnp.asarray(stability_alpha, dtype=jnp.float32),
    )  # spec §17
    next_omega = update_omega(
        omega,
        params,
        g_guard,
        jnp.asarray(rho_omega, dtype=jnp.float32),
    )  # spec §17

    nonfinite = jnp.logical_not(jnp.logical_and(inputs_finite, _tree_all_finite(g_total)))
    g_total = _zero_if_nonfinite(g_total, g_task, nonfinite)
    next_omega = _select_tree(next_omega, omega, jnp.logical_not(nonfinite))
    metrics = GuardCombineMetrics(
        g_task_norm=g_task_norm,
        g_guard_norm=g_guard_norm,
        g_guard_clipped_norm=tree_norm(g_guard),
        g_safe_norm=tree_norm(g_safe),
        g_total_norm=tree_norm(g_total),
        projection_ratio=plasticity_ratio(g_safe, g_task, EPS),
        conflict_dot=conflict_dot,
        nonfinite=nonfinite,
    )
    return g_total, next_omega, metrics


def combine_latent_guard_grads(
    params,
    g_task,
    g_guard,
    omega=None,
    *,
    lambda_total=1.0,
    kappa=0.75,
    stability_alpha=10.0,
    rho_omega=0.99,
    project=True,
):
    """Combine task and latent-memory guard gradients using PMA-C geometry."""
    if omega is None:
        omega = zeros_omega_like(params)
    return _combine_latent_guard_core(
        params,
        g_task,
        g_guard,
        omega,
        jnp.asarray(lambda_total, dtype=jnp.float32),
        jnp.asarray(kappa, dtype=jnp.float32),
        jnp.asarray(stability_alpha, dtype=jnp.float32),
        jnp.asarray(rho_omega, dtype=jnp.float32),
        bool(project),
    )


def _l2_normalize(x, axis=-1):
    x = jnp.asarray(x, dtype=jnp.float32)
    return x / (jnp.linalg.norm(x, axis=axis, keepdims=True) + EPS)


@jax.jit
def hot_novelty(keys_t, bank, cur_game):
    """Vectorized same-game novelty: ``1 - max cosine`` with one masked matmul."""

    keys = _l2_normalize(keys_t)
    bank_keys = _l2_normalize(bank["keys"])
    valid = jnp.asarray(bank["valid"], dtype=bool)
    bank_game = jnp.asarray(bank["game_id"], dtype=jnp.int32)
    cur_game = jnp.asarray(cur_game, dtype=jnp.int32)
    if cur_game.ndim == 0:
        cur_game = jnp.broadcast_to(cur_game, (keys.shape[0],))

    same_game = jnp.logical_and(valid[None, :], bank_game[None, :] == cur_game[:, None])
    sim = keys @ bank_keys.T
    masked = jnp.where(same_game, sim, -1.0)
    max_cos = jnp.max(masked, axis=-1)
    has_same = jnp.any(same_game, axis=-1)
    novelty = jnp.where(has_same, 1.0 - max_cos, 1.0)
    return jnp.clip(novelty, 0.0, 1.0)


@partial(jax.jit, static_argnames=("capacity", "d_k", "d_c", "act_dim"))
def _empty_hot_bank_jit(capacity: int, d_k: int, d_c: int, act_dim: int):
    return {
        "keys": jnp.zeros((capacity, d_k), dtype=jnp.float32),
        "context": jnp.zeros((capacity, d_c), dtype=jnp.float32),
        "teacher_policy": jnp.zeros((capacity, act_dim), dtype=jnp.float32),
        "teacher_value": jnp.zeros((capacity,), dtype=jnp.float32),
        "importance": jnp.zeros((capacity,), dtype=jnp.float32),
        "game_id": jnp.zeros((capacity,), dtype=jnp.int32),
        "source5": jnp.zeros((capacity, 5), dtype=jnp.float32),
        "age": jnp.zeros((capacity,), dtype=jnp.float32),
        "valid": jnp.zeros((capacity,), dtype=bool),
    }


def empty_hot_bank(capacity: int, d_k: int, d_c: int, act_dim: int):
    """Create a reader-compatible fixed-capacity hot bank on the default device."""

    return _empty_hot_bank_jit(int(capacity), int(d_k), int(d_c), int(act_dim))


def _zero_invalid_bank(bank):
    valid = jnp.asarray(bank["valid"], dtype=bool)
    out = {}
    for name, value in bank.items():
        value = jnp.asarray(value)
        if name == "valid":
            out[name] = valid
            continue
        mask = valid
        while mask.ndim < value.ndim:
            mask = mask[..., None]
        out[name] = jnp.where(mask, value, jnp.zeros_like(value))
    return out


@jax.jit
def hot_insert(bank, new_atoms):
    """Insert padded atoms by keeping the top-C valid rows by importance."""

    capacity = int(bank["keys"].shape[0])
    combined = {
        name: jnp.concatenate([jnp.asarray(bank[name]), jnp.asarray(new_atoms[name])], axis=0)
        for name in bank.keys()
    }
    combined_valid = jnp.logical_and(
        jnp.asarray(combined["valid"], dtype=bool),
        jnp.isfinite(jnp.asarray(combined["importance"], dtype=jnp.float32)),
    )
    priority = jnp.where(
        combined_valid,
        jnp.asarray(combined["importance"], dtype=jnp.float32),
        -jnp.inf,
    )
    top_priority, top_idx = jax.lax.top_k(priority, capacity)
    kept = {name: jnp.take(value, top_idx, axis=0) for name, value in combined.items()}
    kept["valid"] = jnp.logical_and(jnp.isfinite(top_priority), top_priority > -jnp.inf)
    return _zero_invalid_bank(kept)


def _hp_get(hp, name: str):
    if isinstance(hp, dict):
        return hp[name]
    return getattr(hp, name)


def _hp_values(hp, bank_arrays):
    capacity = int(bank_arrays["keys"].shape[0])
    if hp is None:
        hp = default_retrieval_hp(min(16, capacity))
    top_k = int(_hp_get(hp, "top_k"))
    if top_k <= 0 or top_k > capacity:
        raise ValueError("retrieval top_k must be in [1, bank capacity]")
    return (
        float(_hp_get(hp, "tau_r")),
        float(_hp_get(hp, "beta_c")),
        float(_hp_get(hp, "beta_I")),
        float(_hp_get(hp, "beta_a")),
        float(_hp_get(hp, "w_rho")),
        float(_hp_get(hp, "w_c")),
        float(_hp_get(hp, "b0")),
        top_k,
    )


def _jit_hp(tau_r, beta_c, beta_I, beta_a, w_rho, w_c, b0, top_k):
    return {
        "tau_r": tau_r,
        "beta_c": beta_c,
        "beta_I": beta_I,
        "beta_a": beta_a,
        "top_k": int(top_k),
        "w_rho": w_rho,
        "w_c": w_c,
        "b0": b0,
    }


def _make_rollout_fn(step_env, num_steps: int):
    """Build the single jitted scan that contains envpool's XLA step."""

    @partial(jax.jit, static_argnames=("top_k",))
    def rollout(
        params,
        handle,
        obs,
        game_id_vec,
        bank,
        mu_g,
        sigma_g,
        rng,
        tau_r,
        beta_c,
        beta_I,
        beta_a,
        w_rho,
        w_c,
        b0,
        top_k,
    ):
        hp = _jit_hp(tau_r, beta_c, beta_I, beta_a, w_rho, w_c, b0, top_k)

        def step(carry, _):
            handle, obs, rng = carry
            rng, action_key = jax.random.split(rng)
            out = mem_apply(params, obs, game_id_vec, bank, hp, mu_g, sigma_g)
            logits_net = out["logits_net"]
            actions = jax.random.categorical(action_key, logits_net, axis=-1).astype(jnp.int32)
            logprobs = _categorical_log_prob(logits_net, actions)
            values = out["v_net"]

            handle, ts = step_env(handle, actions)
            next_obs, rewards, terminated, truncated, info = ts
            done = jnp.logical_or(terminated, truncated)
            _ = info
            row = FastLMRollout(
                obs=obs,
                actions=actions,
                logprobs=logprobs,
                values=values,
                rewards=jnp.asarray(rewards, dtype=jnp.float32),
                dones=done.astype(jnp.float32),
                keys=out["k"],
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


@partial(jax.jit, static_argnames=("top_k",))
def _last_value_jit(
    params,
    obs,
    game_id_vec,
    bank,
    mu_g,
    sigma_g,
    tau_r,
    beta_c,
    beta_I,
    beta_a,
    w_rho,
    w_c,
    b0,
    top_k,
):
    hp = _jit_hp(tau_r, beta_c, beta_I, beta_a, w_rho, w_c, b0, top_k)
    return mem_apply(params, obs, game_id_vec, bank, hp, mu_g, sigma_g)["v_net"]


def _new_value_norm_state():
    return {
        "mu": jnp.asarray(0.0, dtype=jnp.float32),
        "sigma": jnp.asarray(1.0, dtype=jnp.float32),
        "adv_mean": jnp.asarray(0.0, dtype=jnp.float32),
        "adv_mad": jnp.asarray(1.0, dtype=jnp.float32),
        "delta_mean": jnp.asarray(0.0, dtype=jnp.float32),
        "delta_mad": jnp.asarray(1.0, dtype=jnp.float32),
        "ret_mean": jnp.asarray(0.0, dtype=jnp.float32),
        "ret_mad": jnp.asarray(1.0, dtype=jnp.float32),
        "write_initialized": jnp.asarray(False, dtype=bool),
    }


def _as_value_norm_state(value_norm):
    state = _new_value_norm_state()
    if value_norm is None:
        return state
    if isinstance(value_norm, dict):
        for name in state:
            if name in value_norm:
                dtype = bool if name == "write_initialized" else jnp.float32
                state[name] = jnp.asarray(value_norm[name], dtype=dtype)
        return state
    if isinstance(value_norm, RunningValueNorm) or (
        hasattr(value_norm, "mu") and hasattr(value_norm, "sigma")
    ):
        state["mu"] = jnp.asarray(value_norm.mu(), dtype=jnp.float32)
        state["sigma"] = jnp.asarray(value_norm.sigma(), dtype=jnp.float32)
        return state
    raise TypeError("value_norm must be None, a state dict, or RunningValueNorm-like")


def _finite_mean_std(x, fallback_mu, fallback_sigma):
    x = jnp.asarray(x, dtype=jnp.float32).reshape(-1)
    mask = jnp.isfinite(x)
    count = jnp.maximum(jnp.sum(mask.astype(jnp.float32)), 1.0)
    mean = jnp.sum(jnp.where(mask, x, 0.0)) / count
    var = jnp.sum(jnp.where(mask, jnp.square(x - mean), 0.0)) / count
    sigma = jnp.sqrt(jnp.maximum(var, 0.0))
    any_valid = jnp.any(mask)
    return jnp.where(any_valid, mean, fallback_mu), jnp.where(any_valid, sigma, fallback_sigma)


def _update_write_stat(state, prefix: str, x, momentum, eps):
    x = jnp.asarray(x, dtype=jnp.float32).reshape(-1)
    batch_mean, _ = _finite_mean_std(x, state[f"{prefix}_mean"], state[f"{prefix}_mad"])
    finite = jnp.isfinite(x)
    count = jnp.maximum(jnp.sum(finite.astype(jnp.float32)), 1.0)
    batch_mad = jnp.sum(jnp.where(finite, jnp.abs(x - batch_mean), 0.0)) / count
    initialized = jnp.asarray(state["write_initialized"], dtype=bool)
    keep = jnp.asarray(momentum, dtype=jnp.float32)
    mean = jnp.where(
        initialized,
        keep * state[f"{prefix}_mean"] + (1.0 - keep) * batch_mean,
        batch_mean,
    )
    mad = jnp.where(
        initialized,
        keep * state[f"{prefix}_mad"] + (1.0 - keep) * batch_mad,
        batch_mad,
    )
    z = (x - mean) / (mad + jnp.asarray(eps, dtype=jnp.float32))
    return mean, mad, z


@jax.jit
def _value_norm_update_jit(value_norm, returns, value_momentum, sigma_floor):
    batch_mu, batch_sigma = _finite_mean_std(
        returns,
        value_norm["mu"],
        value_norm["sigma"],
    )
    keep = jnp.asarray(value_momentum, dtype=jnp.float32)
    sigma_floor = jnp.asarray(sigma_floor, dtype=jnp.float32)
    updated = dict(value_norm)
    updated["mu"] = keep * value_norm["mu"] + (1.0 - keep) * batch_mu
    updated["sigma"] = jnp.maximum(
        keep * value_norm["sigma"] + (1.0 - keep) * batch_sigma,
        sigma_floor,
    )
    return updated


def _context_embeddings(params, game_ids):
    return jnp.take(params["game_embed"]["embedding"], jnp.asarray(game_ids, dtype=jnp.int32), axis=0)


def _teacher_targets_jax(logits, values, mu_g, sigma_g, temperature):
    p_star = jax.nn.softmax(logits / jnp.asarray(temperature, dtype=jnp.float32), axis=-1)
    v_star = (
        jnp.asarray(values, dtype=jnp.float32) - jnp.asarray(mu_g, dtype=jnp.float32)
    ) / (jnp.asarray(sigma_g, dtype=jnp.float32) + EPS)
    return p_star, v_star


def _source5_jax(high_return, near_life_loss, novelty_hi, failure_recovery):
    high_return, near_life_loss, novelty_hi, failure_recovery = jnp.broadcast_arrays(
        jnp.asarray(high_return, dtype=bool),
        jnp.asarray(near_life_loss, dtype=bool),
        jnp.asarray(novelty_hi, dtype=bool),
        jnp.asarray(failure_recovery, dtype=bool),
    )
    sentinel = jnp.zeros_like(high_return, dtype=bool)
    return jnp.stack(
        [
            high_return.astype(jnp.float32),
            near_life_loss.astype(jnp.float32),
            sentinel.astype(jnp.float32),
            novelty_hi.astype(jnp.float32),
            failure_recovery.astype(jnp.float32),
        ],
        axis=-1,
    )


def _pad_selected_rows(x, write_count: int, max_writes: int):
    selected = jnp.asarray(x)
    out = jnp.zeros((int(max_writes),) + selected.shape[1:], dtype=selected.dtype)
    if int(write_count) > 0:
        out = out.at[: int(write_count)].set(selected)
    return out


@partial(jax.jit, static_argnames=("top_k", "write_batch_size", "write_count", "max_writes"))
def _lm_writes_jit(
    params,
    ema_params,
    bank,
    traj: FastLMRollout,
    last_value,
    advantages,
    returns,
    value_norm,
    game_id,
    rng,
    mu_g,
    sigma_g,
    tau_r,
    beta_c,
    beta_I,
    beta_a,
    w_rho,
    w_c,
    b0,
    top_k,
    write_batch_size,
    write_count,
    max_writes,
    gamma,
    teacher_temperature,
    write_momentum,
    write_eps,
    value_momentum,
    sigma_floor,
):
    hp = _jit_hp(tau_r, beta_c, beta_I, beta_a, w_rho, w_c, b0, top_k)
    batch_size = int(traj.actions.size)
    sample_idx = jax.random.permutation(rng, batch_size)[: int(write_batch_size)]

    flat_obs = traj.obs.reshape((batch_size,) + tuple(traj.obs.shape[2:]))
    flat_rewards = traj.rewards.reshape((batch_size,))
    flat_dones = traj.dones.reshape((batch_size,))
    flat_values = traj.values.reshape((batch_size,))
    flat_advantages = advantages.reshape((batch_size,))
    flat_returns = returns.reshape((batch_size,))

    next_values = jnp.concatenate([traj.values[1:], last_value[None, :]], axis=0)
    next_values = next_values * (1.0 - traj.dones)
    flat_next_values = next_values.reshape((batch_size,))

    obs_w = jnp.take(flat_obs, sample_idx, axis=0)
    rewards_w = jnp.take(flat_rewards, sample_idx, axis=0)
    dones_w = jnp.take(flat_dones, sample_idx, axis=0)
    values_w = jnp.take(flat_values, sample_idx, axis=0)
    next_values_w = jnp.take(flat_next_values, sample_idx, axis=0)
    advantages_w = jnp.take(flat_advantages, sample_idx, axis=0)
    returns_w = jnp.take(flat_returns, sample_idx, axis=0)
    game_ids_w = jnp.full((int(write_batch_size),), jnp.asarray(game_id, dtype=jnp.int32), dtype=jnp.int32)

    logits_net = mem_apply(params, obs_w, game_ids_w, bank, hp, mu_g, sigma_g)["logits_net"]
    keys_w = mem_apply_key(ema_params, obs_w)
    novelty_w = hot_novelty(keys_w, bank, game_ids_w)
    entropy_w = _categorical_entropy(logits_net)
    delta_w = rewards_w + jnp.asarray(gamma, dtype=jnp.float32) * next_values_w - values_w
    life_w = dones_w.astype(jnp.float32)
    forget_w = jnp.zeros_like(life_w, dtype=jnp.float32)

    abs_adv = jnp.abs(advantages_w)
    abs_delta = jnp.abs(delta_w)
    adv_mean, adv_mad, abs_adv_hat = _update_write_stat(
        value_norm,
        "adv",
        abs_adv,
        write_momentum,
        write_eps,
    )
    delta_mean, delta_mad, abs_delta_hat = _update_write_stat(
        value_norm,
        "delta",
        abs_delta,
        write_momentum,
        write_eps,
    )
    ret_mean, ret_mad, ret_hat = _update_write_stat(
        value_norm,
        "ret",
        returns_w,
        write_momentum,
        write_eps,
    )
    scores = write_importance(
        abs_adv_hat,
        abs_delta_hat,
        novelty_w,
        entropy_w,
        life_w,
        ret_hat,
        forget_w,
    )
    scores = jnp.where(jnp.isfinite(scores), scores, -jnp.inf)
    top_scores, top_idx = jax.lax.top_k(scores, int(write_count))

    teacher_policy, teacher_value = _teacher_targets_jax(
        jnp.take(logits_net, top_idx, axis=0),
        jnp.take(returns_w, top_idx, axis=0),
        mu_g,
        sigma_g,
        teacher_temperature,
    )
    novelty_threshold = jnp.quantile(novelty_w, 0.9)
    source5 = _source5_jax(
        high_return=ret_hat > 0.0,
        near_life_loss=life_w > 0.0,
        novelty_hi=novelty_w >= novelty_threshold,
        failure_recovery=jnp.zeros_like(life_w, dtype=bool),
    )

    selected_game_ids = jnp.take(game_ids_w, top_idx, axis=0)
    selected = {
        "keys": jnp.take(keys_w, top_idx, axis=0),
        "context": _context_embeddings(params, selected_game_ids),
        "teacher_policy": teacher_policy,
        "teacher_value": teacher_value,
        "importance": top_scores,
        "game_id": selected_game_ids,
        "source5": jnp.take(source5, top_idx, axis=0),
        "age": jnp.zeros((int(write_count),), dtype=jnp.float32),
        "valid": jnp.isfinite(top_scores),
    }
    new_atoms = {
        name: _pad_selected_rows(value, int(write_count), int(max_writes))
        for name, value in selected.items()
    }
    new_bank = hot_insert(bank, new_atoms)

    new_value_norm = dict(value_norm)
    new_value_norm["adv_mean"] = adv_mean
    new_value_norm["adv_mad"] = adv_mad
    new_value_norm["delta_mean"] = delta_mean
    new_value_norm["delta_mad"] = delta_mad
    new_value_norm["ret_mean"] = ret_mean
    new_value_norm["ret_mad"] = ret_mad
    new_value_norm["write_initialized"] = jnp.asarray(True, dtype=bool)
    new_value_norm = _value_norm_update_jit(
        new_value_norm,
        flat_returns,
        value_momentum,
        sigma_floor,
    )
    metrics = jnp.asarray(
        [
            jnp.sum(new_atoms["valid"].astype(jnp.float32)),
            jnp.sum(new_bank["valid"].astype(jnp.float32)),
            jnp.mean(jnp.where(jnp.isfinite(scores), scores, 0.0)),
        ],
        dtype=jnp.float32,
    )
    return new_bank, new_value_norm, metrics


def lm_writes(
    params,
    ema_params,
    bank,
    traj: FastLMRollout,
    last_value,
    advantages,
    returns,
    value_norm,
    game_id,
    rng,
    mu_g,
    sigma_g,
    hp,
    *,
    write_batch_size: int,
    write_count: int,
    max_writes: int,
    gamma: float,
    teacher_temperature: float,
    write_momentum: float = 0.99,
    write_eps: float = 1.0e-6,
    value_momentum: float = 0.99,
    sigma_floor: float = 1.0e-3,
):
    values = _hp_values(hp, bank)
    return _lm_writes_jit(
        params,
        ema_params,
        bank,
        traj,
        last_value,
        advantages,
        returns,
        value_norm,
        jnp.asarray(game_id, dtype=jnp.int32),
        rng,
        float(mu_g),
        float(sigma_g),
        *values,
        int(write_batch_size),
        int(write_count),
        int(max_writes),
        float(gamma),
        float(teacher_temperature),
        float(write_momentum),
        float(write_eps),
        float(value_momentum),
        float(sigma_floor),
    )


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
        "top_k",
        "n_games",
        "d_k",
        "d_c",
        "d_m",
        "act_dim",
        "guard_project",
    ),
)
def _lm_guarded_update_jit(
    params,
    opt_state,
    batch: TrainBatch,
    game_id,
    bank_arrays,
    mu_g,
    sigma_g,
    rng,
    learning_rate: float,
    update_epochs: int,
    num_minibatches: int,
    minibatch_size: int,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
    tau_r,
    beta_c,
    beta_I,
    beta_a,
    w_rho,
    w_c,
    b0,
    top_k,
    guard_atoms,
    guard_bank,
    omega,
    lambda_total,
    kappa,
    lambda_v,
    stability_alpha,
    rho_omega,
    n_games,
    d_k,
    d_c,
    d_m,
    act_dim,
    guard_project: bool,
):
    hp = _jit_hp(tau_r, beta_c, beta_I, beta_a, w_rho, w_c, b0, top_k)
    dims = (int(n_games), int(d_k), int(d_c), int(d_m), int(act_dim))
    batch_size = int(num_minibatches) * int(minibatch_size)
    tx = optax.chain(
        optax.clip_by_global_norm(float(max_grad_norm)),
        optax.adam(learning_rate=learning_rate),
    )

    def guard_loss_fn(p):
        return latent_conservation_loss(
            p,
            guard_atoms,
            guard_bank,
            hp,
            lambda_v=lambda_v,
            dims=dims,
        )  # spec §11

    def epoch_step(carry, _):
        params, opt_state, rng, omega = carry
        rng, perm_key = jax.random.split(rng)
        permutation = jax.random.permutation(perm_key, batch_size)
        minibatches = _make_minibatches(batch, permutation, int(num_minibatches), int(minibatch_size))

        def minibatch_step(carry, minibatch):
            params, opt_state, omega = carry

            def loss_fn(p):
                return _lm_ppo_loss(
                    p,
                    minibatch,
                    game_id,
                    bank_arrays,
                    hp,
                    mu_g,
                    sigma_g,
                    clip_coef,
                    vf_coef,
                    ent_coef,
                )

            (loss, aux), g_task = jax.value_and_grad(loss_fn, has_aux=True)(params)
            cons_loss, g_guard = jax.value_and_grad(guard_loss_fn)(params)  # spec §11
            g_total, next_omega, guard_metrics = _combine_latent_guard_core(
                params,
                g_task,
                g_guard,
                omega,
                lambda_total,
                kappa,
                stability_alpha,
                rho_omega,
                guard_project,
            )
            finite = jnp.logical_and(jnp.isfinite(loss), jnp.isfinite(cons_loss))
            finite = jnp.logical_and(finite, jnp.logical_not(guard_metrics.nonfinite))
            safe_grads = jax.tree_util.tree_map(
                lambda grad: jnp.where(finite, grad, jnp.zeros_like(grad)),
                g_total,
            )
            updates, new_opt_state = tx.update(safe_grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            params = _select_tree(new_params, params, finite)
            opt_state = _select_tree(new_opt_state, opt_state, finite)
            omega = _select_tree(next_omega, omega, finite)
            safe_loss = jnp.where(jnp.isfinite(loss), loss, 0.0)
            safe_aux = jnp.where(jnp.isfinite(aux), aux, 0.0)
            safe_cons = jnp.where(jnp.isfinite(cons_loss), cons_loss, 0.0)
            metrics = jnp.concatenate(
                [
                    jnp.asarray([safe_loss], dtype=jnp.float32),
                    safe_aux,
                    jnp.asarray(
                        [
                            finite.astype(jnp.float32),
                            safe_cons,
                            guard_metrics.g_task_norm,
                            guard_metrics.g_guard_norm,
                            guard_metrics.g_guard_clipped_norm,
                            guard_metrics.g_safe_norm,
                            guard_metrics.g_total_norm,
                            guard_metrics.projection_ratio,
                            guard_metrics.conflict_dot,
                            guard_metrics.nonfinite.astype(jnp.float32),
                        ],
                        dtype=jnp.float32,
                    ),
                ],
                axis=0,
            )
            return (params, opt_state, omega), metrics

        (params, opt_state, omega), metrics = jax.lax.scan(
            minibatch_step,
            (params, opt_state, omega),
            minibatches,
        )
        return (params, opt_state, rng, omega), jnp.mean(metrics, axis=0)

    (params, opt_state, rng, omega), metrics = jax.lax.scan(
        epoch_step,
        (params, opt_state, rng, omega),
        None,
        length=int(update_epochs),
    )
    return params, opt_state, rng, omega, jnp.mean(metrics, axis=0)


def _guard_field(guard, name: str, default=None):
    if isinstance(guard, dict):
        return guard.get(name, default)
    return getattr(guard, name, default)


def _guard_arrays(value):
    return {name: jnp.asarray(array) for name, array in value.items()}


def _guard_omega(guard, params):
    omega = _guard_field(guard, "omega", None)
    if omega is None:
        return zeros_omega_like(params)
    return jax.tree_util.tree_map(jnp.asarray, omega)


def lm_guarded_update(
    params,
    opt_state,
    batch: TrainBatch,
    game_id,
    bank_arrays,
    mu_g,
    sigma_g,
    rng,
    learning_rate: float,
    update_epochs: int,
    num_minibatches: int,
    minibatch_size: int,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
    *,
    guard,
    omega,
    n_games: int,
    d_k: int,
    d_c: int,
    d_m: int,
    act_dim: int,
    hp=None,
):
    values = _hp_values(hp, bank_arrays)
    return _lm_guarded_update_jit(
        params,
        opt_state,
        batch,
        jnp.asarray(game_id, dtype=jnp.int32),
        bank_arrays,
        float(mu_g),
        float(sigma_g),
        rng,
        float(learning_rate),
        int(update_epochs),
        int(num_minibatches),
        int(minibatch_size),
        float(clip_coef),
        float(vf_coef),
        float(ent_coef),
        float(max_grad_norm),
        *values,
        _guard_arrays(_guard_field(guard, "atoms")),
        _guard_arrays(_guard_field(guard, "bank")),
        omega,
        jnp.asarray(_guard_field(guard, "lambda_total", 1.0), dtype=jnp.float32),
        jnp.asarray(_guard_field(guard, "kappa", 0.75), dtype=jnp.float32),
        jnp.asarray(_guard_field(guard, "lambda_v", 1.0), dtype=jnp.float32),
        jnp.asarray(_guard_field(guard, "stability_alpha", 10.0), dtype=jnp.float32),
        jnp.asarray(_guard_field(guard, "rho_omega", 0.99), dtype=jnp.float32),
        int(n_games),
        int(d_k),
        int(d_c),
        int(d_m),
        int(act_dim),
        bool(_guard_field(guard, "project", True)),
    )


def _completed_returns_from_rollout(traj: FastLMRollout, tracker: EpisodeReturnTracker) -> list[float]:
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


def _block_until_ready(tree):
    leaves = jax.tree_util.tree_leaves(tree)
    for leaf in leaves:
        if hasattr(leaf, "block_until_ready"):
            leaf.block_until_ready()


def _device_scalar_float(x) -> float:
    return float(np.asarray(jax.device_get(x), dtype=np.float32))


def _prepare_hot_bank(hot_bank, cfg):
    if hot_bank is None:
        return empty_hot_bank(int(cfg.hot_capacity), int(cfg.d_k), int(cfg.d_c), ACT_DIM)
    if isinstance(hot_bank, dict):
        return {name: jnp.asarray(value) for name, value in hot_bank.items()}
    if hasattr(hot_bank, "to_retrieval_arrays"):
        return pad_bank(
            hot_bank,
            int(cfg.hot_capacity),
            d_k=int(cfg.d_k),
            d_c=int(cfg.d_c),
            act_dim=ACT_DIM,
        )
    raise TypeError("hot_bank must be None, a reader-compatible dict, or MemoryBank-like")


def _write_shape_config(cfg, batch_size: int):
    write_batch_size = min(_cfg_int(cfg, "write_subsample_size", DEFAULT_WRITE_SUBSAMPLE_SIZE), int(batch_size))
    requested = 0
    if float(cfg.write_top_fraction) > 0.0:
        requested = int(math.ceil(float(cfg.write_top_fraction) * float(write_batch_size)))
    requested = max(requested, int(cfg.write_min_quota))
    default_max = max(requested, 0)
    if not hasattr(cfg, "max_writes_per_segment"):
        max_writes = default_max
    else:
        max_writes = _cfg_int(cfg, "max_writes_per_segment", DEFAULT_MAX_WRITES_PER_SEGMENT)
    max_writes = min(max_writes, write_batch_size)
    write_count = min(max_writes, requested)
    return int(write_batch_size), int(write_count), int(max_writes)


def train_living_memory_fast(
    game,
    game_id,
    n_games,
    cfg,
    seed,
    *,
    init_params=None,
    hot_bank=None,
    ema_params=None,
    value_norm=None,
    guard=None,
) -> dict:
    """Train one Atari game with XLA-scan rollout and jitted GPU hot-bank writes."""

    cfg = FastLMConfig() if cfg is None else cfg
    num_updates = _validate_fast_config(cfg)
    batch_size = int(cfg.num_envs) * int(cfg.num_steps)
    minibatch_size = batch_size // int(cfg.num_minibatches)
    hp = default_retrieval_hp(int(cfg.top_k))
    hp_values = _hp_values(hp, {"keys": jnp.zeros((int(cfg.hot_capacity), int(cfg.d_k)), dtype=jnp.float32)})
    write_batch_size, write_count, max_writes = _write_shape_config(cfg, batch_size)
    write_every = int(cfg.write_every)
    write_enabled = write_every > 0 and write_count > 0 and max_writes > 0

    rng = jax.random.PRNGKey(int(seed))
    rng, init_key = jax.random.split(rng)
    params = init_params
    if params is None:
        params = mem_init(
            init_key,
            int(n_games),
            int(cfg.hot_capacity),
            d_k=int(cfg.d_k),
            d_c=int(cfg.d_c),
            d_m=int(cfg.d_m),
            act_dim=ACT_DIM,
            top_k=int(cfg.top_k),
        )
    if ema_params is None:
        ema_params = params
    hot_bank = _prepare_hot_bank(hot_bank, cfg)
    value_norm_state = _as_value_norm_state(value_norm)
    guard_omega = None if guard is None else _guard_omega(guard, params)

    value_momentum = _cfg_float(cfg, "value_norm_momentum", getattr(value_norm, "momentum", 0.99))
    sigma_floor = _cfg_float(cfg, "value_sigma_floor", getattr(value_norm, "sigma_floor", 1.0e-3))
    write_momentum = _cfg_float(cfg, "write_stats_momentum", 0.99)
    write_eps = _cfg_float(cfg, "write_stats_eps", 1.0e-6)

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
    game_id_vec = jnp.full((int(cfg.num_envs),), int(game_id), dtype=jnp.int32)

    for update in range(1, num_updates + 1):
        update_start = time.perf_counter()
        bank_for_segment = hot_bank
        mu_segment = _device_scalar_float(value_norm_state["mu"])
        sigma_segment = max(_device_scalar_float(value_norm_state["sigma"]), 1.0e-6)

        handle, next_obs, rng, traj = rollout(
            params,
            handle,
            next_obs,
            game_id_vec,
            bank_for_segment,
            float(mu_segment),
            float(sigma_segment),
            rng,
            *hp_values,
        )
        completed_this_update = _completed_returns_from_rollout(traj, tracker)
        if completed_this_update:
            recent_returns.extend(completed_this_update)
        last_curve_value = _mean_or_previous(recent_returns, last_curve_value)
        returns_curve.append(last_curve_value)

        last_value = _last_value_jit(
            params,
            next_obs,
            game_id_vec,
            bank_for_segment,
            float(mu_segment),
            float(sigma_segment),
            *hp_values,
        )
        advantages, returns = gae(
            traj.rewards,
            traj.dones,
            traj.values,
            last_value,
            float(cfg.gamma),
            float(cfg.gae_lambda),
        )

        if write_enabled and update % write_every == 0:
            rng, write_rng = jax.random.split(rng)
            hot_bank, value_norm_state, write_metrics = lm_writes(
                params,
                ema_params,
                bank_for_segment,
                traj,
                last_value,
                advantages,
                returns,
                value_norm_state,
                int(game_id),
                write_rng,
                float(mu_segment),
                float(sigma_segment),
                hp,
                write_batch_size=int(write_batch_size),
                write_count=int(write_count),
                max_writes=int(max_writes),
                gamma=float(cfg.gamma),
                teacher_temperature=float(cfg.teacher_temperature),
                write_momentum=float(write_momentum),
                write_eps=float(write_eps),
                value_momentum=float(value_momentum),
                sigma_floor=float(sigma_floor),
            )
            _block_until_ready((hot_bank, value_norm_state, write_metrics))
        else:
            value_norm_state = _value_norm_update_jit(
                value_norm_state,
                returns,
                float(value_momentum),
                float(sigma_floor),
            )
            _block_until_ready(value_norm_state)

        batch: TrainBatch = _flatten_rollout(
            traj.obs,
            traj.actions,
            traj.logprobs,
            advantages,
            returns,
            traj.values,
            batch_size,
        )
        if guard is None:
            params, opt_state, rng, metrics = lm_update(
                params,
                opt_state,
                batch,
                int(game_id),
                bank_for_segment,
                float(mu_segment),
                float(sigma_segment),
                rng,
                float(_learning_rate(cfg, update, num_updates)),
                int(cfg.update_epochs),
                int(cfg.num_minibatches),
                int(minibatch_size),
                float(cfg.clip_coef),
                float(cfg.vf_coef),
                float(cfg.ent_coef),
                float(cfg.max_grad_norm),
                hp=hp,
            )
        else:
            params, opt_state, rng, guard_omega, metrics = lm_guarded_update(
                params,
                opt_state,
                batch,
                int(game_id),
                bank_for_segment,
                float(mu_segment),
                float(sigma_segment),
                rng,
                float(_learning_rate(cfg, update, num_updates)),
                int(cfg.update_epochs),
                int(cfg.num_minibatches),
                int(minibatch_size),
                float(cfg.clip_coef),
                float(cfg.vf_coef),
                float(cfg.ent_coef),
                float(cfg.max_grad_norm),
                guard=guard,
                omega=guard_omega,
                n_games=int(n_games),
                d_k=int(cfg.d_k),
                d_c=int(cfg.d_c),
                d_m=int(cfg.d_m),
                act_dim=ACT_DIM,
                hp=hp,
            )
        ema_params = ema_update(ema_params, params, float(cfg.tau_key))
        _block_until_ready((metrics, ema_params))

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
    mem_size = int(np.asarray(jax.device_get(jnp.sum(hot_bank["valid"].astype(jnp.int32)))))

    result = {
        "params": params,
        "ema_params": ema_params,
        "hot_bank": hot_bank,
        "value_norm": value_norm_state,
        "returns_curve": [float(v) for v in returns_curve],
        "final_return": final_return,
        "timesteps": timesteps,
        "mem_size": mem_size,
        "steps_per_sec": steps_per_sec,
    }
    if guard_omega is not None:
        result["guard_omega"] = guard_omega
    return result


__all__ = [
    "FastLMConfig",
    "FastLMRollout",
    "GuardCombineMetrics",
    "combine_latent_guard_grads",
    "empty_hot_bank",
    "hot_insert",
    "hot_novelty",
    "lm_guarded_update",
    "lm_writes",
    "train_living_memory_fast",
    "write_importance",
]
