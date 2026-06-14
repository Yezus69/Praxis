"""Fixed-seed sentinel bank for coverage CSN-PPO Phase 2."""

from __future__ import annotations

from typing import Any

import flax
import jax
import jax.numpy as jnp

from agent.csn_ppo import criticality_coverage as cc
from agent.csn_ppo import rollout_mining
from agent.csn_ppo.config import CSNPPOConfig
from agent.csn_ppo.memory import BehavioralMemoryBatch
from praxis import contract


SENTINEL_FAILURE_SOURCE_ID = 2


@flax.struct.dataclass
class SentinelSeed:
    reset_rng: jnp.ndarray              # [N, 2] PRNG keys for CoverEnv.reset
    cluster_id: jnp.ndarray             # [N] int32 sentinel/domain cluster ids
    difficulty: jnp.ndarray             # [N] float32 deterministic difficulty label
    best_coverage: jnp.ndarray          # [C] best historical cluster coverage
    best_collision_rate: jnp.ndarray    # [C] best historical cluster collision rate
    champion_policy_id: jnp.ndarray     # [C] int32 champion policy id


@flax.struct.dataclass
class SentinelTrajectories:
    obs: jnp.ndarray              # [N, T, 28]
    reward: jnp.ndarray           # [N, T]
    coverage: jnp.ndarray         # [N]
    collision_rate: jnp.ndarray   # [N]
    cluster_id: jnp.ndarray       # [N]
    active: jnp.ndarray           # [N, T]


def create_sentinel_bank(rng, size, num_clusters):
    """README section 13: fixed deterministic sentinel worlds assigned to clusters."""
    reset_rng = jax.random.split(rng, int(size))
    idx = jnp.arange(int(size), dtype=jnp.int32)
    cluster_id = idx % jnp.asarray(num_clusters, dtype=jnp.int32)
    denom = jnp.maximum(jnp.asarray(num_clusters - 1, dtype=jnp.float32), 1.0)
    difficulty = cluster_id.astype(jnp.float32) / denom

    return SentinelSeed(
        reset_rng=reset_rng,
        cluster_id=cluster_id.astype(jnp.int32),
        difficulty=difficulty.astype(jnp.float32),
        best_coverage=jnp.full((int(num_clusters),), -jnp.inf, dtype=jnp.float32),
        best_collision_rate=jnp.full((int(num_clusters),), jnp.inf, dtype=jnp.float32),
        champion_policy_id=jnp.full((int(num_clusters),), -1, dtype=jnp.int32),
    )


def _policy_params(params, normalizer_params):
    if hasattr(params, "policy") and hasattr(params, "value"):
        return normalizer_params, params.policy, params.value
    return normalizer_params, params


def _cluster_mean(values, cluster_id, num_clusters):
    one_hot = jax.nn.one_hot(cluster_id, num_clusters, dtype=jnp.float32)
    counts = jnp.maximum(jnp.sum(one_hot, axis=0), 1.0)
    return jnp.sum(one_hot * values[:, None], axis=0) / counts, counts


def _episode_length(env):
    cfg = getattr(env, "_config", None)
    if cfg is not None and hasattr(cfg, "episode_length"):
        return int(cfg.episode_length)
    return int(contract.EPISODE_LENGTH)


def evaluate_sentinel_bank(
    env,
    bank: SentinelSeed,
    make_policy,
    params,
    normalizer_params,
    deterministic=True,
):
    """README section 13: deterministic closed-loop sentinel evaluation."""
    policy = make_policy(
        _policy_params(params, normalizer_params),
        deterministic=deterministic,
    )
    horizon = _episode_length(env)
    num_clusters = bank.best_coverage.shape[0]

    states0 = jax.vmap(env.reset)(bank.reset_rng)
    step_rng = jax.random.fold_in(bank.reset_rng[0], horizon)
    done0 = jnp.zeros((bank.reset_rng.shape[0],), dtype=jnp.float32)

    def scan_step(carry, _):
        state, done, rng = carry
        rng, action_rng = jax.random.split(rng)
        action, _ = policy(state.obs, action_rng)
        action = jnp.where(done[:, None] > 0.0, jnp.zeros_like(action), action)
        next_state = jax.vmap(env.step)(state, action)
        active = 1.0 - done
        coverage_delta = next_state.metrics[contract.METRIC_COVERAGE] * active
        collision_delta = next_state.metrics[contract.METRIC_COLLISION] * active
        reward = next_state.reward * active
        next_done = jnp.maximum(done, next_state.done)
        return (next_state, next_done, rng), (
            state.obs,
            reward,
            coverage_delta,
            collision_delta,
            active,
        )

    (_, _, _), (obs_t, reward_t, coverage_t, collision_t, active_t) = jax.lax.scan(
        scan_step,
        (states0, done0, step_rng),
        (),
        length=horizon,
    )

    obs = jnp.swapaxes(obs_t, 0, 1)
    reward = jnp.swapaxes(reward_t, 0, 1)
    active = jnp.swapaxes(active_t, 0, 1)
    episode_coverage = jnp.sum(coverage_t, axis=0)
    episode_collision_rate = jnp.sum(collision_t, axis=0)
    cluster_coverage, counts = _cluster_mean(
        episode_coverage,
        bank.cluster_id,
        num_clusters,
    )
    cluster_collision_rate, _ = _cluster_mean(
        episode_collision_rate,
        bank.cluster_id,
        num_clusters,
    )

    metrics = {
        "coverage": cluster_coverage,
        "collision_rate": cluster_collision_rate,
        "count": counts,
        "episode_coverage": episode_coverage,
        "episode_collision_rate": episode_collision_rate,
        "sentinel/coverage_mean": jnp.mean(cluster_coverage),
        "sentinel/coverage_min_cluster": jnp.min(cluster_coverage),
        "sentinel/collision_rate_mean": jnp.mean(cluster_collision_rate),
        "sentinel/collision_rate_max_cluster": jnp.max(cluster_collision_rate),
        "sentinel/worst_cluster_id": jnp.argmin(cluster_coverage).astype(jnp.int32),
    }
    trajectories = SentinelTrajectories(
        obs=obs,
        reward=reward,
        coverage=episode_coverage,
        collision_rate=episode_collision_rate,
        cluster_id=bank.cluster_id,
        active=active,
    )
    return metrics, trajectories


def _metric(current_metrics: Any, name: str):
    if isinstance(current_metrics, dict):
        return current_metrics[name]
    return getattr(current_metrics, name)


def detect_sentinel_regressions(
    current_metrics,
    bank: SentinelSeed,
    success_tol,
    collision_tol,
):
    """README section 13 regression rule, coverage-adapted per cluster."""
    coverage = _metric(current_metrics, "coverage")
    collision_rate = _metric(current_metrics, "collision_rate")
    success_bad = coverage < bank.best_coverage - success_tol
    collision_bad = collision_rate > bank.best_collision_rate + collision_tol
    regressed = success_bad | collision_bad
    return {
        "success_bad": success_bad,
        "coverage_bad": success_bad,
        "collision_bad": collision_bad,
        "regressed": regressed,
        "sentinel/regression_count": jnp.sum(regressed.astype(jnp.float32)),
        "sentinel/worst_cluster_id": jnp.argmax(regressed.astype(jnp.int32)).astype(jnp.int32),
    }


def _regression_mask(regressions):
    if isinstance(regressions, dict):
        return regressions["regressed"]
    return getattr(regressions, "regressed")


def _trajectory_field(trajectories, name):
    if isinstance(trajectories, dict):
        return trajectories[name]
    return getattr(trajectories, name)


def _sentinel_criticality(obs, sentinel_failure, cfg, criticality_bonus):
    c = (
        cfg.crit_w_collision * cc.collision_proximity(obs)
        + cfg.crit_w_frontier * cc.frontier_urgency(obs)
        + cfg.crit_w_dynamic * cc.dynamic_obstacle_score(obs)
        + cfg.crit_w_novelty * cc.coverage_novelty(obs)
        + criticality_bonus * sentinel_failure
    )
    return jnp.clip(c, cfg.crit_clip_min, cfg.crit_clip_max)


def mine_failed_sentinel_states(
    failed_trajectories,
    regressions,
    params,
    normalizer_params,
    criticality_bonus=5.0,
    apply_policy_value=None,
    cfg=None,
):
    """README sections 13 and 18: mine regressed sentinel states into memory."""
    cfg = CSNPPOConfig() if cfg is None else cfg
    obs = _trajectory_field(failed_trajectories, "obs")
    cluster_id = _trajectory_field(failed_trajectories, "cluster_id")
    try:
        active = _trajectory_field(failed_trajectories, "active")
    except (AttributeError, KeyError):
        active = jnp.ones(obs.shape[:-1], dtype=jnp.float32)

    n, t = obs.shape[:2]
    obs_flat = obs.reshape((n * t, obs.shape[-1]))
    cluster_flat = jnp.repeat(cluster_id.astype(jnp.int32), t)
    active_flat = active.reshape((n * t,))
    regressed_clusters = _regression_mask(regressions)
    sentinel_failure = regressed_clusters[cluster_flat].astype(jnp.float32) * active_flat

    crit = jax.vmap(
        lambda o, s: _sentinel_criticality(o, s, cfg, criticality_bonus)
    )(obs_flat, sentinel_failure)
    crit = jnp.where(sentinel_failure > 0.0, crit, -jnp.inf)

    k = min(int(cfg.atoms_per_rollout), int(obs_flat.shape[0]))
    selected_crit, idx = jax.lax.top_k(crit, k)
    selected_obs = obs_flat[idx]
    valid = jnp.isfinite(selected_crit)
    c = jnp.where(valid, selected_crit, cfg.crit_clip_min)

    if apply_policy_value is None:
        mean = jnp.zeros((k, contract.ACT_DIM), dtype=jnp.float32)
        logstd = jnp.full((k, contract.ACT_DIM), cfg.teacher_logstd_floor, dtype=jnp.float32)
        value = jnp.zeros((k,), dtype=jnp.float32)
    else:
        labeled = rollout_mining.label_probe_atoms(
            selected_obs,
            params,
            normalizer_params,
            apply_policy_value,
            cfg,
        )
        mean = labeled.mean
        logstd = labeled.logstd
        value = labeled.value

    weight = jnp.where(valid, jax.vmap(lambda x: cc.memory_weight(x, cfg))(c), 0.0)
    kl_budget = jax.vmap(lambda x: cc.kl_budget_from_c(x, cfg))(c)
    value_budget = jax.vmap(lambda x: cc.value_budget_from_c(x, cfg))(c)
    selected_cluster = cluster_flat[idx]
    source_id = jnp.full((k,), SENTINEL_FAILURE_SOURCE_ID, dtype=jnp.int32)

    return BehavioralMemoryBatch(
        obs=selected_obs,
        mean=mean,
        logstd=logstd,
        value=value,
        weight=weight,
        kl_budget=kl_budget,
        value_budget=value_budget,
        cluster_id=selected_cluster.astype(jnp.int32),
        source_id=source_id,
    )
