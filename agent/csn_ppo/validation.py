"""Fixed validation bank for CSN-PPO P6."""

from __future__ import annotations

import flax
import jax
import jax.numpy as jnp

from agent.csn_ppo import coverage_probes
from agent.csn_ppo.guarded_loss import (
    condition_guard_kl_inputs,
    gaussian_kl,
)
from praxis import contract


@flax.struct.dataclass
class ValidationBank:
    current_keys: jnp.ndarray
    history_keys: jnp.ndarray
    sentinel_failure_keys: jnp.ndarray
    synthetic_obs: jnp.ndarray
    best_current: jnp.ndarray
    best_history: jnp.ndarray
    best_failure: jnp.ndarray
    best_synthetic_kl_p95: jnp.ndarray


def _prng_key(rng):
    rng = jnp.asarray(rng)
    if rng.ndim == 0:
        return jax.random.PRNGKey(rng.astype(jnp.uint32))
    if rng.ndim == 2 and rng.shape[0] == 1:
        return rng[0]
    return rng


def _policy_params(params, normalizer_params):
    if hasattr(params, "policy") and hasattr(params, "value"):
        return normalizer_params, params.policy, params.value
    return normalizer_params, params


def _episode_length(env, cfg):
    env_cfg = getattr(env, "_config", None)
    if env_cfg is not None and hasattr(env_cfg, "episode_length"):
        return int(env_cfg.episode_length)
    if hasattr(cfg, "episode_length"):
        return int(cfg.episode_length)
    return int(contract.EPISODE_LENGTH)


def _score_seed_bank(env, keys, policy, cfg):
    """P6 closed-loop validation score on a fixed independent seed bank."""
    horizon = _episode_length(env, cfg)
    states0 = env.reset(keys)
    step_rng = jax.random.fold_in(keys[0], horizon)
    done0 = jnp.zeros((keys.shape[0],), dtype=jnp.float32)

    def scan_step(carry, _):
        state, done, rng = carry
        rng, action_rng = jax.random.split(rng)
        action, _ = policy(state.obs, action_rng)
        action = jnp.where(done[:, None] > 0.0, jnp.zeros_like(action), action)
        next_state = env.step(state, action)
        active = 1.0 - done
        coverage_delta = next_state.metrics[contract.METRIC_COVERAGE] * active
        collision_delta = next_state.metrics[contract.METRIC_COLLISION] * active
        next_done = jnp.maximum(done, next_state.done)
        return (next_state, next_done, rng), (coverage_delta, collision_delta)

    (_, _, _), (coverage_t, collision_t) = jax.lax.scan(
        scan_step,
        (states0, done0, step_rng),
        (),
        length=horizon,
    )
    episode_coverage = jnp.sum(coverage_t, axis=0)
    episode_collision = jnp.sum(collision_t, axis=0)
    return {
        "score": jnp.mean(episode_coverage),
        "coverage": jnp.mean(episode_coverage),
        "collision_rate": jnp.mean(episode_collision),
    }


def _sorted_p95(values):
    sorted_values = jnp.sort(values)
    p95_idx = jnp.asarray(0.95 * (sorted_values.shape[0] - 1), dtype=jnp.int32)
    return sorted_values[p95_idx]


def _synthetic_reference_mean(synthetic_obs, cfg):
    return jax.vmap(lambda obs: coverage_probes.analytic_coverage_teacher(obs, cfg))(
        synthetic_obs
    )


def create_validation_bank(rng, cfg, bank_size=None, synthetic_size=None):
    """P6: build deterministic seed banks and P5 synthetic probes once."""
    rng = _prng_key(rng)
    size = int(cfg.num_eval_envs if bank_size is None else bank_size)
    probe_size = int(
        cfg.synthetic_probe_batch_size if synthetic_size is None else synthetic_size
    )
    current_rng = jax.random.fold_in(rng, 0)
    history_rng = jax.random.fold_in(rng, 1)
    failure_rng = jax.random.fold_in(rng, 2)
    synthetic_rng = jax.random.fold_in(rng, 3)
    return ValidationBank(
        current_keys=jax.random.split(current_rng, size),
        history_keys=jax.random.split(history_rng, size),
        sentinel_failure_keys=jax.random.split(failure_rng, size),
        synthetic_obs=coverage_probes.generate_cover_probes(synthetic_rng, probe_size),
        best_current=jnp.asarray(-jnp.inf, dtype=jnp.float32),
        best_history=jnp.asarray(-jnp.inf, dtype=jnp.float32),
        best_failure=jnp.asarray(-jnp.inf, dtype=jnp.float32),
        best_synthetic_kl_p95=jnp.asarray(jnp.inf, dtype=jnp.float32),
    )


def evaluate_validation_bank(
    env,
    validation_bank: ValidationBank,
    params,
    normalizer_params,
    make_policy,
    apply_policy_value,
    cfg,
):
    """P6 + Invariant 7: evaluate fixed validation using the current normalizer."""
    policy = make_policy(
        _policy_params(params, normalizer_params),
        deterministic=True,
    )
    current = _score_seed_bank(env, validation_bank.current_keys, policy, cfg)
    history = _score_seed_bank(env, validation_bank.history_keys, policy, cfg)
    failure = _score_seed_bank(env, validation_bank.sentinel_failure_keys, policy, cfg)

    pred_mean, pred_logstd, _ = apply_policy_value(
        params,
        normalizer_params,
        validation_bank.synthetic_obs,
    )
    reference_mean = _synthetic_reference_mean(validation_bank.synthetic_obs, cfg)
    reference_logstd = jnp.full_like(reference_mean, cfg.analytic_teacher_logstd)
    t_mean, t_logstd, p_mean, p_logstd = condition_guard_kl_inputs(
        reference_mean,
        reference_logstd,
        pred_mean,
        pred_logstd,
        cfg,
    )
    synthetic_kl = gaussian_kl(
        t_mean,
        t_logstd,
        p_mean,
        p_logstd,
    )
    synthetic_kl = jnp.minimum(
        synthetic_kl,
        jnp.asarray(cfg.max_atom_kl, dtype=synthetic_kl.dtype),
    )
    synthetic_kl_p95 = _sorted_p95(synthetic_kl)

    return {
        "validation/current_score": current["score"],
        "validation/current_collision_rate": current["collision_rate"],
        "validation/history_score": history["score"],
        "validation/history_coverage": history["coverage"],
        "validation/history_collision_rate": history["collision_rate"],
        "validation/sentinel_failure_score": failure["score"],
        "validation/sentinel_failure_collision_rate": failure["collision_rate"],
        "validation/synthetic_guard_kl": jnp.mean(synthetic_kl),
        "validation/synthetic_kl_p95": synthetic_kl_p95,
    }


def validation_best(validation_bank: ValidationBank):
    return {
        "current_score": validation_bank.best_current,
        "history_coverage": validation_bank.best_history,
        "sentinel_failure_score": validation_bank.best_failure,
        "synthetic_kl_p95": validation_bank.best_synthetic_kl_p95,
    }


def validation_regression_signals(metrics, best, cfg):
    history_regressed = (
        metrics["validation/history_coverage"]
        < best["history_coverage"] - cfg.validation_tolerance
    )
    best_synthetic = best.get(
        "synthetic_kl_p95",
        jnp.asarray(jnp.inf, dtype=metrics["validation/synthetic_kl_p95"].dtype),
    )
    synthetic_regressed = (
        metrics["validation/synthetic_kl_p95"]
        > best_synthetic + cfg.validation_kl_margin
    )
    return history_regressed, synthetic_regressed


def validation_regressed(metrics, best, cfg):
    history_regressed, synthetic_regressed = validation_regression_signals(
        metrics,
        best,
        cfg,
    )
    return bool(jnp.asarray(history_regressed) | jnp.asarray(synthetic_regressed))


def update_validation_best(validation_bank: ValidationBank, metrics):
    return validation_bank.replace(
        best_current=jnp.maximum(
            validation_bank.best_current,
            metrics["validation/current_score"],
        ),
        best_history=jnp.maximum(
            validation_bank.best_history,
            metrics["validation/history_coverage"],
        ),
        best_failure=jnp.maximum(
            validation_bank.best_failure,
            metrics["validation/sentinel_failure_score"],
        ),
        best_synthetic_kl_p95=jnp.minimum(
            validation_bank.best_synthetic_kl_p95,
            metrics["validation/synthetic_kl_p95"],
        ),
    )


def validation_guard_regressions(num_clusters):
    return {
        "regressed": jnp.ones((int(num_clusters),), dtype=jnp.bool_),
    }


def select_validation_update(
    metrics,
    best,
    cfg,
    params,
    opt_state,
    normalizer_params,
    best_safe_params,
    best_safe_opt_state,
    best_safe_normalizer_params,
    regression_count=0,
):
    """P6 update-acceptance gate: keep candidate or roll back to last-safe state."""
    regressed = bool(validation_regressed(metrics, best, cfg))
    next_count = int(regression_count) + 1 if regressed else 0
    should_rollback = next_count >= max(int(cfg.validation_patience), 1)
    has_safe = best_safe_params is not None
    if should_rollback and has_safe:
        return (
            best_safe_params,
            best_safe_opt_state,
            best_safe_normalizer_params,
            regressed,
            True,
        )
    return params, opt_state, normalizer_params, regressed, False
