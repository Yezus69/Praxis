import flax
import jax
import jax.numpy as jnp
import numpy as np

from agent.csn_ppo.config import CSNPPOConfig
from agent.csn_ppo.validation import (
    create_validation_bank,
    evaluate_validation_bank,
    select_validation_update,
    validation_regressed,
)
from praxis import contract


@flax.struct.dataclass
class _FakeState:
    obs: jnp.ndarray
    reward: jnp.ndarray
    done: jnp.ndarray
    metrics: dict


class _FakeEnvConfig:
    episode_length = 1


class _FakeEnv:
    _config = _FakeEnvConfig()

    def reset(self, keys):
        batch = keys.shape[0]
        obs = jnp.zeros((batch, contract.OBS_DIM), dtype=jnp.float32)
        seed_feature = jnp.mod(keys[:, 0].astype(jnp.float32), 17.0) / 100.0
        obs = obs.at[:, 0].set(seed_feature)
        zeros = jnp.zeros((batch,), dtype=jnp.float32)
        return _FakeState(
            obs=obs,
            reward=zeros,
            done=zeros,
            metrics={
                contract.METRIC_COVERAGE: zeros,
                contract.METRIC_COLLISION: zeros,
            },
        )

    def step(self, state, action):
        coverage = jnp.clip(0.5 + 0.25 * action[:, 0] + state.obs[:, 0], 0.0, 1.0)
        collision = jnp.zeros_like(coverage)
        return state.replace(
            reward=coverage,
            done=jnp.ones_like(coverage),
            metrics={
                contract.METRIC_COVERAGE: coverage,
                contract.METRIC_COLLISION: collision,
            },
        )


def _unpack_policy_tuple(policy_tuple):
    if len(policy_tuple) == 3:
        normalizer_params, policy_params, _ = policy_tuple
    else:
        normalizer_params, policy_params = policy_tuple
    return normalizer_params, policy_params


def _make_policy(policy_tuple, deterministic=True):
    del deterministic
    normalizer_params, policy_params = _unpack_policy_tuple(policy_tuple)

    def policy(obs, rng):
        del rng
        mean = policy_params["mean"] + normalizer_params["mean_shift"]
        return jnp.broadcast_to(mean, (obs.shape[0], contract.ACT_DIM)), {}

    return policy


def _apply_policy_value(params, normalizer_params, obs):
    mean = params["mean"] + normalizer_params["mean_shift"]
    mean = jnp.broadcast_to(mean, (obs.shape[0], contract.ACT_DIM))
    logstd = jnp.zeros_like(mean)
    value = jnp.zeros((obs.shape[0],), dtype=jnp.float32)
    return mean, logstd, value


def _cfg(**kwargs):
    return CSNPPOConfig(
        episode_length=1,
        num_eval_envs=4,
        synthetic_probe_batch_size=16,
        **kwargs,
    )


def test_6_1_validation_bank_metrics_are_deterministic():
    cfg = _cfg(validation_kl_limit=100.0)
    bank = create_validation_bank(jax.random.PRNGKey(0), cfg, bank_size=4, synthetic_size=16)
    params = {"mean": jnp.asarray([0.2, -0.1], dtype=jnp.float32)}
    normalizer_params = {"mean_shift": jnp.zeros((contract.ACT_DIM,), dtype=jnp.float32)}

    metrics_a = evaluate_validation_bank(
        _FakeEnv(),
        bank,
        params,
        normalizer_params,
        _make_policy,
        _apply_policy_value,
        cfg,
    )
    metrics_b = evaluate_validation_bank(
        _FakeEnv(),
        bank,
        params,
        normalizer_params,
        _make_policy,
        _apply_policy_value,
        cfg,
    )

    assert metrics_a.keys() == metrics_b.keys()
    for key in metrics_a:
        np.testing.assert_allclose(np.asarray(metrics_a[key]), np.asarray(metrics_b[key]))


def test_6_2_validation_regression_selects_last_safe_state():
    cfg = _cfg(validation_tolerance=0.05, validation_patience=3)
    metrics = {
        "validation/history_coverage": jnp.asarray(0.70, dtype=jnp.float32),
        "validation/synthetic_kl_p95": jnp.asarray(0.0, dtype=jnp.float32),
    }
    best = {
        "history_coverage": jnp.asarray(0.80, dtype=jnp.float32),
        "synthetic_kl_p95": jnp.asarray(0.0, dtype=jnp.float32),
    }

    assert validation_regressed(metrics, best, cfg)

    candidate_params = {"id": jnp.asarray(1)}
    candidate_opt_state = {"id": jnp.asarray(1)}
    candidate_normalizer = {"id": jnp.asarray(1)}
    safe_params = {"id": jnp.asarray(0)}
    safe_opt_state = {"id": jnp.asarray(0)}
    safe_normalizer = {"id": jnp.asarray(0)}

    (
        selected_params,
        selected_opt_state,
        selected_normalizer,
        regressed,
        rolled_back,
    ) = select_validation_update(
        metrics,
        best,
        cfg,
        candidate_params,
        candidate_opt_state,
        candidate_normalizer,
        safe_params,
        safe_opt_state,
        safe_normalizer,
        regression_count=0,
    )

    assert regressed
    assert not rolled_back
    assert selected_params is candidate_params
    assert selected_opt_state is candidate_opt_state
    assert selected_normalizer is candidate_normalizer

    (
        selected_params,
        selected_opt_state,
        selected_normalizer,
        regressed,
        rolled_back,
    ) = select_validation_update(
        metrics,
        best,
        cfg,
        candidate_params,
        candidate_opt_state,
        candidate_normalizer,
        safe_params,
        safe_opt_state,
        safe_normalizer,
        regression_count=cfg.validation_patience - 1,
    )

    assert regressed
    assert rolled_back
    assert selected_params is safe_params
    assert selected_opt_state is safe_opt_state
    assert selected_normalizer is safe_normalizer


def test_6_3_synthetic_validation_kl_catches_relative_mean_shift():
    cfg = _cfg(validation_kl_margin=0.1)
    bank = create_validation_bank(jax.random.PRNGKey(7), cfg, bank_size=4, synthetic_size=16)
    shifted_params = {"mean": jnp.asarray([5.0, -5.0], dtype=jnp.float32)}
    normalizer_params = {"mean_shift": jnp.zeros((contract.ACT_DIM,), dtype=jnp.float32)}

    metrics = evaluate_validation_bank(
        _FakeEnv(),
        bank,
        shifted_params,
        normalizer_params,
        _make_policy,
        _apply_policy_value,
        cfg,
    )
    best = {
        "history_coverage": metrics["validation/history_coverage"],
        "synthetic_kl_p95": jnp.asarray(0.0, dtype=jnp.float32),
    }

    assert float(metrics["validation/synthetic_kl_p95"]) > cfg.validation_kl_margin
    assert validation_regressed(metrics, best, cfg)
