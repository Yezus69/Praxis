import jax.numpy as jnp
import numpy as np

from agent.csn_ppo.config import CSNPPOConfig
from agent.csn_ppo.gradient_projection import combine_safe_and_guard_grads
from agent.csn_ppo.guarded_loss import (
    GuardPressureState,
    coefficients_for_buckets,
    update_guard_pressure,
)


def _state(cluster_lambda):
    cluster_lambda = jnp.asarray(cluster_lambda, dtype=jnp.float32)
    return GuardPressureState(
        cluster_lambda=cluster_lambda,
        recovery_count=jnp.zeros(cluster_lambda.shape, dtype=jnp.int32),
    )


def test_3_1_regressed_cluster_lambda_increases():
    cfg = CSNPPOConfig()
    old = jnp.asarray([8.0, 8.0, 8.0, 8.0], dtype=jnp.float32)
    regressions = {
        "regressed": jnp.asarray([False, True, False, False], dtype=jnp.bool_),
    }

    new_state = update_guard_pressure(
        _state(old),
        regressions,
        recovered=jnp.logical_not(regressions["regressed"]),
        cfg=cfg,
    )

    assert float(new_state.cluster_lambda[1]) > float(old[1])
    assert float(new_state.cluster_lambda[0]) <= float(old[0])


def test_3_2_regressed_cluster_lambda_is_capped():
    cfg = CSNPPOConfig()
    state = _state([8.0, 8.0, 8.0, 8.0])
    regressions = {
        "regressed": jnp.asarray([False, True, False, False], dtype=jnp.bool_),
    }

    for _ in range(20):
        state = update_guard_pressure(
            state,
            regressions,
            recovered=jnp.logical_not(regressions["regressed"]),
            cfg=cfg,
        )

    assert float(state.cluster_lambda[1]) <= float(cfg.guard_lambda_max)
    assert float(state.cluster_lambda[1]) == float(cfg.guard_lambda_max)


def test_3_3_guard_combine_uses_per_bucket_coefficients():
    cfg = CSNPPOConfig()
    bucket_names = (
        "collision_boundary",
        "successful_goal",
        "dynamic_obstacle",
        "no_obstacle_straight_line",
    )
    cluster_lambda = jnp.asarray([8.0, 16.0, 8.0, 32.0], dtype=jnp.float32)
    nonuniform_coefs = coefficients_for_buckets(
        bucket_names=bucket_names,
        cluster_guard_lambda=cluster_lambda,
        cfg=cfg,
    )
    uniform_coefs = coefficients_for_buckets(
        bucket_names=bucket_names,
        cluster_guard_lambda=jnp.full((4,), cfg.guard_lambda_base, dtype=jnp.float32),
        cfg=cfg,
    )
    guard_grads = [
        {"x": jnp.eye(4, dtype=jnp.float32)[i]}
        for i in range(4)
    ]
    g_safe = {"x": jnp.zeros((4,), dtype=jnp.float32)}

    nonuniform = combine_safe_and_guard_grads(
        g_safe,
        guard_grads,
        nonuniform_coefs,
    )
    uniform = combine_safe_and_guard_grads(
        g_safe,
        guard_grads,
        uniform_coefs,
    )

    np.testing.assert_allclose(
        np.asarray(nonuniform["x"]),
        np.asarray(cluster_lambda),
    )
    assert float(nonuniform["x"][3]) > float(uniform["x"][3])
    assert float(nonuniform["x"][3] / uniform["x"][3]) == 4.0
