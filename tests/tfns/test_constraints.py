from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax.numpy as jnp
import numpy as np

from tfns.behavior import (
    behavior_components,
    behavior_distance,
    tube_loss,
)
from tfns.protect.constraints import solve_constrained_qp


def test_qp_satisfies_each_conflicting_active_constraint_not_average():
    delta0 = jnp.array([1.0, 6.0], dtype=jnp.float32)
    G = jnp.array(
        [
            [1.0, 0.0],
            [-1.0, 0.2],
        ],
        dtype=jnp.float32,
    )
    m = jnp.array([0.0, 0.0], dtype=jnp.float32)

    delta, info = solve_constrained_qp(delta0, G, m, ridge=1.0e-12)

    assert delta is not None
    assert info["active"] == 2
    delta_np = np.asarray(delta)
    residual = np.asarray(G) @ delta_np - np.asarray(m)
    assert float(np.max(residual)) <= 1.0e-5

    G_avg = jnp.mean(G, axis=0, keepdims=True)
    m_avg = jnp.array([float(jnp.mean(m))], dtype=jnp.float32)
    delta_avg, avg_info = solve_constrained_qp(delta0, G_avg, m_avg, ridge=1.0e-12)

    assert delta_avg is not None
    assert avg_info["active"] == 1
    assert float(jnp.linalg.norm(delta - delta_avg)) > 1.0e-3
    assert float((G @ delta_avg - m)[0]) > 1.0e-3


def test_qp_non_finite_inputs_reject():
    delta0 = jnp.array([0.0, 1.0], dtype=jnp.float32)
    G = jnp.array([[jnp.nan, 0.0]], dtype=jnp.float32)
    m = jnp.array([0.0], dtype=jnp.float32)

    delta, info = solve_constrained_qp(delta0, G, m, ridge=1.0e-6)

    assert delta is None
    assert info["failed"] is True


def test_tube_loss_zero_positive_and_tail_reacts_to_spike():
    tol = 0.5
    inside = tube_loss(jnp.array([0.0, 0.2, 0.5], dtype=jnp.float32), tol)
    assert float(inside["total"]) == 0.0

    outside = tube_loss(jnp.array([0.0, 0.6], dtype=jnp.float32), tol)
    assert float(outside["total"]) > 0.0

    spike = jnp.zeros((20,), dtype=jnp.float32).at[0].set(2.5)
    spiky = tube_loss(spike, tol=0.5, tail_frac=0.10)
    assert float(spiky["tail"]) > float(spiky["mean"])
    assert float(spiky["tail"]) > 1.0e-3


def test_identical_teacher_current_behavior_distance_is_zero():
    logits = jnp.array([[0.2, -0.4, 1.1], [0.0, 0.3, -0.2]], dtype=jnp.float32)
    value = jnp.array([0.5, -1.0], dtype=jnp.float32)
    key = jnp.array([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32)

    comps = behavior_components(logits, value, key, logits, value, key)
    D = behavior_distance(comps, lambda_v=1.0, lambda_q=1.0)

    np.testing.assert_allclose(np.asarray(D), np.zeros((2,), dtype=np.float32), atol=1.0e-4)
