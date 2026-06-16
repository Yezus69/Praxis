import inspect

import numpy as np
import jax
import jax.numpy as jnp

from pmac.rl_update import PMACUpdateConfig, combine_grads, ppo_pmac_update
from pmac.stability import zeros_omega_like
from pmac.tree_utils import tree_dot, tree_norm


def _tree_allclose(a, b, atol=1e-6):
    return all(
        np.allclose(np.asarray(x), np.asarray(y), atol=atol)
        for x, y in zip(
            jax.tree_util.tree_leaves(a),
            jax.tree_util.tree_leaves(b),
        )
    )


def _tree_all_zero(a, atol=1e-6):
    return all(
        np.allclose(np.asarray(x), 0.0, atol=atol)
        for x in jax.tree_util.tree_leaves(a)
    )


def test_conflicting_guard_component_is_projected_away():
    g_new = {"w": jnp.array([-1.0, 0.0])}
    guard = {"w": jnp.array([1.0, 0.0])}
    cfg = PMACUpdateConfig(guard_correction=False, stability=False)

    out, metrics = combine_grads(
        g_new,
        [guard],
        jnp.asarray([0.0], dtype=jnp.float32),
        zeros_omega_like(g_new),
        cfg,
    )

    assert tree_dot(out, guard) >= -1e-5
    assert metrics.projection_ratio < 1.0


def test_aligned_guard_leaves_ppo_gradient_unchanged():
    g_new = {"w": jnp.array([1.0, 2.0])}
    guard = {"w": jnp.array([1.0, 0.0])}
    cfg = PMACUpdateConfig(guard_correction=False, stability=False)

    out, metrics = combine_grads(
        g_new,
        [guard],
        jnp.asarray([0.0], dtype=jnp.float32),
        zeros_omega_like(g_new),
        cfg,
    )

    assert _tree_allclose(out, g_new)
    assert np.isclose(metrics.projection_ratio, 1.0, atol=1e-6)


def test_total_guard_correction_norm_is_bounded():
    g_new = {"w": jnp.array([1.0, 0.0])}
    guard = {"w": jnp.array([100.0, 0.0])}
    cfg = PMACUpdateConfig(
        guard_total_beta=0.25,
        projection=False,
        stability=False,
    )

    _out, metrics = combine_grads(
        g_new,
        [guard],
        jnp.asarray([10.0], dtype=jnp.float32),
        zeros_omega_like(g_new),
        cfg,
    )

    assert metrics.total_guard_norm <= 0.25 * float(np.asarray(tree_norm(g_new))) + 1e-5


def test_stability_scaling_reduces_high_omega_update_magnitude():
    g_new = {"w": jnp.array([2.0, -2.0])}
    omega = {"w": jnp.array([10.0, 10.0])}
    stable_cfg = PMACUpdateConfig(stability=True)
    plain_cfg = PMACUpdateConfig(stability=False)

    stable, _ = combine_grads(g_new, [], jnp.asarray([]), omega, stable_cfg)
    plain, _ = combine_grads(g_new, [], jnp.asarray([]), omega, plain_cfg)

    assert tree_norm(stable) < tree_norm(plain)


def test_nonfinite_guard_gradient_sets_metric_and_zeroes_update():
    g_new = {"w": jnp.array([1.0, 0.0])}
    guard = {"w": jnp.array([jnp.nan, 0.0])}

    out, metrics = combine_grads(
        g_new,
        [guard],
        jnp.asarray([1.0], dtype=jnp.float32),
        zeros_omega_like(g_new),
        PMACUpdateConfig(stability=False),
    )

    assert metrics.nonfinite is True
    assert _tree_all_zero(out)


def test_no_guards_and_no_stability_returns_ppo_gradient():
    g_new = {"w": jnp.array([1.0, -2.0]), "b": jnp.array([0.5])}
    cfg = PMACUpdateConfig(stability=False)

    out, metrics = combine_grads(
        g_new,
        [],
        jnp.asarray([], dtype=jnp.float32),
        zeros_omega_like(g_new),
        cfg,
    )

    assert _tree_allclose(out, g_new)
    assert metrics.conflict_dots == []
    assert metrics.clipped_guard_count == 0


def test_ppo_pmac_update_signature_smoke():
    params = inspect.signature(ppo_pmac_update).parameters
    for name in (
        "params",
        "opt_state",
        "batch",
        "game_onehot",
        "guard_obs",
        "guard_lambdas",
        "omega",
        "cfg",
    ):
        assert name in params
