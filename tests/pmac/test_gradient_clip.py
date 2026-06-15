import numpy as np
import jax.numpy as jnp
import optax

from pmac.continual import clip_global, clip_guard_grad, optimizer_step_if_finite
from pmac.tree_utils import tree_norm


def _norm(tree):
    return float(np.asarray(tree_norm(tree)))


def test_guard_gradient_clip_caps_each_guard_norm_relative_to_new_gradient():
    g_new = {"w": jnp.array([3.0, 4.0])}
    g_guard = {"w": jnp.array([300.0, 400.0])}

    clipped = clip_guard_grad(g_guard, g_new, k=1.0)

    assert _norm(clipped) <= _norm(g_new) + 2e-6


def test_global_clip_caps_final_combined_gradient_norm():
    g_total = {"w": jnp.array([60.0, 80.0])}

    clipped = clip_global(g_total, max_norm=10.0)

    assert np.allclose(_norm(clipped), 10.0, atol=1e-5)


def test_nonfinite_gradient_skips_optimizer_step():
    params = {"w": jnp.array([1.0, -2.0])}
    g_total = {"w": jnp.array([jnp.inf, 1.0])}
    opt = optax.sgd(0.1)
    opt_state = opt.init(params)

    next_params, next_opt_state, info = optimizer_step_if_finite(
        opt, g_total, opt_state, params, max_grad_norm=10.0
    )

    assert info["nonfinite_steps"] == 1
    assert info["clipped"] is False
    assert next_opt_state is opt_state
    assert np.array_equal(np.asarray(next_params["w"]), np.asarray(params["w"]))
