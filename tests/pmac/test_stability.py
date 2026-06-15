import numpy as np
import jax.numpy as jnp

from pmac.stability import scale_by_stability, update_omega, zeros_omega_like


def test_zero_omega_makes_stability_scaling_identity():
    g = {"w": jnp.array([1.0, -2.0]), "b": jnp.array([3.0])}
    omega = zeros_omega_like(g)
    out = scale_by_stability(g, omega, alpha=10.0)

    assert np.allclose(np.asarray(out["w"]), np.asarray(g["w"]))
    assert np.allclose(np.asarray(out["b"]), np.asarray(g["b"]))


def test_larger_omega_reduces_update_magnitude():
    g = {"w": jnp.array([2.0, -2.0])}
    low = {"w": jnp.array([0.0, 0.0])}
    high = {"w": jnp.array([10.0, 10.0])}

    out_low = scale_by_stability(g, low, alpha=1.0)
    out_high = scale_by_stability(g, high, alpha=1.0)
    assert np.all(np.abs(np.asarray(out_high["w"])) < np.abs(np.asarray(out_low["w"])))


def test_update_omega_is_nonnegative_convex_combination():
    omega = {"w": jnp.array([0.2, 2.0])}
    params = {"w": jnp.array([3.0, -4.0])}
    guard_grad = {"w": jnp.array([0.5, -0.25])}
    rho = 0.75

    out = update_omega(omega, params, guard_grad, rho)
    target = np.abs(np.asarray(params["w"]) * np.asarray(guard_grad["w"]))
    old = np.asarray(omega["w"])
    got = np.asarray(out["w"])

    assert np.all(got >= 0.0)
    assert np.all(got <= np.maximum(old, target) + 1e-6)
    assert np.all(got >= np.minimum(old, target) - 1e-6)
