import jax.numpy as jnp

from agent.csn_ppo.gradient_projection import project_conflicting_gradient, tree_dot


def test_projection_removes_conflict():
    g_ppo = {"x": jnp.array([-1.0, 0.0])}
    g_mem = {"x": jnp.array([1.0, 0.0])}
    g_safe = project_conflicting_gradient(g_ppo, [g_mem])
    assert tree_dot(g_safe, g_mem) >= -1e-6


def test_projection_leaves_non_conflict_alone():
    g_ppo = {"x": jnp.array([1.0, 0.0])}
    g_mem = {"x": jnp.array([1.0, 0.0])}
    g_safe = project_conflicting_gradient(g_ppo, [g_mem])
    assert jnp.allclose(g_safe["x"], g_ppo["x"])
