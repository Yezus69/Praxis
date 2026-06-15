import numpy as np
import jax.numpy as jnp

from pmac.projection import plasticity_ratio, project_conflicts
from pmac.tree_utils import tree_add_scaled, tree_dot


def test_conflicting_guard_component_is_removed():
    g_new = {"w": jnp.array([-1.0, 1.0])}
    guard = {"w": jnp.array([1.0, 0.0])}

    projected = project_conflicts(g_new, [guard])
    assert tree_dot(projected, guard) >= -1e-5
    assert np.allclose(np.asarray(projected["w"][0]), 0.0, atol=1e-5)


def test_aligned_guard_leaves_gradient_unchanged():
    g_new = {"w": jnp.array([1.0, 2.0])}
    guard = {"w": jnp.array([1.0, 0.0])}

    projected = project_conflicts(g_new, [guard])
    assert np.allclose(np.asarray(projected["w"]), np.asarray(g_new["w"]))


def test_two_guard_sequential_projection_matches_manual():
    g_new = {"w": jnp.array([-1.0, -2.0])}
    g1 = {"w": jnp.array([1.0, 0.0])}
    g2 = {"w": jnp.array([0.0, 2.0])}

    manual = g_new
    for guard in (g1, g2):
        coeff = jnp.minimum(tree_dot(manual, guard), 0.0) / (tree_dot(guard, guard) + 1e-8)
        manual = tree_add_scaled(manual, guard, -coeff)

    projected = project_conflicts(g_new, [g1, g2])
    assert np.allclose(np.asarray(projected["w"]), np.asarray(manual["w"]))


def test_plasticity_ratio_bounds_and_identity_without_guards():
    g_new = {"w": jnp.array([-1.0, 1.0])}
    projected = project_conflicts(g_new, [])
    ratio = plasticity_ratio(projected, g_new)

    assert np.allclose(np.asarray(ratio), 1.0, atol=1e-6)

    guard = {"w": jnp.array([1.0, 0.0])}
    projected = project_conflicts(g_new, [guard])
    ratio = plasticity_ratio(projected, g_new)
    assert 0.0 <= ratio <= 1.0 + 1e-6
