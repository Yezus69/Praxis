import numpy as np
import jax.numpy as jnp

from pmac import tree_utils as tu


def _flat(tree):
    return np.concatenate([np.asarray(x).reshape(-1) for x in tree["leaves"]])


def test_tree_dot_norm_and_add_scaled_match_flattened_numpy():
    a = {"leaves": [jnp.array([1.0, -2.0]), jnp.array([[3.0], [4.0]])]}
    b = {"leaves": [jnp.array([0.5, 2.0]), jnp.array([[1.0], [-1.0]])]}
    af = _flat(a)
    bf = _flat(b)

    assert np.allclose(np.asarray(tu.tree_dot(a, b)), np.dot(af, bf))
    assert np.allclose(np.asarray(tu.tree_norm(a)), np.linalg.norm(af))

    out = tu.tree_add_scaled(a, b, 0.25)
    assert np.allclose(_flat(out), af + 0.25 * bf)


def test_tree_linear_ops_are_leafwise_and_linear():
    a = {"leaves": [jnp.array([1.0, 2.0]), jnp.array([3.0])]}
    b = {"leaves": [jnp.array([4.0, 5.0]), jnp.array([6.0])]}

    assert np.allclose(_flat(tu.tree_add(a, b)), _flat(a) + _flat(b))
    assert np.allclose(_flat(tu.tree_sub(b, a)), _flat(b) - _flat(a))
    assert np.allclose(_flat(tu.tree_scale(a, 3.0)), 3.0 * _flat(a))
    assert np.allclose(np.asarray(tu.tree_l2sq(a)), np.dot(_flat(a), _flat(a)))
