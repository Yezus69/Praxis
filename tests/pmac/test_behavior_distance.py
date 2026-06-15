import numpy as np
import jax.numpy as jnp

from pmac.behavior_distance import cosine_distance, kl_categorical, mse, value_abs


def test_kl_categorical_invariants_and_temperature():
    teacher = jnp.array([[2.0, -1.0, 0.5], [0.1, 0.2, -0.3]])
    student = jnp.array([[1.5, -0.2, 0.1], [-0.7, 0.6, 0.0]])

    same = kl_categorical(teacher, teacher)
    d1 = kl_categorical(teacher, student, temperature=1.0)
    d2 = kl_categorical(teacher, student, temperature=2.0)

    assert same.shape == (2,)
    assert np.allclose(np.asarray(same), 0.0, atol=1e-6)
    assert np.all(np.asarray(d1) >= -1e-7)
    assert not np.allclose(np.asarray(d1), np.asarray(d2))


def test_mse_cosine_value_shapes_and_self_zero():
    x = jnp.array([[1.0, 2.0], [-3.0, 4.0], [0.0, 0.0]])
    y = jnp.array([[1.5, 2.5], [-3.0, 5.0], [1.0, 0.0]])

    assert mse(x, x).shape == (3,)
    assert np.allclose(np.asarray(mse(x, x)), 0.0)
    assert cosine_distance(x, x).shape == (3,)
    assert np.allclose(np.asarray(cosine_distance(x, x)), 0.0, atol=1e-6)
    assert value_abs(jnp.array([1.0, 2.0]), jnp.array([2.0, -1.0])).shape == (2,)
    assert np.all(np.isfinite(np.asarray(mse(x, y))))
