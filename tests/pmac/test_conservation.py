import numpy as np
import jax.numpy as jnp

from pmac.behavior_distance import mse
from pmac.conservation import AnchorBatch, anchor_loss, conservation_loss


def test_anchor_loss_hinge_zero_and_quadratic_growth():
    d = jnp.array([0.1, 0.5, 1.5])
    tolerance = jnp.array([0.2, 0.5, 0.5])
    weight = jnp.array([1.0, 2.0, 3.0])
    loss = anchor_loss(d, tolerance, weight)

    assert np.allclose(np.asarray(loss[:2]), 0.0)
    assert np.allclose(np.asarray(loss[2]), 3.0 * (1.0**2))


def test_conservation_loss_zero_when_current_equals_teacher_and_positive_when_far():
    def behavior(params, x):
        return params["scale"] * x

    x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    params = {"scale": jnp.array(1.0)}
    teacher = behavior(params, x)
    batch = AnchorBatch(x=x, teacher=teacher, tolerance=0.0, weight=1.0)

    assert np.allclose(np.asarray(conservation_loss(behavior, params, batch, mse)), 0.0)

    far_params = {"scale": jnp.array(3.0)}
    assert conservation_loss(behavior, far_params, batch, mse) > 0.0


def test_conservation_loss_weight_scaling_is_linear():
    def behavior(params, x):
        return params["scale"] * x

    x = jnp.array([[1.0, 0.0], [0.0, 1.0]])
    teacher = x
    params = {"scale": jnp.array(2.0)}
    batch1 = {"x": x, "teacher": teacher, "tolerance": 0.0, "weight": 1.0}
    batch2 = {"x": x, "teacher": teacher, "tolerance": 0.0, "weight": 4.0}

    loss1 = conservation_loss(behavior, params, batch1, mse)
    loss2 = conservation_loss(behavior, params, batch2, mse)
    assert np.allclose(np.asarray(loss2), 4.0 * np.asarray(loss1))
