import numpy as np
import jax
import jax.numpy as jnp

from pmac.adapters.supervised import SupervisedAdapter
from pmac.models.mlp import init_mlp


def test_supervised_adapter_training_and_behavior_contracts():
    adapter = SupervisedAdapter(temperature=2.0)
    params = init_mlp(jax.random.PRNGKey(0), [2, 2], scale=0.01)
    batch = {
        "x": jnp.array([[2.0, 0.0], [0.0, 2.0], [3.0, 0.0], [0.0, 3.0]]),
        "y": jnp.array([0, 1, 0, 1], dtype=jnp.int32),
    }

    initial = adapter.current_loss(params, batch)
    lr = 0.3
    for _ in range(80):
        grads = jax.grad(adapter.current_loss)(params, batch)
        params = jax.tree_util.tree_map(lambda p, g: p - lr * g, params, grads)
    final = adapter.current_loss(params, batch)

    logits = adapter.behavior(params, batch)
    dist = adapter.distance(logits + 0.1, logits, batch)
    acc = adapter.evaluate_skill(params, batch)

    assert final < initial
    assert acc >= 0.99
    assert logits.shape == (4, 2)
    assert dist.shape == (4,)
    assert np.all(np.asarray(dist) >= -1e-7)
