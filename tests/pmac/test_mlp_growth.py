import numpy as np
import jax
import jax.numpy as jnp

from pmac.models.mlp import grow_adapter, init_mlp, mlp_apply, num_params


def test_grow_adapter_leaves_output_unchanged_and_increases_params():
    key = jax.random.PRNGKey(0)
    params = init_mlp(key, [3, 5, 2])
    x = jnp.array([[1.0, 0.0, -1.0], [0.5, 2.0, 1.0]])
    before = mlp_apply(params, x)

    grown = grow_adapter(jax.random.PRNGKey(1), params, hidden_dim=5, rank=2)
    after = mlp_apply(grown, x)

    assert np.allclose(np.asarray(after), np.asarray(before), atol=1e-5)
    assert num_params(grown) > num_params(params)
