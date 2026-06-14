import jax
import jax.numpy as jnp

from agent.csn_ppo.guarded_loss import gaussian_kl
from agent.csn_ppo.synthetic_probes import (
    criticality_score,
    make_probe_no_obstacle,
    pack_contract_obs,
)


def test_probe_obs_shape():
    obs = make_probe_no_obstacle(jax.random.PRNGKey(0))
    assert obs.shape == (27,)


def test_probe_masks_valid():
    obs = make_probe_no_obstacle(jax.random.PRNGKey(0))
    mask = obs[23:27]
    assert jnp.all((mask == 0.0) | (mask == 1.0))


def test_gaussian_kl_consistency_for_unit_mean_shift():
    mean0 = jnp.zeros((1, 2), dtype=jnp.float32)
    logstd0 = jnp.zeros((1, 2), dtype=jnp.float32)
    mean1 = jnp.ones((1, 2), dtype=jnp.float32)
    logstd1 = jnp.zeros((1, 2), dtype=jnp.float32)

    kl = gaussian_kl(mean0, logstd0, mean1, logstd1)

    assert jnp.allclose(kl, jnp.array([1.0], dtype=jnp.float32), atol=1e-6)


def test_criticality_score_increases_with_collision_proximity():
    far_obs = pack_contract_obs(
        goal=(4.0, 0.0, 4.0, 0.0),
        agent_vel=(0.0, 0.0, 0.0),
        obstacles=jnp.zeros((4, 4), dtype=jnp.float32),
        mask=jnp.zeros((4,), dtype=jnp.float32),
    )
    near_collision_obs = pack_contract_obs(
        goal=(4.0, 0.0, 4.0, 0.0),
        agent_vel=(0.0, 0.0, 0.0),
        obstacles=jnp.array(
            [
                [0.1, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
                [0.0, 0.0, 0.0, 0.0],
            ],
            dtype=jnp.float32,
        ),
        mask=jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32),
    )

    far = criticality_score(far_obs, advantage_abs=0.0, novelty=0.0, sentinel_failure=0.0)
    near = criticality_score(
        near_collision_obs,
        advantage_abs=0.0,
        novelty=0.0,
        sentinel_failure=0.0,
    )

    assert near > far
