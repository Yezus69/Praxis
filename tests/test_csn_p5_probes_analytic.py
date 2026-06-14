import jax
import jax.numpy as jnp

from agent.csn_ppo.config import CSNPPOConfig
from agent.csn_ppo.coverage_probes import (
    PROBE_FAMILY_NAMES,
    analytic_coverage_teacher,
    generate_cover_probes,
    make_probe_crossing_obstacle_left_to_right,
    make_probe_obstacle_on_frontier_ray,
    pack_cover_obs,
)
from praxis import contract


CFG = CSNPPOConfig()
_OB0, _OB1 = contract.OBST_SLICE
_MASK0, _MASK1 = contract.MASK_SLICE
_FR0 = contract.FRONTIER_SLICE[0]
_COV0, _COV1 = contract.COVERED_SLICE


def _normalize(v):
    return v / (jnp.linalg.norm(v) + 1e-6)


def _obstacles(obs):
    return obs[_OB0:_OB1].reshape(contract.K, contract.PER_OBSTACLE_DIM)


def _frontier_dir(obs):
    return obs[_FR0:_FR0 + contract.ACT_DIM]


def test_5_1_obstacle_ahead_probe_avoids_forward_collision():
    obs = make_probe_obstacle_on_frontier_ray(jax.random.PRNGKey(0))
    action = analytic_coverage_teacher(obs, CFG)
    obstacle_direction = _normalize(_obstacles(obs)[0, :contract.ACT_DIM])

    assert jnp.dot(action, obstacle_direction) < 0.5


def test_5_2_open_frontier_probe_moves_toward_frontier():
    frontier_direction = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    obs = pack_cover_obs(
        agent_feat=jnp.zeros((contract.AGENT_DIM,), dtype=jnp.float32),
        obstacles=jnp.repeat(
            jnp.asarray([[0.9, 0.9, 0.0, 0.0]], dtype=jnp.float32),
            contract.K,
            axis=0,
        ),
        frontier=jnp.asarray([1.0, 0.0, 0.7], dtype=jnp.float32),
        covered=jnp.asarray(0.2, dtype=jnp.float32),
        mask=jnp.zeros((contract.K,), dtype=jnp.float32),
    )
    action = analytic_coverage_teacher(obs, CFG)

    assert jnp.dot(action, frontier_direction) > 0.8


def test_5_3_crossing_obstacle_probe_has_lateral_component():
    obs = make_probe_crossing_obstacle_left_to_right(jax.random.PRNGKey(1))
    action = analytic_coverage_teacher(obs, CFG)
    frontier_direction = _frontier_dir(obs)
    lateral_component = jnp.abs(
        action[0] * frontier_direction[1] - action[1] * frontier_direction[0]
    )

    assert lateral_component > 0.05


def test_5_4_generated_probes_are_valid_contract_observations():
    count = len(PROBE_FAMILY_NAMES) * 3
    probes = generate_cover_probes(jax.random.PRNGKey(2), count)

    assert probes.shape == (count, contract.OBS_DIM)
    assert jnp.all(jnp.isfinite(probes))

    agent = probes[:, contract.AGENT_SLICE[0]:contract.AGENT_SLICE[1]]
    obstacles = probes[:, _OB0:_OB1].reshape(
        count,
        contract.K,
        contract.PER_OBSTACLE_DIM,
    )
    mask = probes[:, _MASK0:_MASK1]
    frontier = probes[:, contract.FRONTIER_SLICE[0]:contract.FRONTIER_SLICE[1]]
    covered = probes[:, _COV0:_COV1]

    assert jnp.all(agent >= -1.0)
    assert jnp.all(agent <= 1.0)
    assert jnp.all(obstacles >= -1.0)
    assert jnp.all(obstacles <= 1.0)
    assert jnp.all((mask == 0.0) | (mask == 1.0))
    assert jnp.all(frontier[:, contract.ACT_DIM] >= 0.0)
    assert jnp.all(frontier[:, contract.ACT_DIM] <= 1.0)
    assert jnp.all(covered >= 0.0)
    assert jnp.all(covered <= 1.0)

    frontier_norm = jnp.linalg.norm(frontier[:, :contract.ACT_DIM], axis=-1)
    assert jnp.all(frontier_norm > 0.99)
    assert jnp.all(frontier_norm < 1.01)

    obstacle_dist = jnp.linalg.norm(obstacles[:, :, :contract.ACT_DIM], axis=-1)
    sort_key = jnp.where(mask > 0.5, obstacle_dist, 1e6 + obstacle_dist)
    assert jnp.all(jnp.diff(sort_key, axis=1) >= -1e-6)
