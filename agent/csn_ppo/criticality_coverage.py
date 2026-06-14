"""Coverage-observation criticality helpers for CSN-PPO Phase 1b."""

import jax
import jax.numpy as jnp

from praxis import contract


_OB0, _OB1 = contract.OBST_SLICE
_FR0, _FR1 = contract.FRONTIER_SLICE
_COV0 = contract.COVERED_SLICE[0]
_ARENA = contract.ARENA_HALF
_VMAX = contract.AGENT_MAX_SPEED
_COLL_THRESH_M = (
    contract.AGENT_RADIUS + contract.OBSTACLE_RADIUS + contract.COLLISION_MARGIN
)


def _obstacles(obs):
    """Returns [K, 4] normalized obstacle rows from a single 28D coverage obs."""
    return obs[_OB0:_OB1].reshape(contract.K, 4)


def obstacle_distances_m(obs):
    """Obstacle distances in meters; coverage masks are invariant all-ones."""
    o = _obstacles(obs)
    return _ARENA * jnp.sqrt(o[:, 0] ** 2 + o[:, 1] ** 2)


def collision_proximity(obs):
    """1 at contact, linearly decaying to 0 at twice the collision threshold."""
    d_min = jnp.min(obstacle_distances_m(obs))
    return jax.nn.relu(_COLL_THRESH_M * 2.0 - d_min) / (_COLL_THRESH_M * 2.0)


def dynamic_obstacle_score(obs):
    """Maximum obstacle speed in meters/second."""
    o = _obstacles(obs)
    speeds = jnp.sqrt(o[:, 2] ** 2 + o[:, 3] ** 2) * _VMAX
    return jnp.max(speeds)


def frontier_urgency(obs):
    """High when the nearest unvisited frontier is far and coverage is low."""
    fdist = obs[_FR1 - 1]
    covered = obs[_COV0]
    return fdist * (1.0 - covered)


def coverage_novelty(obs):
    """Pure-observation novelty proxy: unexplored area remaining."""
    return 1.0 - obs[_COV0]


def criticality_score(obs, advantage_abs, cfg):
    """README §19 criticality with coverage-specific feature plumbing."""
    c = (
        cfg.crit_w_advantage * advantage_abs
        + cfg.crit_w_collision * collision_proximity(obs)
        + cfg.crit_w_frontier * frontier_urgency(obs)
        + cfg.crit_w_dynamic * dynamic_obstacle_score(obs)
        + cfg.crit_w_novelty * coverage_novelty(obs)
    )
    return jnp.clip(c, cfg.crit_clip_min, cfg.crit_clip_max)


def memory_weight(c, cfg):
    """README §18: w_m = clip(c, w_min, w_max)."""
    return jnp.clip(c, cfg.crit_clip_min, cfg.crit_clip_max)


def kl_budget_from_c(c, cfg):
    """README §18: delta_m = delta0 / (1 + c)."""
    return cfg.guard_kl_budget / (1.0 + c)


def value_budget_from_c(c, cfg):
    """README §18: rho_m = rho0 / (1 + beta * c)."""
    return cfg.value_budget / (1.0 + cfg.value_budget_beta * c)


def cluster_id_for(obs):
    """Coverage cluster ids reused by the Phase 1a guard bucket constants."""
    coll = collision_proximity(obs) > 0.25
    dyn = dynamic_obstacle_score(obs) > 0.3
    near = jnp.min(obstacle_distances_m(obs)) < (_COLL_THRESH_M * 4.0)
    cid = jnp.where(
        coll,
        0,
        jnp.where(dyn & near, 2, jnp.where(near, 3, 1)),
    )
    return cid.astype(jnp.int32)
