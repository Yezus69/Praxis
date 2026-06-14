"""Synthetic probes and analytic teachers on the 28D coverage manifold."""

import jax
import jax.numpy as jnp

from praxis import contract


_EPS = 1e-6
_OB0, _OB1 = contract.OBST_SLICE
_MASK0, _MASK1 = contract.MASK_SLICE
_FR0 = contract.FRONTIER_SLICE[0]


def _normalize(v):
    return v / (jnp.linalg.norm(v) + _EPS)


def _perp(v):
    return jnp.asarray([-v[1], v[0]], dtype=jnp.float32)


def _unit_from_angle(angle):
    return jnp.asarray([jnp.cos(angle), jnp.sin(angle)], dtype=jnp.float32)


def _obstacle_row(rel_pos, rel_vel):
    return jnp.concatenate(
        [
            jnp.asarray(rel_pos, dtype=jnp.float32),
            jnp.asarray(rel_vel, dtype=jnp.float32),
        ]
    )


def _far_obstacles():
    idx = jnp.arange(contract.K, dtype=jnp.float32)
    angles = (2.0 * jnp.pi * idx / contract.K) + (0.25 * jnp.pi)
    pos = 0.90 * jnp.stack([jnp.cos(angles), jnp.sin(angles)], axis=-1)
    vel = jnp.zeros(
        (contract.K, contract.PER_OBSTACLE_DIM - contract.ACT_DIM),
        dtype=jnp.float32,
    )
    return jnp.concatenate([pos, vel], axis=-1)


def _sort_obstacles_with_mask(obstacles, mask):
    dist = jnp.linalg.norm(obstacles[:, :contract.ACT_DIM], axis=-1)
    sort_key = jnp.where(mask > 0.5, dist, 1e6 + dist)
    order = jnp.argsort(sort_key)
    return obstacles[order], mask[order]


def _clean_agent(agent_feat):
    agent_feat = jnp.nan_to_num(
        jnp.asarray(agent_feat, dtype=jnp.float32).reshape(contract.AGENT_DIM)
    )
    pos = jnp.clip(agent_feat[:contract.ACT_DIM], -1.0, 1.0)
    vel = jnp.clip(agent_feat[contract.ACT_DIM:], -1.0, 1.0)
    return jnp.concatenate([pos, vel])


def _clean_obstacles(obstacles):
    obstacles = jnp.nan_to_num(
        jnp.asarray(obstacles, dtype=jnp.float32).reshape(
            contract.K,
            contract.PER_OBSTACLE_DIM,
        )
    )
    pos = jnp.clip(obstacles[:, :contract.ACT_DIM], -1.0, 1.0)
    vel = jnp.clip(obstacles[:, contract.ACT_DIM:], -1.0, 1.0)
    return jnp.concatenate([pos, vel], axis=-1)


def _clean_frontier(frontier):
    frontier = jnp.nan_to_num(
        jnp.asarray(frontier, dtype=jnp.float32).reshape(contract.FRONTIER_DIM)
    )
    direction = _normalize(frontier[:contract.ACT_DIM])
    distance = jnp.clip(frontier[contract.ACT_DIM], 0.0, 1.0)
    return jnp.concatenate([direction, jnp.reshape(distance, (1,))])


def _frontier(fdir, fdist):
    return jnp.concatenate(
        [_normalize(jnp.asarray(fdir, dtype=jnp.float32)), jnp.reshape(fdist, (1,))]
    )


def _random_agent(rng, pos_abs=0.75, vel_abs=0.25):
    kp, kv = jax.random.split(rng)
    pos = jax.random.uniform(
        kp,
        (contract.ACT_DIM,),
        minval=-pos_abs,
        maxval=pos_abs,
    )
    vel = jax.random.uniform(
        kv,
        (contract.AGENT_DIM - contract.ACT_DIM,),
        minval=-vel_abs,
        maxval=vel_abs,
    )
    return jnp.concatenate([pos, vel])


def _random_frontier_dir(rng):
    angle = jax.random.uniform(rng, (), minval=-jnp.pi, maxval=jnp.pi)
    return _unit_from_angle(angle)


def _sign(rng):
    return jnp.where(jax.random.bernoulli(rng), 1.0, -1.0)


def pack_cover_obs(agent_feat, obstacles, frontier, covered, mask=None):
    """Packs one finite, sorted, contract-valid 28D coverage observation."""
    agent = _clean_agent(agent_feat)
    obstacles = _clean_obstacles(obstacles)
    if mask is None:
        mask = jnp.ones((contract.K,), dtype=jnp.float32)
    mask = jnp.where(
        jnp.nan_to_num(jnp.asarray(mask, dtype=jnp.float32).reshape(contract.K)) > 0.5,
        1.0,
        0.0,
    )
    obstacles, mask = _sort_obstacles_with_mask(obstacles, mask)
    frontier = _clean_frontier(frontier)
    covered = jnp.clip(jnp.nan_to_num(jnp.asarray(covered, dtype=jnp.float32)), 0.0, 1.0)
    obs = jnp.concatenate(
        [
            agent,
            obstacles.reshape(-1),
            mask,
            frontier,
            jnp.reshape(covered, (contract.COVERED_DIM,)),
        ]
    )
    return jnp.nan_to_num(obs)


def make_probe_open_frontier_low_coverage(rng):
    """Family 1: open frontier, low coverage."""
    ka, kf, kd, kc = jax.random.split(rng, 4)
    fdir = _random_frontier_dir(kf)
    fdist = jax.random.uniform(kd, (), minval=0.35, maxval=0.95)
    covered = jax.random.uniform(kc, (), minval=0.0, maxval=0.35)
    return pack_cover_obs(_random_agent(ka), _far_obstacles(), _frontier(fdir, fdist), covered)


def make_probe_open_frontier_high_coverage(rng):
    """Family 2: open frontier, high coverage."""
    ka, kf, kd, kc = jax.random.split(rng, 4)
    fdir = _random_frontier_dir(kf)
    fdist = jax.random.uniform(kd, (), minval=0.20, maxval=0.80)
    covered = jax.random.uniform(kc, (), minval=0.60, maxval=0.92)
    return pack_cover_obs(_random_agent(ka), _far_obstacles(), _frontier(fdir, fdist), covered)


def make_probe_obstacle_on_frontier_ray(rng):
    """Family 3: obstacles directly between the agent and frontier."""
    ka, kf, kd, kc = jax.random.split(rng, 4)
    fdir = _random_frontier_dir(kf)
    distances = jnp.asarray([0.08, 0.14, 0.21, 0.30], dtype=jnp.float32)
    obstacles = jax.vmap(
        lambda d: _obstacle_row(fdir * d, jnp.zeros((contract.ACT_DIM,), dtype=jnp.float32))
    )(distances)
    fdist = jax.random.uniform(kd, (), minval=0.35, maxval=0.95)
    covered = jax.random.uniform(kc, (), minval=0.05, maxval=0.65)
    return pack_cover_obs(_random_agent(ka), obstacles, _frontier(fdir, fdist), covered)


def make_probe_crossing_obstacle_left_to_right(rng):
    """Family 4: crossing dynamic obstacle left-to-right."""
    ka, kf, kd, kc = jax.random.split(rng, 4)
    fdir = _random_frontier_dir(kf)
    tangent = _perp(fdir)
    o0 = _obstacle_row(fdir * 0.18 - tangent * 0.03, tangent * 0.85)
    obstacles = _far_obstacles().at[0].set(o0)
    fdist = jax.random.uniform(kd, (), minval=0.30, maxval=0.90)
    covered = jax.random.uniform(kc, (), minval=0.10, maxval=0.70)
    return pack_cover_obs(_random_agent(ka), obstacles, _frontier(fdir, fdist), covered)


def make_probe_crossing_obstacle_right_to_left(rng):
    """Family 5: crossing dynamic obstacle right-to-left."""
    ka, kf, kd, kc = jax.random.split(rng, 4)
    fdir = _random_frontier_dir(kf)
    tangent = _perp(fdir)
    o0 = _obstacle_row(fdir * 0.18 + tangent * 0.03, -tangent * 0.85)
    obstacles = _far_obstacles().at[0].set(o0)
    fdist = jax.random.uniform(kd, (), minval=0.30, maxval=0.90)
    covered = jax.random.uniform(kc, (), minval=0.10, maxval=0.70)
    return pack_cover_obs(_random_agent(ka), obstacles, _frontier(fdir, fdist), covered)


def make_probe_near_wall_frontier_along_wall(rng):
    """Family 6: near wall, frontier points along wall."""
    kx, ky, kv, kd, kc = jax.random.split(rng, 5)
    wall_sign = _sign(kx)
    along_sign = _sign(ky)
    agent = jnp.concatenate(
        [
            jnp.asarray(
                [
                    wall_sign * 0.92,
                    jax.random.uniform(ky, (), minval=-0.75, maxval=0.75),
                ],
                dtype=jnp.float32,
            ),
            jax.random.uniform(kv, (contract.ACT_DIM,), minval=-0.15, maxval=0.15),
        ]
    )
    fdir = jnp.asarray([0.0, along_sign], dtype=jnp.float32)
    fdist = jax.random.uniform(kd, (), minval=0.20, maxval=0.70)
    covered = jax.random.uniform(kc, (), minval=0.15, maxval=0.80)
    return pack_cover_obs(agent, _far_obstacles(), _frontier(fdir, fdist), covered)


def make_probe_corner_escape(rng):
    """Family 7: corner escape."""
    kx, ky, kv, kc = jax.random.split(rng, 4)
    sx = _sign(kx)
    sy = _sign(ky)
    agent = jnp.concatenate(
        [
            jnp.asarray([sx * 0.92, sy * 0.92], dtype=jnp.float32),
            jax.random.uniform(kv, (contract.ACT_DIM,), minval=-0.10, maxval=0.10),
        ]
    )
    fdir = _normalize(jnp.asarray([-sx, -sy], dtype=jnp.float32))
    tangent = _perp(fdir)
    obstacles = _far_obstacles()
    obstacles = obstacles.at[0].set(_obstacle_row(fdir * 0.24 + tangent * 0.10, -fdir * 0.20))
    covered = jax.random.uniform(kc, (), minval=0.25, maxval=0.80)
    return pack_cover_obs(agent, obstacles, _frontier(fdir, 0.45), covered)


def make_probe_obstacle_behind_agent(rng):
    """Family 8: obstacle behind agent, should mostly ignore."""
    ka, kf, kd, kc = jax.random.split(rng, 4)
    fdir = _random_frontier_dir(kf)
    o0 = _obstacle_row(-fdir * 0.16, jnp.zeros((contract.ACT_DIM,), dtype=jnp.float32))
    obstacles = _far_obstacles().at[0].set(o0)
    fdist = jax.random.uniform(kd, (), minval=0.35, maxval=0.90)
    covered = jax.random.uniform(kc, (), minval=0.05, maxval=0.75)
    return pack_cover_obs(_random_agent(ka), obstacles, _frontier(fdir, fdist), covered)


def make_probe_high_speed_near_collision(rng):
    """Family 9: high-speed near-collision."""
    ka, kf, kd, kc = jax.random.split(rng, 4)
    fdir = _random_frontier_dir(kf)
    tangent = _perp(fdir)
    rel = fdir * 0.11 + tangent * 0.02
    o0 = _obstacle_row(rel, -_normalize(rel) * 1.0)
    agent = jnp.concatenate([_random_agent(ka)[:contract.ACT_DIM], fdir * 0.80])
    obstacles = _far_obstacles().at[0].set(o0)
    fdist = jax.random.uniform(kd, (), minval=0.25, maxval=0.85)
    covered = jax.random.uniform(kc, (), minval=0.10, maxval=0.80)
    return pack_cover_obs(agent, obstacles, _frontier(fdir, fdist), covered)


def make_probe_near_complete_final_cell(rng):
    """Family 10: near-complete final-cell state."""
    ka, kf, kd, kc = jax.random.split(rng, 4)
    fdir = _random_frontier_dir(kf)
    fdist = jax.random.uniform(kd, (), minval=0.02, maxval=0.14)
    covered = jax.random.uniform(kc, (), minval=0.92, maxval=0.995)
    return pack_cover_obs(_random_agent(ka), _far_obstacles(), _frontier(fdir, fdist), covered)


def make_probe_stalled_oscillation(rng):
    """Family 11: stalled or oscillating near a frontier."""
    kp, kf, ks, kd, kc = jax.random.split(rng, 5)
    fdir = _random_frontier_dir(kf)
    speed = jax.random.uniform(ks, (), minval=0.35, maxval=0.85)
    pos = jax.random.uniform(kp, (contract.ACT_DIM,), minval=-0.45, maxval=0.45)
    agent = jnp.concatenate([pos, -fdir * speed])
    fdist = jax.random.uniform(kd, (), minval=0.15, maxval=0.55)
    covered = jax.random.uniform(kc, (), minval=0.30, maxval=0.80)
    return pack_cover_obs(agent, _far_obstacles(), _frontier(fdir, fdist), covered)


def make_probe_dense_cluster_escape_tangent(rng):
    """Family 12: dense obstacle cluster with one escape tangent."""
    ka, kf, kd, kc = jax.random.split(rng, 4)
    fdir = _random_frontier_dir(kf)
    tangent = _perp(fdir)
    obstacles = jnp.stack(
        [
            _obstacle_row(fdir * 0.13, -fdir * 0.25),
            _obstacle_row(fdir * 0.20 + tangent * 0.08, -tangent * 0.40),
            _obstacle_row(fdir * 0.25 - tangent * 0.10, tangent * 0.35),
            _obstacle_row(-tangent * 0.22, fdir * 0.15),
        ],
        axis=0,
    )
    fdist = jax.random.uniform(kd, (), minval=0.35, maxval=0.85)
    covered = jax.random.uniform(kc, (), minval=0.15, maxval=0.75)
    return pack_cover_obs(_random_agent(ka), obstacles, _frontier(fdir, fdist), covered)


def make_probe_frontier_behind_agent(rng):
    """Family 13: frontier behind the agent's current motion."""
    kp, kv, kd, kc = jax.random.split(rng, 4)
    vel_dir = _random_frontier_dir(kv)
    pos = jax.random.uniform(kp, (contract.ACT_DIM,), minval=-0.65, maxval=0.65)
    agent = jnp.concatenate([pos, vel_dir * 0.65])
    fdir = -vel_dir
    fdist = jax.random.uniform(kd, (), minval=0.25, maxval=0.85)
    covered = jax.random.uniform(kc, (), minval=0.20, maxval=0.80)
    return pack_cover_obs(agent, _far_obstacles(), _frontier(fdir, fdist), covered)


def make_probe_no_obstacle_straight_motion(rng):
    """Family 14: no-obstacle straight motion."""
    kp, kf, ks, kd, kc = jax.random.split(rng, 5)
    fdir = _random_frontier_dir(kf)
    speed = jax.random.uniform(ks, (), minval=0.10, maxval=0.60)
    pos = jax.random.uniform(kp, (contract.ACT_DIM,), minval=-0.60, maxval=0.60)
    agent = jnp.concatenate([pos, fdir * speed])
    fdist = jax.random.uniform(kd, (), minval=0.35, maxval=0.95)
    covered = jax.random.uniform(kc, (), minval=0.00, maxval=0.70)
    return pack_cover_obs(
        agent,
        _far_obstacles(),
        _frontier(fdir, fdist),
        covered,
        mask=jnp.zeros((contract.K,), dtype=jnp.float32),
    )


def make_probe_padded_mask_edge_cases(rng):
    """Family 15: padded/mask edge cases."""
    ka, kf, kd, kc = jax.random.split(rng, 4)
    fdir = _random_frontier_dir(kf)
    tangent = _perp(fdir)
    obstacles = _far_obstacles()
    obstacles = obstacles.at[0].set(_obstacle_row(fdir * 0.18 + tangent * 0.05, tangent * 0.40))
    obstacles = obstacles.at[1].set(_obstacle_row(-fdir * 0.30, jnp.zeros((contract.ACT_DIM,), dtype=jnp.float32)))
    mask = jnp.asarray([1.0, 1.0, 0.0, 0.0], dtype=jnp.float32)
    fdist = jax.random.uniform(kd, (), minval=0.20, maxval=0.80)
    covered = jax.random.uniform(kc, (), minval=0.05, maxval=0.85)
    return pack_cover_obs(_random_agent(ka), obstacles, _frontier(fdir, fdist), covered, mask=mask)


def analytic_coverage_teacher(obs, cfg):
    """P5 analytic teacher: frontier pursuit plus guarded repel/tangent avoidance."""
    frontier = obs[_FR0:_FR0 + contract.ACT_DIM]
    desired = frontier / (jnp.linalg.norm(frontier) + _EPS)
    obstacles = obs[_OB0:_OB1].reshape(contract.K, contract.PER_OBSTACLE_DIM)
    mask = obs[_MASK0:_MASK1]
    safe_dist = jnp.maximum(jnp.asarray(cfg.synthetic_safe_dist, dtype=jnp.float32), _EPS)
    avoid = jnp.zeros_like(desired)

    for k in range(contract.K):
        rel = obstacles[k, :contract.ACT_DIM]
        dist = jnp.linalg.norm(rel) + _EPS
        rel_dir = rel / dist
        in_front = jnp.dot(rel_dir, desired) > 0.3
        close = dist < safe_dist
        active = mask[k] > 0.5
        repel = -rel_dir
        tangent = _perp(rel_dir)
        tangent = jnp.where(jnp.dot(tangent, desired) < 0.0, -tangent, tangent)
        strength = jnp.clip((safe_dist - dist) / safe_dist, 0.0, 1.0)
        contribution = strength * (0.7 * repel + 0.3 * tangent)
        avoid = avoid + jnp.where(active & in_front & close, contribution, 0.0)

    action = desired + avoid
    action = action / (jnp.linalg.norm(action) + _EPS)
    return jnp.clip(action, -1.0, 1.0)


def obstacle_collision_risk(obs, action, cfg):
    """Minimum viable P5 action risk: close active obstacles in the action direction."""
    obstacles = obs[_OB0:_OB1].reshape(contract.K, contract.PER_OBSTACLE_DIM)
    mask = obs[_MASK0:_MASK1]
    rel = obstacles[:, :contract.ACT_DIM]
    dist = jnp.linalg.norm(rel, axis=-1) + _EPS
    rel_dir = rel / dist[:, None]
    action_dir = action / (jnp.linalg.norm(action) + _EPS)
    safe_dist = jnp.maximum(jnp.asarray(cfg.synthetic_safe_dist, dtype=jnp.float32), _EPS)
    close_strength = jnp.clip((safe_dist - dist) / safe_dist, 0.0, 1.0)
    toward = jnp.maximum(jnp.sum(rel_dir * action_dir[None, :], axis=-1), 0.0)
    active = mask > 0.5
    return jnp.max(jnp.where(active, close_strength * toward, 0.0))


def safer_of(obs, policy_teacher, analytic_teacher, cfg):
    """P5 safer-of rule: choose analytic only when policy has higher collision risk."""
    policy_risk = obstacle_collision_risk(obs, policy_teacher, cfg)
    analytic_risk = obstacle_collision_risk(obs, analytic_teacher, cfg)
    return jnp.where(policy_risk > analytic_risk, analytic_teacher, policy_teacher)


PROBE_FAMILY_NAMES = (
    "open_frontier_low_coverage",
    "open_frontier_high_coverage",
    "obstacle_on_frontier_ray",
    "crossing_obstacle_left_to_right",
    "crossing_obstacle_right_to_left",
    "near_wall_frontier_along_wall",
    "corner_escape",
    "obstacle_behind_agent",
    "high_speed_near_collision",
    "near_complete_final_cell",
    "stalled_oscillation",
    "dense_cluster_escape_tangent",
    "frontier_behind_agent",
    "no_obstacle_straight_motion",
    "padded_mask_edge_cases",
)

_PROBE_FNS = (
    make_probe_open_frontier_low_coverage,
    make_probe_open_frontier_high_coverage,
    make_probe_obstacle_on_frontier_ray,
    make_probe_crossing_obstacle_left_to_right,
    make_probe_crossing_obstacle_right_to_left,
    make_probe_near_wall_frontier_along_wall,
    make_probe_corner_escape,
    make_probe_obstacle_behind_agent,
    make_probe_high_speed_near_collision,
    make_probe_near_complete_final_cell,
    make_probe_stalled_oscillation,
    make_probe_dense_cluster_escape_tangent,
    make_probe_frontier_behind_agent,
    make_probe_no_obstacle_straight_motion,
    make_probe_padded_mask_edge_cases,
)

# Backward-compatible names used by earlier coverage probe tests and call sites.
make_probe_open_explore = make_probe_open_frontier_low_coverage
make_probe_obstacle_ahead = make_probe_obstacle_on_frontier_ray
make_probe_near_complete = make_probe_near_complete_final_cell


def generate_cover_probes(rng, batch_size):
    """Builds a fixed-size [batch_size, 28] synthetic coverage probe batch."""
    keys = jax.random.split(rng, batch_size)
    family_id = jnp.mod(jnp.arange(batch_size, dtype=jnp.int32), len(_PROBE_FNS))
    return jax.vmap(lambda k, fid: jax.lax.switch(fid, _PROBE_FNS, k))(keys, family_id)
