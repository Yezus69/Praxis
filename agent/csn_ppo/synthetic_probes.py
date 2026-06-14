"""Contract-space synthetic probes and analytic teachers for CSN-PPO."""

import jax
import jax.numpy as jnp


def pack_contract_obs(goal, agent_vel, obstacles, mask):
    """Pack Praxis 27D observation.

    Args:
        goal: tuple/list (dx, dy, dist, heading_err)
        agent_vel: tuple/list (vx, vy, omega)
        obstacles: array-like [4, 4], each row (px, py, vx, vy)
        mask: array-like [4]
    """
    return jnp.concatenate([
        jnp.asarray(goal, dtype=jnp.float32),
        jnp.asarray(agent_vel, dtype=jnp.float32),
        jnp.asarray(obstacles, dtype=jnp.float32).reshape(-1),
        jnp.asarray(mask, dtype=jnp.float32),
    ], axis=0)


def sort_and_pad_obstacles(obstacles, max_k=4):
    """Sort active obstacles by distance and pad to K=4."""
    obstacles = jnp.asarray(obstacles, dtype=jnp.float32).reshape((-1, 4))
    d = jnp.sqrt(obstacles[:, 0] ** 2 + obstacles[:, 1] ** 2)
    order = jnp.argsort(d)
    obstacles = obstacles[order]

    active_n = min(obstacles.shape[0], max_k)
    clipped = obstacles[:max_k]
    pad_n = max_k - clipped.shape[0]

    if pad_n > 0:
        padded = jnp.pad(clipped, ((0, pad_n), (0, 0)))
    else:
        padded = clipped

    mask = jnp.concatenate([
        jnp.ones((active_n,), dtype=jnp.float32),
        jnp.zeros((max_k - active_n,), dtype=jnp.float32),
    ])

    return padded, mask


def make_probe_blocked_path(rng):
    rng_goal, rng_obs, rng_vel = jax.random.split(rng, 3)

    # Sample goal vector.
    angle = jax.random.uniform(rng_goal, (), minval=-jnp.pi, maxval=jnp.pi)
    dist = jax.random.uniform(rng_goal, (), minval=2.0, maxval=8.0)
    dx = dist * jnp.cos(angle)
    dy = dist * jnp.sin(angle)
    heading_err = angle

    # Unit vector toward goal and perpendicular vector.
    gx = dx / (dist + 1e-8)
    gy = dy / (dist + 1e-8)
    nx = -gy
    ny = gx

    # Obstacle lies near the line from agent to goal.
    alpha = jax.random.uniform(rng_obs, (), minval=0.25, maxval=0.75)
    lateral = jax.random.normal(rng_obs, ()) * 0.15
    ox = alpha * dx + lateral * nx
    oy = alpha * dy + lateral * ny

    # Obstacle velocity crosses the path.
    speed = jax.random.uniform(rng_vel, (), minval=0.2, maxval=1.0)
    direction = jnp.where(jax.random.bernoulli(rng_vel), 1.0, -1.0)
    ovx = direction * speed * nx
    ovy = direction * speed * ny

    obstacles = jnp.array([
        [ox, oy, ovx, ovy],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
        [0.0, 0.0, 0.0, 0.0],
    ], dtype=jnp.float32)

    mask = jnp.array([1.0, 0.0, 0.0, 0.0], dtype=jnp.float32)

    agent_vel = jnp.array([0.0, 0.0, 0.0], dtype=jnp.float32)

    return pack_contract_obs(
        goal=(dx, dy, dist, heading_err),
        agent_vel=agent_vel,
        obstacles=obstacles,
        mask=mask,
    )


def make_probe_no_obstacle(rng):
    angle = jax.random.uniform(rng, (), minval=-jnp.pi, maxval=jnp.pi)
    dist = jax.random.uniform(rng, (), minval=1.0, maxval=8.0)

    dx = dist * jnp.cos(angle)
    dy = dist * jnp.sin(angle)
    heading_err = angle

    return pack_contract_obs(
        goal=(dx, dy, dist, heading_err),
        agent_vel=(0.0, 0.0, 0.0),
        obstacles=jnp.zeros((4, 4), dtype=jnp.float32),
        mask=jnp.zeros((4,), dtype=jnp.float32),
    )


def analytic_no_obstacle_teacher(obs, speed=1.0):
    dx, dy, dist, _heading_err = obs[0], obs[1], obs[2], obs[3]
    direction = jnp.array([dx, dy]) / (dist + 1e-8)
    return jnp.clip(speed * direction, -1.0, 1.0)


def analytic_obstacle_teacher(obs):
    dx, dy, dist, _heading_err = obs[0], obs[1], obs[2], obs[3]
    goal_dir = jnp.array([dx, dy]) / (dist + 1e-8)

    obstacle_block = obs[7:7 + 16].reshape(4, 4)
    mask = obs[23:27]

    desired = goal_dir

    for k in range(4):
        px, py, ovx, ovy = obstacle_block[k]
        active = mask[k]
        rel = jnp.array([px, py])
        obs_dist = jnp.linalg.norm(rel) + 1e-8
        obs_dir = rel / obs_dist

        # Obstacle is considered blocking if it is in front of the agent and close to goal ray.
        forward = jnp.dot(obs_dir, goal_dir)
        lateral_dist = jnp.linalg.norm(rel - jnp.dot(rel, goal_dir) * goal_dir)
        blocking = (active > 0.5) & (forward > 0.3) & (lateral_dist < 0.75)

        # Evade perpendicular to obstacle direction.
        perp = jnp.array([-obs_dir[1], obs_dir[0]])
        evade_strength = active * blocking.astype(jnp.float32) * jnp.clip(1.5 / obs_dist, 0.0, 2.0)
        desired = desired + evade_strength * perp

    desired = desired / (jnp.linalg.norm(desired) + 1e-8)
    return jnp.clip(desired, -1.0, 1.0)


def make_teacher_distribution(action_mean, logstd=-2.0):
    return action_mean, jnp.full_like(action_mean, logstd)


def obstacle_distances(obs):
    obstacles = obs[7:23].reshape(4, 4)
    mask = obs[23:27]
    d = jnp.sqrt(obstacles[:, 0] ** 2 + obstacles[:, 1] ** 2)
    # Inactive obstacles get large distance.
    return jnp.where(mask > 0.5, d, 1e6)


def collision_proximity(obs, radius=0.75):
    d_min = jnp.min(obstacle_distances(obs))
    return jax.nn.relu(radius - d_min) / radius


def success_proximity(obs, goal_radius=0.5):
    dist_to_goal = obs[2]
    return jax.nn.relu(goal_radius - dist_to_goal) / goal_radius


def dynamic_obstacle_score(obs):
    obstacles = obs[7:23].reshape(4, 4)
    mask = obs[23:27]
    speeds = jnp.sqrt(obstacles[:, 2] ** 2 + obstacles[:, 3] ** 2)
    return jnp.max(mask * speeds)


def criticality_score(obs, advantage_abs, novelty, sentinel_failure):
    c = (
        1.0 * advantage_abs
        + 3.0 * collision_proximity(obs)
        + 2.0 * success_proximity(obs)
        + 1.0 * dynamic_obstacle_score(obs)
        + 1.0 * novelty
        + 5.0 * sentinel_failure
    )
    return jnp.clip(c, 0.1, 10.0)


def memory_weight(c, w_min=0.1, w_max=10.0):
    return jnp.clip(c, w_min, w_max)


def kl_budget(c, delta0=0.02):
    return delta0 / (1.0 + c)


def value_budget(c, rho0=0.25, beta=1.0):
    return rho0 / (1.0 + beta * c)
