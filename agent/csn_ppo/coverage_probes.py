"""Synthetic probes on the 28D coverage observation manifold."""

import jax
import jax.numpy as jnp

from praxis import contract


def pack_cover_obs(agent_feat, obstacles, frontier, covered):
    """Packs one valid 28D coverage observation."""
    mask = jnp.ones((contract.K,), jnp.float32)
    obs = jnp.concatenate(
        [
            agent_feat.astype(jnp.float32),
            obstacles.reshape(-1).astype(jnp.float32),
            mask,
            frontier.astype(jnp.float32),
            jnp.reshape(covered, (1,)).astype(jnp.float32),
        ]
    )
    return jnp.nan_to_num(obs)


def make_probe_open_explore(rng):
    """Clear-frontier, low-coverage exploration state."""
    ka, kf, kc = jax.random.split(rng, 3)
    agent = jax.random.uniform(ka, (4,), minval=-0.8, maxval=0.8)
    far = jnp.full((4, 4), 0.0).at[:, 0].set(0.9).at[:, 1].set(0.9)
    fang = jax.random.uniform(kf, (), minval=-jnp.pi, maxval=jnp.pi)
    fdir = jnp.array([jnp.cos(fang), jnp.sin(fang)])
    fdist = jax.random.uniform(kf, (), minval=0.3, maxval=0.9)
    frontier = jnp.concatenate([fdir, fdist[None]])
    covered = jax.random.uniform(kc, (), minval=0.0, maxval=0.4)
    return pack_cover_obs(agent, far, frontier, covered)


def make_probe_obstacle_ahead(rng):
    """Obstacle near the frontier direction."""
    ka, ko, kf, kc = jax.random.split(rng, 4)
    agent = jax.random.uniform(ka, (4,), minval=-0.6, maxval=0.6)
    fang = jax.random.uniform(kf, (), minval=-jnp.pi, maxval=jnp.pi)
    fdir = jnp.array([jnp.cos(fang), jnp.sin(fang)])
    fdist = jax.random.uniform(kf, (), minval=0.2, maxval=0.7)
    near = jax.random.uniform(ko, (), minval=0.10, maxval=0.20)
    ov = jax.random.uniform(ko, (2,), minval=-0.5, maxval=0.5)
    o0 = jnp.concatenate([fdir * near, ov])
    far = jnp.array([0.9, 0.9, 0.0, 0.0])
    obstacles = jnp.stack([o0, far, far, far])
    covered = jax.random.uniform(kc, (), minval=0.1, maxval=0.7)
    frontier = jnp.concatenate([fdir, fdist[None]])
    return pack_cover_obs(agent, obstacles, frontier, covered)


def make_probe_near_complete(rng):
    """High-coverage finishing state."""
    ka, kf, kc = jax.random.split(rng, 3)
    agent = jax.random.uniform(ka, (4,), minval=-0.9, maxval=0.9)
    far = jnp.array([0.9, 0.9, 0.0, 0.0])
    obstacles = jnp.stack([far, far, far, far])
    fang = jax.random.uniform(kf, (), minval=-jnp.pi, maxval=jnp.pi)
    fdir = jnp.array([jnp.cos(fang), jnp.sin(fang)])
    fdist = jax.random.uniform(kf, (), minval=0.0, maxval=0.2)
    frontier = jnp.concatenate([fdir, fdist[None]])
    covered = jax.random.uniform(kc, (), minval=0.7, maxval=0.99)
    return pack_cover_obs(agent, obstacles, frontier, covered)


_PROBE_FNS = (
    make_probe_open_explore,
    make_probe_obstacle_ahead,
    make_probe_near_complete,
)


def generate_cover_probes(rng, batch_size):
    """Builds a fixed-size [batch_size, 28] synthetic coverage probe batch."""
    keys = jax.random.split(rng, batch_size)
    per = batch_size // len(_PROBE_FNS)
    chunks = []
    for i, fn in enumerate(_PROBE_FNS):
        sub = jax.lax.map(fn, keys[i * per:(i + 1) * per])
        chunks.append(sub)
    out = jnp.concatenate(chunks, axis=0)
    if out.shape[0] < batch_size:
        extra = jax.lax.map(_PROBE_FNS[0], keys[out.shape[0]:])
        out = jnp.concatenate([out, extra], axis=0)
    return out
