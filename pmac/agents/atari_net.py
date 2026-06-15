"""Flax Nature-DQN actor-critic for Atari PPO."""

from __future__ import annotations

import math

import flax.linen as nn
import jax
import jax.numpy as jnp

from pmac.envs.atari_envpool import ACT_DIM


def _orthogonal(scale: float):
    return nn.initializers.orthogonal(float(scale))


def _prepare_obs(obs):
    obs = jnp.asarray(obs)
    was_integer = jnp.issubdtype(obs.dtype, jnp.integer)
    if obs.ndim < 4:
        raise ValueError(f"expected batched Atari obs, got shape {obs.shape}")
    if obs.shape[-1] != 4:
        if obs.shape[-3] != 4:
            raise ValueError(f"expected NCHW or NHWC Atari obs, got shape {obs.shape}")
        obs = jnp.moveaxis(obs, -3, -1)
    obs = obs.astype(jnp.float32)
    if was_integer:
        obs = obs / 255.0
    return obs


class AtariActorCritic(nn.Module):
    """Shared Nature-CNN policy/value model conditioned on a game one-hot."""

    n_games: int
    act_dim: int = ACT_DIM

    @nn.compact
    def __call__(self, obs, game_onehot):
        obs = _prepare_obs(obs)
        game_onehot = jnp.asarray(game_onehot, dtype=jnp.float32)
        if game_onehot.ndim == 1:
            game_onehot = jnp.broadcast_to(game_onehot, (obs.shape[0], game_onehot.shape[-1]))

        conv_init = _orthogonal(math.sqrt(2.0))
        dense_init = _orthogonal(math.sqrt(2.0))
        x = nn.Conv(
            features=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            kernel_init=conv_init,
            bias_init=nn.initializers.zeros,
        )(obs)
        x = nn.relu(x)
        x = nn.Conv(
            features=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            kernel_init=conv_init,
            bias_init=nn.initializers.zeros,
        )(x)
        x = nn.relu(x)
        x = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=conv_init,
            bias_init=nn.initializers.zeros,
        )(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = nn.Dense(
            features=512,
            kernel_init=dense_init,
            bias_init=nn.initializers.zeros,
        )(x)
        x = nn.relu(x)
        x = jnp.concatenate([x, game_onehot], axis=-1)
        logits = nn.Dense(
            features=int(self.act_dim),
            kernel_init=_orthogonal(0.01),
            bias_init=nn.initializers.zeros,
        )(x)
        value = nn.Dense(
            features=1,
            kernel_init=_orthogonal(1.0),
            bias_init=nn.initializers.zeros,
        )(x)
        return logits, jnp.squeeze(value, axis=-1)


def init_atari(key, n_games: int):
    """Initialize Atari actor-critic params on a dummy 84x84x4 batch."""
    net = AtariActorCritic(n_games=int(n_games), act_dim=ACT_DIM)
    dummy_obs = jnp.zeros((1, 84, 84, 4), dtype=jnp.float32)
    dummy_game = jax.nn.one_hot(0, int(n_games), dtype=jnp.float32)
    variables = net.init(key, dummy_obs, dummy_game)
    return variables["params"]


def atari_apply(params, obs, game_onehot):
    """Apply the shared Atari policy/value model."""
    obs = jnp.asarray(obs)
    single = obs.ndim == 3
    if single:
        obs = obs[jnp.newaxis, ...]
    lead_shape = obs.shape[:-3]
    obs_flat = obs.reshape((-1,) + obs.shape[-3:])

    game_onehot = jnp.asarray(game_onehot, dtype=jnp.float32)
    n_games = int(game_onehot.shape[-1])
    if game_onehot.ndim == 1:
        game_flat = jnp.broadcast_to(game_onehot, (obs_flat.shape[0], n_games))
    else:
        game_flat = game_onehot.reshape((-1, n_games))

    net = AtariActorCritic(n_games=n_games, act_dim=ACT_DIM)
    logits, value = net.apply({"params": params}, obs_flat, game_flat)
    logits = logits.reshape(lead_shape + (logits.shape[-1],))
    value = value.reshape(lead_shape)
    if single:
        return logits[0], value[0]
    return logits, value


__all__ = ["AtariActorCritic", "atari_apply", "init_atari"]
