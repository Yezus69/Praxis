"""Shared plain-pytree CNN actor-critic for padded MinAtar observations."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _init_weight(key, shape, std: float):
    return jax.random.normal(key, tuple(int(v) for v in shape), dtype=jnp.float32) * float(std)


def init_ac(key, c_in, n_games, act_max, hidden=128):
    """Initialize a shared CNN actor-critic pytree."""
    c_in = int(c_in)
    n_games = int(n_games)
    act_max = int(act_max)
    hidden = int(hidden)
    conv_out = 16
    keys = jax.random.split(key, 4)
    conv_std = (2.0 / float(3 * 3 * c_in)) ** 0.5
    dense_in = 10 * 10 * conv_out + n_games
    dense_std = (2.0 / float(dense_in)) ** 0.5
    head_std = (1.0 / float(hidden)) ** 0.5
    return {
        "conv": {
            "w": _init_weight(keys[0], (3, 3, c_in, conv_out), conv_std),
            "b": jnp.zeros((conv_out,), dtype=jnp.float32),
        },
        "dense": {
            "w": _init_weight(keys[1], (dense_in, hidden), dense_std),
            "b": jnp.zeros((hidden,), dtype=jnp.float32),
        },
        "policy": {
            "w": _init_weight(keys[2], (hidden, act_max), head_std),
            "b": jnp.zeros((act_max,), dtype=jnp.float32),
        },
        "value": {
            "w": _init_weight(keys[3], (hidden, 1), head_std),
            "b": jnp.zeros((1,), dtype=jnp.float32),
        },
        "action_masks": jnp.ones((n_games, act_max), dtype=jnp.float32),
    }


def set_action_masks(params, action_masks):
    """Return params with per-game valid-action masks attached."""
    new_params = dict(params)
    new_params["action_masks"] = jnp.asarray(action_masks, dtype=jnp.float32)
    return new_params


def _mask_logits(params, logits, game_onehot):
    masks = params.get("action_masks")
    if masks is None:
        return logits
    action_mask = game_onehot @ masks
    return jnp.where(action_mask > 0.5, logits, jnp.asarray(-1.0e9, dtype=logits.dtype))


def ac_apply(params, obs, game_onehot):
    """Apply CNN trunk and masked policy/value heads."""
    obs = jnp.asarray(obs, dtype=jnp.float32)
    game_onehot = jnp.asarray(game_onehot, dtype=jnp.float32)
    single = obs.ndim == 3
    lead_shape = () if single else obs.shape[:-3]
    obs_b = obs[jnp.newaxis, ...] if single else obs.reshape((-1,) + obs.shape[-3:])

    if game_onehot.ndim == 1:
        ctx = jnp.broadcast_to(game_onehot, lead_shape + (game_onehot.shape[-1],))
    else:
        ctx = game_onehot
    ctx_b = ctx[jnp.newaxis, ...] if single else ctx.reshape((-1, ctx.shape[-1]))

    h = jax.lax.conv_general_dilated(
        obs_b,
        params["conv"]["w"],
        window_strides=(1, 1),
        padding="SAME",
        dimension_numbers=("NHWC", "HWIO", "NHWC"),
    )
    h = jax.nn.relu(h + params["conv"]["b"])
    h = h.reshape((h.shape[0], -1))
    h = jnp.concatenate([h, ctx_b], axis=-1)
    h = jax.nn.relu(h @ params["dense"]["w"] + params["dense"]["b"])
    logits = h @ params["policy"]["w"] + params["policy"]["b"]
    logits = _mask_logits(params, logits, ctx_b)
    value = jnp.squeeze(h @ params["value"]["w"] + params["value"]["b"], axis=-1)

    if single:
        return logits[0], value[0]
    return logits.reshape(lead_shape + (logits.shape[-1],)), value.reshape(lead_shape)


__all__ = ["ac_apply", "init_ac", "set_action_masks"]
