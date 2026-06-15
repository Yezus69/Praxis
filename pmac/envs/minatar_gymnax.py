"""gymnax MinAtar registry and vectorized helpers."""

from __future__ import annotations

import warnings
from dataclasses import dataclass
from typing import Any

warnings.filterwarnings("ignore")

import gymnax
import jax
import jax.numpy as jnp


GAMES = [
    "Breakout-MinAtar",
    "Asterix-MinAtar",
    "Freeway-MinAtar",
    "SpaceInvaders-MinAtar",
    # Seaquest-MinAtar is NOT registered in gymnax 0.0.9 — omit to avoid a crash.
]


@dataclass(frozen=True)
class GameSpec:
    name: str
    env: Any
    params: Any
    num_actions: int
    channels: int
    mask: jnp.ndarray
    game_id: int
    c_max: int
    act_max: int


def _num_actions(env, params) -> int:
    if hasattr(env, "num_actions"):
        return int(env.num_actions)
    return int(env.action_space(params).n)


def make_games(names) -> list[GameSpec]:
    """Create MinAtar game specs with shared channel/action dimensions."""
    names = list(names)
    if not names:
        raise ValueError("at least one game name is required")

    raw = []
    probe_key = jax.random.PRNGKey(0)
    for game_id, name in enumerate(names):
        env, params = gymnax.make(str(name))
        obs, _ = env.reset(probe_key, params)
        if len(obs.shape) != 3 or obs.shape[0] != 10 or obs.shape[1] != 10:
            raise ValueError(f"{name} produced unexpected observation shape {obs.shape}")
        raw.append(
            {
                "name": str(name),
                "env": env,
                "params": params,
                "num_actions": _num_actions(env, params),
                "channels": int(obs.shape[-1]),
                "game_id": int(game_id),
            }
        )

    c_max = max(item["channels"] for item in raw)
    act_max = max(item["num_actions"] for item in raw)
    specs = []
    for item in raw:
        mask = jnp.arange(act_max, dtype=jnp.int32) < int(item["num_actions"])
        specs.append(
            GameSpec(
                name=item["name"],
                env=item["env"],
                params=item["params"],
                num_actions=int(item["num_actions"]),
                channels=int(item["channels"]),
                mask=mask,
                game_id=int(item["game_id"]),
                c_max=int(c_max),
                act_max=int(act_max),
            )
        )
    return specs


def pad_obs(obs, c_max):
    """Pad ``(..., 10, 10, C)`` observations to ``(..., 10, 10, C_MAX)``."""
    obs = jnp.asarray(obs, dtype=jnp.float32)
    channels = int(obs.shape[-1])
    c_max = int(c_max)
    if channels > c_max:
        raise ValueError(f"obs has {channels} channels, c_max={c_max}")
    if channels == c_max:
        return obs
    pad_width = ((0, 0),) * (obs.ndim - 1) + ((0, c_max - channels),)
    return jnp.pad(obs, pad_width, mode="constant")


def vreset(env, params, keys):
    """Vectorized gymnax reset."""
    return jax.vmap(lambda key: env.reset(key, params))(keys)


def vstep(env, params, keys, state, actions):
    """Vectorized gymnax step."""
    return jax.vmap(lambda key, s, a: env.step(key, s, a, params))(keys, state, actions)


__all__ = ["GAMES", "GameSpec", "make_games", "pad_obs", "vreset", "vstep"]
