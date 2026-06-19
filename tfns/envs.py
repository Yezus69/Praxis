"""Task-free Atari environment helpers."""

from __future__ import annotations

import jax.numpy as jnp

from pmac.envs.atari_envpool import (
    ACT_DIM,
    ATARI_GAMES,
    EpisodeReturnTracker,
    make_eval_env,
    make_train_env,
    norm_obs,
)

FIVE_GAMES = ["SpaceInvaders-v5", "Breakout-v5", "BeamRider-v5", "Asterix-v5", "Qbert-v5"]
EIGHT_GAMES = FIVE_GAMES + ["Pong-v5", "Seaquest-v5", "DemonAttack-v5"]


def clip_reward(r):
    """Apply the fixed task-independent recurrent reward transform."""

    return jnp.clip(r, -1, 1)


__all__ = [
    "ACT_DIM",
    "ATARI_GAMES",
    "EIGHT_GAMES",
    "EpisodeReturnTracker",
    "FIVE_GAMES",
    "clip_reward",
    "make_eval_env",
    "make_train_env",
    "norm_obs",
]
