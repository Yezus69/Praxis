"""envpool Atari helpers for fixed-action PPO experiments."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np


ATARI_GAMES = [
    "Pong-v5",
    "Breakout-v5",
    "SpaceInvaders-v5",
    "BeamRider-v5",
    "Asterix-v5",
    "Seaquest-v5",
    "Qbert-v5",
    "DemonAttack-v5",
]
ACT_DIM = 18


def make_train_env(game: str, num_envs: int, seed: int):
    """Create an envpool Atari training env with uniform 18-action space."""
    import envpool

    return envpool.make(
        str(game),
        env_type="gymnasium",
        num_envs=int(num_envs),
        seed=int(seed),
        full_action_space=True,
        episodic_life=True,
        reward_clip=True,
    )


def make_train_env_xla(game: str, num_envs: int, seed: int):
    """Create an envpool Atari training env configured for XLA stepping."""
    import envpool

    return envpool.make(
        str(game),
        env_type="gymnasium",
        num_envs=int(num_envs),
        batch_size=int(num_envs),
        seed=int(seed),
        full_action_space=True,
        episodic_life=True,
        reward_clip=True,
    )


def make_eval_env(game: str, num_envs: int, seed: int):
    """Create an envpool Atari eval env with true episode scores."""
    import envpool

    return envpool.make(
        str(game),
        env_type="gymnasium",
        num_envs=int(num_envs),
        seed=int(seed),
        full_action_space=True,
        episodic_life=False,
        reward_clip=False,
    )


def norm_obs(obs):
    """Convert envpool ``(N, 4, 84, 84)`` uint8 frames to NHWC float32."""
    obs = np.asarray(obs)
    if obs.ndim < 4 or obs.shape[-3] != 4:
        raise ValueError(f"expected obs shape (..., 4, 84, 84), got {obs.shape}")
    obs = np.moveaxis(obs, -3, -1)
    return obs.astype(np.float32) / 255.0


def _info_array(info: Any, key: str):
    if isinstance(info, dict) and key in info:
        return np.asarray(info[key])
    return None


@dataclass
class EpisodeReturnTracker:
    """Track true completed-game returns from vector Atari env rewards."""

    num_envs: int

    def __post_init__(self):
        self.returns = np.zeros((int(self.num_envs),), dtype=np.float32)

    def reset(self):
        self.returns[...] = 0.0

    def update(self, rewards, terminated, truncated, info) -> list[float]:
        rewards = np.asarray(rewards, dtype=np.float32)
        terminated = np.asarray(terminated, dtype=bool)
        truncated = np.asarray(truncated, dtype=bool)
        true_done = _info_array(info, "terminated")
        if true_done is None:
            true_done = np.logical_or(terminated, truncated)
        true_done = np.asarray(true_done, dtype=bool)

        self.returns += rewards
        completed = self.returns[true_done].astype(float).tolist()
        self.returns[true_done] = 0.0
        return completed


__all__ = [
    "ACT_DIM",
    "ATARI_GAMES",
    "EpisodeReturnTracker",
    "make_eval_env",
    "make_train_env",
    "make_train_env_xla",
    "norm_obs",
]
