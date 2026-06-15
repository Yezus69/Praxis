"""Greedy bounded Atari evaluation."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from pmac.agents.ppo_atari import jit_greedy_policy
from pmac.envs.atari_envpool import EpisodeReturnTracker, make_eval_env


def evaluate_atari(params, game, game_id, n_games, n_episodes=20, seed=0) -> float:
    """Evaluate a greedy Atari policy over true full-game returns."""
    n_episodes = int(n_episodes)
    num_envs = max(1, min(8, n_episodes))
    max_steps = int(math.ceil(float(n_episodes * 30000) / float(num_envs)))
    env = make_eval_env(str(game), num_envs, int(seed))
    obs, _ = env.reset()
    obs = np.asarray(obs, dtype=np.uint8)
    game_onehot = jax.nn.one_hot(int(game_id), int(n_games), dtype=jnp.float32)
    tracker = EpisodeReturnTracker(num_envs)
    completed_returns: list[float] = []

    for _ in range(max_steps):
        actions, _, _ = jit_greedy_policy(params, obs, game_onehot)
        actions_np = np.asarray(jax.device_get(actions), dtype=np.int32)
        obs, rewards, terminated, truncated, info = env.step(actions_np)
        obs = np.asarray(obs, dtype=np.uint8)
        completed = tracker.update(rewards, terminated, truncated, info)
        for episode_return in completed:
            if len(completed_returns) < n_episodes:
                completed_returns.append(float(episode_return))

    if not completed_returns:
        return 0.0
    first_returns = np.asarray(completed_returns[:n_episodes], dtype=np.float32)
    return float(np.mean(first_returns))


__all__ = ["evaluate_atari"]
