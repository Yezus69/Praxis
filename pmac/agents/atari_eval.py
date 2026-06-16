"""Greedy bounded Atari evaluation."""

from __future__ import annotations

import math

import jax
import jax.numpy as jnp
import numpy as np

from pmac.agents.ppo_atari import jit_greedy_policy
from pmac.envs.atari_envpool import EpisodeReturnTracker, make_eval_env


def evaluate_atari(
    params,
    game,
    game_id,
    n_games,
    n_episodes=4,
    seed=0,
    max_steps_per_episode=30_000,
    eval_envs=16,
    eval_steps_cap=6_000,
) -> float:
    """Evaluate a greedy Atari policy with bounded vectorized envpool rollout."""
    n_episodes = int(n_episodes)
    num_envs = int(eval_envs)
    eval_steps_cap = int(eval_steps_cap)
    max_steps_per_episode = int(max_steps_per_episode)
    if n_episodes <= 0:
        raise ValueError("n_episodes must be positive")
    if num_envs <= 0:
        raise ValueError("eval_envs must be positive")
    if eval_steps_cap <= 0:
        raise ValueError("eval_steps_cap must be positive")
    if max_steps_per_episode <= 0:
        raise ValueError("max_steps_per_episode must be positive")

    episode_budget_steps = int(math.ceil(float(n_episodes * max_steps_per_episode) / float(num_envs)))
    max_steps = max(1, min(eval_steps_cap, episode_budget_steps))
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
        if len(completed_returns) >= n_episodes:
            break

    if not completed_returns:
        return float(np.mean(np.asarray(tracker.returns, dtype=np.float32)))
    first_returns = np.asarray(completed_returns[:n_episodes], dtype=np.float32)
    return float(np.mean(first_returns))


__all__ = ["evaluate_atari"]
