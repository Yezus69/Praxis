"""envpool Atari adapter for task-free recurrent PPO rollouts."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from tfns.envs import make_eval_env, make_train_env


_RAW_REWARD_KEYS = (
    "reward_raw",
    "raw_reward",
    "unclipped_reward",
    "original_reward",
    "env_reward",
)
_FIRE_ACTION = 1


def _reset_result(result: Any) -> tuple[Any, dict[str, Any]]:
    if isinstance(result, tuple) and len(result) == 2:
        return result[0], result[1]
    return result, {}


def _nhwc_uint8(obs: Any) -> np.ndarray:
    arr = np.asarray(obs)
    if arr.ndim != 4:
        raise ValueError(f"expected envpool observation with 4 dims, got {arr.shape}")
    if arr.shape[1] == 4:
        arr = np.moveaxis(arr, 1, -1)
    if arr.shape[-1] != 4:
        raise ValueError(f"expected 4-frame observation stack, got {arr.shape}")
    return np.ascontiguousarray(arr.astype(np.uint8, copy=False))


def _vec(value: Any, dtype: Any, num_envs: int, name: str) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        arr = np.full((int(num_envs),), arr.item(), dtype=dtype)
    arr = arr.reshape(-1)
    if int(arr.shape[0]) != int(num_envs):
        raise ValueError(f"{name} must have shape ({num_envs},), got {arr.shape}")
    return arr


def _info_value(info: Any, key: str) -> Any | None:
    if isinstance(info, dict) and key in info:
        return info[key]
    return None


def _info_bool(info: Any, key: str, fallback: Any, num_envs: int) -> np.ndarray:
    value = _info_value(info, key)
    if value is None:
        value = fallback
    return _vec(value, np.bool_, num_envs, key)


def _raw_reward(info: Any, reward: np.ndarray, num_envs: int) -> np.ndarray:
    for key in _RAW_REWARD_KEYS:
        value = _info_value(info, key)
        if value is not None:
            return _vec(value, np.float32, num_envs, key)
    return reward.astype(np.float32, copy=False)


@dataclass(frozen=True)
class AtariEnvHandle:
    """Small handle returned with the callable stepper."""

    env: Any
    game: str
    num_envs: int
    seed: int
    training: bool
    fire_reset: bool = True

    def close(self) -> None:
        close = getattr(self.env, "close", None)
        if callable(close):
            close()


class AtariEnvStep:
    """Callable adapter matching ``tfns.ppo.rollout.collect_rollout``."""

    def __init__(
        self,
        env: Any,
        game: str,
        num_envs: int,
        seed: int,
        training: bool,
        fire_reset: bool = True,
    ):
        self.handle = AtariEnvHandle(
            env=env,
            game=str(game),
            num_envs=int(num_envs),
            seed=int(seed),
            training=bool(training),
            fire_reset=bool(fire_reset),
        )
        self._returns = np.zeros((int(num_envs),), dtype=np.float32)
        self._obs = _nhwc_uint8(_reset_result(env.reset())[0])
        self._needs_fire = np.full((int(num_envs),), bool(fire_reset), dtype=np.bool_)

    @property
    def obs(self) -> np.ndarray:
        return self._obs

    @obs.setter
    def obs(self, value: Any) -> None:
        self._obs = _nhwc_uint8(value)

    @property
    def current_obs(self) -> np.ndarray:
        return self._obs

    @current_obs.setter
    def current_obs(self, value: Any) -> None:
        self.obs = value

    def get_obs(self) -> np.ndarray:
        return self._obs

    def reset(self) -> np.ndarray:
        obs, _ = _reset_result(self.handle.env.reset())
        self._returns[...] = 0.0
        self._needs_fire[...] = bool(self.handle.fire_reset)
        self._obs = _nhwc_uint8(obs)
        return self._obs

    def __call__(self, action_array: Any):
        action = _vec(action_array, np.int32, self.handle.num_envs, "action")
        fired = self._needs_fire.copy()
        exec_action = action.copy()
        exec_action[fired] = _FIRE_ACTION
        obs, reward, terminated, truncated, info = self.handle.env.step(exec_action)

        reward = _vec(reward, np.float32, self.handle.num_envs, "reward")
        terminated = _vec(terminated, np.bool_, self.handle.num_envs, "terminated")
        truncated = _vec(truncated, np.bool_, self.handle.num_envs, "truncated")

        ppo_done = np.logical_or(terminated, truncated)
        true_done = np.logical_or(
            _info_bool(info, "terminated", terminated, self.handle.num_envs),
            truncated,
        )
        self._needs_fire = np.where(
            np.logical_or(ppo_done, true_done),
            bool(self.handle.fire_reset),
            False,
        ).astype(np.bool_)
        raw_reward = _raw_reward(info, reward, self.handle.num_envs)
        reward_clipped = np.clip(reward, -1.0, 1.0).astype(np.float32)

        self._returns += raw_reward
        episode_returns = self._returns[true_done].astype(float).tolist()
        self._returns[true_done] = 0.0

        self._obs = _nhwc_uint8(obs)
        extra = {
            "reward_raw": raw_reward,
            "reward_unclipped": raw_reward,
            "episode_returns": episode_returns,
            "true_done": true_done,
            "terminated": terminated,
            "truncated": truncated,
            "lives": _info_value(info, "lives"),
            "info": info,
            "fired": fired.astype(np.bool_),
            "exec_action": exec_action.astype(np.int32),
        }
        return self._obs, reward_clipped, ppo_done.astype(np.bool_), true_done.astype(np.bool_), extra


def make_atari_env_step(
    game: str,
    num_envs: int,
    seed: int,
    *,
    training: bool = True,
    fire_reset: bool = True,
) -> tuple[AtariEnvStep, AtariEnvHandle]:
    """Create an envpool Atari stepper using verified terminal semantics.

    ``fire_reset`` sends full-action-space FIRE on reset/life starts. This is
    required by Breakout-style starts and benign for games that do not need it.
    """

    if training:
        env = make_train_env(str(game), int(num_envs), int(seed))
    else:
        env = make_eval_env(str(game), int(num_envs), int(seed))
    stepper = AtariEnvStep(
        env,
        str(game),
        int(num_envs),
        int(seed),
        bool(training),
        bool(fire_reset),
    )
    return stepper, stepper.handle


__all__ = ["AtariEnvHandle", "AtariEnvStep", "make_atari_env_step"]
