"""Small deterministic environments for TFNS integration tests."""

from __future__ import annotations

from typing import Any

import numpy as np


DEFAULT_OBS_HW = 84
FRAME_STACK = 4


def _actions(action_array: Any, num_envs: int) -> np.ndarray:
    actions = np.asarray(action_array, dtype=np.int32).reshape(-1)
    if actions.shape != (int(num_envs),):
        raise ValueError(f"expected action shape ({num_envs},), got {actions.shape}")
    return actions


def _obs_shape(obs_hw: int) -> tuple[int, int, int]:
    hw = int(obs_hw)
    if hw <= 0:
        raise ValueError("obs_hw must be positive")
    return (hw, hw, FRAME_STACK)


def _base_frame(num_envs: int, obs_hw: int = DEFAULT_OBS_HW, value: int = 18) -> np.ndarray:
    return np.full((int(num_envs),) + _obs_shape(obs_hw), np.uint8(value), dtype=np.uint8)


def _scale_coord(coord: int, obs_hw: int) -> int:
    return int(round(float(coord) * float(obs_hw) / float(DEFAULT_OBS_HW)))


def _scaled_slice(start: int, stop: int, obs_hw: int) -> slice:
    lo = max(0, min(int(obs_hw), _scale_coord(start, obs_hw)))
    hi = max(0, min(int(obs_hw), _scale_coord(stop, obs_hw)))
    if hi <= lo:
        hi = min(int(obs_hw), lo + 1)
    return slice(lo, hi)


def _paint(obs: np.ndarray, rows: slice, cols: slice, value: int) -> None:
    obs[:, rows, cols, :] = np.uint8(value)


class TwoContextMemoryEnv:
    """Two-context POMDP where only the first observation reveals the context."""

    def __init__(
        self,
        num_envs: int,
        episode_length: int = 6,
        seed: int = 0,
        obs_hw: int = DEFAULT_OBS_HW,
    ):
        if int(episode_length) < 2:
            raise ValueError("episode_length must be at least 2")
        self.num_envs = int(num_envs)
        self.episode_length = int(episode_length)
        self.obs_hw = int(obs_hw)
        _obs_shape(self.obs_hw)
        self.rng = np.random.default_rng(int(seed))
        self.context = np.zeros((self.num_envs,), dtype=np.int32)
        self.t = np.zeros((self.num_envs,), dtype=np.int32)
        self.obs = _base_frame(self.num_envs, self.obs_hw)
        self.reset()

    def _sample_context(self, count: int) -> np.ndarray:
        context = np.arange(int(count), dtype=np.int32) % 2
        self.rng.shuffle(context)
        return context

    def reset(self) -> np.ndarray:
        self.context = self._sample_context(self.num_envs)
        self.t = np.zeros((self.num_envs,), dtype=np.int32)
        self.obs = self._render()
        return self.obs

    def current_obs(self) -> np.ndarray:
        return self.obs

    def _reset_done(self, done: np.ndarray) -> None:
        if not np.any(done):
            return
        count = int(np.sum(done))
        self.context[done] = self._sample_context(count)
        self.t[done] = 0

    def _render(self) -> np.ndarray:
        obs = _base_frame(self.num_envs, self.obs_hw, value=16)
        obs[
            :,
            _scaled_slice(34, 50, self.obs_hw),
            _scaled_slice(34, 50, self.obs_hw),
            :,
        ] = np.uint8(96)
        cue_rows = np.flatnonzero(self.t == 0)
        if cue_rows.size:
            context_zero = cue_rows[self.context[cue_rows] == 0]
            context_one = cue_rows[self.context[cue_rows] == 1]
            top_left = (
                _scaled_slice(6, 34, self.obs_hw),
                _scaled_slice(6, 34, self.obs_hw),
            )
            bottom_right = (
                _scaled_slice(50, 78, self.obs_hw),
                _scaled_slice(50, 78, self.obs_hw),
            )
            obs[context_zero, top_left[0], top_left[1], :] = np.uint8(248)
            obs[context_zero, bottom_right[0], bottom_right[1], :] = np.uint8(36)
            obs[context_one, top_left[0], top_left[1], :] = np.uint8(36)
            obs[context_one, bottom_right[0], bottom_right[1], :] = np.uint8(248)
        return obs

    def __call__(self, action_array: Any):
        actions = _actions(action_array, self.num_envs)
        decision = self.t >= 1
        reward = (decision & (actions == self.context)).astype(np.float32)
        done = (self.t + 1) >= self.episode_length
        self.t = self.t + 1
        self._reset_done(done)
        self.obs = self._render()
        extra = {"terminal": done.copy()}
        return self.obs, reward, done.astype(np.bool_), done.astype(np.bool_), extra


class ConflictingTaskEnv:
    """Two tasks with identical observations and opposite rewarded actions."""

    def __init__(
        self,
        num_envs: int,
        episode_length: int = 2,
        seed: int = 0,
        task: int = 0,
        obs_hw: int = DEFAULT_OBS_HW,
    ):
        if int(episode_length) < 1:
            raise ValueError("episode_length must be positive")
        self.num_envs = int(num_envs)
        self.episode_length = int(episode_length)
        self.obs_hw = int(obs_hw)
        _obs_shape(self.obs_hw)
        self.rng = np.random.default_rng(int(seed))
        self.task = int(task)
        self.t = np.zeros((self.num_envs,), dtype=np.int32)
        self.obs = _base_frame(self.num_envs, self.obs_hw)
        self.set_task(self.task)

    def set_task(self, task: int) -> np.ndarray:
        task = int(task)
        if task not in (0, 1):
            raise ValueError("task must be 0 or 1")
        self.task = task
        return self.reset()

    def reset(self) -> np.ndarray:
        self.t = np.zeros((self.num_envs,), dtype=np.int32)
        self.obs = self._render()
        return self.obs

    def current_obs(self) -> np.ndarray:
        return self.obs

    def _render(self) -> np.ndarray:
        obs = _base_frame(self.num_envs, self.obs_hw, value=24)
        _paint(
            obs,
            _scaled_slice(10, 74, self.obs_hw),
            _scaled_slice(10, 18, self.obs_hw),
            210,
        )
        _paint(
            obs,
            _scaled_slice(10, 74, self.obs_hw),
            _scaled_slice(66, 74, self.obs_hw),
            210,
        )
        _paint(
            obs,
            _scaled_slice(34, 50, self.obs_hw),
            _scaled_slice(34, 50, self.obs_hw),
            96,
        )
        _paint(
            obs,
            _scaled_slice(20, 28, self.obs_hw),
            _scaled_slice(28, 56, self.obs_hw),
            150,
        )
        _paint(
            obs,
            _scaled_slice(56, 64, self.obs_hw),
            _scaled_slice(28, 56, self.obs_hw),
            150,
        )
        return obs

    def __call__(self, action_array: Any):
        actions = _actions(action_array, self.num_envs)
        reward = (actions == self.task).astype(np.float32)
        done = (self.t + 1) >= self.episode_length
        self.t = np.where(done, 0, self.t + 1).astype(np.int32)
        self.obs = self._render()
        extra = {"terminal": done.copy()}
        return self.obs, reward, done.astype(np.bool_), done.astype(np.bool_), extra


class DelayedRewardEnv:
    """Delayed-credit environment where only the first action controls return."""

    def __init__(
        self,
        num_envs: int,
        episode_length: int = 6,
        seed: int = 0,
        terminal_reward: float = 1.0,
        obs_hw: int = DEFAULT_OBS_HW,
    ):
        if int(episode_length) < 2:
            raise ValueError("episode_length must be at least 2")
        self.num_envs = int(num_envs)
        self.episode_length = int(episode_length)
        self.obs_hw = int(obs_hw)
        _obs_shape(self.obs_hw)
        self.terminal_reward = float(terminal_reward)
        self.rng = np.random.default_rng(int(seed))
        self.t = np.zeros((self.num_envs,), dtype=np.int32)
        self.first_action_was_zero = np.zeros((self.num_envs,), dtype=np.bool_)
        self.obs = _base_frame(self.num_envs, self.obs_hw)
        self.reset()

    def reset(self) -> np.ndarray:
        self.t = np.zeros((self.num_envs,), dtype=np.int32)
        self.first_action_was_zero = np.zeros((self.num_envs,), dtype=np.bool_)
        self.obs = self._render()
        return self.obs

    def current_obs(self) -> np.ndarray:
        return self.obs

    def _render(self) -> np.ndarray:
        obs = _base_frame(self.num_envs, self.obs_hw, value=22)
        _paint(
            obs,
            _scaled_slice(38, 46, self.obs_hw),
            _scaled_slice(38, 46, self.obs_hw),
            72,
        )
        return obs

    def __call__(self, action_array: Any):
        actions = _actions(action_array, self.num_envs)
        at_start = self.t == 0
        self.first_action_was_zero = np.where(
            at_start,
            actions == 0,
            self.first_action_was_zero,
        )
        done = (self.t + 1) >= self.episode_length
        reward = np.where(done & self.first_action_was_zero, self.terminal_reward, 0.0).astype(
            np.float32
        )
        self.t = np.where(done, 0, self.t + 1).astype(np.int32)
        self.first_action_was_zero = np.where(done, False, self.first_action_was_zero)
        self.obs = self._render()
        extra = {"terminal": done.copy()}
        return self.obs, reward, done.astype(np.bool_), done.astype(np.bool_), extra


__all__ = [
    "ConflictingTaskEnv",
    "DelayedRewardEnv",
    "TwoContextMemoryEnv",
]
