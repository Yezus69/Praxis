from __future__ import annotations

import numpy as np

from tfns.train.atari_env import AtariEnvStep


class _ScriptedAtariEnv:
    def __init__(self):
        self.actions: list[np.ndarray] = []
        self.t = 0

    def reset(self):
        self.t = 0
        return np.zeros((2, 84, 84, 4), dtype=np.uint8), {}

    def step(self, action):
        self.actions.append(np.asarray(action, dtype=np.int32).copy())
        terminated_script = [
            np.array([False, True], dtype=np.bool_),
            np.array([False, False], dtype=np.bool_),
            np.array([False, False], dtype=np.bool_),
        ]
        true_terminated_script = [
            np.array([False, False], dtype=np.bool_),
            np.array([True, False], dtype=np.bool_),
            np.array([False, False], dtype=np.bool_),
        ]
        t = min(self.t, len(terminated_script) - 1)
        self.t += 1
        obs = np.full((2, 84, 84, 4), self.t, dtype=np.uint8)
        reward = np.zeros((2,), dtype=np.float32)
        truncated = np.zeros((2,), dtype=np.bool_)
        info = {"terminated": true_terminated_script[t]}
        return obs, reward, terminated_script[t], truncated, info


def test_atari_env_step_force_fires_at_start_life_loss_and_true_reset():
    env = _ScriptedAtariEnv()
    stepper = AtariEnvStep(env, "Breakout-v5", 2, 0, True, fire_reset=True)

    _obs, _reward, ppo_done, reset, extra = stepper(np.array([4, 5], dtype=np.int32))
    np.testing.assert_array_equal(env.actions[-1], np.array([1, 1], dtype=np.int32))
    np.testing.assert_array_equal(extra["fired"], np.array([True, True], dtype=np.bool_))
    np.testing.assert_array_equal(extra["exec_action"], np.array([1, 1], dtype=np.int32))
    np.testing.assert_array_equal(ppo_done, np.array([False, True], dtype=np.bool_))
    np.testing.assert_array_equal(reset, np.array([False, False], dtype=np.bool_))

    _obs, _reward, _ppo_done, reset, extra = stepper(np.array([4, 5], dtype=np.int32))
    np.testing.assert_array_equal(env.actions[-1], np.array([4, 1], dtype=np.int32))
    np.testing.assert_array_equal(extra["fired"], np.array([False, True], dtype=np.bool_))
    np.testing.assert_array_equal(reset, np.array([True, False], dtype=np.bool_))

    _obs, _reward, _ppo_done, _reset, extra = stepper(np.array([4, 5], dtype=np.int32))
    np.testing.assert_array_equal(env.actions[-1], np.array([1, 5], dtype=np.int32))
    np.testing.assert_array_equal(extra["fired"], np.array([True, False], dtype=np.bool_))


def test_atari_env_step_fire_reset_can_be_disabled():
    env = _ScriptedAtariEnv()
    stepper = AtariEnvStep(env, "Breakout-v5", 2, 0, True, fire_reset=False)

    _obs, _reward, _ppo_done, _reset, extra = stepper(np.array([4, 5], dtype=np.int32))

    np.testing.assert_array_equal(env.actions[-1], np.array([4, 5], dtype=np.int32))
    np.testing.assert_array_equal(extra["fired"], np.array([False, False], dtype=np.bool_))
    np.testing.assert_array_equal(extra["exec_action"], np.array([4, 5], dtype=np.int32))
