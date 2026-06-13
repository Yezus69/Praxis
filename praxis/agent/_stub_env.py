"""praxis/agent/_stub_env.py — OPTIONAL dummy env for ``--stub`` smoke runs ONLY.

================================ STUB — NOT THE REAL ENV ================================
The real environment is ``praxis.envs.NavEnv`` (owned by Agent-A). This module exists
ONLY so ``praxis.train`` can be smoke-imported / smoke-run before Agent-A's env lands,
behind the explicit ``--stub`` flag. The DEFAULT training path uses the real NavEnv.

Goals of the stub:
  * Shape-correct: ``observation_size == contract.OBS_DIM (27)``, ``action_size == 2``.
  * Functional Brax/MJX-style API: ``reset(rng) -> State``, ``step(state, action) -> State``.
  * Emits the contract metric keys (``success``, ``collision``) + reward-component keys
    into ``state.metrics`` every step, and ``info['truncation']`` semantics, so that the
    trainer's progress_fn / Brax eval can surface ``eval/episode_success`` etc.
  * Produces a *trivially learnable* signal (reward = -||action - target||, success when
    close, collision otherwise on termination) so smoke curves are non-degenerate.

It is pure JAX + a Brax ``envs.Env`` subclass so ``wrapper.wrap_for_brax_training``
accepts it exactly like the real MjxEnv. It does NOT import mujoco / mjx.
========================================================================================
"""

from __future__ import annotations

import jax
import jax.numpy as jp
from brax import envs as brax_envs
from brax.envs.base import State

from praxis import contract


class StubNavEnv(brax_envs.Env):
    """Minimal pure-JAX stand-in for NavEnv. For ``--stub`` smoke runs only.

    The "task": pick an action close to a fixed per-episode target direction. Reward
    is dense (negative distance to target) so PPO reward visibly rises within a smoke
    run. ``success`` fires when the agent stays near the target; we terminate at
    ``EPISODE_LENGTH`` as a truncation. There is no physics here whatsoever.
    """

    def __init__(self, episode_length: int = contract.EPISODE_LENGTH):
        self._episode_length = int(episode_length)
        # A fixed 2D target the policy must match (normalized into action range).
        self._target = jp.array([0.5, -0.5], dtype=jp.float32)

    # --- Brax envs.Env interface ------------------------------------------------ #
    @property
    def observation_size(self) -> int:  # noqa: D401 - simple property
        return contract.OBS_DIM  # 27

    @property
    def action_size(self) -> int:
        return contract.ACT_DIM  # 2

    @property
    def backend(self) -> str:
        return "stub"

    def _obs(self, rng: jax.Array, last_action: jax.Array) -> jax.Array:
        """Build a 27-d obs. Encode the target in the goal slot so it's learnable."""
        obs = jp.zeros((contract.OBS_DIM,), dtype=jp.float32)
        # goal slice [0,4): dx, dy, dist, heading_err — stash target dir + a little noise.
        noise = 0.01 * jax.random.normal(rng, (contract.OBS_DIM,))
        obs = obs.at[0:2].set(self._target)
        obs = obs.at[2].set(jp.linalg.norm(self._target))
        # vel slice [4,7): last action echoed back so the policy can be reactive.
        obs = obs.at[4:6].set(last_action)
        return obs + noise

    def _metrics(self) -> dict:
        """Zeroed metric dict carrying every key the contract / progress_fn expects."""
        m = {
            contract.METRIC_SUCCESS: jp.zeros(()),
            contract.METRIC_COLLISION: jp.zeros(()),
        }
        for k in contract.METRIC_REWARD_COMPONENTS:
            m[k] = jp.zeros(())
        return m

    def reset(self, rng: jax.Array) -> State:
        rng, obs_rng = jax.random.split(rng)
        obs = self._obs(obs_rng, jp.zeros((contract.ACT_DIM,)))
        info = {
            "rng": rng,
            "step": jp.zeros(()),
            "truncation": jp.zeros(()),
        }
        return State(
            pipeline_state=None,  # no physics pipeline in the stub
            obs=obs,
            reward=jp.zeros(()),
            done=jp.zeros(()),
            metrics=self._metrics(),
            info=info,
        )

    def step(self, state: State, action: jax.Array) -> State:
        rng, obs_rng = jax.random.split(state.info["rng"])
        action = jp.clip(action, -contract.ACTION_LIMIT, contract.ACTION_LIMIT)

        # Dense, learnable reward: closer to target => higher reward.
        err = jp.linalg.norm(action - self._target)
        progress = -err  # in [-~2.1, 0]
        success = (err < 0.1).astype(jp.float32)
        # "collision" is a stand-in failure flag; never fires in the stub (kept 0)
        # so collision_rate trends to ~0, mirroring the desired real-run shape.
        collision = jp.zeros(())

        reward = (
            contract.DEFAULT_REWARD_WEIGHTS["k1"] * progress
            - contract.DEFAULT_REWARD_WEIGHTS["k3"]
            + contract.DEFAULT_REWARD_WEIGHTS["k4"] * success
        )

        step = state.info["step"] + 1.0
        timeout = step >= self._episode_length
        done = timeout.astype(jp.float32)
        # Episode end here is a TRUNCATION (timeout), so truncation=1 on done.
        truncation = done

        metrics = dict(state.metrics)
        metrics[contract.METRIC_SUCCESS] = success
        metrics[contract.METRIC_COLLISION] = collision
        metrics["reward_progress"] = contract.DEFAULT_REWARD_WEIGHTS["k1"] * progress
        metrics["reward_collision"] = -contract.DEFAULT_REWARD_WEIGHTS["k2"] * collision
        metrics["reward_time"] = jp.asarray(-contract.DEFAULT_REWARD_WEIGHTS["k3"])
        metrics["reward_success"] = contract.DEFAULT_REWARD_WEIGHTS["k4"] * success

        obs = self._obs(obs_rng, action)
        info = dict(state.info)
        info["rng"] = rng
        info["step"] = step
        info["truncation"] = truncation

        return state.replace(
            obs=obs,
            reward=reward,
            done=done,
            metrics=metrics,
            info=info,
        )


def make_stub_env(episode_length: int = contract.EPISODE_LENGTH) -> StubNavEnv:
    """Factory mirroring how train.py builds the real env."""
    return StubNavEnv(episode_length=episode_length)


__all__ = ["StubNavEnv", "make_stub_env"]
