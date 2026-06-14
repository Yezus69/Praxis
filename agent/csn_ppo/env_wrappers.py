"""CSN-PPO training environment wrappers."""

from __future__ import annotations

from typing import Any

import jax
import jax.numpy as jnp
from brax.envs.wrappers import training as brax_training

from agent.csn_ppo.curriculum import (
    CurriculumState,
    init_curriculum_state,
    sample_world_difficulties,
)
from praxis import contract


def _broadcast_leaf(x: jax.Array, num_envs: int) -> jax.Array:
    x = jnp.asarray(x)
    return jnp.broadcast_to(x, (num_envs,) + x.shape)


def _first_leaf(x: jax.Array) -> jax.Array:
    return jnp.asarray(x)[0]


class _CurriculumVmapWrapper:
    """Vectorizes CoverEnv while preserving difficulty-aware resets."""

    def __init__(self, env: Any):
        self.env = env

    @property
    def action_size(self) -> int:
        return self.env.action_size

    @property
    def observation_size(self) -> int:
        return contract.OBS_DIM

    @property
    def backend(self) -> str:
        return getattr(self.env, "backend", "mjx")

    @property
    def unwrapped(self) -> Any:
        return self.env

    def reset(self, rng: jax.Array, difficulty: jax.Array | None = None) -> Any:
        if difficulty is None:
            return jax.vmap(self.env.reset)(rng)
        return jax.vmap(self.env.reset)(rng, difficulty)

    def step(self, state: Any, action: jax.Array) -> Any:
        return jax.vmap(self.env.step)(state, action)


class _DifficultyEpisodeWrapper(brax_training.EpisodeWrapper):
    """Brax EpisodeWrapper reset extended with per-env difficulties."""

    def reset(self, rng: jax.Array, difficulty: jax.Array | None = None) -> Any:
        state = self.env.reset(rng, difficulty)
        batch_shape = rng.shape[:-1]
        state.info["steps"] = jnp.zeros(batch_shape)
        state.info["truncation"] = jnp.zeros(batch_shape)
        state.info["episode_done"] = jnp.zeros(batch_shape)
        episode_metrics = {
            "sum_reward": jnp.zeros(batch_shape),
            "length": jnp.zeros(batch_shape),
        }
        for metric_name in state.metrics.keys():
            episode_metrics[metric_name] = jnp.zeros(batch_shape)
        state.info["episode_metrics"] = episode_metrics
        return state


def _select_done(done: jax.Array, reset_value: Any, step_value: Any) -> Any:
    def select_leaf(reset_leaf, step_leaf):
        cond = done.astype(jnp.bool_)
        while cond.ndim < jnp.asarray(step_leaf).ndim:
            cond = cond[..., None]
        return jnp.where(cond, reset_leaf, step_leaf)

    return jax.tree_util.tree_map(select_leaf, reset_value, step_value)


class CurriculumBraxTrainingWrapper:
    """Brax training wrapper with curriculum-aware full autoresets."""

    def __init__(
        self,
        env: Any,
        default_curriculum_state: CurriculumState | None = None,
        episode_length: int | None = None,
        action_repeat: int = 1,
    ):
        self.env = env
        episode_length = (
            episode_length
            if episode_length is not None
            else getattr(
                getattr(env, "_config", None),
                "episode_length",
                contract.EPISODE_LENGTH,
            )
        )
        self._vector_env = _CurriculumVmapWrapper(env)
        self._episode_env = _DifficultyEpisodeWrapper(
            self._vector_env,
            int(episode_length),
            int(action_repeat),
        )
        self._default_curriculum_state = (
            default_curriculum_state
            if default_curriculum_state is not None
            else init_curriculum_state({})
        )

    @property
    def action_size(self) -> int:
        return self.env.action_size

    @property
    def observation_size(self) -> int:
        return contract.OBS_DIM

    @property
    def backend(self) -> str:
        return getattr(self.env, "backend", "mjx")

    @property
    def unwrapped(self) -> Any:
        return self.env

    def _num_envs(self, state: Any) -> int:
        return int(state.done.shape[0])

    def _attach_curriculum_info(
        self,
        state: Any,
        curriculum_state: CurriculumState,
        sentinel_failure_difficulty: jax.Array,
        next_difficulty: jax.Array,
    ) -> Any:
        num_envs = self._num_envs(state)
        info = dict(state.info)
        info["curriculum_state"] = jax.tree_util.tree_map(
            lambda x: _broadcast_leaf(x, num_envs),
            curriculum_state,
        )
        info["sentinel_failure_difficulty"] = _broadcast_leaf(
            jnp.reshape(jnp.asarray(sentinel_failure_difficulty, dtype=jnp.float32), (-1,)),
            num_envs,
        )
        info["next_difficulty"] = jnp.reshape(
            jnp.asarray(next_difficulty, dtype=jnp.float32),
            (num_envs,),
        )
        return state.replace(info=info)

    def set_curriculum_info(
        self,
        state: Any,
        curriculum_state: CurriculumState,
        sentinel_failure_difficulty: jax.Array,
        next_difficulty: jax.Array,
    ) -> Any:
        return self._attach_curriculum_info(
            state,
            curriculum_state,
            sentinel_failure_difficulty,
            next_difficulty,
        )

    def _clear_done_for_step(self, state: Any) -> Any:
        done = state.done.astype(jnp.bool_)
        info = dict(state.info)
        if "steps" in info:
            info["steps"] = jnp.where(done, jnp.zeros_like(info["steps"]), info["steps"])
        return state.replace(done=jnp.zeros_like(state.done), info=info)

    def reset(
        self,
        rng: jax.Array,
        difficulty: jax.Array | None = None,
        curriculum_state: CurriculumState | None = None,
        sentinel_failure_difficulty: jax.Array | None = None,
    ) -> Any:
        rng = jnp.asarray(rng)
        if rng.ndim == 1:
            rng = rng[None, :]
        num_envs = int(rng.shape[0])
        curriculum_state = (
            self._default_curriculum_state
            if curriculum_state is None
            else curriculum_state
        )
        if difficulty is None:
            difficulty = jnp.full(
                (num_envs,),
                curriculum_state.current_difficulty,
                dtype=jnp.float32,
            )
        difficulty = jnp.reshape(jnp.asarray(difficulty, dtype=jnp.float32), (num_envs,))
        if sentinel_failure_difficulty is None:
            sentinel_failure_difficulty = jnp.asarray(
                [curriculum_state.current_difficulty],
                dtype=jnp.float32,
            )

        state = self._episode_env.reset(rng, difficulty)
        return self._attach_curriculum_info(
            state,
            curriculum_state,
            sentinel_failure_difficulty,
            difficulty,
        )

    def step(self, state: Any, action: jax.Array) -> Any:
        state = self._clear_done_for_step(state)
        stepped = self._episode_env.step(state, action)
        num_envs = self._num_envs(stepped)
        curriculum_state = jax.tree_util.tree_map(
            _first_leaf,
            stepped.info["curriculum_state"],
        )
        sentinel_failure_difficulty = _first_leaf(
            stepped.info["sentinel_failure_difficulty"]
        )
        reset_rng = stepped.info["rng"]
        difficulty_rng = jax.random.fold_in(
            reset_rng[0],
            jnp.sum(stepped.info["step"].astype(jnp.uint32)),
        )
        next_difficulty = sample_world_difficulties(
            curriculum_state,
            difficulty_rng,
            num_envs,
            sentinel_failure_difficulty,
        )
        stepped = self._attach_curriculum_info(
            stepped,
            curriculum_state,
            sentinel_failure_difficulty,
            next_difficulty,
        )
        reset_state = self._episode_env.reset(reset_rng, next_difficulty)
        reset_state = self._attach_curriculum_info(
            reset_state,
            curriculum_state,
            sentinel_failure_difficulty,
            next_difficulty,
        )

        done = stepped.done
        info = _select_done(done, reset_state.info, stepped.info)
        for key in ("steps", "truncation", "time_out", "episode_done", "episode_metrics"):
            if key in stepped.info:
                info[key] = stepped.info[key]
        return stepped.replace(
            data=_select_done(done, reset_state.data, stepped.data),
            obs=_select_done(done, reset_state.obs, stepped.obs),
            info=info,
        )
