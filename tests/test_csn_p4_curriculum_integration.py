from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from agent.csn_ppo.curriculum import (
    COMPONENT_FRONTIER,
    COMPONENT_HISTORY,
    COMPONENT_SENTINEL_FAILURE,
    CurriculumState,
    freeze_or_slow_curriculum,
    init_curriculum_state,
    maybe_advance_curriculum,
    sample_world_difficulties_with_components,
)
from praxis.envs.cover_env import CoverEnv, default_config


@dataclass(frozen=True)
class DummyCurriculumConfig:
    curriculum_initial_difficulty: float = 0.40
    curriculum_history_size: int = 8
    frontier_fraction: float = 0.70
    history_fraction: float = 0.20
    sentinel_failure_fraction: float = 0.10
    curriculum_advance_step: float = 0.05
    curriculum_success_threshold: float = 0.80
    curriculum_guard_violation_threshold: float = 0.0
    curriculum_collision_rate_threshold: float = 0.0
    curriculum_kl_threshold: float = 0.03
    curriculum_value_error_threshold: float = 0.25


def _state_with_history() -> CurriculumState:
    state = init_curriculum_state(DummyCurriculumConfig())
    return state.replace(
        current_difficulty=jnp.asarray(0.40, dtype=jnp.float32),
        history=jnp.asarray(
            [0.05, 0.10, 0.20, 0.30, 0.80, 0.90, 0.95, 1.00],
            dtype=jnp.float32,
        ),
        history_count=jnp.asarray(4, dtype=jnp.int32),
        history_index=jnp.asarray(4, dtype=jnp.int32),
    )


def test_4_1_cover_env_reset_difficulty_changes_obstacle_motion():
    env = CoverEnv(default_config())
    keys = jax.random.split(jax.random.PRNGKey(41), 128)

    easy = jax.vmap(lambda key: env.reset(key, jnp.asarray(0.0)))(keys)
    hard = jax.vmap(lambda key: env.reset(key, jnp.asarray(1.0)))(keys)

    easy_amp = jnp.mean(easy.info["obst_amp"])
    hard_amp = jnp.mean(hard.info["obst_amp"])
    easy_speed = jnp.mean(easy.info["obst_peak_speed"])
    hard_speed = jnp.mean(hard.info["obst_peak_speed"])

    assert float(hard_amp) > float(easy_amp)
    assert float(hard_speed) > float(easy_speed)


def test_4_2_curriculum_sampler_preserves_70_20_10_mixture():
    state = _state_with_history()
    _, components = sample_world_difficulties_with_components(
        state,
        jax.random.PRNGKey(42),
        100_000,
        jnp.asarray([0.90], dtype=jnp.float32),
    )

    frontier = jnp.mean((components == COMPONENT_FRONTIER).astype(jnp.float32))
    history = jnp.mean((components == COMPONENT_HISTORY).astype(jnp.float32))
    sentinel_failure = jnp.mean(
        (components == COMPONENT_SENTINEL_FAILURE).astype(jnp.float32)
    )

    assert abs(float(frontier) - 0.70) < 0.03
    assert abs(float(history) - 0.20) < 0.03
    assert abs(float(sentinel_failure) - 0.10) < 0.03


def test_4_3_freeze_or_slow_curriculum_sets_frozen():
    state = init_curriculum_state(DummyCurriculumConfig())
    frozen = freeze_or_slow_curriculum(state)

    assert bool(frozen.frozen) is True


def test_4_4_maybe_advance_curriculum_requires_current_and_history_pass():
    state = init_curriculum_state(DummyCurriculumConfig())
    held = maybe_advance_curriculum(
        state,
        {
            "current_pass": jnp.asarray(True),
            "historical_pass": jnp.asarray(False),
        },
    )
    advanced = maybe_advance_curriculum(
        state,
        {
            "current_pass": jnp.asarray(True),
            "historical_pass": jnp.asarray(True),
        },
    )

    assert float(held.current_difficulty) == float(state.current_difficulty)
    assert float(advanced.current_difficulty) > float(state.current_difficulty)
