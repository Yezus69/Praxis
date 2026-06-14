from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from agent.csn_ppo.curriculum import (
    CurriculumState,
    difficulty_to_env_params,
    freeze_or_slow_curriculum,
    init_curriculum_state,
    maybe_advance_curriculum,
    sample_world_difficulties,
)


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


def test_sample_world_difficulties_matches_section_22_mixture_fractions():
    state = _state_with_history()
    draws = sample_world_difficulties(
        state,
        jax.random.PRNGKey(0),
        50_000,
        jnp.asarray(0.90, dtype=jnp.float32),
    )

    frontier_fraction = jnp.mean(jnp.isclose(draws, 0.40))
    history_fraction = jnp.mean(
        jnp.isclose(draws, 0.05)
        | jnp.isclose(draws, 0.10)
        | jnp.isclose(draws, 0.20)
        | jnp.isclose(draws, 0.30)
    )
    sentinel_failure_fraction = jnp.mean(jnp.isclose(draws, 0.90))

    assert abs(float(frontier_fraction) - 0.70) < 0.03
    assert abs(float(history_fraction) - 0.20) < 0.03
    assert abs(float(sentinel_failure_fraction) - 0.10) < 0.02


def test_maybe_advance_curriculum_advances_only_when_current_and_history_pass():
    state = init_curriculum_state(DummyCurriculumConfig())
    pass_metrics = {
        "current_pass": jnp.asarray(True),
        "historical_pass": jnp.asarray(True),
    }
    advanced = maybe_advance_curriculum(state, pass_metrics)

    assert float(advanced.current_difficulty) > float(state.current_difficulty)
    assert int(advanced.history_count) == 1
    assert float(advanced.history[0]) == float(state.current_difficulty)

    hold_metrics = {
        "current_pass": jnp.asarray(True),
        "historical_pass": jnp.asarray(False),
    }
    held = maybe_advance_curriculum(state, hold_metrics)

    assert float(held.current_difficulty) == float(state.current_difficulty)
    assert int(held.history_count) == 0


def test_freeze_stops_curriculum_advancement():
    state = init_curriculum_state(DummyCurriculumConfig())
    frozen = freeze_or_slow_curriculum(state)
    metrics = {
        "current_pass": jnp.asarray(True),
        "historical_pass": jnp.asarray(True),
    }
    advanced = maybe_advance_curriculum(frozen, metrics)

    assert bool(advanced.frozen)
    assert float(advanced.current_difficulty) == float(state.current_difficulty)
    assert int(advanced.history_count) == 0


def test_curriculum_state_is_jax_pytree():
    state = _state_with_history()
    leaves, treedef = jax.tree_util.tree_flatten(state)
    rebuilt = jax.tree_util.tree_unflatten(treedef, leaves)

    assert isinstance(rebuilt, CurriculumState)
    assert jnp.allclose(rebuilt.current_difficulty, state.current_difficulty)
    assert jnp.allclose(rebuilt.history, state.history)
    assert bool(rebuilt.frozen) == bool(state.frozen)


def test_difficulty_to_env_params_keeps_obstacle_count_fixed():
    easy = difficulty_to_env_params(jnp.asarray(0.0))
    hard = difficulty_to_env_params(jnp.asarray(1.0))

    assert int(easy["max_obstacles"]) == 4
    assert int(hard["max_obstacles"]) == 4
    assert float(hard["moving_obstacle_speed"]) > float(easy["moving_obstacle_speed"])
    assert float(hard["moving_obstacle_amplitude"]) > float(
        easy["moving_obstacle_amplitude"]
    )
    assert float(hard["frac_moving"]) > float(easy["frac_moving"])
    assert float(hard["start_goal_spread"]) > float(easy["start_goal_spread"])
