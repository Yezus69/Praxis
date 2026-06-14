"""Curriculum mixture for CSN-PPO.

Implements the section-22 curriculum sampler and advancement gate as a
standalone JAX pytree module.  The sampler keeps all arrays fixed-size for
JIT/update-path compatibility.
"""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
from flax import struct


DEFAULT_FRONTIER_FRACTION = 0.70
DEFAULT_HISTORY_FRACTION = 0.20
DEFAULT_SENTINEL_FAILURE_FRACTION = 0.10
DEFAULT_HISTORY_SIZE = 32
DEFAULT_ADVANCE_STEP = 0.05
DEFAULT_SUCCESS_THRESHOLD = 0.80
DEFAULT_GUARD_VIOLATION_THRESHOLD = 0.0
DEFAULT_COLLISION_RATE_THRESHOLD = 0.0
DEFAULT_KL_THRESHOLD = 0.03
DEFAULT_VALUE_ERROR_THRESHOLD = 0.25
MAX_OBSTACLES = 4
COMPONENT_FRONTIER = 0
COMPONENT_HISTORY = 1
COMPONENT_SENTINEL_FAILURE = 2


@struct.dataclass
class CurriculumState:
    """Fixed-shape curriculum state for section 22."""

    current_difficulty: jax.Array
    history: jax.Array
    history_index: jax.Array
    history_count: jax.Array
    frozen: jax.Array
    frontier_fraction: jax.Array
    history_fraction: jax.Array
    sentinel_failure_fraction: jax.Array
    advance_step: jax.Array
    success_threshold: jax.Array
    guard_violation_threshold: jax.Array
    collision_rate_threshold: jax.Array
    kl_threshold: jax.Array
    value_error_threshold: jax.Array


def _config_value(config: Any, names: tuple[str, ...], default: Any) -> Any:
    for name in names:
        if isinstance(config, Mapping) and name in config:
            return config[name]
        if hasattr(config, name):
            return getattr(config, name)
    return default


def _metric_value(metrics: Any, names: tuple[str, ...]) -> tuple[Any, bool]:
    for name in names:
        if isinstance(metrics, Mapping) and name in metrics:
            return metrics[name], True
        if hasattr(metrics, name):
            return getattr(metrics, name), True
    return None, False


def _as_float(value: Any) -> jax.Array:
    return jnp.asarray(value, dtype=jnp.float32)


def _as_bool(value: Any) -> jax.Array:
    return jnp.asarray(value, dtype=jnp.bool_)


def init_curriculum_state(config: Any) -> CurriculumState:
    """Create the section-22 curriculum state from config."""

    history_size = int(
        _config_value(
            config,
            ("curriculum_history_size", "history_size", "difficulty_history_size"),
            DEFAULT_HISTORY_SIZE,
        )
    )
    history_size = max(history_size, 1)
    initial_difficulty = _config_value(
        config,
        ("current_difficulty", "initial_difficulty", "curriculum_initial_difficulty"),
        0.0,
    )

    return CurriculumState(
        current_difficulty=jnp.clip(_as_float(initial_difficulty), 0.0, 1.0),
        history=jnp.zeros((history_size,), dtype=jnp.float32),
        history_index=jnp.asarray(0, dtype=jnp.int32),
        history_count=jnp.asarray(0, dtype=jnp.int32),
        frozen=jnp.asarray(False, dtype=jnp.bool_),
        frontier_fraction=_as_float(
            _config_value(config, ("frontier_fraction",), DEFAULT_FRONTIER_FRACTION)
        ),
        history_fraction=_as_float(
            _config_value(config, ("history_fraction",), DEFAULT_HISTORY_FRACTION)
        ),
        sentinel_failure_fraction=_as_float(
            _config_value(
                config,
                ("sentinel_failure_fraction",),
                DEFAULT_SENTINEL_FAILURE_FRACTION,
            )
        ),
        advance_step=_as_float(
            _config_value(
                config,
                ("curriculum_advance_step", "difficulty_step", "advance_step"),
                DEFAULT_ADVANCE_STEP,
            )
        ),
        success_threshold=_as_float(
            _config_value(
                config,
                ("curriculum_success_threshold", "sentinel_success_threshold"),
                DEFAULT_SUCCESS_THRESHOLD,
            )
        ),
        guard_violation_threshold=_as_float(
            _config_value(
                config,
                (
                    "curriculum_guard_violation_threshold",
                    "sentinel_guard_violation_threshold",
                ),
                DEFAULT_GUARD_VIOLATION_THRESHOLD,
            )
        ),
        collision_rate_threshold=_as_float(
            _config_value(
                config,
                ("curriculum_collision_rate_threshold", "sentinel_collision_threshold"),
                DEFAULT_COLLISION_RATE_THRESHOLD,
            )
        ),
        kl_threshold=_as_float(
            _config_value(
                config,
                ("curriculum_kl_threshold", "kl_budget", "sentinel_kl_threshold"),
                DEFAULT_KL_THRESHOLD,
            )
        ),
        value_error_threshold=_as_float(
            _config_value(
                config,
                (
                    "curriculum_value_error_threshold",
                    "value_budget",
                    "sentinel_value_error_threshold",
                ),
                DEFAULT_VALUE_ERROR_THRESHOLD,
            )
        ),
    )


def _mixture_logits(state: CurriculumState) -> jax.Array:
    fractions = jnp.asarray(
        [
            state.frontier_fraction,
            state.history_fraction,
            state.sentinel_failure_fraction,
        ],
        dtype=jnp.float32,
    )
    fractions = jnp.maximum(fractions, 0.0)
    default_fractions = jnp.asarray(
        [
            DEFAULT_FRONTIER_FRACTION,
            DEFAULT_HISTORY_FRACTION,
            DEFAULT_SENTINEL_FAILURE_FRACTION,
        ],
        dtype=jnp.float32,
    )
    fraction_sum = jnp.sum(fractions)
    fractions = jnp.where(fraction_sum > 0.0, fractions / fraction_sum, default_fractions)
    return jnp.where(fractions > 0.0, jnp.log(fractions), -jnp.inf)


def _sample_history_difficulties(
    state: CurriculumState, rng: jax.Array, num_envs: int
) -> jax.Array:
    history_capacity = state.history.shape[0]
    history_positions = jnp.arange(history_capacity, dtype=jnp.int32)
    valid_history = (history_positions < state.history_count) & (
        state.history < state.current_difficulty
    )
    has_history = jnp.any(valid_history)
    history_logits = jnp.where(valid_history, 0.0, -jnp.inf)
    history_logits = jnp.where(has_history, history_logits, jnp.zeros_like(history_logits))
    history_indices = jax.random.categorical(rng, history_logits, shape=(num_envs,))
    history_draws = jnp.take(state.history, history_indices, mode="clip")
    return jnp.where(has_history, history_draws, state.current_difficulty)


def _sample_sentinel_failure_difficulties(
    sentinel_failure_difficulty: jax.Array, rng: jax.Array, num_envs: int
) -> jax.Array:
    values = jnp.reshape(
        jnp.clip(_as_float(sentinel_failure_difficulty), 0.0, 1.0),
        (-1,),
    )
    indices = jax.random.randint(rng, (num_envs,), 0, values.shape[0])
    return jnp.take(values, indices, mode="clip")


def sample_world_difficulties_with_components(
    state: CurriculumState,
    rng: jax.Array,
    num_envs: int,
    sentinel_failure_difficulty: jax.Array,
) -> tuple[jax.Array, jax.Array]:
    """Sample difficulties and mixture labels from the section-22 curriculum mixture."""

    component_key, history_key, sentinel_key = jax.random.split(rng, 3)
    components = jax.random.categorical(
        component_key, _mixture_logits(state), shape=(num_envs,)
    )
    frontier_draws = jnp.full(
        (num_envs,), state.current_difficulty, dtype=jnp.float32
    )
    history_draws = _sample_history_difficulties(state, history_key, num_envs)
    sentinel_failure_draws = _sample_sentinel_failure_difficulties(
        sentinel_failure_difficulty,
        sentinel_key,
        num_envs,
    )
    difficulties = jnp.where(
        components == COMPONENT_FRONTIER,
        frontier_draws,
        jnp.where(components == COMPONENT_HISTORY, history_draws, sentinel_failure_draws),
    )
    return difficulties, components


def sample_world_difficulties(
    state: CurriculumState,
    rng: jax.Array,
    num_envs: int,
    sentinel_failure_difficulty: jax.Array,
) -> jax.Array:
    """Sample per-env difficulties from the section-22 curriculum mixture."""

    difficulties, _ = sample_world_difficulties_with_components(
        state,
        rng,
        num_envs,
        sentinel_failure_difficulty,
    )
    return difficulties


def difficulty_to_env_params(difficulty: jax.Array) -> dict[str, jax.Array]:
    """Map a scalar difficulty to cover-env randomization parameters."""

    difficulty = jnp.clip(_as_float(difficulty), 0.0, 1.0)
    return {
        "moving_obstacle_speed": 0.10 + 0.90 * difficulty,
        "moving_obstacle_amplitude": 0.05 + 0.45 * difficulty,
        "frac_moving": difficulty,
        "start_goal_spread": 0.25 + 0.75 * difficulty,
        "max_obstacles": jnp.asarray(MAX_OBSTACLES, dtype=jnp.int32),
    }


def _all_conditions(conditions: tuple[jax.Array, ...]) -> jax.Array:
    passed = jnp.asarray(True, dtype=jnp.bool_)
    for condition in conditions:
        passed = passed & _as_bool(condition)
    return passed


def _slice_pass(
    sentinel_metrics: Any, prefixes: tuple[str, ...], state: CurriculumState
) -> jax.Array:
    explicit_names = tuple(name for prefix in prefixes for name in (f"{prefix}_pass",))
    explicit_value, has_explicit = _metric_value(sentinel_metrics, explicit_names)
    if has_explicit:
        return _as_bool(explicit_value)

    conditions = []
    for prefix in prefixes:
        success, has_success = _metric_value(
            sentinel_metrics,
            (
                f"{prefix}_success_rate",
                f"{prefix}_success",
            ),
        )
        if has_success:
            conditions.append(_as_float(success) >= state.success_threshold)

        guard_rate, has_guard_rate = _metric_value(
            sentinel_metrics,
            (
                f"{prefix}_guard_violation_rate",
                f"{prefix}_hinge_violation_rate",
                f"{prefix}_constraint_violation_rate",
            ),
        )
        if has_guard_rate:
            conditions.append(_as_float(guard_rate) <= state.guard_violation_threshold)

        collision_rate, has_collision_rate = _metric_value(
            sentinel_metrics,
            (
                f"{prefix}_collision_rate",
                f"{prefix}_failure_rate",
            ),
        )
        if has_collision_rate:
            conditions.append(_as_float(collision_rate) <= state.collision_rate_threshold)

        kl_value, has_kl = _metric_value(
            sentinel_metrics,
            (
                f"{prefix}_kl",
                f"{prefix}_mean_kl",
            ),
        )
        if has_kl:
            conditions.append(_as_float(kl_value) <= state.kl_threshold)

        value_error, has_value_error = _metric_value(
            sentinel_metrics,
            (
                f"{prefix}_value_error",
                f"{prefix}_value_loss",
            ),
        )
        if has_value_error:
            conditions.append(_as_float(value_error) <= state.value_error_threshold)

    if not conditions:
        return jnp.asarray(False, dtype=jnp.bool_)
    return _all_conditions(tuple(conditions))


def maybe_advance_curriculum(
    state: CurriculumState, sentinel_metrics: Any
) -> CurriculumState:
    """Advance only when current and historical sentinel slices pass."""

    current_pass = _slice_pass(sentinel_metrics, ("current", "frontier"), state)
    historical_pass = _slice_pass(sentinel_metrics, ("historical", "history"), state)
    should_advance = (~state.frozen) & current_pass & historical_pass

    history_capacity = state.history.shape[0]
    write_index = jnp.mod(state.history_index, history_capacity)
    proposed_history = state.history.at[write_index].set(state.current_difficulty)
    proposed_history_index = jnp.mod(write_index + 1, history_capacity)
    proposed_history_count = jnp.minimum(
        state.history_count + jnp.asarray(1, dtype=jnp.int32),
        jnp.asarray(history_capacity, dtype=jnp.int32),
    )
    proposed_difficulty = jnp.clip(state.current_difficulty + state.advance_step, 0.0, 1.0)

    return state.replace(
        current_difficulty=jnp.where(
            should_advance, proposed_difficulty, state.current_difficulty
        ),
        history=jnp.where(should_advance, proposed_history, state.history),
        history_index=jnp.where(
            should_advance, proposed_history_index, state.history_index
        ),
        history_count=jnp.where(
            should_advance, proposed_history_count, state.history_count
        ),
    )


def freeze_or_slow_curriculum(state: CurriculumState) -> CurriculumState:
    """Freeze advancement after a sentinel regression."""

    return state.replace(frozen=jnp.asarray(True, dtype=jnp.bool_))
