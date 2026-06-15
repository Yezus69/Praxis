"""Tiny deterministic goal-conditioned gridworld for continual RL."""

from __future__ import annotations

from dataclasses import dataclass
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np


@dataclass(frozen=True)
class GridWorldConfig:
    grid_size: int = 5
    horizon: int = 25
    step_penalty: float = -0.01
    goal_reward: float = 1.0


class GridWorldState(NamedTuple):
    position: jnp.ndarray
    goal_id: jnp.ndarray
    step: jnp.ndarray
    done: jnp.ndarray
    reached: jnp.ndarray


class RolloutBatch(NamedTuple):
    obs: jnp.ndarray
    actions: jnp.ndarray
    rewards: jnp.ndarray
    done: jnp.ndarray
    reached: jnp.ndarray
    mask: jnp.ndarray
    logits: jnp.ndarray
    values: jnp.ndarray


def default_goal_cells(num_goals: int, grid_size: int = 5) -> np.ndarray:
    """Return deterministic goal cells, corners first, then spread over the grid."""
    num_goals = int(num_goals)
    grid_size = int(grid_size)
    if num_goals <= 0:
        raise ValueError("num_goals must be positive")
    num_cells = grid_size * grid_size
    if num_goals > num_cells:
        raise ValueError("num_goals cannot exceed grid_size * grid_size")

    corners = [0, grid_size - 1, num_cells - grid_size, num_cells - 1]
    candidates = []
    for cell in corners:
        if 0 <= cell < num_cells and cell not in candidates:
            candidates.append(cell)
    for cell in np.linspace(0, num_cells - 1, num_cells, dtype=np.int32):
        value = int(cell)
        if value not in candidates:
            candidates.append(value)
        if len(candidates) >= num_goals:
            break
    return np.asarray(candidates[:num_goals], dtype=np.int32)


def normalize_goal_cells(goals, grid_size: int = 5) -> np.ndarray:
    """Normalize cell ids or ``(row, col)`` goal coordinates to cell ids."""
    if goals is None:
        return default_goal_cells(4, grid_size)
    if isinstance(goals, (int, np.integer)):
        return default_goal_cells(int(goals), grid_size)

    arr = np.asarray(goals, dtype=np.int32)
    if arr.ndim == 0:
        return default_goal_cells(int(arr), grid_size)
    if arr.ndim == 2 and arr.shape[1] == 2:
        rows = arr[:, 0]
        cols = arr[:, 1]
        if np.any(rows < 0) or np.any(rows >= grid_size):
            raise ValueError("goal row outside grid")
        if np.any(cols < 0) or np.any(cols >= grid_size):
            raise ValueError("goal column outside grid")
        cells = rows * int(grid_size) + cols
    elif arr.ndim == 1:
        cells = arr
    else:
        raise ValueError("goals must be an int, cell-id vector, or (row, col) array")

    num_cells = int(grid_size) * int(grid_size)
    if cells.size == 0:
        raise ValueError("at least one goal is required")
    if np.any(cells < 0) or np.any(cells >= num_cells):
        raise ValueError("goal cell outside grid")
    if len(np.unique(cells)) != int(cells.size):
        raise ValueError("goal cells must be unique")
    return np.asarray(cells, dtype=np.int32)


def _policy_outputs(output):
    if isinstance(output, dict):
        return output["policy_logits"], output["value"]
    logits, value = output
    return logits, value


class GridWorld:
    """Vectorized deterministic gridworld with one-hot state and goal context."""

    def __init__(
        self,
        grid_size: int = 5,
        horizon: int = 25,
        goal_cells=None,
        step_penalty: float = -0.01,
        goal_reward: float = 1.0,
    ):
        self.config = GridWorldConfig(
            grid_size=int(grid_size),
            horizon=int(horizon),
            step_penalty=float(step_penalty),
            goal_reward=float(goal_reward),
        )
        self.goal_cells_np = normalize_goal_cells(
            default_goal_cells(4, grid_size) if goal_cells is None else goal_cells,
            grid_size,
        )
        self.goal_cells = jnp.asarray(self.goal_cells_np, dtype=jnp.int32)
        self.num_goals = int(self.goal_cells_np.shape[0])
        self.num_cells = int(self.config.grid_size * self.config.grid_size)
        self.obs_dim = int(self.num_cells + self.num_goals)
        self.num_actions = 4
        if self.num_cells <= 1:
            raise ValueError("grid_size must produce at least two cells")

    @property
    def horizon(self) -> int:
        return int(self.config.horizon)

    @property
    def grid_size(self) -> int:
        return int(self.config.grid_size)

    def goal_cell(self, goal_id):
        return jnp.take(self.goal_cells, jnp.asarray(goal_id, dtype=jnp.int32), axis=0)

    def state_from_position(self, position, goal_id, step=0, done=False, reached=False):
        position = jnp.asarray(position, dtype=jnp.int32)
        goal_id = jnp.asarray(goal_id, dtype=jnp.int32)
        shape = jnp.broadcast_shapes(position.shape, goal_id.shape)
        position = jnp.broadcast_to(position, shape)
        goal_id = jnp.broadcast_to(goal_id, shape)
        return GridWorldState(
            position=position,
            goal_id=goal_id,
            step=jnp.broadcast_to(jnp.asarray(step, dtype=jnp.int32), shape),
            done=jnp.broadcast_to(jnp.asarray(done, dtype=bool), shape),
            reached=jnp.broadcast_to(jnp.asarray(reached, dtype=bool), shape),
        )

    def reset(self, key, goal_id):
        """Sample non-goal starts for each requested goal id."""
        goal_id = jnp.asarray(goal_id, dtype=jnp.int32)
        goal_cell = self.goal_cell(goal_id)
        raw = jax.random.randint(
            key,
            goal_id.shape,
            minval=0,
            maxval=self.num_cells - 1,
            dtype=jnp.int32,
        )
        position = raw + (raw >= goal_cell).astype(jnp.int32)
        zeros_i = jnp.zeros(goal_id.shape, dtype=jnp.int32)
        zeros_b = jnp.zeros(goal_id.shape, dtype=bool)
        return GridWorldState(position, goal_id, zeros_i, zeros_b, zeros_b)

    def observe(self, state: GridWorldState) -> jnp.ndarray:
        pos = jax.nn.one_hot(state.position, self.num_cells, dtype=jnp.float32)
        goal = jax.nn.one_hot(state.goal_id, self.num_goals, dtype=jnp.float32)
        return jnp.concatenate([pos, goal], axis=-1)

    def all_observations_for_goal(self, goal_id) -> jnp.ndarray:
        """Return observations for all non-goal positions for one goal id."""
        goal_id = jnp.asarray(goal_id, dtype=jnp.int32)
        goal_cell = self.goal_cell(goal_id)
        raw = jnp.arange(self.num_cells - 1, dtype=jnp.int32)
        positions = raw + (raw >= goal_cell).astype(jnp.int32)
        goal_ids = jnp.full((self.num_cells - 1,), goal_id, dtype=jnp.int32)
        return self.observe(self.state_from_position(positions, goal_ids))

    def step(self, state: GridWorldState, action) -> tuple[GridWorldState, jnp.ndarray]:
        action = jnp.asarray(action, dtype=jnp.int32)
        row = state.position // self.grid_size
        col = state.position % self.grid_size

        next_row = jnp.where(action == 0, row - 1, row)
        next_row = jnp.where(action == 1, row + 1, next_row)
        next_col = jnp.where(action == 2, col - 1, col)
        next_col = jnp.where(action == 3, col + 1, next_col)
        next_row = jnp.clip(next_row, 0, self.grid_size - 1)
        next_col = jnp.clip(next_col, 0, self.grid_size - 1)
        next_position = next_row * self.grid_size + next_col
        next_position = jnp.where(state.done, state.position, next_position)

        goal_cell = self.goal_cell(state.goal_id)
        reached_now = jnp.logical_and(jnp.logical_not(state.done), next_position == goal_cell)
        next_reached = jnp.logical_or(state.reached, reached_now)
        next_step = state.step + jnp.where(state.done, 0, 1).astype(jnp.int32)
        timeout = next_step >= self.horizon
        next_done = jnp.logical_or(jnp.logical_or(state.done, reached_now), timeout)

        reward = jnp.where(reached_now, self.config.goal_reward, self.config.step_penalty)
        reward = jnp.where(state.done, 0.0, reward).astype(jnp.float32)
        next_state = GridWorldState(
            position=next_position.astype(jnp.int32),
            goal_id=state.goal_id,
            step=next_step.astype(jnp.int32),
            done=next_done,
            reached=next_reached,
        )
        return next_state, reward

    def rollout(self, key, policy_fn, params, goal_id, batch_size: int, greedy: bool = False):
        """Roll out a vectorized batch of episodes with ``lax.scan`` over time."""
        batch_size = int(batch_size)
        key_reset, key_action = jax.random.split(key)
        goal_ids = jnp.full((batch_size,), jnp.asarray(goal_id, dtype=jnp.int32))
        init_state = self.reset(key_reset, goal_ids)

        def body(carry, _):
            state, action_key = carry
            obs = self.observe(state)
            logits, value = _policy_outputs(policy_fn(params, obs))
            active = jnp.logical_not(state.done)
            action_key, subkey = jax.random.split(action_key)
            if greedy:
                action = jnp.argmax(logits, axis=-1).astype(jnp.int32)
            else:
                action = jax.random.categorical(subkey, logits, axis=-1).astype(jnp.int32)
            next_state, reward = self.step(state, action)
            step = RolloutBatch(
                obs=obs,
                actions=action,
                rewards=reward,
                done=next_state.done,
                reached=next_state.reached,
                mask=active.astype(jnp.float32),
                logits=logits,
                values=jnp.asarray(value),
            )
            return (next_state, action_key), step

        (_, _), traj = jax.lax.scan(body, (init_state, key_action), None, length=self.horizon)
        return traj


__all__ = [
    "GridWorld",
    "GridWorldConfig",
    "GridWorldState",
    "RolloutBatch",
    "default_goal_cells",
    "normalize_goal_cells",
]
