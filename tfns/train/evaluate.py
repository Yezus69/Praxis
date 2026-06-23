"""Closed-loop Atari evaluation helpers."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from functools import partial
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tfns.envs import ACT_DIM
from tfns.train.atari_env import make_atari_env_step


DEFAULT_EVAL_MAX_STEPS = 20_000


def _std(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1))


def _sem(values: Sequence[float]) -> float:
    arr = np.asarray(values, dtype=np.float64)
    if arr.size <= 1:
        return 0.0
    return float(np.std(arr, ddof=1) / np.sqrt(arr.size))


def _default_adapter_dormant(agent: Any, adapter_dormant: Any | None) -> jnp.ndarray:
    if adapter_dormant is not None:
        return jnp.asarray(adapter_dormant, dtype=bool)
    return jnp.ones((int(agent.adapter_config.num_adapters),), dtype=bool)


def _resolve_max_steps(max_steps: int | None) -> int:
    if max_steps is None:
        return int(DEFAULT_EVAL_MAX_STEPS)
    value = int(max_steps)
    if value < 0:
        raise ValueError("max_steps must be non-negative")
    return value


def _step_raw_reward(extra: Any, reward_clipped: Any, num_envs: int) -> np.ndarray:
    reward = None
    if isinstance(extra, Mapping):
        reward = extra.get("reward_raw", extra.get("reward_unclipped"))
    if reward is None:
        reward = reward_clipped
    arr = np.asarray(reward, dtype=np.float32).reshape(-1)
    if int(arr.shape[0]) != int(num_envs):
        arr = np.asarray(reward_clipped, dtype=np.float32).reshape(-1)
    if int(arr.shape[0]) != int(num_envs):
        raise ValueError(f"reward must have shape ({num_envs},), got {arr.shape}")
    return arr


def _executed_action(extra: Any, fallback: np.ndarray) -> np.ndarray:
    if isinstance(extra, Mapping):
        value = extra.get("exec_action")
        if value is not None:
            return np.asarray(value, dtype=np.int32)
    return np.asarray(fallback, dtype=np.int32)


def _eval_result(
    returns: Sequence[float],
    running_returns: Any,
    *,
    n_episodes: int,
    capped: bool,
    total_transitions: int = 0,
) -> dict[str, Any]:
    """Build an evaluation report.

    Per spec section 19, an evaluation is *valid* only when the requested number
    of complete episodes finished. Partial in-progress returns are never
    substituted into the scientific score: when invalid, ``mean`` is NaN so any
    downstream gate fails loudly rather than silently passing on a partial
    average. The in-progress mean is still surfaced as ``partial_mean`` for
    diagnostics only.
    """

    used = [float(value) for value in returns[: int(n_episodes)]]
    completed = np.asarray(used, dtype=np.float64)
    valid = bool(completed.size >= int(n_episodes) and int(n_episodes) > 0)
    mean = float(np.mean(completed)) if completed.size else float("nan")
    if not valid:
        # Do not let a partial average masquerade as a score in any gate.
        mean = float("nan")
    in_progress = np.asarray(running_returns, dtype=np.float64).reshape(-1)
    partial_mean = float(np.mean(in_progress)) if in_progress.size else float("nan")
    return {
        "mean": mean,
        "std": _std(used),
        "sem": _sem(used),
        "n": int(completed.size),
        "n_requested": int(n_episodes),
        "returns": used,
        "capped": bool(capped),
        "valid": valid,
        "partial_mean": partial_mean,
        "total_transitions": int(total_transitions),
    }


@partial(jax.jit, static_argnames=("agent", "greedy"))
def _policy_action(
    agent: Any,
    params: Any,
    obs: Any,
    prev_action: Any,
    prev_reward_clipped: Any,
    reset: Any,
    hidden: Any,
    rng: Any,
    greedy: bool,
    adapter_dormant: Any,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    out = agent.apply(
        {"params": params},
        obs,
        prev_action,
        prev_reward_clipped,
        reset,
        hidden,
        adapter_dormant=adapter_dormant,
    )
    if greedy:
        action = jnp.argmax(out.logits, axis=-1).astype(jnp.int32)
    else:
        action = jax.random.categorical(rng, out.logits, axis=-1).astype(jnp.int32)
    return action, out.h_next.astype(jnp.float32)


def evaluate_game(
    agent: Any,
    params: Any,
    game: str,
    *,
    num_envs: int,
    n_episodes: int,
    seed: int,
    greedy: bool = False,
    adapter_dormant: Any | None = None,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Evaluate one Atari game with true returns and no task or memory input.

    ``max_steps`` counts individual environment transitions across vectorized
    calls, so each ``env_step`` call adds ``num_envs`` steps. ``None`` selects
    ``DEFAULT_EVAL_MAX_STEPS``; unbounded evaluation is intentionally unsupported.
    """

    env_count = int(num_envs)
    target_episodes = int(n_episodes)
    step_cap = _resolve_max_steps(max_steps)
    env_step, handle = make_atari_env_step(game, env_count, int(seed), training=False)
    try:
        obs = jnp.asarray(env_step.obs)
        hidden = agent.init_hidden(env_count, dtype=jnp.float32)
        prev_action = jnp.zeros((env_count,), dtype=jnp.int32)
        prev_reward = jnp.zeros((env_count,), dtype=jnp.float32)
        prev_reset = jnp.ones((env_count,), dtype=bool)
        dormant = _default_adapter_dormant(agent, adapter_dormant)
        rng = jax.random.PRNGKey(int(seed))
        returns: list[float] = []
        running_returns = np.zeros((env_count,), dtype=np.float32)
        steps = 0

        while len(returns) < target_episodes and steps < step_cap:
            rng, action_key = jax.random.split(rng)
            action, h_next = _policy_action(
                agent,
                params,
                obs,
                prev_action,
                prev_reward,
                prev_reset,
                hidden,
                action_key,
                bool(greedy),
                dormant,
            )
            action_np = np.asarray(jax.device_get(action), dtype=np.int32)
            next_obs, reward_clipped, _ppo_done, reset, extra = env_step(action_np)
            exec_action = _executed_action(extra, action_np)

            returns.extend(float(value) for value in extra.get("episode_returns", ()))
            running_returns += _step_raw_reward(extra, reward_clipped, env_count)
            running_returns[np.asarray(reset, dtype=np.bool_)] = 0.0
            steps += env_count
            reset_jnp = jnp.asarray(reset, dtype=bool)
            hidden = jnp.where(reset_jnp[:, None], jnp.zeros_like(h_next), h_next)
            obs = jnp.asarray(next_obs)
            prev_action = jnp.asarray(exec_action, dtype=jnp.int32)
            prev_reward = jnp.asarray(reward_clipped, dtype=jnp.float32)
            prev_reset = reset_jnp

        return _eval_result(
            returns,
            running_returns,
            n_episodes=target_episodes,
            capped=len(returns) < target_episodes,
            total_transitions=steps,
        )
    finally:
        handle.close()


def random_score(
    game: str,
    *,
    num_envs: int,
    n_episodes: int,
    seed: int,
    max_steps: int | None = None,
) -> dict[str, Any]:
    """Return true-score stats for a uniform random 18-action policy.

    ``max_steps`` counts individual environment transitions across vectorized
    calls, so each ``env_step`` call adds ``num_envs`` steps. ``None`` selects
    ``DEFAULT_EVAL_MAX_STEPS``; unbounded evaluation is intentionally unsupported.
    """

    env_count = int(num_envs)
    target_episodes = int(n_episodes)
    step_cap = _resolve_max_steps(max_steps)
    env_step, handle = make_atari_env_step(game, env_count, int(seed), training=False)
    try:
        rng = np.random.default_rng(int(seed))
        returns: list[float] = []
        running_returns = np.zeros((env_count,), dtype=np.float32)
        steps = 0
        while len(returns) < target_episodes and steps < step_cap:
            action = rng.integers(0, int(ACT_DIM), size=env_count, dtype=np.int32)
            _obs, reward_clipped, _ppo_done, reset, extra = env_step(action)
            returns.extend(float(value) for value in extra.get("episode_returns", ()))
            running_returns += _step_raw_reward(extra, reward_clipped, env_count)
            running_returns[np.asarray(reset, dtype=np.bool_)] = 0.0
            steps += env_count
        return _eval_result(
            returns,
            running_returns,
            n_episodes=target_episodes,
            capped=len(returns) < target_episodes,
            total_transitions=steps,
        )
    finally:
        handle.close()


def retention(S_cur: Any, S_rand: Any, S_best: Any, eps: float = 1.0e-8) -> Any:
    """Section 21.6 retention, intentionally unclamped."""

    return (np.asarray(S_cur) - np.asarray(S_rand)) / (
        np.asarray(S_best) - np.asarray(S_rand) + float(eps)
    )


def normalized_progress(S: Any, S_rand: Any, S_single: Any, eps: float = 1.0e-8) -> Any:
    """Random-normalized progress against a matched single-task reference."""

    return (np.asarray(S) - np.asarray(S_rand)) / (
        np.asarray(S_single) - np.asarray(S_rand) + float(eps)
    )


def _get_meta(meta: Any, names: tuple[str, ...], default: Any = None) -> Any:
    if isinstance(meta, Mapping):
        for name in names:
            if name in meta:
                return meta[name]
        return default
    for name in names:
        if hasattr(meta, name):
            return getattr(meta, name)
    return default


def _learned_items(learned_games_meta: Any) -> list[tuple[Any, Any]]:
    if isinstance(learned_games_meta, Mapping):
        return list(learned_games_meta.items())
    if isinstance(learned_games_meta, str):
        return [(learned_games_meta, {"game": learned_games_meta})]
    items = []
    for index, meta in enumerate(list(learned_games_meta)):
        if isinstance(meta, str):
            items.append((meta, {"game": meta}))
            continue
        key = _get_meta(meta, ("key", "game_key", "name"), index)
        items.append((key, meta))
    return items


def make_closed_loop_eval_fn(
    learned_games_meta: Any,
    agent: Any,
    *,
    num_envs: int,
    n_episodes: int,
    seed: int,
    max_steps: int | None = None,
    adapter_dormant: Any | None = None,
):
    """Return a live-param evaluator for consolidation gates.

    ``adapter_dormant`` is the single current global adapter mask. It is used
    for every learned game so routing is determined solely by current recurrent
    content; no per-game adapter mask is selected from metadata (that would be a
    future task-identity leak). All active adapters stay globally available.
    """

    items = _learned_items(learned_games_meta)
    step_cap = _resolve_max_steps(max_steps)
    global_dormant = None if adapter_dormant is None else jnp.asarray(adapter_dormant, dtype=bool)

    def eval_fn(params: Any) -> dict[Any, Any]:
        result: dict[Any, Any] = {}
        current_progress = None
        explicit_current = False
        all_valid = True

        for index, (game_key, meta) in enumerate(items):
            game = _get_meta(meta, ("game", "env", "atari_game"), game_key)
            eval_seed = int(seed) + 10_003 * index
            score_info = evaluate_game(
                agent,
                params,
                str(game),
                num_envs=int(num_envs),
                n_episodes=int(n_episodes),
                seed=eval_seed,
                greedy=False,
                adapter_dormant=global_dormant,
                max_steps=step_cap,
            )
            score = float(score_info["mean"])
            S_rand = _get_meta(meta, ("S_random", "random", "random_score"), None)
            random_capped = False
            if S_rand is None:
                random_info = random_score(
                    str(game),
                    num_envs=int(num_envs),
                    n_episodes=int(n_episodes),
                    seed=eval_seed + 1_000_003,
                    max_steps=step_cap,
                )
                S_rand = float(random_info["mean"])
                random_capped = bool(random_info["capped"])
            S_best = _get_meta(meta, ("S_best", "best", "best_score"), score)
            S_single = _get_meta(meta, ("S_single", "single", "single_score"), S_best)

            progress_value = float(normalized_progress(score, S_rand, S_single))
            retention_value = float(retention(score, S_rand, S_best))
            entry_valid = bool(score_info.get("valid", score_info["n"] >= int(n_episodes)))
            all_valid = all_valid and entry_valid
            result[game_key] = {
                "score": score,
                "progress": progress_value,
                "retention": retention_value,
                "std": float(score_info.get("std", 0.0)),
                "sem": float(score_info["sem"]),
                "n": int(score_info["n"]),
                "valid": entry_valid,
                "capped": bool(score_info.get("capped", False)),
                "random_capped": bool(random_capped),
                "total_transitions": int(score_info.get("total_transitions", 0)),
            }

            is_current = bool(_get_meta(meta, ("current", "is_current", "current_game"), False))
            if is_current or not explicit_current:
                current_progress = progress_value
                explicit_current = explicit_current or is_current

        result["current_progress"] = float(current_progress) if current_progress is not None else 0.0
        result["all_valid"] = bool(all_valid)
        return result

    return eval_fn


__all__ = [
    "DEFAULT_EVAL_MAX_STEPS",
    "evaluate_game",
    "make_closed_loop_eval_fn",
    "normalized_progress",
    "random_score",
    "retention",
]
