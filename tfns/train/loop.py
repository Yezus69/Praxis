"""Reusable training-loop helpers for task-free integration runs."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

from tfns.config import PPOConfig, TFNSConfig
from tfns.consolidate.lifecycle import apply_rejection_feedback, consolidate
from tfns.consolidate.state import ContinualState
from tfns.credit import ReturnPredictor, make_predictor_optimizer
from tfns.detect import PageHinkleyDetector
from tfns.memory.bank import SequenceMemoryBank
from tfns.protect.bases import empty_basis
from tfns.protect.projection import build_protected_modules
from tfns.train.block import train_block


def _cfg_section(cfg: Any, name: str, default: Any = None) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _cfg_value(obj: Any, name: str, default: Any) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _current_obs(env_step: Any) -> Any:
    for name in ("obs", "current_obs"):
        if hasattr(env_step, name):
            value = getattr(env_step, name)
            return value() if callable(value) else value
    if hasattr(env_step, "get_obs"):
        return env_step.get_obs()
    raise ValueError("env_step must expose current observation")


def make_optimizer(cfg: Any) -> optax.GradientTransformation:
    """Return the PPO optimizer."""

    return optax.adam(cfg.ppo.lr)


def init_state(agent: Any, agent_params: Any, cfg: Any, rng: Any, num_envs: int) -> ContinualState:
    """Build the initial mutable continual-training state."""

    cfg = cfg or TFNSConfig()
    tx = make_optimizer(cfg)
    modules = build_protected_modules(agent_params, agent.model_config)
    bases = {name: empty_basis(module.d_aug) for name, module in modules.items()}

    rng = jnp.asarray(rng if rng is not None else jax.random.PRNGKey(0))
    rng, pred_key = jax.random.split(rng)
    predictor = ReturnPredictor(act_dim=int(agent.model_config.act_dim))
    predictor_tx = make_predictor_optimizer(_cfg_section(cfg, "credit", None))
    feature_dim = int(
        _cfg_value(_cfg_section(cfg, "model", None), "dense_dim", agent.model_config.dense_dim)
    )
    features = jnp.zeros((1, int(num_envs), feature_dim), dtype=jnp.float32)
    actions = jnp.zeros((1, int(num_envs)), dtype=jnp.int32)
    rewards = jnp.zeros((1, int(num_envs)), dtype=jnp.float32)
    resets = jnp.ones((1, int(num_envs)), dtype=bool)
    h0 = predictor.init_hidden(int(num_envs))
    predictor_params = predictor.init(pred_key, features, actions, rewards, resets, h0)["params"]

    return ContinualState(
        params=agent_params,
        opt_state=tx.init(agent_params),
        ema_params=agent_params,
        bases=bases,
        memory=SequenceMemoryBank(_cfg_section(cfg, "memory", None)),
        predictor_params=predictor_params,
        predictor_opt_state=predictor_tx.init(predictor_params),
        detector_state=PageHinkleyDetector(_cfg_section(cfg, "detect", None)).init(),
        adapter_dormant=jnp.ones((int(agent.adapter_config.num_adapters),), dtype=bool),
        robust_stats={},
        protected_clusters=[],
        rng=rng,
        block_index=0,
        skills={},
        rollout_carry=None,
    )


def run_blocks(
    state: ContinualState,
    agent: Any,
    tx: optax.GradientTransformation,
    env_step: Any,
    cfg: Any,
    n_blocks: int,
    *,
    sentinel_clusters: Sequence[Any] | None = None,
    constraint_clusters: Sequence[Any] | None = None,
    enable_shaping: bool = False,
    progress_frac: float | None = None,
    steps_done: int | None = None,
    steps_per_game: int | None = None,
) -> tuple[ContinualState, list[dict[str, Any]]]:
    """Run ``train_block`` repeatedly, threading state and rollout carry."""

    telemetry: list[dict[str, Any]] = []
    ppo_cfg = _cfg_section(cfg, "ppo", PPOConfig())
    block_env_steps = int(_cfg_value(ppo_cfg, "num_envs", PPOConfig.num_envs)) * int(
        _cfg_value(ppo_cfg, "rollout_len", PPOConfig.rollout_len)
    )
    completed_steps = None if steps_done is None else int(steps_done)
    total_steps_target = None if steps_per_game is None else int(steps_per_game)

    for _ in range(int(n_blocks)):
        block_progress_frac = progress_frac
        if block_progress_frac is None and completed_steps is not None and total_steps_target is not None:
            block_progress_frac = float(completed_steps) / float(max(1, total_steps_target))
        block_sentinels = (
            list(state.protected_clusters)
            if sentinel_clusters is None
            else list(sentinel_clusters)
        )
        state, block_info = train_block(
            state,
            agent,
            tx,
            env_step,
            cfg,
            sentinel_clusters=block_sentinels,
            constraint_clusters=constraint_clusters,
            enable_shaping=enable_shaping,
            progress_frac=block_progress_frac,
            steps_done=completed_steps,
            total_steps_target=total_steps_target,
        )
        telemetry.append(block_info)
        if completed_steps is not None:
            completed_steps += block_env_steps
    return state, telemetry


def _recent_important_records(state: ContinualState, cfg: Any) -> list[Any]:
    memory = getattr(state, "memory", None)
    if memory is None or not hasattr(memory, "records"):
        return []
    records = list(memory.records())
    if not records:
        return []

    consolidate_cfg = _cfg_section(cfg, "consolidate", None)
    count = int(_cfg_value(consolidate_cfg, "candidate_records", 16))
    recent_window = int(_cfg_value(consolidate_cfg, "candidate_recent_window", max(count * 4, count)))
    recent = records[-max(count, recent_window) :]
    ranked = sorted(
        recent,
        key=lambda rec: (
            float(getattr(rec, "seq_importance", 0.0)),
            int(getattr(rec, "episode_id", 0)),
            int(getattr(rec, "chunk_index", 0)),
        ),
        reverse=True,
    )
    return ranked[: max(1, min(count, len(ranked)))]


def consolidate_skill(
    state: ContinualState,
    agent: Any,
    tx: optax.GradientTransformation,
    eval_fn: Any,
    candidate_records: Sequence[Any] | None,
    cfg: Any,
    *,
    S_random: float,
    S_single: float,
    score_windows: Any,
    learned_game_keys: Any,
) -> tuple[ContinualState, bool, dict[str, Any]]:
    """Thin consolidation wrapper with candidate selection and rejection feedback."""

    records = (
        _recent_important_records(state, cfg)
        if candidate_records is None
        else list(candidate_records)
    )
    if not records:
        return state, False, {"reason": "no_candidate_records"}

    state, accepted, report = consolidate(
        state,
        agent,
        tx,
        eval_fn,
        records,
        cfg,
        S_random=S_random,
        S_single=S_single,
        score_windows=score_windows,
        learned_game_keys=learned_game_keys,
    )
    if not accepted:
        state = apply_rejection_feedback(state, report, cfg)
    return state, accepted, report


def evaluate_skill(agent: Any, params: Any, env_step_eval: Any, n_episodes: int) -> float:
    """Evaluate a policy with greedy actions and no task or memory inputs."""

    if hasattr(env_step_eval, "reset"):
        obs = env_step_eval.reset()
        if obs is None:
            obs = _current_obs(env_step_eval)
    else:
        obs = _current_obs(env_step_eval)

    obs = jnp.asarray(obs)
    num_envs = int(obs.shape[0])
    hidden = agent.init_hidden(num_envs, dtype=jnp.float32)
    prev_action = jnp.zeros((num_envs,), dtype=jnp.int32)
    prev_reward = jnp.zeros((num_envs,), dtype=jnp.float32)
    prev_reset = jnp.ones((num_envs,), dtype=bool)
    episode_returns = np.zeros((num_envs,), dtype=np.float32)
    completed: list[float] = []

    max_steps = int(n_episodes) * int(getattr(env_step_eval, "episode_length", 1024)) * 4
    max_steps = max(max_steps, int(n_episodes) + num_envs)
    steps = 0
    while len(completed) < int(n_episodes) and steps < max_steps:
        out = agent.apply(
            {"params": params},
            obs,
            prev_action,
            prev_reward,
            prev_reset,
            hidden,
        )
        action = np.asarray(jax.device_get(jnp.argmax(out.logits, axis=-1)), dtype=np.int32)
        next_obs, reward, ppo_done, reset, _ = env_step_eval(action)
        reward_np = np.asarray(reward, dtype=np.float32)
        done_np = np.asarray(ppo_done, dtype=np.bool_)
        episode_returns += reward_np
        for env_index in np.flatnonzero(done_np):
            completed.append(float(episode_returns[int(env_index)]))
            episode_returns[int(env_index)] = 0.0
            if len(completed) >= int(n_episodes):
                break

        obs = jnp.asarray(next_obs)
        hidden = out.h_next.astype(jnp.float32)
        prev_action = jnp.asarray(action, dtype=jnp.int32)
        prev_reward = jnp.asarray(reward_np, dtype=jnp.float32)
        prev_reset = jnp.asarray(reset, dtype=bool)
        steps += 1

    if not completed:
        return 0.0
    return float(np.mean(np.asarray(completed[: int(n_episodes)], dtype=np.float32)))


__all__ = [
    "consolidate_skill",
    "evaluate_skill",
    "init_state",
    "make_optimizer",
    "run_blocks",
]
