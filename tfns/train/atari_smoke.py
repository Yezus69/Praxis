"""Two-game real Atari smoke driver for the task-free continual path."""

from __future__ import annotations

import argparse
import json
import time
from collections.abc import Mapping
from datetime import datetime
from functools import partial
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tfns.config import (
    AdapterConfig,
    AuxConfig,
    BehaviorConfig,
    ConsolidateConfig,
    MemoryConfig,
    ModelConfig,
    PPOConfig,
    ProtectConfig,
    ReplayConfig,
    TFNSConfig,
)
from tfns.envs import ACT_DIM, FIVE_GAMES
from tfns.model.agent import RecurrentAgent
from tfns.train.atari_env import make_atari_env_step
from tfns.train.evaluate import DEFAULT_EVAL_MAX_STEPS, evaluate_game, make_closed_loop_eval_fn
from tfns.train.loop import consolidate_skill, init_state, make_optimizer, run_blocks


def _divisor_at_most(value: int, target: int) -> int:
    value = max(1, int(value))
    for candidate in range(min(value, int(target)), 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _smoke_cfg(num_envs: int, rollout_len: int) -> TFNSConfig:
    seq_chunk = _divisor_at_most(rollout_len, 16)
    burn_in = min(4, max(0, seq_chunk // 4))
    return TFNSConfig(
        model=ModelConfig(
            act_dim=int(ACT_DIM),
            conv_channels=(16, 32, 32),
            dense_dim=128,
            action_embed_dim=16,
            gru_hidden=128,
            key_dim=128,
            ema_decay=0.90,
        ),
        adapter=AdapterConfig(num_adapters=4, rank=8, top_k=1),
        aux=AuxConfig(aux_coef=0.02),
        ppo=PPOConfig(
            num_envs=int(num_envs),
            rollout_len=int(rollout_len),
            gamma=0.99,
            gae_lambda=0.95,
            ent_coef=0.01,
            vf_coef=0.5,
            max_grad_norm=0.5,
            lr=2.5e-4,
            update_epochs=1,
            seq_chunk=seq_chunk,
        ),
        replay=ReplayConfig(
            seq_len=max(4, seq_chunk),
            burn_in=burn_in,
            protected_region=max(1, max(4, seq_chunk) - burn_in),
            replay_frac_start=0.25,
            batch_size=max(1, min(4, int(num_envs))),
        ),
        memory=MemoryConfig(byte_budget=1 << 28, min_per_cluster=0, max_clusters=32),
        behavior=BehaviorConfig(kl_tol=0.25, value_tol=10.0, key_cos_tol=0.50),
        protect=ProtectConfig(residual_energy=0.95, max_rank_frac=0.80),
        consolidate=ConsolidateConfig(
            learned_threshold=0.10,
            stable_windows=2,
            retention_accept=0.10,
            slow_replay_steps=1,
            slow_replay_max_update_norm=0.01,
        ),
    )


def _init_agent_state(cfg: TFNSConfig, env_step: Any, seed: int):
    agent = RecurrentAgent(model_config=cfg.model, adapter_config=cfg.adapter)
    obs = jnp.asarray(env_step.obs)
    prev_action = jnp.zeros((cfg.ppo.num_envs,), dtype=jnp.int32)
    prev_reward = jnp.zeros((cfg.ppo.num_envs,), dtype=jnp.float32)
    reset = jnp.ones((cfg.ppo.num_envs,), dtype=bool)
    hidden = agent.init_hidden(cfg.ppo.num_envs, dtype=jnp.float32)
    params = agent.init(
        jax.random.PRNGKey(int(seed)),
        obs,
        prev_action,
        prev_reward,
        reset,
        hidden,
    )["params"]
    tx = make_optimizer(cfg)
    state = init_state(agent, params, cfg, jax.random.PRNGKey(int(seed) + 1), cfg.ppo.num_envs)
    return agent, tx, state


def _basis_ranks(state: Any) -> dict[str, int]:
    return {
        str(name): int(np.asarray(basis).shape[1])
        for name, basis in sorted((state.bases or {}).items(), key=lambda item: item[0])
    }


def _projection_summary(telemetry: list[dict[str, Any]]) -> dict[str, Any]:
    rows = [
        {
            "block_index": int(row.get("block_index", idx)),
            "candidate_delta_norm": float(row.get("candidate_delta_norm", 0.0)),
            "projected_delta_norm": float(row.get("projected_delta_norm", 0.0)),
            "applied_norm": float(row.get("applied_norm", 0.0)),
        }
        for idx, row in enumerate(telemetry)
    ]
    active = any(
        abs(row["candidate_delta_norm"] - row["projected_delta_norm"]) > 1.0e-8
        for row in rows
    )
    return {"projection_active": bool(active), "blocks": rows}


def _memory_report(state: Any) -> dict[str, int]:
    memory = getattr(state, "memory", None)
    return {
        "bytes": int(memory.bytes_used()) if hasattr(memory, "bytes_used") else 0,
        "count": int(len(memory)) if memory is not None else 0,
        "clusters": int(len(memory.clusters())) if hasattr(memory, "clusters") else 0,
    }


@partial(jax.jit, static_argnames=("agent",))
def _greedy_step(
    agent_vars,
    obs,
    prev_action,
    prev_reward,
    reset,
    hidden,
    adapter_dormant,
    *,
    agent,
):
    out = agent.apply(
        agent_vars,
        obs,
        prev_action,
        prev_reward,
        reset,
        hidden,
        adapter_dormant=adapter_dormant,
    )
    return jnp.argmax(out.logits, axis=-1).astype(jnp.int32), out.h_next.astype(jnp.float32)


def _action_trace(
    agent: Any,
    params: Any,
    game: str,
    *,
    num_envs: int,
    steps: int,
    seed: int,
    memory_payload: Any = None,
    adapter_dormant: Any | None = None,
) -> np.ndarray:
    _ = memory_payload
    env_step, handle = make_atari_env_step(game, int(num_envs), int(seed), training=False)
    try:
        obs = jnp.asarray(env_step.obs)
        hidden = agent.init_hidden(int(num_envs), dtype=jnp.float32)
        prev_action = jnp.zeros((int(num_envs),), dtype=jnp.int32)
        prev_reward = jnp.zeros((int(num_envs),), dtype=jnp.float32)
        prev_reset = jnp.ones((int(num_envs),), dtype=bool)
        dormant = (
            jnp.asarray(adapter_dormant, dtype=bool)
            if adapter_dormant is not None
            else jnp.ones((int(agent.adapter_config.num_adapters),), dtype=bool)
        )
        actions = []
        for _step in range(int(steps)):
            action, h_next = _greedy_step(
                {"params": params},
                obs,
                prev_action,
                prev_reward,
                prev_reset,
                hidden,
                dormant,
                agent=agent,
            )
            action_np = np.asarray(jax.device_get(action), dtype=np.int32)
            next_obs, reward_clipped, _ppo_done, reset, extra = env_step(action_np)
            exec_action = (
                np.asarray(extra.get("exec_action"), dtype=np.int32)
                if isinstance(extra, Mapping) and extra.get("exec_action") is not None
                else action_np
            )
            actions.append(exec_action)
            reset_jnp = jnp.asarray(reset, dtype=bool)
            hidden = jnp.where(reset_jnp[:, None], jnp.zeros_like(h_next), h_next)
            obs = jnp.asarray(next_obs)
            prev_action = jnp.asarray(exec_action, dtype=jnp.int32)
            prev_reward = jnp.asarray(reward_clipped, dtype=jnp.float32)
            prev_reset = reset_jnp
        return np.stack(actions, axis=0)
    finally:
        handle.close()


def _jsonable(value: Any) -> Any:
    if isinstance(value, dict):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, int, float, bool)) or value is None:
        return value
    if isinstance(value, np.generic):
        return value.item()
    try:
        arr = np.asarray(jax.device_get(value))
    except Exception:
        return str(value)
    if arr.ndim == 0:
        return arr.item()
    return arr.tolist()


def _calibrated_meta(game: str, score: float) -> dict[str, Any]:
    margin = max(1.0, abs(float(score)) * 0.10 + 1.0)
    return {
        "game": str(game),
        "S_random": float(score) - margin,
        "S_single": float(score) + 0.05 * margin,
        "S_best": float(score) + 0.01 * margin,
        "current": True,
    }


def main(argv: list[str] | None = None) -> dict[str, Any]:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--games", nargs="+", default=FIVE_GAMES[:2])
    parser.add_argument("--num-envs", type=int, default=16)
    parser.add_argument("--blocks-per-game", type=int, default=2)
    parser.add_argument("--rollout-len", type=int, default=32)
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--eval-max-steps", type=int, default=DEFAULT_EVAL_MAX_STEPS)
    parser.add_argument("--trace-steps", type=int, default=32)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--output-dir", default="tfns_runs/atari_smoke")
    args = parser.parse_args(argv)
    if int(args.eval_max_steps) < 0:
        raise ValueError("--eval-max-steps must be non-negative")

    games = list(args.games)[:2]
    if len(games) != 2:
        raise ValueError("atari smoke requires exactly two games")

    cfg = _smoke_cfg(args.num_envs, args.rollout_len)
    env_a, handle_a = make_atari_env_step(games[0], cfg.ppo.num_envs, args.seed, training=True)
    try:
        agent, tx, state = _init_agent_state(cfg, env_a, args.seed)
        base_ranks_initial = _basis_ranks(state)

        t0 = time.perf_counter()
        state, telemetry_a = run_blocks(
            state,
            agent,
            tx,
            env_a,
            cfg,
            int(args.blocks_per_game),
        )
        elapsed_a = max(1.0e-8, time.perf_counter() - t0)
    finally:
        handle_a.close()

    score_a = evaluate_game(
        agent,
        state.params,
        games[0],
        num_envs=max(1, min(cfg.ppo.num_envs, args.num_envs)),
        n_episodes=max(1, int(args.eval_episodes)),
        seed=int(args.seed) + 2000,
        adapter_dormant=state.adapter_dormant,
        max_steps=int(args.eval_max_steps),
    )["mean"]
    learned_meta = {games[0]: _calibrated_meta(games[0], float(score_a))}
    eval_fn = make_closed_loop_eval_fn(
        learned_meta,
        agent,
        num_envs=max(1, min(cfg.ppo.num_envs, args.num_envs)),
        n_episodes=max(1, int(args.eval_episodes)),
        seed=int(args.seed) + 3000,
        max_steps=int(args.eval_max_steps),
    )
    score_windows = [float(score_a), float(score_a)]
    state, consolidation_accepted, consolidation_report = consolidate_skill(
        state,
        agent,
        tx,
        eval_fn,
        candidate_records=None,
        cfg=cfg,
        S_random=learned_meta[games[0]]["S_random"],
        S_single=learned_meta[games[0]]["S_single"],
        score_windows=score_windows,
        learned_game_keys=learned_meta,
    )
    base_ranks_after_consolidation = _basis_ranks(state)

    state.rollout_carry = None
    env_b, handle_b = make_atari_env_step(games[1], cfg.ppo.num_envs, args.seed + 100, training=True)
    try:
        t1 = time.perf_counter()
        state, telemetry_b = run_blocks(
            state,
            agent,
            tx,
            env_b,
            cfg,
            int(args.blocks_per_game),
            sentinel_clusters=state.protected_clusters,
        )
        elapsed_b = max(1.0e-8, time.perf_counter() - t1)
    finally:
        handle_b.close()

    evals = {
        game: evaluate_game(
            agent,
            state.params,
            game,
            num_envs=max(1, min(cfg.ppo.num_envs, args.num_envs)),
            n_episodes=max(1, int(args.eval_episodes)),
            seed=int(args.seed) + 4000 + index,
            adapter_dormant=state.adapter_dormant,
            max_steps=int(args.eval_max_steps),
        )
        for index, game in enumerate(games)
    }

    trace_envs = max(1, min(4, cfg.ppo.num_envs))
    trace_normal = _action_trace(
        agent,
        state.params,
        games[0],
        num_envs=trace_envs,
        steps=int(args.trace_steps),
        seed=int(args.seed) + 5000,
        memory_payload=state.memory,
        adapter_dormant=state.adapter_dormant,
    )
    trace_disabled = _action_trace(
        agent,
        state.params,
        games[0],
        num_envs=trace_envs,
        steps=int(args.trace_steps),
        seed=int(args.seed) + 5000,
        memory_payload=None,
        adapter_dormant=state.adapter_dormant,
    )
    trace_corrupt = _action_trace(
        agent,
        state.params,
        games[0],
        num_envs=trace_envs,
        steps=int(args.trace_steps),
        seed=int(args.seed) + 5000,
        memory_payload={"records": "shuffled", "teacher_policy": "corrupted"},
        adapter_dormant=state.adapter_dormant,
    )
    trace_identical = bool(
        np.array_equal(trace_normal, trace_disabled)
        and np.array_equal(trace_normal, trace_corrupt)
    )

    env_steps_per_game = int(args.blocks_per_game) * int(cfg.ppo.num_envs) * int(cfg.ppo.rollout_len)
    report = {
        "games": games,
        "seed": int(args.seed),
        "config": {
            "num_envs": int(cfg.ppo.num_envs),
            "rollout_len": int(cfg.ppo.rollout_len),
            "seq_chunk": int(cfg.ppo.seq_chunk),
            "blocks_per_game": int(args.blocks_per_game),
            "eval_episodes": int(args.eval_episodes),
            "eval_max_steps": int(args.eval_max_steps),
        },
        "throughput": {
            games[0]: {
                "env_steps": env_steps_per_game,
                "seconds": elapsed_a,
                "sps": env_steps_per_game / elapsed_a,
            },
            games[1]: {
                "env_steps": env_steps_per_game,
                "seconds": elapsed_b,
                "sps": env_steps_per_game / elapsed_b,
            },
        },
        "basis_ranks_initial": base_ranks_initial,
        "basis_ranks_after_consolidation": base_ranks_after_consolidation,
        "basis_grew": any(
            base_ranks_after_consolidation.get(name, 0) > base_ranks_initial.get(name, 0)
            for name in base_ranks_after_consolidation
        ),
        "projection": _projection_summary(telemetry_b),
        "consolidation": {
            "accepted": bool(consolidation_accepted),
            "report": consolidation_report,
        },
        "memory": _memory_report(state),
        "eval": evals,
        "memory_disabled_action_trace_identical": trace_identical,
        "memory_trace_shape": list(trace_normal.shape),
        "active_adapter_count": int(np.sum(~np.asarray(state.adapter_dormant, dtype=np.bool_))),
    }

    out_dir = Path(args.output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"atari_smoke_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    out_path.write_text(json.dumps(_jsonable(report), indent=2, sort_keys=True), encoding="utf-8")

    summary = {
        "games": games,
        "basis_grew": report["basis_grew"],
        "projection_active": report["projection"]["projection_active"],
        "consolidation_accepted": bool(consolidation_accepted),
        "memory": report["memory"],
        "memory_disabled_action_trace_identical": trace_identical,
        "sps": {game: report["throughput"][game]["sps"] for game in games},
        "report": str(out_path),
    }
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return report


if __name__ == "__main__":
    main()
