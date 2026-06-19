"""Five-game sequential curriculum driver for TFNS Atari experiments."""

from __future__ import annotations

import argparse
import dataclasses
import json
import math
import time
from collections import deque
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax

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
from tfns.consolidate import certify
from tfns.envs import ACT_DIM, EIGHT_GAMES, FIVE_GAMES
from tfns.memory.bank import SequenceMemoryBank
from tfns.model.agent import RecurrentAgent
from tfns.train import atari_env, evaluate, loop


_PROGRESS_KEYS = (
    "loss",
    "pg_loss",
    "v_loss",
    "entropy",
    "approx_kl",
    "aux_loss",
    "replay_tube_total",
    "raw_grad_norm",
    "projected_delta_norm",
    "applied_norm",
    "predictor_val_error",
    "detector_changed",
    "memory_admitted",
)


class _ReturnTrackingEnvStep:
    """Delegate env calls while retaining completed true episode returns."""

    def __init__(self, env_step: Any):
        self.env_step = env_step
        self.block_returns: list[float] = []
        self.all_returns: list[float] = []
        self.recent_returns: deque[float] = deque(maxlen=128)

    @property
    def obs(self) -> Any:
        return self.env_step.obs

    @obs.setter
    def obs(self, value: Any) -> None:
        self.env_step.obs = value

    @property
    def current_obs(self) -> Any:
        return self.env_step.current_obs

    @current_obs.setter
    def current_obs(self, value: Any) -> None:
        self.env_step.current_obs = value

    def get_obs(self) -> Any:
        return self.env_step.get_obs()

    def reset(self) -> Any:
        return self.env_step.reset()

    def begin_block(self) -> None:
        self.block_returns.clear()

    def recent_score(self) -> float:
        if not self.recent_returns:
            return 0.0
        return float(np.mean(np.asarray(self.recent_returns, dtype=np.float64)))

    def __call__(self, action: Any) -> Any:
        result = self.env_step(action)
        extra = result[4] if isinstance(result, tuple) and len(result) >= 5 else {}
        returns = extra.get("episode_returns", ()) if isinstance(extra, Mapping) else ()
        for value in returns:
            score = float(value)
            self.block_returns.append(score)
            self.all_returns.append(score)
            self.recent_returns.append(score)
        return result


def _jsonable(value: Any) -> Any:
    if isinstance(value, Mapping):
        return {str(key): _jsonable(item) for key, item in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return _jsonable(dataclasses.asdict(value))
    if isinstance(value, (list, tuple)):
        return [_jsonable(item) for item in value]
    if isinstance(value, (str, bool)) or value is None:
        return value
    if isinstance(value, (int, np.integer)):
        return int(value)
    if isinstance(value, (float, np.floating)):
        out = float(value)
        return out if math.isfinite(out) else None
    try:
        arr = np.asarray(jax.device_get(value))
    except Exception:
        return str(value)
    if arr.ndim == 0:
        item = arr.item()
        return _jsonable(item)
    return arr.tolist()


def _persist_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(_jsonable(payload), indent=2, sort_keys=True), encoding="utf-8")
    tmp.replace(path)


def _read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        return {}
    return json.loads(path.read_text(encoding="utf-8"))


def _append_progress(out_dir: Path, row: Mapping[str, Any]) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    with (out_dir / "progress.jsonl").open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(_jsonable(row), sort_keys=True) + "\n")


def _divisor_at_most(value: int, target: int) -> int:
    value = max(1, int(value))
    target = max(1, min(value, int(target)))
    for candidate in range(target, 0, -1):
        if value % candidate == 0:
            return candidate
    return 1


def _with_rollout_len(cfg: TFNSConfig, rollout_len: int) -> TFNSConfig:
    rollout_len = max(1, int(rollout_len))
    seq_chunk = _divisor_at_most(rollout_len, min(int(cfg.ppo.seq_chunk), rollout_len))
    burn_in = min(int(cfg.replay.burn_in), max(0, seq_chunk - 1))
    replay_len = max(1, min(int(cfg.replay.seq_len), rollout_len))
    protected_region = max(1, min(int(cfg.replay.protected_region), replay_len - burn_in))
    return dataclasses.replace(
        cfg,
        ppo=dataclasses.replace(cfg.ppo, rollout_len=rollout_len, seq_chunk=seq_chunk),
        replay=dataclasses.replace(
            cfg.replay,
            seq_len=replay_len,
            burn_in=burn_in,
            protected_region=protected_region,
        ),
    )


def _base_cfg(args: argparse.Namespace) -> TFNSConfig:
    base = TFNSConfig()
    num_minibatches = int(getattr(args, "num_minibatches", base.ppo.num_minibatches))
    if args.smoke:
        num_envs = min(int(args.num_envs), 4)
        rollout_len = min(int(args.rollout_len), 32)
        seq_chunk = _divisor_at_most(rollout_len, min(int(base.ppo.seq_chunk), rollout_len))
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
                num_envs=num_envs,
                rollout_len=rollout_len,
                lr=2.5e-4,
                update_epochs=int(args.update_epochs),
                num_minibatches=num_minibatches,
                seq_chunk=seq_chunk,
            ),
            replay=ReplayConfig(
                seq_len=max(1, seq_chunk),
                burn_in=burn_in,
                protected_region=max(1, max(1, seq_chunk) - burn_in),
                replay_frac_start=0.25,
                batch_size=max(1, min(4, num_envs)),
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

    seq_chunk = _divisor_at_most(int(args.rollout_len), min(int(base.ppo.seq_chunk), int(args.rollout_len)))
    replay_len = max(1, min(int(base.replay.seq_len), int(args.rollout_len)))
    burn_in = min(int(base.replay.burn_in), max(0, replay_len - 1))
    return dataclasses.replace(
        base,
        model=dataclasses.replace(base.model, act_dim=int(ACT_DIM)),
        ppo=dataclasses.replace(
            base.ppo,
            num_envs=int(args.num_envs),
            rollout_len=int(args.rollout_len),
            update_epochs=int(args.update_epochs),
            num_minibatches=num_minibatches,
            seq_chunk=seq_chunk,
        ),
        replay=dataclasses.replace(
            base.replay,
            seq_len=replay_len,
            burn_in=burn_in,
            protected_region=max(1, replay_len - burn_in),
        ),
    )


def _parse_games(raw: Sequence[str] | None, *, smoke: bool) -> list[str]:
    if not raw:
        games = list(FIVE_GAMES)
    elif len(raw) == 1 and raw[0].upper() == "FIVE_GAMES":
        games = list(FIVE_GAMES)
    elif len(raw) == 1 and raw[0].upper() == "EIGHT_GAMES":
        games = list(EIGHT_GAMES)
    elif len(raw) == 1 and "," in raw[0]:
        games = [part.strip() for part in raw[0].split(",") if part.strip()]
    else:
        games = [str(game) for game in raw]
    return games[:2] if smoke else games


def _eval_num_envs(cfg: TFNSConfig, eval_episodes: int) -> int:
    return max(1, min(int(cfg.ppo.num_envs), max(1, int(eval_episodes))))


def _init_agent_state(cfg: TFNSConfig, game: str, seed: int) -> tuple[Any, optax.GradientTransformation, Any]:
    env_step, handle = atari_env.make_atari_env_step(game, int(cfg.ppo.num_envs), int(seed), training=True)
    try:
        agent = RecurrentAgent(model_config=cfg.model, adapter_config=cfg.adapter)
        obs = jnp.asarray(env_step.obs)
        prev_action = jnp.zeros((int(cfg.ppo.num_envs),), dtype=jnp.int32)
        prev_reward = jnp.zeros((int(cfg.ppo.num_envs),), dtype=jnp.float32)
        reset = jnp.ones((int(cfg.ppo.num_envs),), dtype=bool)
        hidden = agent.init_hidden(int(cfg.ppo.num_envs), dtype=jnp.float32)
        params = agent.init(
            jax.random.PRNGKey(int(seed)),
            obs,
            prev_action,
            prev_reward,
            reset,
            hidden,
        )["params"]
        tx = loop.make_optimizer(cfg)
        state = loop.init_state(
            agent,
            params,
            cfg,
            jax.random.PRNGKey(int(seed) + 1),
            int(cfg.ppo.num_envs),
        )
        return agent, tx, state
    finally:
        handle.close()


def _memory_report(state: Any) -> dict[str, int]:
    memory = getattr(state, "memory", None)
    return {
        "bytes": int(memory.bytes_used()) if hasattr(memory, "bytes_used") else 0,
        "count": int(len(memory)) if memory is not None else 0,
        "clusters": int(len(memory.clusters())) if hasattr(memory, "clusters") else 0,
    }


def _basis_report(state: Any) -> dict[str, dict[str, float | int]]:
    out: dict[str, dict[str, float | int]] = {}
    for name, basis in sorted((getattr(state, "bases", None) or {}).items(), key=lambda item: str(item[0])):
        arr = np.asarray(basis)
        d_aug = int(arr.shape[0]) if arr.ndim >= 1 else 0
        rank = int(arr.shape[1]) if arr.ndim >= 2 else 0
        out[str(name)] = {
            "rank": rank,
            "d_aug": d_aug,
            "free_rank_fraction": float(1.0 - rank / max(1, d_aug)),
        }
    return out


def _active_adapter_count(state: Any) -> int:
    dormant = getattr(state, "adapter_dormant", None)
    if dormant is None:
        return 0
    return int(np.sum(~np.asarray(dormant, dtype=np.bool_)))


def _clear_plain_state(state: Any, cfg: TFNSConfig) -> Any:
    state.bases = {}
    state.protected_clusters = []
    state.memory = SequenceMemoryBank(cfg.memory)
    state.robust_stats = {}
    return state


def _compact_block(info: Mapping[str, Any]) -> dict[str, Any]:
    return {key: _jsonable(info[key]) for key in _PROGRESS_KEYS if key in info}


def random_scores(
    games: Sequence[str],
    eval_episodes: int,
    seed: int,
    *,
    num_envs: int = 1,
    out_dir: str | Path | None = None,
    max_steps: int | None = None,
) -> dict[str, float]:
    """Return cached uniform-random Atari scores and persist after each miss."""

    max_steps_value = evaluate.DEFAULT_EVAL_MAX_STEPS if max_steps is None else int(max_steps)
    if max_steps_value < 0:
        raise ValueError("max_steps must be non-negative")
    path = Path(out_dir) / f"random_seed{int(seed)}.json" if out_dir is not None else None
    payload = _read_json(path) if path is not None else {}
    cache_matches = (
        int(payload.get("seed", seed)) == int(seed)
        and int(payload.get("eval_episodes", eval_episodes)) == int(eval_episodes)
        and int(payload.get("num_envs", num_envs)) == int(num_envs)
        and int(payload.get("max_steps", max_steps_value)) == max_steps_value
    )
    scores = (
        {str(game): float(score) for game, score in (payload.get("scores", {}) or {}).items()}
        if cache_matches
        else {}
    )

    for index, game in enumerate(games):
        if str(game) in scores:
            continue
        score_info = evaluate.random_score(
            str(game),
            num_envs=int(num_envs),
            n_episodes=int(eval_episodes),
            seed=int(seed) + 1009 * index,
            max_steps=max_steps_value,
        )
        scores[str(game)] = float(score_info["mean"])
        if path is not None:
            _persist_json(
                path,
                {
                    "seed": int(seed),
                    "eval_episodes": int(eval_episodes),
                    "num_envs": int(num_envs),
                    "max_steps": max_steps_value,
                    "scores": scores,
                },
            )

    return {str(game): float(scores[str(game)]) for game in games}


def train_one_game(
    state: Any,
    agent: Any,
    tx: optax.GradientTransformation,
    game: str,
    cfg: TFNSConfig,
    steps_per_game: int,
    seed: int,
    *,
    protect: bool,
    learned_meta: Mapping[str, Any] | None,
    out_dir: str | Path,
) -> tuple[Any, dict[str, Any]]:
    """Train one Atari game for a bounded step budget and append JSONL progress."""

    steps_per_game = int(steps_per_game)
    if steps_per_game <= 0:
        raise ValueError("steps_per_game must be positive")

    out_path = Path(out_dir)
    env_step, handle = atari_env.make_atari_env_step(
        str(game),
        int(cfg.ppo.num_envs),
        int(seed),
        training=True,
    )
    tracked_env = _ReturnTrackingEnvStep(env_step)
    state.rollout_carry = None
    steps_done = 0
    blocks: list[dict[str, Any]] = []

    try:
        while steps_done < steps_per_game:
            remaining = int(steps_per_game) - int(steps_done)
            rollout_len = min(int(cfg.ppo.rollout_len), remaining // int(cfg.ppo.num_envs))
            if rollout_len <= 0:
                break

            block_cfg = cfg if rollout_len == int(cfg.ppo.rollout_len) else _with_rollout_len(cfg, rollout_len)
            if not protect:
                state = _clear_plain_state(state, block_cfg)

            tracked_env.begin_block()
            start = time.perf_counter()
            state, telemetry = loop.run_blocks(
                state,
                agent,
                tx,
                tracked_env,
                block_cfg,
                1,
                sentinel_clusters=None if protect else [],
                constraint_clusters=None if protect else [],
            )
            elapsed = max(1.0e-8, time.perf_counter() - start)
            steps_this = int(block_cfg.ppo.num_envs) * int(block_cfg.ppo.rollout_len)
            steps_done += steps_this

            if not protect:
                state = _clear_plain_state(state, block_cfg)

            block_info = telemetry[-1] if telemetry else {}
            mem = _memory_report(state)
            progress = {
                "time": time.time(),
                "game": str(game),
                "protect": bool(protect),
                "steps_done": int(steps_done),
                "target_steps": int(steps_per_game),
                "block_env_steps": int(steps_this),
                "recent_score": float(tracked_env.recent_score()),
                "block_completed_episodes": int(len(tracked_env.block_returns)),
                "env_SPS": float(steps_this / elapsed),
                "memory_count": int(mem["count"]),
                "mem_count": int(mem["count"]),
                "memory_bytes": int(mem["bytes"]),
                "memory_clusters": int(mem["clusters"]),
                "accept_count": int(block_info.get("accept_count", 0)),
                "reject_count": int(block_info.get("reject_count", 0)),
                "consolidation_status": "not_run",
                "protected_cluster_count": int(len(getattr(state, "protected_clusters", []) or [])),
                "learned_game_count": int(len(learned_meta or {})),
                "block_index": int(block_info.get("block_index", getattr(state, "block_index", 0))),
            }
            _append_progress(out_path, progress)
            blocks.append({**progress, "telemetry": _compact_block(block_info)})

    finally:
        handle.close()

    return state, {
        "game": str(game),
        "steps_done": int(steps_done),
        "target_steps": int(steps_per_game),
        "blocks": blocks,
        "budget_unused_steps": int(max(0, steps_per_game - steps_done)),
        "completed_episode_returns": [float(value) for value in tracked_env.all_returns],
    }


def _load_refs(path: str | Path | None) -> dict[str, dict[str, float]]:
    if path is None:
        return {}
    ref_path = Path(path)
    if not ref_path.exists():
        raise FileNotFoundError(f"refs JSON not found: {ref_path}")
    payload = _read_json(ref_path)
    raw = payload.get("refs", payload)
    refs: dict[str, dict[str, float]] = {}
    if not isinstance(raw, Mapping):
        return refs
    for game, row in raw.items():
        if not isinstance(row, Mapping):
            continue
        single = row.get("S_single", row.get("single", row.get("single_score")))
        random = row.get("S_random", row.get("random", row.get("random_score")))
        refs[str(game)] = {}
        if single is not None:
            refs[str(game)]["S_single"] = float(single)
        if random is not None:
            refs[str(game)]["S_random"] = float(random)
    return refs


def _reference_for_game(
    game: str,
    refs: Mapping[str, Mapping[str, float]],
    random_by_game: Mapping[str, float],
) -> tuple[float, float, bool]:
    ref = refs.get(str(game), {})
    random_score = float(ref.get("S_random", random_by_game[str(game)]))
    if "S_single" in ref:
        return random_score, float(ref["S_single"]), True
    return random_score, random_score + 1.0, False


def _progress_value(score: float, S_random: float, S_single: float, has_ref: bool) -> float:
    if has_ref:
        return float(evaluate.normalized_progress(score, S_random, S_single))
    return float(score - S_random)


def _eval_primary_greedy(
    agent: Any,
    state: Any,
    game: str,
    cfg: TFNSConfig,
    *,
    eval_episodes: int,
    seed: int,
    max_steps: int | None,
) -> dict[str, Any]:
    num_envs = _eval_num_envs(cfg, eval_episodes)
    stochastic = evaluate.evaluate_game(
        agent,
        state.params,
        str(game),
        num_envs=num_envs,
        n_episodes=int(eval_episodes),
        seed=int(seed),
        greedy=False,
        adapter_dormant=getattr(state, "adapter_dormant", None),
        max_steps=max_steps,
    )
    greedy = evaluate.evaluate_game(
        agent,
        state.params,
        str(game),
        num_envs=num_envs,
        n_episodes=int(eval_episodes),
        seed=int(seed) + 500_003,
        greedy=True,
        adapter_dormant=getattr(state, "adapter_dormant", None),
        max_steps=max_steps,
    )
    return {
        "stochastic": stochastic,
        "greedy": greedy,
        "score": float(stochastic["mean"]),
    }


def _eval_windows(
    agent: Any,
    state: Any,
    game: str,
    cfg: TFNSConfig,
    *,
    eval_episodes: int,
    seed: int,
    windows: int,
    max_steps: int | None,
) -> list[dict[str, Any]]:
    return [
        _eval_primary_greedy(
            agent,
            state,
            game,
            cfg,
            eval_episodes=eval_episodes,
            seed=int(seed) + 10_003 * index,
            max_steps=max_steps,
        )
        for index in range(max(1, int(windows)))
    ]


def _window_scores(windows: Sequence[Mapping[str, Any]]) -> list[float]:
    return [float(row["stochastic"]["mean"]) for row in windows]


def _mean_window_score(windows: Sequence[Mapping[str, Any]]) -> float:
    scores = _window_scores(windows)
    return float(np.mean(np.asarray(scores, dtype=np.float64))) if scores else 0.0


def _retention_row(
    agent: Any,
    state: Any,
    games: Sequence[str],
    cfg: TFNSConfig,
    random_by_game: Mapping[str, float],
    best_scores: dict[str, float],
    *,
    eval_episodes: int,
    seed: int,
    after_game: str,
    max_steps: int | None,
) -> dict[str, Any]:
    scores: dict[str, float] = {}
    evals: dict[str, Any] = {}
    retention_by_game: dict[str, float] = {}
    forgetting: dict[str, float] = {}

    for index, game in enumerate(games):
        eval_row = _eval_primary_greedy(
            agent,
            state,
            str(game),
            cfg,
            eval_episodes=eval_episodes,
            seed=int(seed) + 20_011 * index,
            max_steps=max_steps,
        )
        score = float(eval_row["score"])
        best_scores[str(game)] = max(float(best_scores.get(str(game), score)), score)
        value = float(evaluate.retention(score, random_by_game[str(game)], best_scores[str(game)]))
        scores[str(game)] = score
        evals[str(game)] = eval_row
        retention_by_game[str(game)] = value
        forgetting[str(game)] = float(1.0 - value)

    return {
        "after_game": str(after_game),
        "games": [str(game) for game in games],
        "scores": scores,
        "eval": evals,
        "retention": retention_by_game,
        "normalized_forgetting": forgetting,
    }


def _retention_summary(matrix: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    final = dict(matrix[-1].get("retention", {})) if matrix else {}
    final_values = [float(value) for value in final.values()]
    row_means = [
        float(np.mean(np.asarray(list(row.get("retention", {}).values()), dtype=np.float64)))
        for row in matrix
        if row.get("retention")
    ]
    worst_game = min(final, key=lambda key: final[key]) if final else None
    return {
        "mean_retention": float(np.mean(np.asarray(final_values, dtype=np.float64))) if final_values else None,
        "worst_game": worst_game,
        "worst_game_retention": float(final[worst_game]) if worst_game is not None else None,
        "per_game_retention": final,
        "normalized_forgetting": {game: float(1.0 - value) for game, value in final.items()},
        "retention_auc": float(np.mean(np.asarray(row_means, dtype=np.float64))) if row_means else None,
    }


def _final_report(
    state: Any,
    matrix: Sequence[Mapping[str, Any]],
    progress_by_game: Mapping[str, Any],
) -> dict[str, Any]:
    return {
        "retention": _retention_summary(matrix),
        "per_game_normalized_progress": dict(progress_by_game),
        "protected_basis": _basis_report(state),
        "memory": _memory_report(state),
        "active_adapter_count": _active_adapter_count(state),
    }


def _result_header(
    mode: str,
    args: argparse.Namespace,
    cfg: TFNSConfig,
    games: Sequence[str],
    random_by_game: Mapping[str, float],
) -> dict[str, Any]:
    return {
        "mode": str(mode),
        "seed": int(args.seed),
        "eval_seed": int(args.eval_seed),
        "smoke": bool(args.smoke),
        "games": [str(game) for game in games],
        "steps_per_game": int(args.steps_per_game),
        "random_scores": dict(random_by_game),
        "config": {
            "num_envs": int(cfg.ppo.num_envs),
            "rollout_len": int(cfg.ppo.rollout_len),
            "update_epochs": int(cfg.ppo.update_epochs),
            "num_minibatches": int(cfg.ppo.num_minibatches),
            "seq_chunk": int(cfg.ppo.seq_chunk),
            "eval_episodes": int(args.eval_episodes),
            "eval_max_steps": int(args.eval_max_steps),
        },
    }


def _run_refs(
    args: argparse.Namespace,
    cfg: TFNSConfig,
    games: Sequence[str],
    out_dir: Path,
    random_by_game: Mapping[str, float],
) -> dict[str, Any]:
    result = _result_header("refs", args, cfg, games, random_by_game)
    result["refs"] = {}
    path = out_dir / f"refs_seed{int(args.seed)}.json"
    _persist_json(path, result)

    for index, game in enumerate(games):
        agent, tx, state = _init_agent_state(cfg, str(game), int(args.seed) + 30_001 * index)
        state, train_info = train_one_game(
            state,
            agent,
            tx,
            str(game),
            cfg,
            int(args.steps_per_game),
            int(args.seed) + 40_009 * index,
            protect=False,
            learned_meta={},
            out_dir=out_dir,
        )
        windows = _eval_windows(
            agent,
            state,
            str(game),
            cfg,
            eval_episodes=int(args.eval_episodes),
            seed=int(args.eval_seed) + 50_021 * index,
            windows=max(1, int(cfg.consolidate.stable_windows)),
            max_steps=int(args.eval_max_steps),
        )
        scores = _window_scores(windows)
        result["refs"][str(game)] = {
            "S_single": float(np.mean(np.asarray(scores, dtype=np.float64))) if scores else 0.0,
            "S_random": float(random_by_game[str(game)]),
            "eval_curve": windows,
            "train": train_info,
        }
        _persist_json(path, result)

    return result


def _learned_meta_row(
    game: str,
    S_random: float,
    S_single: float,
    S_best: float,
    state: Any,
    *,
    current: bool = False,
) -> dict[str, Any]:
    return {
        "game": str(game),
        "S_random": float(S_random),
        "S_single": float(S_single),
        "S_best": float(S_best),
        "current": bool(current),
        "adapter_dormant": getattr(state, "adapter_dormant", None),
    }


def _run_curriculum(
    args: argparse.Namespace,
    cfg: TFNSConfig,
    games: Sequence[str],
    out_dir: Path,
    random_by_game: Mapping[str, float],
    refs: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    cfg = dataclasses.replace(
        cfg,
        consolidate=dataclasses.replace(
            cfg.consolidate,
            learned_threshold=float(args.learned_threshold),
        ),
    )
    result = _result_header("curriculum", args, cfg, games, random_by_game)
    result.update(
        {
            "game_results": {},
            "consolidation": {},
            "failure_to_learn": [],
            "retention_matrix": [],
            "per_game_normalized_progress": {},
        }
    )
    path = out_dir / f"curriculum_seed{int(args.seed)}.json"
    _persist_json(path, result)

    agent, tx, state = _init_agent_state(cfg, str(games[0]), int(args.seed))
    score_windows: dict[str, list[float]] = {str(game): [] for game in games}
    best_scores: dict[str, float] = {}
    certified_meta: dict[str, dict[str, Any]] = {}
    protected_meta: dict[str, dict[str, Any]] = {}

    for index, game in enumerate(games):
        game = str(game)
        state, train_info = train_one_game(
            state,
            agent,
            tx,
            game,
            cfg,
            int(args.steps_per_game),
            int(args.seed) + 100_003 * index,
            protect=True,
            learned_meta=protected_meta,
            out_dir=out_dir,
        )
        eval_windows = _eval_windows(
            agent,
            state,
            game,
            cfg,
            eval_episodes=int(args.eval_episodes),
            seed=int(args.eval_seed) + 110_017 * index,
            windows=max(1, int(cfg.consolidate.stable_windows)),
            max_steps=int(args.eval_max_steps),
        )
        scores = _window_scores(eval_windows)
        score_windows[game].extend(scores)
        score = _mean_window_score(eval_windows)
        S_random, S_single, has_ref = _reference_for_game(game, refs, random_by_game)
        progress = _progress_value(score, S_random, S_single, has_ref)
        learned, learn_info = certify.is_learned(score_windows[game], S_random, S_single, cfg)
        result["per_game_normalized_progress"][game] = {
            "value": float(progress),
            "basis": "single_task_ref" if has_ref else "random_margin",
            "score": float(score),
            "S_random": float(S_random),
            "S_single": float(S_single),
        }

        consolidation = {"ran": False, "accepted": False, "report": None}
        if not learned:
            result["failure_to_learn"].append(
                {"game": game, "learned": learn_info, "score": float(score)}
            )
        else:
            current_best = max(float(best_scores.get(game, score)), float(score))
            certified_meta[game] = _learned_meta_row(
                game,
                S_random,
                S_single,
                current_best,
                state,
                current=False,
            )
            best_scores[game] = current_best
            eval_meta = dict(protected_meta)
            eval_meta[game] = _learned_meta_row(
                game,
                S_random,
                S_single,
                current_best,
                state,
                current=True,
            )
            eval_fn = evaluate.make_closed_loop_eval_fn(
                eval_meta,
                agent,
                num_envs=_eval_num_envs(cfg, int(args.eval_episodes)),
                n_episodes=int(args.eval_episodes),
                seed=int(args.eval_seed) + 120_011 * index,
                max_steps=int(args.eval_max_steps),
            )
            state, accepted, report = loop.consolidate_skill(
                state,
                agent,
                tx,
                eval_fn,
                candidate_records=None,
                cfg=cfg,
                S_random=S_random,
                S_single=S_single,
                score_windows=score_windows[game],
                learned_game_keys=protected_meta,
            )
            consolidation = {"ran": True, "accepted": bool(accepted), "report": report}
            if accepted:
                protected_meta[game] = _learned_meta_row(
                    game,
                    S_random,
                    S_single,
                    best_scores[game],
                    state,
                    current=False,
                )

        row = _retention_row(
            agent,
            state,
            list(certified_meta),
            cfg,
            random_by_game,
            best_scores,
            eval_episodes=int(args.eval_episodes),
            seed=int(args.eval_seed) + 130_021 * index,
            after_game=game,
            max_steps=int(args.eval_max_steps),
        )
        for learned_game in certified_meta:
            certified_meta[learned_game]["S_best"] = float(best_scores[learned_game])
            if learned_game in protected_meta:
                protected_meta[learned_game]["S_best"] = float(best_scores[learned_game])

        result["game_results"][game] = {
            "train": train_info,
            "eval_windows": eval_windows,
            "score": float(score),
            "learned": bool(learned),
            "learned_info": learn_info,
        }
        result["consolidation"][game] = consolidation
        result["retention_matrix"].append(row)
        result["final"] = _final_report(
            state,
            result["retention_matrix"],
            result["per_game_normalized_progress"],
        )
        _persist_json(path, result)

    return result


def _run_plain(
    args: argparse.Namespace,
    cfg: TFNSConfig,
    games: Sequence[str],
    out_dir: Path,
    random_by_game: Mapping[str, float],
    refs: Mapping[str, Mapping[str, float]],
) -> dict[str, Any]:
    result = _result_header("plain", args, cfg, games, random_by_game)
    result.update(
        {
            "game_results": {},
            "failure_to_learn": [],
            "retention_matrix": [],
            "per_game_normalized_progress": {},
            "consolidation": {},
        }
    )
    path = out_dir / f"plain_seed{int(args.seed)}.json"
    _persist_json(path, result)

    agent, tx, state = _init_agent_state(cfg, str(games[0]), int(args.seed))
    state = _clear_plain_state(state, cfg)
    score_windows: dict[str, list[float]] = {str(game): [] for game in games}
    best_scores: dict[str, float] = {}
    trained_games: list[str] = []

    for index, game in enumerate(games):
        game = str(game)
        state, train_info = train_one_game(
            state,
            agent,
            tx,
            game,
            cfg,
            int(args.steps_per_game),
            int(args.seed) + 200_003 * index,
            protect=False,
            learned_meta={},
            out_dir=out_dir,
        )
        eval_windows = _eval_windows(
            agent,
            state,
            game,
            cfg,
            eval_episodes=int(args.eval_episodes),
            seed=int(args.eval_seed) + 210_017 * index,
            windows=max(1, int(cfg.consolidate.stable_windows)),
            max_steps=int(args.eval_max_steps),
        )
        scores = _window_scores(eval_windows)
        score_windows[game].extend(scores)
        score = _mean_window_score(eval_windows)
        S_random, S_single, has_ref = _reference_for_game(game, refs, random_by_game)
        progress = _progress_value(score, S_random, S_single, has_ref)
        learned, learn_info = certify.is_learned(score_windows[game], S_random, S_single, cfg)
        if not learned:
            result["failure_to_learn"].append(
                {"game": game, "learned": learn_info, "score": float(score)}
            )
        result["per_game_normalized_progress"][game] = {
            "value": float(progress),
            "basis": "single_task_ref" if has_ref else "random_margin",
            "score": float(score),
            "S_random": float(S_random),
            "S_single": float(S_single),
        }
        trained_games.append(game)

        row = _retention_row(
            agent,
            state,
            trained_games,
            cfg,
            random_by_game,
            best_scores,
            eval_episodes=int(args.eval_episodes),
            seed=int(args.eval_seed) + 220_021 * index,
            after_game=game,
            max_steps=int(args.eval_max_steps),
        )
        result["game_results"][game] = {
            "train": train_info,
            "eval_windows": eval_windows,
            "score": float(score),
            "learned": bool(learned),
            "learned_info": learn_info,
        }
        result["consolidation"][game] = {"ran": False, "accepted": False, "report": "plain_baseline"}
        result["retention_matrix"].append(row)
        result["final"] = _final_report(
            state,
            result["retention_matrix"],
            result["per_game_normalized_progress"],
        )
        _persist_json(path, result)

    return result


def _parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=("curriculum", "refs", "plain"), default="curriculum")
    parser.add_argument("--games", nargs="+", default=None)
    parser.add_argument("--steps-per-game", type=int, default=None)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--num-envs", type=int, default=128)
    parser.add_argument("--rollout-len", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=2)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--eval-episodes", type=int, default=None)
    parser.add_argument("--eval-seed", type=int, default=None)
    parser.add_argument("--eval-max-steps", type=int, default=evaluate.DEFAULT_EVAL_MAX_STEPS)
    parser.add_argument("--learned-threshold", type=float, default=0.9)
    parser.add_argument("--out-dir", default="tfns_runs/curriculum")
    parser.add_argument("--refs-json", default=None)
    parser.add_argument("--smoke", action="store_true")
    return parser


def _normalize_args(args: argparse.Namespace) -> argparse.Namespace:
    if int(args.num_envs) <= 0:
        raise ValueError("--num-envs must be positive")
    if int(args.rollout_len) <= 0:
        raise ValueError("--rollout-len must be positive")
    if int(args.update_epochs) <= 0:
        raise ValueError("--update-epochs must be positive")
    if int(args.num_minibatches) <= 0:
        raise ValueError("--num-minibatches must be positive")
    if int(args.eval_max_steps) < 0:
        raise ValueError("--eval-max-steps must be non-negative")
    if args.smoke:
        args.num_envs = min(int(args.num_envs), 4)
        args.rollout_len = min(int(args.rollout_len), 32)
        if args.eval_episodes is None:
            args.eval_episodes = 4
        if args.steps_per_game is None:
            args.steps_per_game = 2 * int(args.num_envs) * int(args.rollout_len)
    else:
        if args.eval_episodes is None:
            args.eval_episodes = 30
        if args.steps_per_game is None:
            raise ValueError("--steps-per-game is required outside --smoke")
    if args.eval_seed is None:
        args.eval_seed = int(args.seed) + 1_000_003
    if int(args.steps_per_game) < int(args.num_envs):
        raise ValueError("--steps-per-game must cover at least one vector env step")
    return args


def main(argv: Sequence[str] | None = None) -> dict[str, Any]:
    args = _normalize_args(_parser().parse_args(argv))
    games = _parse_games(args.games, smoke=bool(args.smoke))
    if not games:
        raise ValueError("at least one game is required")
    cfg = _base_cfg(args)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    random_by_game = random_scores(
        games,
        int(args.eval_episodes),
        int(args.eval_seed),
        num_envs=_eval_num_envs(cfg, int(args.eval_episodes)),
        out_dir=out_dir,
        max_steps=int(args.eval_max_steps),
    )
    refs = _load_refs(args.refs_json)

    if args.mode == "refs":
        result = _run_refs(args, cfg, games, out_dir, random_by_game)
        artifact = out_dir / f"refs_seed{int(args.seed)}.json"
    elif args.mode == "curriculum":
        result = _run_curriculum(args, cfg, games, out_dir, random_by_game, refs)
        artifact = out_dir / f"curriculum_seed{int(args.seed)}.json"
    else:
        result = _run_plain(args, cfg, games, out_dir, random_by_game, refs)
        artifact = out_dir / f"plain_seed{int(args.seed)}.json"

    summary = {
        "mode": args.mode,
        "seed": int(args.seed),
        "games": games,
        "artifact": str(artifact),
        "progress_log": str(out_dir / "progress.jsonl"),
        "final": result.get("final", {}),
    }
    print(json.dumps(_jsonable(summary), indent=2, sort_keys=True))
    return result


if __name__ == "__main__":
    main()
