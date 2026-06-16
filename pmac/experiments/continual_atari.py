"""Continual envpool Atari PPO: matched warm-start baseline versus PMA-C."""

from __future__ import annotations

import argparse
import json
import math
import time
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import NamedTuple

import jax
import jax.numpy as jnp
import numpy as np

from pmac.agents.atari_eval import evaluate_atari
from pmac.agents.atari_net import atari_apply, init_atari
from pmac.agents.ppo_atari import AtariPPOConfig, train_ppo_atari
from pmac.anchors import AnchorStore
from pmac.atlas import Atlas
from pmac.checkpoint import ChampionStore
from pmac.deployment import DeployedPolicy, DeploymentDecision
from pmac.envs.atari_envpool import ACT_DIM, make_train_env
from pmac.evaluation import aggregate_retention, make_skill_scores
from pmac.sentinels import SentinelStore

warnings.filterwarnings("ignore")


ALLOWED_ATARI_ABLATIONS = {None, "none", "no_conservation", "no_replay"}
ALLOWED_GUARD_NORMS = {"length", "none"}


@dataclass(frozen=True)
class ContinualAtariConfig:
    per_game_steps: int = 4_000_000
    num_envs: int = 64
    num_steps: int = 128
    update_epochs: int = 4
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 2.5e-4
    anneal_lr: bool = True
    eval_episodes: int = 4
    eval_envs: int = 16
    eval_max_steps_per_episode: int = 30_000
    eval_steps_cap: int = 6_000
    guard_coef: float = 1.0
    guard_norm: str = "length"
    guard_batch: int = 256
    anchor_buffer_per_game: int = 2048
    value_coef: float = 1.0
    guard_tolerance: float = 0.01
    ablation: str | None = None
    deploy_eval_episodes: int = 12
    sentinel_eval_episodes: int = 8
    cert_frac: float = 0.05
    cert_abs_floor: float = 0.0
    learned_delta_frac: float = 0.05
    allowed_regression_frac: float = 0.02


class AtariAnchorBuffer(NamedTuple):
    obs_uint8: np.ndarray
    game_onehot: np.ndarray
    teacher_logits: np.ndarray
    teacher_value: np.ndarray


def _parse_csv(text):
    return [part.strip() for part in str(text).split(",") if part.strip()]


def _parse_seeds(text):
    return [int(part) for part in _parse_csv(text)]


def _parse_ablations(text):
    return [None if value == "none" else value for value in _parse_csv(text)]


def _seed_int(seed) -> int:
    arr = np.asarray(seed, dtype=np.uint32).reshape(-1)
    mixed = 0
    for value in arr:
        mixed = (1664525 * mixed + int(value) + 1013904223) % (2**32)
    return int(mixed)


def _to_ppo_config(cfg: ContinualAtariConfig) -> AtariPPOConfig:
    return AtariPPOConfig(
        total_timesteps=int(cfg.per_game_steps),
        num_envs=int(cfg.num_envs),
        num_steps=int(cfg.num_steps),
        update_epochs=int(cfg.update_epochs),
        num_minibatches=int(cfg.num_minibatches),
        gamma=float(cfg.gamma),
        gae_lambda=float(cfg.gae_lambda),
        clip_coef=float(cfg.clip_coef),
        ent_coef=float(cfg.ent_coef),
        vf_coef=float(cfg.vf_coef),
        max_grad_norm=float(cfg.max_grad_norm),
        lr=float(cfg.lr),
        anneal_lr=bool(cfg.anneal_lr),
    )


def _validate_continual_config(cfg: ContinualAtariConfig) -> None:
    steps_per_update = int(cfg.num_envs) * int(cfg.num_steps)
    if steps_per_update <= 0:
        raise ValueError("num_envs*num_steps must be positive")
    if int(cfg.per_game_steps) // steps_per_update <= 0:
        raise ValueError("per_game_steps must cover at least one PPO update")
    if int(cfg.update_epochs) <= 0:
        raise ValueError("update_epochs must be positive")
    if int(cfg.num_minibatches) <= 0:
        raise ValueError("num_minibatches must be positive")
    if steps_per_update % int(cfg.num_minibatches) != 0:
        raise ValueError("num_envs*num_steps must be divisible by num_minibatches")
    if int(cfg.eval_episodes) <= 0:
        raise ValueError("eval_episodes must be positive")
    if int(cfg.eval_envs) <= 0:
        raise ValueError("eval_envs must be positive")
    if int(cfg.eval_max_steps_per_episode) <= 0:
        raise ValueError("eval_max_steps_per_episode must be positive")
    if int(cfg.eval_steps_cap) <= 0:
        raise ValueError("eval_steps_cap must be positive")
    if int(cfg.deploy_eval_episodes) <= 0:
        raise ValueError("deploy_eval_episodes must be positive")
    if int(cfg.sentinel_eval_episodes) <= 0:
        raise ValueError("sentinel_eval_episodes must be positive")
    if int(cfg.guard_batch) <= 0:
        raise ValueError("guard_batch must be positive")
    if int(cfg.anchor_buffer_per_game) <= 0:
        raise ValueError("anchor_buffer_per_game must be positive")
    if int(cfg.num_envs) <= 0:
        raise ValueError("num_envs must be positive")
    if float(cfg.cert_frac) < 0.0:
        raise ValueError("cert_frac must be non-negative")
    if float(cfg.cert_abs_floor) < 0.0:
        raise ValueError("cert_abs_floor must be non-negative")
    if float(cfg.learned_delta_frac) < 0.0:
        raise ValueError("learned_delta_frac must be non-negative")
    if float(cfg.allowed_regression_frac) < 0.0:
        raise ValueError("allowed_regression_frac must be non-negative")
    ablation = None if cfg.ablation == "none" else cfg.ablation
    if ablation not in ALLOWED_ATARI_ABLATIONS:
        raise ValueError(f"unknown Atari PMA-C ablation: {cfg.ablation}")
    if str(cfg.guard_norm) not in ALLOWED_GUARD_NORMS:
        raise ValueError(f"unknown Atari guard_norm: {cfg.guard_norm}")


def _init_seed(seed: int) -> int:
    return int(seed) + 7


def _train_seed(seed: int, task_i: int) -> int:
    return int(seed) + 100_003 * int(task_i)


def _eval_seed(seed: int, task_i: int) -> int:
    return int(seed) + 200_003 * int(task_i) + 17


def _sentinel_eval_seed(seed: int, game_id: int) -> int:
    return int(seed) + 600_011 * int(game_id) + 53


def _deploy_eval_seed(seed: int, game_id: int) -> int:
    return int(seed) + 700_001 * int(game_id) + 71


def _random_deploy_seed(seed: int, game_id: int) -> int:
    return int(seed) + 800_011 * int(game_id) + 97


def _anchor_seed(seed: int, task_i: int) -> int:
    return int(seed) + 300_007 * int(task_i) + 31


def _random_eval_seed(seed: int) -> int:
    return int(seed) + 500_009


def _fresh_params(seed: int, n_games: int):
    return init_atari(jax.random.PRNGKey(_init_seed(seed)), int(n_games))


def _guard_from_buffers(
    buffers: list[AtariAnchorBuffer],
    cfg: ContinualAtariConfig,
    ablation,
):
    selected = [] if ablation == "no_conservation" else list(buffers)
    if ablation == "no_replay" and selected:
        selected = selected[-1:]
    if not selected:
        return None
    lengths = [int(buf.obs_uint8.shape[0]) for buf in selected]
    prior_offsets = np.concatenate(
        [np.asarray([0], dtype=np.int32), np.cumsum(np.asarray(lengths, dtype=np.int32))]
    )
    n_prior = max(1, len(selected))
    if str(cfg.guard_norm) == "length":
        effective_guard_coef = float(cfg.guard_coef) / float(n_prior)
    else:
        effective_guard_coef = float(cfg.guard_coef)

    return {
        "obs_uint8": np.concatenate([buf.obs_uint8 for buf in selected], axis=0),
        "game_onehot": np.concatenate([buf.game_onehot for buf in selected], axis=0),
        "teacher_logits": np.concatenate([buf.teacher_logits for buf in selected], axis=0),
        "teacher_value": np.concatenate([buf.teacher_value for buf in selected], axis=0),
        "prior_offsets": prior_offsets,
        "guard_coef": float(effective_guard_coef),
        "value_coef": float(cfg.value_coef),
        "guard_tolerance": float(cfg.guard_tolerance),
        "guard_batch": int(cfg.guard_batch),
    }


@jax.jit
def _jit_anchor_policy(params, obs, game_onehot, rng):
    rng, action_key = jax.random.split(rng)
    logits, value = atari_apply(params, obs, game_onehot)
    actions = jax.random.categorical(action_key, logits, axis=-1).astype(jnp.int32)
    return actions, logits, value, rng


def _collect_anchor_buffer(
    params,
    game,
    game_id: int,
    n_games: int,
    cfg: ContinualAtariConfig,
    seed: int,
) -> AtariAnchorBuffer:
    n = int(cfg.anchor_buffer_per_game)
    anchor_envs = max(1, min(int(cfg.num_envs), n))
    anchor_steps = int(math.ceil(float(n) / float(anchor_envs)))
    total = int(anchor_steps * anchor_envs)
    rng = jax.random.PRNGKey(int(seed))
    game_onehot = jax.nn.one_hot(int(game_id), int(n_games), dtype=jnp.float32)

    env = make_train_env(str(game), anchor_envs, int(seed))
    obs, _ = env.reset()
    obs = np.asarray(obs, dtype=np.uint8)

    obs_buf = np.zeros((total, 4, 84, 84), dtype=np.uint8)
    logits_buf = np.zeros((total, ACT_DIM), dtype=np.float32)
    value_buf = np.zeros((total,), dtype=np.float32)

    for step in range(anchor_steps):
        start = int(step * anchor_envs)
        stop = int(start + anchor_envs)
        obs_buf[start:stop] = obs
        actions, logits, value, rng = _jit_anchor_policy(params, obs, game_onehot, rng)
        logits_buf[start:stop] = np.asarray(jax.device_get(logits), dtype=np.float32)
        value_buf[start:stop] = np.asarray(jax.device_get(value), dtype=np.float32)
        actions_np = np.asarray(jax.device_get(actions), dtype=np.int32)
        obs, _, _, _, _ = env.step(actions_np)
        obs = np.asarray(obs, dtype=np.uint8)

    goh = np.broadcast_to(
        np.asarray(jax.device_get(game_onehot), dtype=np.float32),
        (total, int(n_games)),
    ).copy()
    return AtariAnchorBuffer(
        obs_uint8=obs_buf[:n].copy(),
        game_onehot=goh[:n].copy(),
        teacher_logits=logits_buf[:n].copy(),
        teacher_value=value_buf[:n].copy(),
    )


def _softmax_confidence(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    probs = np.exp(shifted) / np.sum(np.exp(shifted), axis=-1, keepdims=True)
    return np.max(probs, axis=-1).astype(np.float32)


def _certify_game(
    params,
    game,
    game_id: int,
    task_i: int,
    eval_score: float,
    buffer: AtariAnchorBuffer,
    cfg: ContinualAtariConfig,
    atlas: Atlas,
    champions: ChampionStore,
):
    teacher = np.concatenate(
        [buffer.teacher_logits, buffer.teacher_value[:, None]],
        axis=-1,
    )
    n = int(buffer.obs_uint8.shape[0])
    anchors = AnchorStore(cfg.anchor_buffer_per_game)
    anchors.add(
        buffer.obs_uint8,
        teacher,
        np.full((n,), float(cfg.guard_tolerance), dtype=np.float32),
        np.ones((n,), dtype=np.float32),
        _softmax_confidence(buffer.teacher_logits),
        contexts=buffer.game_onehot,
        skill_ids=[str(game)] * n,
        labels=np.full((n,), int(game_id), dtype=np.int32),
    )
    sent_n = min(n, int(cfg.eval_episodes))
    sentinels = SentinelStore(
        x=buffer.obs_uint8[:sent_n],
        y=np.full((sent_n,), int(game_id), dtype=np.int32),
        seeds=np.arange(sent_n, dtype=np.int32),
    )
    champion = champions.freeze(
        params,
        route=str(game),
        meta={"skill_id": str(game), "task_index": int(task_i)},
    )
    return atlas.create_or_update_node(
        str(game),
        context_key=str(game),
        anchors=anchors,
        sentinels=sentinels,
        status="protected",
        champion_ref=champion,
        best_score=float(eval_score),
        current_score=float(eval_score),
        retention=1.0,
        allowed_regression=0.0,
        current_certified=True,
        current_last_score=float(eval_score),
        fallback_route_id=str(game),
        fallback_score=float(eval_score),
        needs_repair=False,
        last_certified_step=int(task_i),
        guard_lambda=float(cfg.guard_coef),
        certified_impls=[str(game)],
    )


def _evaluate_all_games_with_cap(
    params,
    games,
    n_games,
    eval_episodes,
    seed,
    max_steps_per_episode: int,
    eval_envs: int,
    eval_steps_cap: int,
) -> np.ndarray:
    games = list(games)
    if not games:
        return np.asarray([], dtype=np.float32)
    base_seed = _seed_int(seed)
    scores = []
    for game_id, game in enumerate(games):
        scores.append(
            evaluate_atari(
                params,
                game,
                game_id,
                int(n_games),
                n_episodes=int(eval_episodes),
                seed=base_seed + 9973 * int(game_id),
                max_steps_per_episode=int(max_steps_per_episode),
                eval_envs=int(eval_envs),
                eval_steps_cap=int(eval_steps_cap),
            )
        )
    return np.asarray(scores, dtype=np.float32)


def evaluate_all_games(
    params,
    games,
    n_games,
    eval_episodes,
    seed,
    eval_envs: int = 16,
    eval_steps_cap: int = 6_000,
) -> np.ndarray:
    """Greedy true-score evaluation on every Atari game."""
    return _evaluate_all_games_with_cap(
        params,
        games,
        n_games,
        eval_episodes,
        seed,
        30_000,
        eval_envs,
        eval_steps_cap,
    )


def _deployment_eval_score(
    params,
    game,
    game_id: int,
    n_games: int,
    eval_episodes: int,
    seed: int,
    cfg: ContinualAtariConfig,
) -> float:
    return float(
        evaluate_atari(
            params,
            game,
            int(game_id),
            int(n_games),
            n_episodes=int(eval_episodes),
            seed=int(seed),
            max_steps_per_episode=int(cfg.eval_max_steps_per_episode),
            eval_envs=int(cfg.eval_envs),
            eval_steps_cap=int(cfg.eval_steps_cap),
        )
    )


def _learned_delta_for_game(best: float, random_score: float, cfg: ContinualAtariConfig) -> float:
    """Use an absolute Atari margin so low peaks cannot count as learned."""
    del best
    return float(max(1.0, float(cfg.learned_delta_frac) * abs(float(random_score))))


def _allowed_regression_for_game(
    best: float, random_score: float, cfg: ContinualAtariConfig
) -> float:
    return float(float(cfg.allowed_regression_frac) * max(float(best) - float(random_score), 0.0))


def evaluate_deployment(
    current_params,
    games,
    n_games,
    random_scores,
    *,
    atlas,
    cfg,
    seed,
    best_scores=None,
    random_params=None,
) -> dict:
    """Run the bounded deployed-vs-current Atari evaluation pass.

    All scores (champion / current / random) are re-measured live under one consistent
    protocol so the reported retentions cannot be gamed by seed/episode-count mismatch:
      * sentinel seed A  -> ROUTE DECISION only (never reported)
      * deploy seed B     -> reported champion_B / current_B / random_B
    Honesty guarantees:
      * best[g] = max(champion_B, current_B) (true peak under the deploy protocol) so a
        current net that exceeds its champion can never inflate retention above 1.0.
      * deployed_B = current_B iff the sentinel certified current (current_A >= champion_A
        - cert_tol); else champion_B. This is a real route choice, NOT an offline max.
      * a forgotten skill routes to its champion => deployed_B == champion_B == best =>
        deployed_retention == 1.0 exactly (the no-forgetting safety invariant).
    """
    games = list(games)
    random_scores = np.asarray(random_scores, dtype=np.float32)
    if random_scores.shape[0] != len(games):
        raise ValueError("random_scores must match games")
    if atlas is None:
        if best_scores is None:
            raise ValueError("best_scores are required when atlas is None")
        best_scores = np.asarray(best_scores, dtype=np.float32)
        if best_scores.shape[0] != len(games):
            raise ValueError("best_scores must match games")

    policy = None if atlas is None else DeployedPolicy(current_params, atlas)
    skill_scores = []
    decisions = []
    best_values = []
    champion_b_values = []
    current_b_values = []
    deployed_b_values = []
    random_b_values = []
    current_certified_values = []

    for game_id, game in enumerate(games):
        skill_id = str(game)
        seed_a = _sentinel_eval_seed(seed, game_id)
        seed_b = _deploy_eval_seed(seed, game_id)

        # Stable random baseline under the SAME deploy protocol (fresh-init net), so the
        # normalization denominator is not corrupted by the coarse per-game eval random.
        if random_params is not None:
            random_score = _deployment_eval_score(
                random_params,
                game,
                game_id,
                n_games,
                int(cfg.deploy_eval_episodes),
                _random_deploy_seed(seed, game_id),
                cfg,
            )
        else:
            random_score = float(random_scores[game_id])

        current_b = _deployment_eval_score(
            current_params,
            game,
            game_id,
            n_games,
            int(cfg.deploy_eval_episodes),
            seed_b,
            cfg,
        )

        if atlas is None:
            champion_b = float(current_b)
            best = float(max(float(best_scores[game_id]), float(current_b)))
            current_certified = True
            decision = DeploymentDecision(
                skill_id=skill_id,
                route_type="current",
                route_id="current",
                reason="baseline_no_champion",
                current_certified=True,
                fallback_used=False,
            )
        else:
            assert policy is not None
            if skill_id not in atlas.nodes:
                policy.select_route(skill_id, False)
            node = atlas.nodes[skill_id]
            champion = getattr(node, "champion_ref", None)
            if champion is None or not hasattr(champion, "params") or champion.params is None:
                policy.select_route(skill_id, False)
            champion_params = champion.params
            champion_a = _deployment_eval_score(
                champion_params,
                game,
                game_id,
                n_games,
                int(cfg.sentinel_eval_episodes),
                seed_a,
                cfg,
            )
            current_a = _deployment_eval_score(
                current_params,
                game,
                game_id,
                n_games,
                int(cfg.sentinel_eval_episodes),
                seed_a,
                cfg,
            )
            champion_b = _deployment_eval_score(
                champion_params,
                game,
                game_id,
                n_games,
                int(cfg.deploy_eval_episodes),
                seed_b,
                cfg,
            )
            # True peak under the deploy protocol: the better of the certified champion and
            # the current net. Guarantees retention in [0,1] and exact 1.0 for the route taken
            # when it is the peak (forgotten skill -> champion is the peak -> retention 1.0).
            best = float(max(float(champion_b), float(current_b)))
            cert_tol = (
                float(cfg.cert_frac) * max(float(champion_a) - float(random_score), 0.0)
                + float(cfg.cert_abs_floor)
            )
            current_certified = bool(float(current_a) >= float(champion_a) - float(cert_tol))
            allowed_regression = _allowed_regression_for_game(best, random_score, cfg)
            route_id = getattr(champion, "route", None)
            node.current_certified = current_certified
            node.current_last_score = float(current_a)
            node.current_score = float(current_b)
            node.best_score = float(best)
            node.allowed_regression = float(allowed_regression)
            node.fallback_route_id = str(route_id if route_id is not None else skill_id)
            node.fallback_score = float(champion_b)
            decision = policy.select_route(skill_id, current_certified)
            policy.resolve_params(decision)

        deployed_b = float(current_b if decision.route_type == "current" else champion_b)
        allowed_regression = _allowed_regression_for_game(best, random_score, cfg)
        learned_delta = _learned_delta_for_game(best, random_score, cfg)
        scores = make_skill_scores(
            skill_id,
            best=best,
            current=current_b,
            champion=champion_b,
            deployed=deployed_b,
            random_score=random_score,
            route_type=decision.route_type,
            current_certified=current_certified,
            learned_delta=learned_delta,
            allowed_regression=allowed_regression,
        )
        skill_scores.append(scores)
        decisions.append(decision)
        best_values.append(float(best))
        champion_b_values.append(float(champion_b))
        current_b_values.append(float(current_b))
        deployed_b_values.append(float(deployed_b))
        random_b_values.append(float(random_score))
        current_certified_values.append(bool(current_certified))

    return {
        "skill_scores": [asdict(score) for score in skill_scores],
        "decisions": [asdict(decision) for decision in decisions],
        "route_usage": DeployedPolicy.route_usage(decisions),
        "aggregate": aggregate_retention(skill_scores),
        "best": best_values,
        "champion_B": champion_b_values,
        "current_B": current_b_values,
        "deployed_B": deployed_b_values,
        "random_B": random_b_values,
        "current_certified": current_certified_values,
    }


def compute_atari_metrics(return_matrix, random_scores) -> dict:
    returns = np.asarray(return_matrix, dtype=np.float32)
    random_scores = np.asarray(random_scores, dtype=np.float32)
    if returns.ndim != 2 or returns.shape[0] == 0 or returns.shape[1] == 0:
        raise ValueError("return_matrix must be a non-empty 2D array")
    if random_scores.shape[0] != returns.shape[1]:
        raise ValueError("random_scores must match return_matrix columns")

    learned = np.diag(returns).astype(np.float32)
    final = returns[-1].astype(np.float32)
    denom = learned - random_scores + np.float32(1.0e-6)
    norm_retention = np.clip((final - random_scores) / denom, 0.0, 1.5)
    norm_forgetting = np.maximum(0.0, (learned - final) / denom)
    return {
        "mean_final_return": float(np.mean(final)),
        "raw_mean_final_return": float(np.mean(final)),
        "norm_retention": norm_retention.astype(float).tolist(),
        "mean_norm_retention": float(np.mean(norm_retention)),
        "worst_norm_retention": float(np.min(norm_retention)),
        "norm_forgetting": float(np.mean(norm_forgetting)),
        "mean_retention": float(np.mean(norm_retention)),
        "worst_retention": float(np.min(norm_retention)),
        "forgetting": float(np.mean(norm_forgetting)),
        "learned": learned.astype(float).tolist(),
        "final": final.astype(float).tolist(),
        "random_scores": random_scores.astype(float).tolist(),
        "return_matrix": returns.astype(float).tolist(),
    }


def _result(
    return_matrix,
    random_scores,
    mode: str,
    cfg: ContinualAtariConfig,
    seed: int,
    wall_s: float,
    extra=None,
) -> dict:
    return_matrix = np.asarray(return_matrix, dtype=np.float32)
    random_scores = np.asarray(random_scores, dtype=np.float32)
    metrics = compute_atari_metrics(return_matrix, random_scores)
    learned = np.diag(return_matrix).astype(np.float32)
    final = return_matrix[-1].astype(np.float32)
    return {
        "mode": mode,
        "return_matrix": return_matrix,
        "learned": learned,
        "learned_returns": learned,
        "final": final,
        "final_returns": final,
        "random_scores": random_scores,
        "metrics": metrics,
        "wall_s": float(wall_s),
        "extra": {
            "seed": int(seed),
            "config": asdict(cfg),
            "wall_s": float(wall_s),
            **dict(extra or {}),
        },
    }


def _total_updates_from_train(train: dict, ppo_cfg: AtariPPOConfig) -> int:
    batch_size = int(ppo_cfg.num_envs) * int(ppo_cfg.num_steps)
    return int(train["timesteps"]) // batch_size


def run_atari_baseline(
    games,
    cfg: ContinualAtariConfig | None = None,
    seed: int = 0,
) -> dict:
    """Sequential warm-start Atari PPO without PMA-C protection."""
    started = time.perf_counter()
    cfg = cfg or ContinualAtariConfig()
    _validate_continual_config(cfg)
    games = list(games)
    if not games:
        raise ValueError("at least one Atari game is required")

    n_games = len(games)
    ppo_cfg = _to_ppo_config(cfg)
    random_scores = _evaluate_all_games_with_cap(
        _fresh_params(seed, n_games),
        games,
        n_games,
        cfg.eval_episodes,
        _random_eval_seed(seed),
        cfg.eval_max_steps_per_episode,
        cfg.eval_envs,
        cfg.eval_steps_cap,
    )
    return_matrix = np.zeros((n_games, n_games), dtype=np.float32)
    curves = {}
    total_updates = 0
    params = None

    for task_i, game in enumerate(games):
        train = train_ppo_atari(
            game,
            task_i,
            n_games,
            ppo_cfg,
            _train_seed(seed, task_i),
            init_params=params,
        )
        params = train["params"]
        total_updates += _total_updates_from_train(train, ppo_cfg)
        curves[str(game)] = train["returns_curve"]
        return_matrix[task_i] = _evaluate_all_games_with_cap(
            params,
            games,
            n_games,
            cfg.eval_episodes,
            _eval_seed(seed, task_i),
            cfg.eval_max_steps_per_episode,
            cfg.eval_envs,
            cfg.eval_steps_cap,
        )

    wall_s = time.perf_counter() - started
    deployment = evaluate_deployment(
        params,
        games,
        n_games,
        random_scores,
        atlas=None,
        cfg=cfg,
        seed=seed,
        best_scores=np.max(return_matrix, axis=0),
        random_params=_fresh_params(seed, n_games),
    )
    result = _result(
        return_matrix,
        random_scores,
        "baseline",
        cfg,
        seed,
        wall_s,
        extra={
            "game_order": [str(game) for game in games],
            "updates": int(total_updates),
            "returns_curves": curves,
            "guard_enabled": False,
            "guard_source": "none",
            "deployment": deployment,
        },
    )
    result["metrics"]["deployed"] = deployment["aggregate"]
    return result


def run_atari_pmac(
    games,
    cfg: ContinualAtariConfig | None = None,
    seed: int = 0,
    ablation=None,
) -> dict:
    """Sequential warm-start Atari PPO with optional PMA-C frame-anchor conservation."""
    started = time.perf_counter()
    cfg = cfg or ContinualAtariConfig()
    ablation = cfg.ablation if ablation is None else ablation
    ablation = None if ablation == "none" else ablation
    if ablation not in ALLOWED_ATARI_ABLATIONS:
        raise ValueError(f"unknown Atari PMA-C ablation: {ablation}")
    _validate_continual_config(cfg)
    games = list(games)
    if not games:
        raise ValueError("at least one Atari game is required")

    n_games = len(games)
    ppo_cfg = _to_ppo_config(cfg)
    random_scores = _evaluate_all_games_with_cap(
        _fresh_params(seed, n_games),
        games,
        n_games,
        cfg.eval_episodes,
        _random_eval_seed(seed),
        cfg.eval_max_steps_per_episode,
        cfg.eval_envs,
        cfg.eval_steps_cap,
    )
    atlas = Atlas()
    champions = ChampionStore()
    buffers: list[AtariAnchorBuffer] = []
    return_matrix = np.zeros((n_games, n_games), dtype=np.float32)
    curves = {}
    guard_curves = {}
    total_updates = 0
    guard_enabled = ablation != "no_conservation"
    guard_effective_coefs: list[float] = []
    params = None

    for task_i, game in enumerate(games):
        guard = _guard_from_buffers(buffers, cfg, ablation) if guard_enabled else None
        guard_effective_coefs.append(0.0 if guard is None else float(guard["guard_coef"]))
        train = train_ppo_atari(
            game,
            task_i,
            n_games,
            ppo_cfg,
            _train_seed(seed, task_i),
            init_params=params,
            guard=guard,
        )
        params = train["params"]
        total_updates += _total_updates_from_train(train, ppo_cfg)
        curves[str(game)] = train["returns_curve"]
        guard_curves[str(game)] = train.get("guard_curve", [])
        return_matrix[task_i] = _evaluate_all_games_with_cap(
            params,
            games,
            n_games,
            cfg.eval_episodes,
            _eval_seed(seed, task_i),
            cfg.eval_max_steps_per_episode,
            cfg.eval_envs,
            cfg.eval_steps_cap,
        )

        if guard_enabled:
            buffer = _collect_anchor_buffer(
                params,
                game,
                task_i,
                n_games,
                cfg,
                _anchor_seed(seed, task_i),
            )
            buffers.append(buffer)
            _certify_game(
                params,
                game,
                task_i,
                task_i,
                float(return_matrix[task_i, task_i]),
                buffer,
                cfg,
                atlas,
                champions,
            )

    wall_s = time.perf_counter() - started
    mode = "pmac" if ablation is None else f"pmac_{ablation}"
    if not guard_enabled:
        guard_source = "none"
    elif ablation == "no_replay":
        guard_source = "most_recent_prior"
    else:
        guard_source = "all_prior"
    deployment_atlas = atlas if guard_enabled else None
    deployment = evaluate_deployment(
        params,
        games,
        n_games,
        random_scores,
        atlas=deployment_atlas,
        cfg=cfg,
        seed=seed,
        best_scores=None if deployment_atlas is not None else np.max(return_matrix, axis=0),
        random_params=_fresh_params(seed, n_games),
    )
    result = _result(
        return_matrix,
        random_scores,
        mode,
        cfg,
        seed,
        wall_s,
        extra={
            "ablation": ablation,
            "game_order": [str(game) for game in games],
            "updates": int(total_updates),
            "returns_curves": curves,
            "guard_loss_curves": guard_curves,
            "protected_skills": list(atlas.nodes.keys()),
            "anchor_counts": [int(buf.obs_uint8.shape[0]) for buf in buffers],
            "guard_enabled": bool(guard_enabled),
            "guard_source": guard_source,
            "guard_norm": str(cfg.guard_norm),
            "guard_effective_coefs": [float(v) for v in guard_effective_coefs],
            "deployment": deployment,
        },
    )
    result["metrics"]["deployed"] = deployment["aggregate"]
    return result


def _jsonify_result(result: dict) -> dict:
    return {
        "mode": result["mode"],
        "return_matrix": np.asarray(result["return_matrix"]).astype(float).tolist(),
        "learned": np.asarray(result["learned"]).astype(float).tolist(),
        "learned_returns": np.asarray(result["learned_returns"]).astype(float).tolist(),
        "final": np.asarray(result["final"]).astype(float).tolist(),
        "final_returns": np.asarray(result["final_returns"]).astype(float).tolist(),
        "random_scores": np.asarray(result["random_scores"]).astype(float).tolist(),
        "metrics": result["metrics"],
        "wall_s": float(result["wall_s"]),
        "extra": result["extra"],
    }


def _aggregate(results_by_mode):
    aggregate = {}
    for mode, results in results_by_mode.items():
        stats = {}
        for key, value in results[0]["metrics"].items():
            if isinstance(value, dict):
                continue
            if isinstance(value, (list, tuple)):
                continue
            arr = np.asarray([result["metrics"][key] for result in results], dtype=np.float64)
            stats[key] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
        deployed = results[0]["metrics"].get("deployed")
        if isinstance(deployed, dict):
            deployed_stats = {}
            for key, value in deployed.items():
                if isinstance(value, (dict, list, tuple, str)):
                    continue
                arr = np.asarray(
                    [result["metrics"]["deployed"][key] for result in results],
                    dtype=np.float64,
                )
                deployed_stats[key] = {
                    "mean": float(np.mean(arr)),
                    "std": float(np.std(arr)),
                }
            stats["deployed"] = deployed_stats
        wall = np.asarray([result["wall_s"] for result in results], dtype=np.float64)
        stats["wall_s"] = {"mean": float(np.mean(wall)), "std": float(np.std(wall))}
        aggregate[mode] = stats
    return aggregate


def _game0_normalized_sequence(result):
    returns = np.asarray(result["return_matrix"], dtype=np.float32)
    learned = np.asarray(result["learned"], dtype=np.float32)
    random_scores = np.asarray(result["random_scores"], dtype=np.float32)
    denom = learned[0] - random_scores[0] + np.float32(1.0e-6)
    return (returns[:, 0] - random_scores[0]) / denom


def _plot_results(first_seed_results, games, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    modes = list(first_seed_results.keys())
    first = first_seed_results[modes[0]]
    n_games = int(np.asarray(first["return_matrix"]).shape[1])
    x = np.arange(n_games)
    width = 0.8 / max(1, len(modes))
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2))
    fig.suptitle("Continual Atari: PMA-C vs Baseline")

    for i, mode in enumerate(modes):
        result = first_seed_results[mode]
        offset = (i - (len(modes) - 1) / 2.0) * width
        retention = np.asarray(result["metrics"]["norm_retention"], dtype=np.float32)
        axes[0].bar(x + offset, retention, width=width, label=mode)
        axes[1].plot(_game0_normalized_sequence(result), marker="o", label=mode)

    axes[0].set_title("Per-Game Normalized Retention")
    axes[0].set_xlabel("Game")
    axes[0].set_ylabel("Retention")
    axes[0].set_ylim(bottom=0.0)
    axes[0].set_xticks(x)
    axes[0].set_xticklabels([str(game).replace("-v5", "") for game in games], rotation=25, ha="right")
    axes[0].legend(fontsize=8)

    axes[1].set_title("Game 0 Normalized Score Across Training")
    axes[1].set_xlabel("After Game")
    axes[1].set_ylabel("Normalized Score")
    axes[1].set_xticks(x)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _print_result(result: dict):
    metrics = result["metrics"]
    learned = ", ".join(f"{v:.3f}" for v in np.asarray(result["learned"]))
    final = ", ".join(f"{v:.3f}" for v in np.asarray(result["final"]))
    deployed = metrics.get("deployed")
    deployed_text = ""
    if isinstance(deployed, dict):
        deployed_text = (
            f" deployed_mean_retention={deployed['mean_deployed_retention']:.3f} "
            f"deployed_worst_retention={deployed['worst_deployed_retention']:.3f}"
        )
    print(
        f"{result['mode']} seed={int(result['extra']['seed'])} "
        f"wall_s={float(result['wall_s']):.3f} "
        f"learned=[{learned}] final=[{final}] "
        f"mean_norm_retention={metrics['mean_norm_retention']:.3f} "
        f"worst_norm_retention={metrics['worst_norm_retention']:.3f} "
        f"norm_forgetting={metrics['norm_forgetting']:.3f} "
        f"mean_final_return={metrics['mean_final_return']:.3f}"
        f"{deployed_text}"
    )


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--games",
        default="Pong-v5,Breakout-v5,SpaceInvaders-v5,BeamRider-v5",
    )
    parser.add_argument("--per-game-steps", type=int, default=4_000_000)
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--ablations", default="none")
    parser.add_argument("--guard-coef", type=float, default=1.0)
    parser.add_argument("--guard-norm", choices=sorted(ALLOWED_GUARD_NORMS), default="length")
    parser.add_argument("--out", default="runs/atari_continual")
    parser.add_argument("--eval-episodes", type=int, default=4)
    parser.add_argument("--eval-envs", type=int, default=16)
    parser.add_argument("--eval-max-steps-per-episode", type=int, default=30_000)
    parser.add_argument("--eval-steps-cap", type=int, default=6_000)
    parser.add_argument("--deploy-eval-episodes", type=int, default=12)
    parser.add_argument("--sentinel-eval-episodes", type=int, default=8)
    parser.add_argument("--cert-frac", type=float, default=0.05)
    parser.add_argument("--cert-abs-floor", type=float, default=0.0)
    parser.add_argument("--learned-delta-frac", type=float, default=0.05)
    parser.add_argument("--allowed-regression-frac", type=float, default=0.02)
    parser.add_argument("--num-envs", type=int, default=64)
    parser.add_argument("--num-steps", type=int, default=128)
    parser.add_argument("--update-epochs", type=int, default=4)
    parser.add_argument("--num-minibatches", type=int, default=4)
    parser.add_argument("--lr", type=float, default=2.5e-4)
    parser.add_argument("--clip-coef", type=float, default=0.1)
    parser.add_argument("--ent-coef", type=float, default=0.01)
    parser.add_argument("--vf-coef", type=float, default=0.5)
    parser.add_argument("--max-grad-norm", type=float, default=0.5)
    parser.add_argument("--anchor-buffer-per-game", type=int, default=2048)
    parser.add_argument("--guard-batch", type=int, default=256)
    parser.add_argument("--guard-tolerance", type=float, default=0.01)
    parser.add_argument("--value-coef", type=float, default=1.0)
    parser.add_argument("--no-anneal-lr", dest="anneal_lr", action="store_false")
    parser.set_defaults(anneal_lr=True)
    args = parser.parse_args(argv)

    games = _parse_csv(args.games)
    if not games:
        parser.error("at least one game is required")
    seeds = _parse_seeds(args.seeds)
    ablations = _parse_ablations(args.ablations)
    invalid_ablations = [value for value in ablations if value not in ALLOWED_ATARI_ABLATIONS]
    if invalid_ablations:
        parser.error(
            "unknown ablation(s): "
            + ", ".join(str(value) for value in invalid_ablations)
            + "; valid values are none,no_conservation,no_replay"
        )

    cfg = ContinualAtariConfig(
        per_game_steps=int(args.per_game_steps),
        num_envs=int(args.num_envs),
        num_steps=int(args.num_steps),
        update_epochs=int(args.update_epochs),
        num_minibatches=int(args.num_minibatches),
        lr=float(args.lr),
        clip_coef=float(args.clip_coef),
        ent_coef=float(args.ent_coef),
        vf_coef=float(args.vf_coef),
        max_grad_norm=float(args.max_grad_norm),
        anneal_lr=bool(args.anneal_lr),
        eval_episodes=int(args.eval_episodes),
        eval_envs=int(args.eval_envs),
        eval_max_steps_per_episode=int(args.eval_max_steps_per_episode),
        eval_steps_cap=int(args.eval_steps_cap),
        deploy_eval_episodes=int(args.deploy_eval_episodes),
        sentinel_eval_episodes=int(args.sentinel_eval_episodes),
        cert_frac=float(args.cert_frac),
        cert_abs_floor=float(args.cert_abs_floor),
        learned_delta_frac=float(args.learned_delta_frac),
        allowed_regression_frac=float(args.allowed_regression_frac),
        guard_coef=float(args.guard_coef),
        guard_norm=str(args.guard_norm),
        guard_batch=int(args.guard_batch),
        anchor_buffer_per_game=int(args.anchor_buffer_per_game),
        guard_tolerance=float(args.guard_tolerance),
        value_coef=float(args.value_coef),
    )
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    raw = {
        "seeds": seeds,
        "games": games,
        "config": asdict(cfg),
        "runs": {},
    }
    results_by_mode = {}
    first_seed_results = {}

    for seed in seeds:
        seed_results = {}
        baseline = run_atari_baseline(games, cfg, seed)
        _print_result(baseline)
        seed_results[baseline["mode"]] = baseline
        results_by_mode.setdefault(baseline["mode"], []).append(baseline)

        pmac = run_atari_pmac(games, cfg, seed, None)
        _print_result(pmac)
        seed_results[pmac["mode"]] = pmac
        results_by_mode.setdefault(pmac["mode"], []).append(pmac)

        for ablation in ablations:
            if ablation is None:
                continue
            result = run_atari_pmac(games, cfg, seed, ablation)
            _print_result(result)
            seed_results[result["mode"]] = result
            results_by_mode.setdefault(result["mode"], []).append(result)

        if not first_seed_results:
            first_seed_results = dict(seed_results)
        raw["runs"][str(seed)] = {
            "results": {mode: _jsonify_result(result) for mode, result in seed_results.items()}
        }

    raw["aggregate"] = _aggregate(results_by_mode)
    results_path = out_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)
    plot_path = out_dir / "comparison.png"
    _plot_results(first_seed_results, games, plot_path)
    print(f"wrote {results_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()


__all__ = [
    "ALLOWED_ATARI_ABLATIONS",
    "ALLOWED_GUARD_NORMS",
    "AtariAnchorBuffer",
    "ContinualAtariConfig",
    "compute_atari_metrics",
    "evaluate_all_games",
    "evaluate_deployment",
    "run_atari_baseline",
    "run_atari_pmac",
]
