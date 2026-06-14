"""Minimal one-cluster champion teacher for CSN-PPO Phase 1b."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import numpy as np


@dataclass
class ChampionState:
    normalizer_params: Any | None
    params: Any | None
    champion_best_coverage: float = float("-inf")
    champion_wins: int = 0


def init_champion() -> ChampionState:
    return ChampionState(normalizer_params=None, params=None)


def has_champion(champion: ChampionState) -> bool:
    return champion.params is not None and champion.normalizer_params is not None


def _copy_tree_to_host(tree):
    return jax.tree_util.tree_map(lambda x: np.array(jax.device_get(x), copy=True), tree)


def snapshot_params(normalizer_params, params):
    """Copies the frozen normalizer/policy/value champion snapshot to the host."""
    normalizer_snapshot = _copy_tree_to_host(normalizer_params)
    policy_snapshot = _copy_tree_to_host(params.policy)
    value_snapshot = _copy_tree_to_host(params.value)
    return normalizer_snapshot, params.replace(policy=policy_snapshot, value=value_snapshot)


def teacher_snapshot(champion: ChampionState, normalizer_params, params):
    """Returns champion teacher params, falling back to current params before one exists."""
    if has_champion(champion):
        return champion.normalizer_params, champion.params
    return normalizer_params, params


def _coverage_from_metrics(eval_metrics) -> float:
    for key in ("eval/episode_coverage", "eval/coverage"):
        if key in eval_metrics:
            return float(np.asarray(eval_metrics[key]))
    return float("nan")


def maybe_update_champion(champion, eval_metrics, normalizer_params, params, cfg):
    """README §14 champion ratchet using deterministic eval coverage."""
    eval_coverage = _coverage_from_metrics(eval_metrics)
    if not np.isfinite(eval_coverage):
        return ChampionState(
            normalizer_params=champion.normalizer_params,
            params=champion.params,
            champion_best_coverage=champion.champion_best_coverage,
            champion_wins=0,
        ), {"mosaic/champion_updates": 0.0, "mosaic/champion_wins": 0.0}

    if eval_coverage > champion.champion_best_coverage + cfg.champion_min_margin:
        wins = champion.champion_wins + 1
    else:
        wins = 0

    updated = wins >= cfg.champion_patience
    if updated:
        normalizer_snapshot, params_snapshot = snapshot_params(normalizer_params, params)
        next_champion = ChampionState(
            normalizer_params=normalizer_snapshot,
            params=params_snapshot,
            champion_best_coverage=eval_coverage,
            champion_wins=0,
        )
    else:
        next_champion = ChampionState(
            normalizer_params=champion.normalizer_params,
            params=champion.params,
            champion_best_coverage=champion.champion_best_coverage,
            champion_wins=wins,
        )

    return next_champion, {
        "mosaic/champion_updates": 1.0 if updated else 0.0,
        "mosaic/champion_wins": float(next_champion.champion_wins),
        "mosaic/champion_best_coverage": float(next_champion.champion_best_coverage),
        "mosaic/has_champion": 1.0 if has_champion(next_champion) else 0.0,
    }
