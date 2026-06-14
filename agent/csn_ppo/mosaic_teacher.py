"""Minimal one-cluster champion teacher for CSN-PPO Phase 1b."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Iterable, Mapping

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


@dataclass(frozen=True)
class ClusterChampion:
    """Host-side champion snapshot for one sentinel cluster."""

    param_snapshot: Any | None
    normalizer_snapshot: Any | None
    best_coverage: float
    best_collision_rate: float
    best_return: float
    consecutive_wins: int
    policy_id: Any | None
    cluster_id: int


@dataclass(frozen=True)
class MosaicChampions:
    """Per-cluster mosaic teacher state for CSN-PPO README section 14."""

    champions: tuple[ClusterChampion, ...]
    fallback_normalizer_snapshot: Any | None = None
    fallback_param_snapshot: Any | None = None


def _snapshot_teacher(normalizer_params: Any, params: Any) -> tuple[Any, Any]:
    """Snapshot normalizer and params through the existing one-cluster helper."""

    if normalizer_params is None or params is None:
        return normalizer_params, params
    return snapshot_params(normalizer_params, params)


def _metric_value(metrics: Any, name: str) -> Any:
    if isinstance(metrics, Mapping):
        return metrics[name]
    return getattr(metrics, name)


def _is_vector_cluster_metrics(cluster_metrics: Any) -> bool:
    return (
        isinstance(cluster_metrics, Mapping)
        and "coverage" in cluster_metrics
        and "collision_rate" in cluster_metrics
        and "mean_return" in cluster_metrics
    )


def _iter_cluster_metrics(cluster_metrics: Any) -> Iterable[tuple[int, Any]]:
    if _is_vector_cluster_metrics(cluster_metrics):
        num_clusters = int(np.asarray(cluster_metrics["coverage"]).shape[0])
        for cluster_id in range(num_clusters):
            yield cluster_id, {
                "coverage": cluster_metrics["coverage"][cluster_id],
                "collision_rate": cluster_metrics["collision_rate"][cluster_id],
                "mean_return": cluster_metrics["mean_return"][cluster_id],
            }
        return

    if isinstance(cluster_metrics, Mapping):
        for cluster_id, metrics in cluster_metrics.items():
            yield int(cluster_id), metrics
        return

    for index, metrics in enumerate(cluster_metrics):
        if isinstance(metrics, Mapping) and "cluster_id" in metrics:
            cluster_id = metrics["cluster_id"]
        else:
            cluster_id = getattr(metrics, "cluster_id", index)
        yield int(cluster_id), metrics


def init_mosaic_champions(num_clusters: int) -> MosaicChampions:
    """Initialize one empty champion slot per sentinel cluster.

    README section 14: the mosaic teacher keeps an independent champion per
    cluster. Empty slots start with no snapshot and unreachable best scores so
    the first qualifying policy can promote after the required patience.
    """

    champions = tuple(
        ClusterChampion(
            param_snapshot=None,
            normalizer_snapshot=None,
            best_coverage=float("-inf"),
            best_collision_rate=float("inf"),
            best_return=float("-inf"),
            consecutive_wins=0,
            policy_id=None,
            cluster_id=cluster_id,
        )
        for cluster_id in range(num_clusters)
    )
    return MosaicChampions(champions=champions)


def _maybe_update_one_cluster(
    metrics: Any,
    current_params: Any,
    current_normalizer: Any,
    champion: ClusterChampion,
    config: Any,
    current_policy_id: Any,
) -> ClusterChampion:
    """README section 14 per-cluster champion update rule."""

    champion_min_margin = float(getattr(config, "champion_min_margin", 0.02))
    champion_patience = int(getattr(config, "champion_patience", 3))

    coverage = float(_metric_value(metrics, "coverage"))
    collision_rate = float(_metric_value(metrics, "collision_rate"))
    mean_return = float(_metric_value(metrics, "mean_return"))

    better_success = coverage > champion.best_coverage + champion_min_margin
    no_collision_regression = collision_rate <= champion.best_collision_rate + 0.01
    better_return = mean_return > champion.best_return

    if better_success and no_collision_regression and better_return:
        consecutive_wins = champion.consecutive_wins + 1
    else:
        consecutive_wins = 0

    if consecutive_wins >= champion_patience:
        normalizer_snapshot, params_snapshot = _snapshot_teacher(current_normalizer, current_params)
        return ClusterChampion(
            param_snapshot=params_snapshot,
            normalizer_snapshot=normalizer_snapshot,
            best_coverage=coverage,
            best_collision_rate=collision_rate,
            best_return=mean_return,
            consecutive_wins=0,
            policy_id=current_policy_id,
            cluster_id=champion.cluster_id,
        )

    return replace(champion, consecutive_wins=consecutive_wins)


def maybe_update_champions(
    cluster_metrics: Any,
    current_params: Any,
    current_normalizer: Any,
    champions: MosaicChampions,
    config: Any,
    current_policy_id: Any,
) -> MosaicChampions:
    """Apply the README section 14 champion rule independently per cluster."""

    updated = list(champions.champions)
    for cluster_id, metrics in _iter_cluster_metrics(cluster_metrics):
        if cluster_id < 0 or cluster_id >= len(updated):
            raise IndexError(f"cluster_id {cluster_id} outside champion table")
        updated[cluster_id] = _maybe_update_one_cluster(
            metrics=metrics,
            current_params=current_params,
            current_normalizer=current_normalizer,
            champion=updated[cluster_id],
            config=config,
            current_policy_id=current_policy_id,
        )

    fallback_param_snapshot = champions.fallback_param_snapshot
    fallback_normalizer_snapshot = champions.fallback_normalizer_snapshot
    if fallback_param_snapshot is None or fallback_normalizer_snapshot is None:
        normalizer_snapshot, params_snapshot = _snapshot_teacher(current_normalizer, current_params)
        if fallback_param_snapshot is None:
            fallback_param_snapshot = params_snapshot
        if fallback_normalizer_snapshot is None:
            fallback_normalizer_snapshot = normalizer_snapshot

    return MosaicChampions(
        champions=tuple(updated),
        fallback_normalizer_snapshot=fallback_normalizer_snapshot,
        fallback_param_snapshot=fallback_param_snapshot,
    )


def get_cluster_teacher(champions: MosaicChampions, cluster_id: int) -> tuple[Any | None, Any | None]:
    """Return the teacher snapshot for labeling one cluster's memory atoms."""

    champion = champions.champions[int(cluster_id)]
    if champion.param_snapshot is not None or champion.normalizer_snapshot is not None:
        return champion.normalizer_snapshot, champion.param_snapshot
    return champions.fallback_normalizer_snapshot, champions.fallback_param_snapshot
