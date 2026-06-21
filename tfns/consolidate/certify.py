"""Closed-loop certification gates for consolidation."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np


def _get(obj: Any, name: str, default: Any) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _consolidate_cfg(cfg: Any) -> Any:
    return _get(cfg, "consolidate", cfg)


def random_normalized_progress(S: Any, S_random: Any, S_single: Any, eps: float = 1.0e-8) -> Any:
    """Return section-18 random-normalized progress.

    The denominator is the matched single-task score improvement over random,
    so negative-score games keep the same orientation.
    """

    return (np.asarray(S) - np.asarray(S_random)) / (
        np.asarray(S_single) - np.asarray(S_random) + float(eps)
    )


def is_learned(
    score_windows: Any,
    S_random: float,
    S_single: float,
    cfg: Any,
) -> tuple[bool, dict[str, Any]]:
    """Return whether recent independent evaluation windows certify learning."""

    c_cfg = _consolidate_cfg(cfg)
    threshold = float(_get(c_cfg, "learned_threshold", 0.9))
    stable_windows = int(_get(c_cfg, "stable_windows", 2))
    stability_std_max = float(_get(c_cfg, "stability_std_max", max(0.05, 0.10 * threshold)))
    margin_frac = float(_get(c_cfg, "random_margin_frac", 0.01))

    scores = np.asarray(score_windows, dtype=np.float64).reshape(-1)
    progress = random_normalized_progress(scores, S_random, S_single)
    all_finite = bool(
        scores.size > 0
        and np.all(np.isfinite(scores))
        and np.isfinite(float(S_random))
        and np.isfinite(float(S_single))
        and np.all(np.isfinite(progress))
    )
    enough_windows = bool(scores.size >= stable_windows)

    if enough_windows:
        recent_scores = scores[-stable_windows:]
        recent_progress = progress[-stable_windows:]
    else:
        recent_scores = scores
        recent_progress = progress

    recent_progress_mean = float(np.mean(recent_progress)) if recent_progress.size else float("nan")
    recent_progress_std = float(np.std(recent_progress)) if recent_progress.size else float("nan")
    recent_progress_cleared = bool(np.all(recent_progress >= threshold))
    progress_pass = bool(
        enough_windows
        and all_finite
        and recent_progress_cleared
        and recent_progress_mean >= threshold
    )
    stability_pass = bool(enough_windows and all_finite and recent_progress_std <= stability_std_max)
    all_recent_windows_cleared = bool(enough_windows and all_finite and recent_progress_cleared)
    high_mean_progress_pass = bool(
        enough_windows and all_finite and recent_progress_mean >= max(threshold, 0.85)
    )
    stable_pass = bool(
        stability_pass or all_recent_windows_cleared or high_mean_progress_pass
    )
    score_margin = max(1.0e-6, margin_frac * abs(float(S_single) - float(S_random)))
    random_margin_pass = bool(
        enough_windows and all_finite and float(np.mean(recent_scores)) >= float(S_random) + score_margin
    )
    learned = bool(all_finite and progress_pass and stable_pass and random_margin_pass)

    info = {
        "learned": learned,
        "threshold": threshold,
        "stable_windows": stable_windows,
        "all_finite": all_finite,
        "enough_windows": enough_windows,
        "progress": progress.astype(np.float32),
        "recent_progress": np.asarray(recent_progress, dtype=np.float32),
        "current_progress": float(recent_progress[-1]) if recent_progress.size else float("nan"),
        "recent_progress_mean": recent_progress_mean,
        "recent_progress_std": recent_progress_std,
        "progress_pass": progress_pass,
        "stable_pass": stable_pass,
        "random_margin": score_margin,
        "random_margin_pass": random_margin_pass,
    }
    return learned, info


def _retention_value(value: Any) -> float:
    if isinstance(value, Mapping):
        if "retention" in value:
            return float(value["retention"])
        if {"current", "random", "single"} <= set(value):
            return float(random_normalized_progress(value["current"], value["random"], value["single"]))
        if {"score", "random", "single"} <= set(value):
            return float(random_normalized_progress(value["score"], value["random"], value["single"]))
        if {"current", "random", "best"} <= set(value):
            return float(random_normalized_progress(value["current"], value["random"], value["best"]))
    return float(value)


def closed_loop_gate(
    retention_by_game: Mapping[Any, Any],
    current_progress: float,
    cfg: Any,
) -> tuple[bool, dict[str, Any]]:
    """Accept only if every learned game and the current behavior pass MIN gates."""

    c_cfg = _consolidate_cfg(cfg)
    retention_threshold = float(_get(c_cfg, "retention_accept", 0.90))
    current_threshold = float(_get(c_cfg, "learned_threshold", 0.9))

    per_game: dict[Any, dict[str, Any]] = {}
    worst_game = None
    worst_retention = float("inf")
    all_games_pass = True
    all_finite = np.isfinite(float(current_progress))

    for game, raw in retention_by_game.items():
        retention = _retention_value(raw)
        finite = bool(np.isfinite(retention))
        passed = bool(finite and retention >= retention_threshold)
        per_game[game] = {
            "retention": retention,
            "threshold": retention_threshold,
            "pass": passed,
        }
        all_games_pass = all_games_pass and passed
        all_finite = all_finite and finite
        if finite and retention < worst_retention:
            worst_game = game
            worst_retention = retention
        elif worst_game is None:
            worst_game = game

    current_pass = bool(np.isfinite(float(current_progress)) and current_progress >= current_threshold)
    accepted = bool(all_finite and all_games_pass and current_pass)
    info = {
        "accepted": accepted,
        "retention_threshold": retention_threshold,
        "current_threshold": current_threshold,
        "current_progress": float(current_progress),
        "current_pass": current_pass,
        "all_games_pass": bool(all_games_pass),
        "all_finite": bool(all_finite),
        "per_game": per_game,
        "worst_game": worst_game,
        "worst_retention": worst_retention if worst_game is not None else None,
    }
    return accepted, info


__all__ = [
    "closed_loop_gate",
    "is_learned",
    "random_normalized_progress",
]
