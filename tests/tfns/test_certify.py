from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import numpy as np

from tfns.config import ConsolidateConfig, TFNSConfig
from tfns.consolidate.certify import (
    closed_loop_gate,
    is_learned,
    random_normalized_progress,
)


def test_random_normalized_progress_handles_negative_score_games():
    progress = random_normalized_progress(-10.0, S_random=-20.0, S_single=-5.0)
    np.testing.assert_allclose(progress, 10.0 / (15.0 + 1.0e-8))


def test_is_learned_rejects_weak_or_unstable_and_accepts_strong_stable():
    cfg = TFNSConfig(consolidate=ConsolidateConfig(learned_threshold=0.9, stable_windows=2))

    weak, weak_info = is_learned([6.0, 6.5], 0.0, 10.0, cfg)
    assert weak is False
    assert weak_info["progress_pass"] is False

    unstable, unstable_info = is_learned([9.5, 7.5], 0.0, 10.0, cfg)
    assert unstable is False
    assert unstable_info["progress_pass"] is False

    strong, strong_info = is_learned([9.2, 9.4], 0.0, 10.0, cfg)
    assert strong is True
    assert strong_info["stable_pass"] is True


def test_is_learned_accepts_noisy_recent_windows_when_all_clear_threshold():
    cfg = {"learned_threshold": 0.9, "stable_windows": 2, "stability_std_max": 0.01}

    learned, info = is_learned([9.1, 11.0], S_random=0.0, S_single=10.0, cfg=cfg)

    assert info["recent_progress_std"] > cfg["stability_std_max"]
    assert info["progress_pass"] is True
    assert info["stable_pass"] is True
    assert learned is True


def test_is_learned_mean_stability_relaxation_still_requires_progress_pass():
    cfg = {"learned_threshold": 0.8, "stable_windows": 2, "stability_std_max": 0.01}

    learned, info = is_learned([10.1, 7.1], S_random=0.0, S_single=10.0, cfg=cfg)

    assert info["recent_progress_mean"] >= 0.85
    assert info["recent_progress_std"] > cfg["stability_std_max"]
    assert info["stable_pass"] is True
    assert info["progress_pass"] is False
    assert learned is False


def test_is_learned_requires_score_above_random_margin():
    cfg = TFNSConfig(consolidate=ConsolidateConfig(learned_threshold=0.9, stable_windows=2))

    learned, info = is_learned([0.0, 0.0], S_random=0.0, S_single=0.0, cfg=cfg)

    assert learned is False
    assert info["random_margin_pass"] is False


def test_closed_loop_gate_rejects_any_retention_below_threshold_and_accepts_all_pass():
    cfg = TFNSConfig(
        consolidate=ConsolidateConfig(
            learned_threshold=0.9,
            stable_windows=2,
            retention_accept=0.9,
        )
    )

    reject, reject_info = closed_loop_gate({"a": 0.95, "b": 0.89}, 0.92, cfg)
    assert reject is False
    assert reject_info["worst_game"] == "b"
    assert reject_info["per_game"]["b"]["pass"] is False

    accept, accept_info = closed_loop_gate({"a": 0.95, "b": 0.90}, 0.92, cfg)
    assert accept is True
    assert accept_info["all_games_pass"] is True


def test_closed_loop_gate_min_score_orientation_rejects_below_threshold_current_score():
    cfg = TFNSConfig(
        consolidate=ConsolidateConfig(
            learned_threshold=0.9,
            stable_windows=2,
            retention_accept=0.9,
        )
    )

    accepted, info = closed_loop_gate({"old": 1.0}, current_progress=0.89, cfg=cfg)

    assert accepted is False
    assert info["current_pass"] is False
