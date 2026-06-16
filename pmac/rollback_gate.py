"""Rollback acceptance gate for protected-game retention and current-game validation."""

from __future__ import annotations

from dataclasses import dataclass
import math

from pmac.evaluation import normalized_retention


@dataclass
class GateConfig:
    r_min: float = 0.9
    current_regress_frac: float = 0.1
    delta_abs: dict[str, float] | float = 0.0
    max_violation_rate: float = 0.25
    retrieval_floor: float = 0.0
    min_new_progress: float = 0.0


@dataclass
class GateDecision:
    accept: bool
    regressed_games: list[str]
    reasons: list[str]


def _delta_for(delta_abs: dict[str, float] | float, game: str) -> float:
    if isinstance(delta_abs, dict):
        return float(delta_abs.get(game, 0.0))
    return float(delta_abs)


def _is_finite(value) -> bool:
    if value is None:
        return False
    return math.isfinite(float(value))


def evaluate_gate(
    *,
    protected: dict[str, dict],
    current: dict,
    violation_rate: float,
    retrieval_alignment: float,
    cfg: GateConfig,
) -> GateDecision:
    """Accept iff protected retention, validation, conservation, retrieval, and progress pass.

    ``retrieval_alignment`` is a quality scalar where higher is better. If the caller has a
    contrastive retrieval loss instead, it should pass the negated loss.
    """
    regressed_games = []
    reasons = []

    for game, scores in protected.items():
        game = str(game)
        score = float(scores["current"])
        best = float(scores["best"])
        random_score = float(scores["random"])
        retention = normalized_retention(score, best, random_score)  # spec §19
        delta_g = _delta_for(cfg.delta_abs, game)
        if retention < float(cfg.r_min) or score < best - delta_g:  # spec §19
            regressed_games.append(game)

    if regressed_games:
        reasons.append("protected_regression")

    val_best = current.get("val_best")
    if _is_finite(val_best):
        val_best = float(val_best)
        val_current = float(current["val_current"])
        random_score = float(current["random"])
        tolerance = float(cfg.current_regress_frac) * max(val_best - random_score, 1.0e-6)  # spec §26
        if not math.isfinite(val_current) or val_current < val_best - tolerance:  # spec §26
            reasons.append("current_val_regression")

    if float(violation_rate) > float(cfg.max_violation_rate):  # spec §19
        reasons.append("violation_rate")

    if float(retrieval_alignment) < float(cfg.retrieval_floor):  # spec §19
        reasons.append("retrieval_alignment")

    if float(current["progress"]) < float(cfg.min_new_progress):  # spec §19
        reasons.append("new_game_progress")

    return GateDecision(
        accept=not reasons,
        regressed_games=regressed_games,
        reasons=reasons,
    )


def on_reject_actions(decision) -> dict:
    """Return follow-up actions for the training loop after a rejected candidate."""
    return {
        "increase_risk_games": list(decision.regressed_games),
        "increase_review_games": list(decision.regressed_games),
        "write_failure_memories": True,
        "raise_retrieval_confidence": True,
    }


__all__ = [
    "GateConfig",
    "GateDecision",
    "evaluate_gate",
    "on_reject_actions",
]
