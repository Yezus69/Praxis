"""Risk-normalized guard allocation and old-game review sampling."""

from __future__ import annotations

import numpy as np


def forgetting_risk(best, current, random_score, eps=1e-6) -> float:
    """Return raw-score forgetting pressure for one protected game."""
    best = float(best)
    current = float(current)
    random_score = float(random_score)
    eps = float(eps)
    return float(max(0.0, (best - current) / (abs(best - random_score) + eps)))  # spec §16


def risk_score(forget_risk, violation_rate, alpha_v=1.0, rho_floor=0.05) -> float:
    """Combine forgetting, conservation violations, and floor pressure."""
    return float(float(forget_risk) + float(alpha_v) * float(violation_rate) + float(rho_floor))  # spec §16


def allocate_guard(u: dict[str, float], lambda_total, eps=1e-6) -> dict[str, float]:
    """Allocate a fixed total guard budget across protected games by risk."""
    if not u:
        return {}
    denom = sum(float(value) for value in u.values()) + float(eps)
    return {
        str(game): float(lambda_total) * float(value) / denom  # spec §16
        for game, value in u.items()
    }


def review_probs(u: dict[str, float], eps=1e-6) -> dict[str, float]:
    """Return old-game review probabilities from the same risk scores as guards."""
    if not u:
        return {}
    denom = sum(float(value) for value in u.values()) + float(eps)
    return {str(game): float(value) / denom for game, value in u.items()}  # spec §18


def sample_review_games(u: dict[str, float], n, rng) -> list[str]:
    """Sample review games with replacement according to risk-normalized review probability."""
    n = int(n)
    if n <= 0 or not u:
        return []

    probs = review_probs(u)
    total_prob = sum(probs.values())
    if total_prob <= 0.0:
        return []

    games = list(probs)
    weights = np.asarray([probs[game] / total_prob for game in games], dtype=float)  # spec §18
    if hasattr(rng, "choice"):
        try:
            picked = rng.choice(games, size=n, replace=True, p=weights)
            return [str(game) for game in picked]
        except TypeError:
            pass
    if hasattr(rng, "choices"):
        return [str(game) for game in rng.choices(games, weights=weights, k=n)]
    raise TypeError("rng must provide numpy-style choice or random-style choices")


def review_loss_weights(u: dict[str, float], lambda_review) -> dict[str, float]:
    """Return coefficients for protected-game PPO review losses."""
    probs = review_probs(u)
    return {
        game: float(lambda_review) * float(prob)  # spec §18
        for game, prob in probs.items()
    }


def build_risk(per_game: dict[str, dict]) -> dict[str, float]:
    """Build risk scores from per-game raw scores and conservation violation rates."""
    out = {}
    for game, scores in per_game.items():
        forget = forgetting_risk(scores["best"], scores["current"], scores["random"])  # spec §16
        out[str(game)] = risk_score(forget, scores["violation_rate"])  # spec §16
    return out


__all__ = [
    "allocate_guard",
    "build_risk",
    "forgetting_risk",
    "review_loss_weights",
    "review_probs",
    "risk_score",
    "sample_review_games",
]
