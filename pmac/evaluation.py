"""PMA-C score accounting and deployed-vs-current retention metrics."""

from __future__ import annotations

from dataclasses import dataclass


LEARNED_DELTA_DEFAULT = 1.0e-3
"""Default absolute margin over random used by generic callers."""


@dataclass
class SkillScores:
    skill_id: str
    best_score: float
    current_score: float
    champion_score: float
    deployed_score: float
    random_score: float | None = None
    current_retention: float = 0.0
    deployed_retention: float = 0.0
    champion_retention: float = 0.0
    regressed_current: bool = False
    regressed_deployed: bool = False
    learned: bool = True
    route_type: str = "current"


RETENTION_CLIP_MAX = 1.5
"""Report-side clip for random-normalized retention (matches compute_atari_metrics).

A skill that forgot *below* its random baseline reads 0.0 (retained 0% of skill), and a
near-random denominator cannot explode the ratio into a huge magnitude. Raw scores
(best/current/champion/deployed/random) are always stored, so raw ratios are recoverable.
"""


def normalized_retention(score, best, random_score, eps=1e-6) -> float:
    """Return random-normalized retention, clipped to [0, RETENTION_CLIP_MAX].

    Exact best-score retention is 1.0 (the route-to-champion no-forgetting guarantee), and a
    degenerate best<=random returns 0.0.
    """
    score = float(score)
    best = float(best)
    random_score = float(random_score)
    if best <= random_score:
        return 0.0
    if score == best:
        return 1.0
    ratio = (score - random_score) / (best - random_score + float(eps))
    return float(min(max(ratio, 0.0), RETENTION_CLIP_MAX))


def is_learned(best, random_score, learned_delta) -> bool:
    """Return whether best is above random by an absolute learned margin."""
    return bool(float(best) > float(random_score) + float(learned_delta))


def make_skill_scores(
    skill_id,
    *,
    best,
    current,
    champion,
    deployed,
    random_score,
    route_type,
    current_certified,
    learned_delta,
    allowed_regression,
) -> SkillScores:
    """Build per-skill current, champion, and deployed retention accounting."""
    del current_certified
    best = float(best)
    current = float(current)
    champion = float(champion)
    deployed = float(deployed)
    random_score = float(random_score)
    allowed_regression = float(allowed_regression)
    return SkillScores(
        skill_id=str(skill_id),
        best_score=best,
        current_score=current,
        champion_score=champion,
        deployed_score=deployed,
        random_score=random_score,
        current_retention=normalized_retention(current, best, random_score),
        deployed_retention=normalized_retention(deployed, best, random_score),
        champion_retention=normalized_retention(champion, best, random_score),
        regressed_current=bool(current < best - allowed_regression),
        regressed_deployed=bool(deployed < best - allowed_regression),
        learned=is_learned(best, random_score, learned_delta),
        route_type=str(route_type),
    )


def _mean(values) -> float:
    values = list(values)
    if not values:
        return 0.0
    return float(sum(float(value) for value in values) / float(len(values)))


def aggregate_retention(scores: list[SkillScores]) -> dict:
    """Aggregate retention over learned skills only."""
    scores = list(scores)
    learned_scores = [score for score in scores if bool(score.learned)]
    not_learned = [score.skill_id for score in scores if not bool(score.learned)]
    out = {
        "n_learned": int(len(learned_scores)),
        "n_total": int(len(scores)),
        "n_not_learned": int(len(not_learned)),
        "not_learned_skills": list(not_learned),
    }
    if not learned_scores:
        out.update(
            {
                "mean_current_retention": 0.0,
                "worst_current_retention": 0.0,
                "mean_deployed_retention": 0.0,
                "worst_deployed_retention": 0.0,
                "mean_champion_retention": 0.0,
                "worst_champion_retention": 0.0,
                "any_deployed_regressed": False,
                "any_current_regressed": False,
                "mean_best": 0.0,
                "mean_current": 0.0,
                "mean_deployed": 0.0,
            }
        )
        return out

    current_retention = [score.current_retention for score in learned_scores]
    deployed_retention = [score.deployed_retention for score in learned_scores]
    champion_retention = [score.champion_retention for score in learned_scores]
    out.update(
        {
            "mean_current_retention": _mean(current_retention),
            "worst_current_retention": float(min(current_retention)),
            "mean_deployed_retention": _mean(deployed_retention),
            "worst_deployed_retention": float(min(deployed_retention)),
            "mean_champion_retention": _mean(champion_retention),
            "worst_champion_retention": float(min(champion_retention)),
            "any_deployed_regressed": bool(
                any(score.regressed_deployed for score in learned_scores)
            ),
            "any_current_regressed": bool(any(score.regressed_current for score in learned_scores)),
            "mean_best": _mean(score.best_score for score in learned_scores),
            "mean_current": _mean(score.current_score for score in learned_scores),
            "mean_deployed": _mean(score.deployed_score for score in learned_scores),
        }
    )
    return out


__all__ = [
    "LEARNED_DELTA_DEFAULT",
    "SkillScores",
    "aggregate_retention",
    "is_learned",
    "make_skill_scores",
    "normalized_retention",
]
