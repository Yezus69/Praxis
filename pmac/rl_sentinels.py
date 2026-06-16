"""Fixed RL sentinel sets for PMA-C online audits."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass
class RLSentinelSet:
    skill_id: str
    env_name: str
    game_id: int
    eval_seeds: list[int]
    random_score: float
    best_score: float
    allowed_regression: float
    current_allowed_regression: float
    eval_episodes_fast: int
    eval_episodes_full: int


def build_rl_sentinels(
    skill_id,
    game_id,
    env_name,
    *,
    random_score,
    best_score,
    allowed_regression_frac=0.02,
    current_allowed_regression_frac=0.10,
    n_fast=4,
    n_full=20,
    seed=0,
) -> RLSentinelSet:
    """Construct a fixed RL sentinel set.

    Regression tolerances are fractions of max(best_score - random_score, 0).
    """
    n_fast = int(n_fast)
    n_full = int(n_full)
    if n_fast < 0 or n_full < 0:
        raise ValueError("sentinel episode counts must be non-negative")

    score_gap = max(float(best_score) - float(random_score), 0.0)
    n_seeds = max(n_fast, n_full)
    rng = np.random.default_rng(int(seed))
    eval_seeds = [
        int(value)
        for value in rng.integers(0, np.iinfo(np.int32).max, size=n_seeds, dtype=np.int64)
    ]
    return RLSentinelSet(
        skill_id=str(skill_id),
        env_name=str(env_name),
        game_id=int(game_id),
        eval_seeds=eval_seeds,
        random_score=float(random_score),
        best_score=float(best_score),
        allowed_regression=float(allowed_regression_frac) * score_gap,
        current_allowed_regression=float(current_allowed_regression_frac) * score_gap,
        eval_episodes_fast=n_fast,
        eval_episodes_full=n_full,
    )


__all__ = ["RLSentinelSet", "build_rl_sentinels"]
