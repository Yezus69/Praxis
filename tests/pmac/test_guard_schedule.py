import numpy as np
import pytest

from pmac.guard_schedule import (
    allocate_guard,
    build_risk,
    forgetting_risk,
    review_loss_weights,
    review_probs,
    risk_score,
    sample_review_games,
)


def test_forgetting_risk_uses_raw_scores_and_clamps_only_risk_floor():
    assert forgetting_risk(10.0, 10.0, 2.0) == 0.0
    assert forgetting_risk(10.0, 7.0, 2.0) == pytest.approx(3.0 / (8.0 + 1.0e-6))
    assert forgetting_risk(10.0, 12.0, 2.0) == 0.0


def test_allocate_guard_normalizes_budget_and_preserves_risk_order():
    u = {"pong": 0.25, "breakout": 0.75}
    allocation = allocate_guard(u, lambda_total=2.0)

    assert sum(allocation.values()) == pytest.approx(2.0, abs=3.0e-6)
    assert allocation["breakout"] > allocation["pong"]
    assert allocate_guard({}, lambda_total=2.0) == {}

    risks = build_risk(
        {
            "pong": {"best": 10.0, "current": 10.0, "random": 0.0, "violation_rate": 0.0},
            "breakout": {"best": 4.0, "current": 4.0, "random": 0.0, "violation_rate": 0.0},
        }
    )
    assert all(value > 0.0 for value in risks.values())
    assert risk_score(0.0, 0.0) == pytest.approx(0.05)


def test_review_probs_sampling_and_loss_weights_follow_same_risk_scores():
    u = {"pong": 1.0, "breakout": 3.0}
    probs = review_probs(u)

    assert sum(probs.values()) == pytest.approx(1.0)
    assert probs["breakout"] > probs["pong"]
    assert sample_review_games(u, 0, np.random.default_rng(0)) == []

    rng = np.random.default_rng(7)
    samples = sample_review_games(u, 4000, rng)
    breakout_freq = samples.count("breakout") / len(samples)
    assert breakout_freq == pytest.approx(0.75, abs=0.04)

    weights = review_loss_weights(u, lambda_review=0.2)
    assert weights["pong"] == pytest.approx(0.2 * probs["pong"])
    assert weights["breakout"] == pytest.approx(0.2 * probs["breakout"])
