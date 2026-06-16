import math

import numpy as np
import pytest

from pmac.anchors import AnchorStore
from pmac.atlas import Atlas
from pmac.guard_pressure import (
    GuardPressureConfig,
    GuardPressureState,
    compute_guard_lambdas,
    update_recovery,
)
from pmac.sentinels import SentinelStore


def _store():
    store = AnchorStore(capacity=1)
    store.add(
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.ones(1, dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )
    return store


def _sentinels():
    return SentinelStore(np.zeros((1, 2), dtype=np.float32), np.zeros(1, dtype=np.int32))


def _add_skill(
    atlas,
    skill_id,
    *,
    best_score=1.0,
    current_score=1.0,
    status="protected",
    interference_neighbors=None,
):
    node = atlas.create_or_update_node(
        skill_id,
        skill_id,
        _store(),
        _sentinels(),
        status=status,
        best_score=best_score,
        current_score=current_score,
    )
    if interference_neighbors:
        node.interference_neighbors.update(str(skill) for skill in interference_neighbors)
    return node


def _atlas_with_protected_count(n):
    atlas = Atlas()
    for idx in range(n):
        _add_skill(atlas, f"old-{idx}")
    _add_skill(atlas, "current", status="learning")
    return atlas


def test_total_lambda_stays_bounded_as_skill_count_increases():
    cfg = GuardPressureConfig(lambda_total=1.0)

    for n in (2, 10, 100):
        lambdas = compute_guard_lambdas(
            _atlas_with_protected_count(n),
            "current",
            cfg=cfg,
        )

        assert len(lambdas) == n
        assert sum(lambdas.values()) <= cfg.lambda_total + 1.0e-9


def test_regressed_skill_gets_higher_lambda_than_clean_skill():
    atlas = Atlas()
    _add_skill(atlas, "regressed", best_score=1.0, current_score=0.2)
    _add_skill(atlas, "clean", best_score=1.0, current_score=1.0)

    lambdas = compute_guard_lambdas(
        atlas,
        "current",
        sentinel_metrics={"regressed": 1.0, "clean": 0.0},
    )

    assert lambdas["regressed"] > lambdas["clean"]


def test_update_recovery_waits_for_patience_and_boosts_regression():
    cfg = GuardPressureConfig(
        lambda_min_per_skill=0.02,
        lambda_max_per_skill=0.5,
        up_factor=1.5,
        down_factor=0.5,
        recovery_patience=3,
    )
    state = GuardPressureState(
        skill_lambda={"clean": 0.2, "regressed": 0.2},
        recovery_count={"clean": 0, "regressed": 2},
        regression_count={"clean": 0, "regressed": 1},
        last_audit_step={"clean": 10, "regressed": 10},
    )

    once = update_recovery(state, {"clean": False}, cfg)
    twice = update_recovery(once, {"clean": False}, cfg)
    thrice = update_recovery(twice, {"clean": False}, cfg)

    assert once.skill_lambda["clean"] == pytest.approx(0.2)
    assert twice.skill_lambda["clean"] == pytest.approx(0.2)
    assert thrice.skill_lambda["clean"] == pytest.approx(0.1)
    assert thrice.recovery_count["clean"] == 3

    boosted = update_recovery(state, {"regressed": True}, cfg)

    assert boosted.skill_lambda["regressed"] == pytest.approx(0.3)
    assert boosted.recovery_count["regressed"] == 0
    assert boosted.regression_count["regressed"] == 2
    assert state.skill_lambda["regressed"] == pytest.approx(0.2)
    assert state.recovery_count["regressed"] == 2


def test_interference_neighbor_gets_higher_lambda():
    atlas = Atlas()
    _add_skill(atlas, "neighbor", interference_neighbors={"current"})
    _add_skill(atlas, "unrelated")

    lambdas = compute_guard_lambdas(atlas, "current")

    assert lambdas["neighbor"] > lambdas["unrelated"]


def test_hundred_skills_respect_length_cap_and_total_budget():
    cfg = GuardPressureConfig(lambda_total=1.0, lambda_max_per_skill=0.5)
    n = 100

    lambdas = compute_guard_lambdas(
        _atlas_with_protected_count(n),
        "current",
        cfg=cfg,
    )

    length_cap = cfg.lambda_max_per_skill / math.sqrt(n)
    assert max(lambdas.values()) <= length_cap + 1.0e-9
    assert sum(lambdas.values()) <= cfg.lambda_total + 1.0e-9
