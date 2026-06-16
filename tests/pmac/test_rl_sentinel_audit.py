import numpy as np
import pytest

from pmac.atlas import Atlas
from pmac.checkpoint import Champion, SafeCheckpoint
from pmac.deployment import InvariantViolation
from pmac.rl_audit import (
    CURRENT_REGRESSION,
    HARD_FAILURE,
    PASS,
    apply_audit_actions,
    audit_atlas,
    audit_skill,
)
from pmac.rl_sentinels import RLSentinelSet, build_rl_sentinels


def _sentinels(skill_id="s", *, best=100.0, allowed=2.0, current_allowed=10.0):
    return RLSentinelSet(
        skill_id=skill_id,
        env_name="Breakout",
        game_id=0,
        eval_seeds=[10, 11, 12, 13, 14],
        random_score=0.0,
        best_score=best,
        allowed_regression=allowed,
        current_allowed_regression=current_allowed,
        eval_episodes_fast=2,
        eval_episodes_full=5,
    )


def _evaluate(scores):
    def evaluate_fn(params, skill):
        return scores[(str(params), str(skill.skill_id))]

    return evaluate_fn


def _atlas_with_node(skill_id="s", champion_params="champion", sentinels=None):
    atlas = Atlas()
    champion = None
    if champion_params is not None:
        champion = Champion(
            params=champion_params,
            route=f"{skill_id}-champion",
            meta={"skill_id": skill_id},
        )
    atlas.create_or_update_node(
        skill_id,
        skill_id,
        anchors=None,
        sentinels=_sentinels(skill_id) if sentinels is None else sentinels,
        champion_ref=champion,
        best_score=100.0,
        allowed_regression=2.0,
    )
    return atlas


def test_current_regression_uses_champion_fallback_and_marks_repair():
    sentinels = _sentinels()
    result = audit_skill(
        sentinels,
        "current",
        "champion",
        _evaluate({("current", "s"): 85.0, ("champion", "s"): 100.0}),
    )

    assert result.status == CURRENT_REGRESSION
    assert result.deployed_score == 100.0
    assert result.regressed is True

    atlas = _atlas_with_node()
    restored, summary = apply_audit_actions(
        atlas,
        {"s": result},
        SafeCheckpoint({"w": np.asarray([1.0], dtype=np.float32)}),
    )

    node = atlas.nodes["s"]
    assert restored is None
    assert summary["current_regression"] == ["s"]
    assert node.needs_repair is True
    assert node.current_certified is False
    assert node.fallback_route_id == "s-champion"


def test_hard_failure_when_current_and_champion_fail_and_missing_champion_raises():
    sentinels = _sentinels()
    result = audit_skill(
        sentinels,
        "current",
        "champion",
        _evaluate({("current", "s"): 80.0, ("champion", "s"): 90.0}),
    )

    assert result.status == HARD_FAILURE
    assert result.deployed_score == 90.0

    atlas = _atlas_with_node()
    restored, summary = apply_audit_actions(
        atlas,
        {"s": result},
        SafeCheckpoint({"w": np.asarray([1.0], dtype=np.float32)}),
    )
    assert restored is not None
    assert summary["hard_failure"] == ["s"]

    missing_champion_atlas = _atlas_with_node(champion_params=None)
    with pytest.raises(InvariantViolation):
        apply_audit_actions(
            missing_champion_atlas,
            {"s": result},
            SafeCheckpoint({"w": np.asarray([1.0], dtype=np.float32)}),
        )


def test_hard_failure_restores_params_from_safe_checkpoint():
    result = audit_skill(
        _sentinels(),
        "current",
        "champion",
        _evaluate({("current", "s"): 80.0, ("champion", "s"): 90.0}),
    )
    checkpoint = SafeCheckpoint({"w": np.asarray([1.0, 2.0], dtype=np.float32)})

    restored, summary = apply_audit_actions(_atlas_with_node(), {"s": result}, checkpoint)

    assert summary["restored"] is True
    assert np.asarray(restored["w"]).tolist() == [1.0, 2.0]


def test_pass_certifies_current_and_clears_repair():
    result = audit_skill(
        _sentinels(),
        "current",
        "champion",
        _evaluate({("current", "s"): 95.0, ("champion", "s"): 100.0}),
    )
    atlas = _atlas_with_node()
    atlas.nodes["s"].needs_repair = True
    atlas.nodes["s"].current_certified = False

    restored, summary = apply_audit_actions(
        atlas,
        {"s": result},
        SafeCheckpoint({"w": np.asarray([1.0], dtype=np.float32)}),
    )

    assert result.status == PASS
    assert restored is None
    assert summary["pass"] == ["s"]
    assert atlas.nodes["s"].current_certified is True
    assert atlas.nodes["s"].needs_repair is False


def test_build_rl_sentinels_regression_tolerances_use_nonnegative_gap():
    sentinels = build_rl_sentinels(
        "s",
        1,
        "Breakout",
        random_score=10.0,
        best_score=60.0,
        allowed_regression_frac=0.02,
        current_allowed_regression_frac=0.10,
        n_fast=4,
        n_full=8,
        seed=7,
    )

    assert sentinels.allowed_regression == pytest.approx(1.0)
    assert sentinels.current_allowed_regression == pytest.approx(5.0)
    assert len(sentinels.eval_seeds) == 8

    degenerate = build_rl_sentinels("s", 1, "Breakout", random_score=10.0, best_score=5.0)
    assert degenerate.allowed_regression == 0.0
    assert degenerate.current_allowed_regression == 0.0


def test_audit_atlas_audits_all_protected_rl_sentinel_nodes():
    atlas = Atlas()
    atlas.create_or_update_node(
        "a",
        "a",
        anchors=None,
        sentinels=_sentinels("a"),
        champion_ref=Champion(params="champion-a", route="a", meta={"skill_id": "a"}),
    )
    atlas.create_or_update_node(
        "b",
        "b",
        anchors=None,
        sentinels=_sentinels("b"),
        champion_ref=Champion(params="champion-b", route="b", meta={"skill_id": "b"}),
    )
    atlas.create_or_update_node(
        "learning",
        "learning",
        anchors=None,
        sentinels=_sentinels("learning"),
        status="learning",
    )

    results = audit_atlas(
        atlas,
        "current",
        {"a": "champion-a", "b": "champion-b"},
        _evaluate(
            {
                ("current", "a"): 100.0,
                ("champion-a", "a"): 100.0,
                ("current", "b"): 80.0,
                ("champion-b", "b"): 100.0,
            }
        ),
    )

    assert set(results) == {"a", "b"}
    assert all(isinstance(result.skill_id, str) for result in results.values())
    assert results["a"].status == PASS
    assert results["b"].status == CURRENT_REGRESSION
