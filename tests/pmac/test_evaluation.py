import pytest

from pmac.evaluation import (
    SkillScores,
    aggregate_retention,
    is_learned,
    make_skill_scores,
    normalized_retention,
)


def test_normalized_retention_basic_and_degenerate():
    assert normalized_retention(7.0, 10.0, 2.0) == pytest.approx(5.0 / (8.0 + 1.0e-6))
    assert normalized_retention(10.0, 10.0, 2.0) == 1.0
    assert normalized_retention(1.0, 1.0, 1.0) == 0.0
    assert normalized_retention(2.0, 1.0, 3.0) == 0.0


def test_is_learned_uses_absolute_margin():
    assert is_learned(11.1, 10.0, 1.0)
    assert not is_learned(11.0, 10.0, 1.0)


def test_aggregate_excludes_not_learned_and_empty_is_safe():
    scores = [
        SkillScores(
            "learned-a",
            best_score=10.0,
            current_score=8.0,
            champion_score=10.0,
            deployed_score=10.0,
            random_score=0.0,
            current_retention=0.8,
            deployed_retention=1.0,
            champion_retention=1.0,
            learned=True,
        ),
        SkillScores(
            "learned-b",
            best_score=20.0,
            current_score=8.0,
            champion_score=20.0,
            deployed_score=18.0,
            random_score=0.0,
            current_retention=0.4,
            deployed_retention=0.9,
            champion_retention=1.0,
            regressed_current=True,
            learned=True,
        ),
        SkillScores(
            "not-learned",
            best_score=1.0,
            current_score=1.0,
            champion_score=1.0,
            deployed_score=1.0,
            random_score=0.5,
            learned=False,
        ),
    ]

    aggregate = aggregate_retention(scores)
    assert aggregate["n_total"] == 3
    assert aggregate["n_learned"] == 2
    assert aggregate["n_not_learned"] == 1
    assert aggregate["not_learned_skills"] == ["not-learned"]
    assert aggregate["mean_current_retention"] == pytest.approx(0.6)
    assert aggregate["worst_current_retention"] == pytest.approx(0.4)
    assert aggregate["mean_deployed_retention"] == pytest.approx(0.95)
    assert aggregate["worst_deployed_retention"] == pytest.approx(0.9)
    assert aggregate["mean_champion_retention"] == pytest.approx(1.0)
    assert aggregate["worst_champion_retention"] == pytest.approx(1.0)
    assert aggregate["any_current_regressed"] is True
    assert aggregate["any_deployed_regressed"] is False
    assert aggregate["mean_best"] == pytest.approx(15.0)
    assert aggregate["mean_current"] == pytest.approx(8.0)
    assert aggregate["mean_deployed"] == pytest.approx(14.0)

    empty = aggregate_retention([])
    assert empty["n_total"] == 0
    assert empty["n_learned"] == 0
    assert empty["mean_deployed_retention"] == 0.0
    assert empty["any_deployed_regressed"] is False


def test_make_skill_scores_flags_regressions_and_champion_retention():
    scores = make_skill_scores(
        "s",
        best=10.0,
        current=8.0,
        champion=10.0,
        deployed=9.0,
        random_score=0.0,
        route_type="champion",
        current_certified=False,
        learned_delta=1.0,
        allowed_regression=1.5,
    )

    assert scores.learned is True
    assert scores.route_type == "champion"
    assert scores.regressed_current is True
    assert scores.regressed_deployed is False
    assert scores.current_retention == pytest.approx(8.0 / (10.0 + 1.0e-6))
    assert scores.deployed_retention == pytest.approx(9.0 / (10.0 + 1.0e-6))
    assert scores.champion_retention == 1.0
