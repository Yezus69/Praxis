from types import SimpleNamespace

import pytest

from pmac.checkpoint import Champion
from pmac.deployment import DeployedPolicy, DeploymentDecision, InvariantViolation


def _atlas(champion=None, needs_repair=False):
    node = SimpleNamespace(
        champion_ref=champion,
        needs_repair=needs_repair,
        current_certified=False,
        fallback_route_id=None,
    )
    return SimpleNamespace(nodes={"s": node})


def _champion(params=None):
    return Champion(params={"w": 1} if params is None else params, route="s", meta={"skill_id": "s"})


def test_current_certified_routes_to_current_params():
    current_params = {"w": 0}
    policy = DeployedPolicy(current_params, _atlas(_champion()))

    decision = policy.select_route("s", current_certified=True)

    assert decision.route_type == "current"
    assert decision.fallback_used is False
    assert policy.resolve_params(decision) is current_params


def test_uncertified_current_routes_to_champion_params():
    champion_params = {"w": 1}
    policy = DeployedPolicy({"w": 0}, _atlas(_champion(champion_params)))

    decision = policy.select_route("s", current_certified=False)

    assert decision.route_type == "champion"
    assert decision.fallback_used is True
    assert policy.resolve_params(decision) is champion_params


def test_missing_champion_when_uncertified_raises_invariant_violation():
    policy = DeployedPolicy({"w": 0}, _atlas(champion=None))

    with pytest.raises(InvariantViolation):
        policy.select_route("s", current_certified=False)


def test_needs_repair_forces_champion_even_when_current_certified():
    champion_params = {"w": 1}
    policy = DeployedPolicy({"w": 0}, _atlas(_champion(champion_params), needs_repair=True))

    decision = policy.select_route("s", current_certified=True)

    assert decision.route_type == "champion"
    assert decision.reason == "needs_repair"
    assert decision.fallback_used is True
    assert policy.resolve_params(decision) is champion_params


def test_route_usage_reports_fractions_and_per_skill_fallbacks():
    decisions = [
        DeploymentDecision("a", "current", "current", "current_certified", True, False),
        DeploymentDecision("b", "champion", "b", "current_not_certified", False, True),
    ]

    usage = DeployedPolicy.route_usage(decisions)

    assert usage["current_fraction"] == pytest.approx(0.5)
    assert usage["champion_fraction"] == pytest.approx(0.5)
    assert usage["n_current"] == 1
    assert usage["n_champion"] == 1
    assert usage["fallback_used_per_skill"] == {"a": False, "b": True}
    assert usage["routes"] == {"a": "current", "b": "champion"}
