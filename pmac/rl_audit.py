"""Standalone RL sentinel audit and rollback actions for PMA-C."""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any

from pmac.deployment import InvariantViolation
from pmac.rl_sentinels import RLSentinelSet


PASS = "PASS"
CURRENT_REGRESSION = "CURRENT_REGRESSION"
HARD_FAILURE = "HARD_FAILURE"


@dataclass
class AuditResult:
    skill_id: str
    status: str
    current_score: float
    champion_score: float
    deployed_score: float
    regressed: bool
    reason: str


def _mode_sentinels(sentinels: RLSentinelSet, *, fast: bool) -> RLSentinelSet:
    n_eval = sentinels.eval_episodes_fast if fast else sentinels.eval_episodes_full
    return replace(sentinels, eval_seeds=list(sentinels.eval_seeds[: int(n_eval)]))


def _score_or_missing(params, sentinels: RLSentinelSet, evaluate_fn) -> tuple[float, bool]:
    if params is None:
        return float("-inf"), True
    return float(evaluate_fn(params, sentinels)), False


def audit_skill(
    sentinels: RLSentinelSet,
    current_params,
    champion_params,
    evaluate_fn,
    *,
    fast=True,
) -> AuditResult:
    """Score current and champion policies on a fixed sentinel set."""
    eval_sentinels = _mode_sentinels(sentinels, fast=bool(fast))
    current_score, current_missing = _score_or_missing(current_params, eval_sentinels, evaluate_fn)
    champion_score, champion_missing = _score_or_missing(champion_params, eval_sentinels, evaluate_fn)
    del current_missing

    best_score = float(sentinels.best_score)
    deployed_threshold = best_score - float(sentinels.allowed_regression)
    current_threshold = best_score - float(sentinels.current_allowed_regression)
    current_regressed = bool(current_score < current_threshold)
    deployed_score = champion_score if current_regressed else current_score

    if champion_missing:
        status = HARD_FAILURE
        reason = "missing champion fallback"
    elif champion_score < deployed_threshold:
        status = HARD_FAILURE
        reason = (
            f"champion_score {champion_score:.6g} < deployed threshold "
            f"{deployed_threshold:.6g}"
        )
    elif current_regressed:
        status = CURRENT_REGRESSION
        reason = (
            f"current_score {current_score:.6g} < current threshold "
            f"{current_threshold:.6g}; champion fallback passes"
        )
    else:
        status = PASS
        reason = (
            f"current_score {current_score:.6g} >= current threshold "
            f"{current_threshold:.6g}"
        )

    return AuditResult(
        skill_id=str(sentinels.skill_id),
        status=status,
        current_score=float(current_score),
        champion_score=float(champion_score),
        deployed_score=float(deployed_score),
        regressed=current_regressed,
        reason=reason,
    )


def _protected_nodes(atlas) -> list[Any]:
    if hasattr(atlas, "protected_nodes"):
        return list(atlas.protected_nodes())
    return [
        node
        for node in getattr(atlas, "nodes", {}).values()
        if getattr(node, "status", "protected") == "protected"
    ]


def _node_rl_sentinels(node) -> RLSentinelSet:
    for attr in ("rl_sentinels", "fast_sentinels", "sentinels"):
        sentinels = getattr(node, attr, None)
        if isinstance(sentinels, RLSentinelSet):
            return sentinels
    raise ValueError(f"protected skill {getattr(node, 'skill_id', '<unknown>')} has no RL sentinels")


def _champion_params_for(node, champion_params_by_skill):
    skill_id = str(getattr(node, "skill_id"))
    if champion_params_by_skill is not None and skill_id in champion_params_by_skill:
        return champion_params_by_skill[skill_id]
    champion = getattr(node, "champion_ref", None)
    return getattr(champion, "params", None)


def audit_atlas(
    atlas,
    current_params,
    champion_params_by_skill,
    evaluate_fn,
    *,
    fast=True,
) -> dict[str, AuditResult]:
    """Audit every protected atlas skill with an injected scorer."""
    results = {}
    for node in _protected_nodes(atlas):
        sentinels = _node_rl_sentinels(node)
        skill_id = str(sentinels.skill_id)
        champion_params = _champion_params_for(node, champion_params_by_skill)
        results[skill_id] = audit_skill(
            sentinels,
            current_params,
            champion_params,
            evaluate_fn,
            fast=fast,
        )
    return results


def _results_iter(audit_results):
    if isinstance(audit_results, dict):
        return list(audit_results.values())
    return list(audit_results)


def _atlas_node(atlas, skill_id: str):
    nodes = getattr(atlas, "nodes", {})
    if skill_id not in nodes:
        raise InvariantViolation(f"missing protected skill node: {skill_id}")
    return nodes[skill_id]


def _has_executable_champion(node) -> bool:
    champion = getattr(node, "champion_ref", None)
    return bool(champion is not None and getattr(champion, "params", None) is not None)


def _set_fallback_route(node, skill_id: str) -> None:
    champion = getattr(node, "champion_ref", None)
    route_id = getattr(champion, "route", None)
    if route_id is None:
        route_id = getattr(champion, "meta", {}).get("skill_id", skill_id)
    node.fallback_route_id = str(route_id)


def _increase_guard_pressure(guard_pressure, skill_ids: list[str]) -> None:
    if guard_pressure is None or not skill_ids:
        return
    if hasattr(guard_pressure, "increase"):
        guard_pressure.increase(skill_ids)
        return
    if callable(guard_pressure):
        guard_pressure(skill_ids)
        return
    if isinstance(guard_pressure, dict):
        for skill_id in skill_ids:
            guard_pressure[skill_id] = guard_pressure.get(skill_id, 0) + 1


def apply_audit_actions(atlas, audit_results, safe_checkpoint, guard_pressure=None):
    """Apply P3 audit actions to atlas state and restore from checkpoint on hard failure."""
    restored_params = None
    summary = {
        "pass": [],
        "current_regression": [],
        "hard_failure": [],
        "restored": False,
        "guard_pressure": [],
    }
    guard_pressure_skills = []

    for result in _results_iter(audit_results):
        skill_id = str(result.skill_id)
        node = _atlas_node(atlas, skill_id)
        node.current_score = float(result.current_score)
        node.fallback_score = float(result.champion_score)

        if result.status == HARD_FAILURE:
            if not _has_executable_champion(node):
                raise InvariantViolation(f"missing executable champion params for skill: {skill_id}")
            node.needs_repair = True
            node.current_certified = False
            _set_fallback_route(node, skill_id)
            summary["hard_failure"].append(skill_id)
            guard_pressure_skills.append(skill_id)
            if restored_params is None:
                restored_params = safe_checkpoint.restore()
                summary["restored"] = True
        elif result.status == CURRENT_REGRESSION:
            if not _has_executable_champion(node):
                raise InvariantViolation(f"missing executable champion params for skill: {skill_id}")
            node.needs_repair = True
            node.current_certified = False
            _set_fallback_route(node, skill_id)
            summary["current_regression"].append(skill_id)
            guard_pressure_skills.append(skill_id)
        elif result.status == PASS:
            node.current_certified = True
            node.needs_repair = False
            summary["pass"].append(skill_id)
        else:
            raise ValueError(f"unsupported audit status for {skill_id}: {result.status}")

    _increase_guard_pressure(guard_pressure, guard_pressure_skills)
    summary["guard_pressure"] = list(guard_pressure_skills)
    return restored_params, summary


__all__ = [
    "PASS",
    "CURRENT_REGRESSION",
    "HARD_FAILURE",
    "AuditResult",
    "audit_skill",
    "audit_atlas",
    "apply_audit_actions",
]
