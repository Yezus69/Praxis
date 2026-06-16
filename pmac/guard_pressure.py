"""Risk-adaptive guard pressure for protected PMA-C skills."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np


@dataclass
class GuardPressureConfig:
    lambda_total: float = 1.0
    lambda_min_per_skill: float = 0.02
    lambda_max_per_skill: float = 0.5
    risk_forgetting_weight: float = 2.0
    risk_interference_weight: float = 1.0
    risk_age_weight: float = 0.25
    risk_sentinel_weight: float = 4.0
    up_factor: float = 1.5
    down_factor: float = 0.98
    recovery_patience: int = 3


@dataclass
class GuardPressureState:
    skill_lambda: dict[str, float]
    recovery_count: dict[str, int]
    regression_count: dict[str, int]
    last_audit_step: dict[str, int]


def _protected_nodes(atlas: Any) -> list[Any]:
    if hasattr(atlas, "protected_nodes"):
        return list(atlas.protected_nodes())
    nodes = getattr(atlas, "nodes", {})
    return [node for node in nodes.values() if getattr(node, "status", "protected") == "protected"]


def _lookup(mapping: Mapping[Any, Any] | None, skill_id: str, default: float) -> Any:
    if mapping is None:
        return default
    if skill_id in mapping:
        return mapping[skill_id]
    return mapping.get(str(skill_id), default)


def _finite_float(value: Any, default: float = 0.0) -> float:
    try:
        out = float(value)
    except (TypeError, ValueError):
        return float(default)
    if not np.isfinite(out):
        return float(default)
    return out


def _unit_interval(value: Any, default: float = 0.0) -> float:
    return float(np.clip(_finite_float(value, default), 0.0, 1.0))


def compute_guard_lambdas(
    atlas: Any,
    current_skill_id: Any,
    *,
    sentinel_metrics: Mapping[Any, Any] | None = None,
    interference: Mapping[Any, Any] | None = None,
    ages: Mapping[Any, Any] | None = None,
    cfg: GuardPressureConfig = GuardPressureConfig(),
) -> dict[str, float]:
    """Compute P2.1 per-skill guard pressure for protected non-current skills."""

    current_skill_id = str(current_skill_id)
    protected = [
        node
        for node in _protected_nodes(atlas)
        if str(getattr(node, "skill_id")) != current_skill_id
    ]
    if not protected:
        return {}

    lambda_total = max(0.0, _finite_float(cfg.lambda_total, 0.0))
    if lambda_total <= 0.0:
        return {str(getattr(node, "skill_id")): 0.0 for node in protected}

    skill_ids: list[str] = []
    risks: list[float] = []
    for node in protected:
        skill_id = str(getattr(node, "skill_id"))
        skill_ids.append(skill_id)

        forgetting = max(0.0, _finite_float(node.forgetting_risk(), 0.0))
        neighbors = getattr(node, "interference_neighbors", set())
        default_interference = 1.0 if current_skill_id in neighbors else 0.0
        if interference is None:
            interference_score = default_interference
        else:
            interference_score = _unit_interval(
                _lookup(interference, skill_id, default_interference),
                default_interference,
            )
        age = _unit_interval(_lookup(ages, skill_id, 0.0), 0.0)
        sentinel = _unit_interval(_lookup(sentinel_metrics, skill_id, 0.0), 0.0)

        risk = (
            1.0
            + float(cfg.risk_forgetting_weight) * forgetting
            + float(cfg.risk_interference_weight) * interference_score
            + float(cfg.risk_age_weight) * age
            + float(cfg.risk_sentinel_weight) * sentinel
        )
        risks.append(max(0.0, risk))

    risks_arr = np.asarray(risks, dtype=np.float64)
    risk_sum = float(np.sum(risks_arr))
    if risk_sum <= 0.0:
        risks_arr = np.ones_like(risks_arr)
        risk_sum = float(np.sum(risks_arr))

    eps = 1.0e-12
    lambdas = lambda_total * risks_arr / (risk_sum + eps)

    lower = max(0.0, _finite_float(cfg.lambda_min_per_skill, 0.0))
    upper = max(0.0, _finite_float(cfg.lambda_max_per_skill, 0.0))
    lambdas = np.clip(lambdas, lower, upper)

    length_cap = upper / float(np.sqrt(len(skill_ids)))
    lambdas = np.minimum(lambdas, length_cap)

    total = float(np.sum(lambdas))
    if total > lambda_total:
        lambdas = lambdas * (lambda_total / (total + eps))

    return {skill_id: float(value) for skill_id, value in zip(skill_ids, lambdas)}


def update_recovery(
    state: GuardPressureState,
    audit_results: Mapping[Any, bool],
    cfg: GuardPressureConfig = GuardPressureConfig(),
) -> GuardPressureState:
    """Apply P2.3 recovery patience without mutating the input state."""

    skill_lambda = dict(state.skill_lambda)
    recovery_count = dict(state.recovery_count)
    regression_count = dict(state.regression_count)
    last_audit_step = dict(state.last_audit_step)

    lambda_min = max(0.0, _finite_float(cfg.lambda_min_per_skill, 0.0))
    lambda_max = max(lambda_min, _finite_float(cfg.lambda_max_per_skill, lambda_min))
    patience = max(1, int(cfg.recovery_patience))

    for raw_skill_id, skill_regressed in audit_results.items():
        skill_id = str(raw_skill_id)
        old_lambda = _finite_float(skill_lambda.get(skill_id, lambda_min), lambda_min)

        if bool(skill_regressed):
            skill_lambda[skill_id] = min(old_lambda * float(cfg.up_factor), lambda_max)
            recovery_count[skill_id] = 0
            regression_count[skill_id] = int(regression_count.get(skill_id, 0)) + 1
            continue

        clean_count = int(recovery_count.get(skill_id, 0)) + 1
        recovery_count[skill_id] = clean_count
        regression_count.setdefault(skill_id, int(regression_count.get(skill_id, 0)))
        if clean_count >= patience:
            skill_lambda[skill_id] = max(old_lambda * float(cfg.down_factor), lambda_min)
        else:
            skill_lambda[skill_id] = old_lambda

    return GuardPressureState(
        skill_lambda=skill_lambda,
        recovery_count=recovery_count,
        regression_count=regression_count,
        last_audit_step=last_audit_step,
    )


__all__ = [
    "GuardPressureConfig",
    "GuardPressureState",
    "compute_guard_lambdas",
    "update_recovery",
]
