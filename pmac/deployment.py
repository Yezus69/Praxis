"""Deterministic deployed routing for protected PMA-C skills."""

from __future__ import annotations

from dataclasses import dataclass


class InvariantViolation(RuntimeError):
    pass


@dataclass
class DeploymentDecision:
    skill_id: str
    route_type: str
    route_id: str
    reason: str
    current_certified: bool
    fallback_used: bool


class DeployedPolicy:
    """Routes per skill to current shared params or a frozen certified champion."""

    def __init__(self, current_params, atlas, router=None):
        self.current_params = current_params
        self.atlas = atlas
        self.router = router

    def _node(self, skill_id):
        key = str(skill_id)
        if self.atlas is None or key not in getattr(self.atlas, "nodes", {}):
            raise InvariantViolation(f"missing protected skill node: {key}")
        return self.atlas.nodes[key]

    @staticmethod
    def _champion_route_id(champion, skill_id) -> str:
        route_id = getattr(champion, "route", None)
        if route_id is None:
            route_id = getattr(champion, "meta", {}).get("skill_id", skill_id)
        return str(route_id)

    @staticmethod
    def _champion_params(champion, skill_id):
        if champion is None or not hasattr(champion, "params") or champion.params is None:
            raise InvariantViolation(f"missing executable champion params for skill: {skill_id}")
        return champion.params

    def select_route(self, skill_id, current_certified: bool) -> DeploymentDecision:
        node = self._node(skill_id)
        skill_id = str(skill_id)
        current_certified = bool(current_certified)
        needs_repair = bool(getattr(node, "needs_repair", False))
        node.current_certified = current_certified

        if current_certified and not needs_repair:
            if self.current_params is None:
                raise InvariantViolation("missing executable current params")
            return DeploymentDecision(
                skill_id=skill_id,
                route_type="current",
                route_id="current",
                reason="current_certified",
                current_certified=True,
                fallback_used=False,
            )

        champion = getattr(node, "champion_ref", None)
        self._champion_params(champion, skill_id)
        reason = "needs_repair" if needs_repair else "current_not_certified"
        route_id = self._champion_route_id(champion, skill_id)
        node.fallback_route_id = route_id
        return DeploymentDecision(
            skill_id=skill_id,
            route_type="champion",
            route_id=route_id,
            reason=reason,
            current_certified=current_certified,
            fallback_used=True,
        )

    def resolve_params(self, decision: DeploymentDecision):
        if decision.route_type == "current":
            if self.current_params is None:
                raise InvariantViolation("missing executable current params")
            return self.current_params
        if decision.route_type == "champion":
            node = self._node(decision.skill_id)
            champion = getattr(node, "champion_ref", None)
            return self._champion_params(champion, decision.skill_id)
        raise InvariantViolation(f"unsupported deployed route type: {decision.route_type}")

    @staticmethod
    def route_usage(decisions: list[DeploymentDecision]) -> dict:
        decisions = list(decisions)
        total = float(len(decisions))
        n_current = sum(1 for decision in decisions if decision.route_type == "current")
        n_champion = sum(1 for decision in decisions if decision.route_type == "champion")
        n_expert = sum(1 for decision in decisions if decision.route_type == "expert")
        n_consolidated = sum(1 for decision in decisions if decision.route_type == "consolidated")
        denom = total if total > 0.0 else 1.0
        return {
            "current_fraction": float(n_current / denom),
            "champion_fraction": float(n_champion / denom),
            "expert_fraction": float(n_expert / denom),
            "consolidated_fraction": float(n_consolidated / denom),
            "fallback_used_per_skill": {
                str(decision.skill_id): bool(decision.fallback_used) for decision in decisions
            },
            "n_current": int(n_current),
            "n_champion": int(n_champion),
            "n_expert": int(n_expert),
            "n_consolidated": int(n_consolidated),
            "routes": {str(decision.skill_id): str(decision.route_type) for decision in decisions},
        }


__all__ = [
    "DeployedPolicy",
    "DeploymentDecision",
    "InvariantViolation",
]
