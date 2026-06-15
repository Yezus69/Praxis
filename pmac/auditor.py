"""PMA-C acceptance gate from spec section 16."""

from __future__ import annotations

from dataclasses import dataclass, field

from pmac.conservation import conservation_loss


@dataclass
class Audit:
    accept: bool
    current_delta: float
    conservation_ok: bool
    sentinel_ok: bool
    regressed_nodes: list
    metrics: dict = field(default_factory=dict)


def _score(params, source_eval, adapter):
    if callable(source_eval):
        return float(source_eval(params))
    return float(adapter.evaluate_skill(params, source_eval))


def _guard_loss(params, node, adapter):
    if len(node.anchors) == 0:
        return 0.0
    batch = node.anchors.all_batch()
    behavior_fn = lambda p, x: adapter.behavior(p, {"x": x})
    distance_fn = lambda teacher, cur: adapter.distance(cur, teacher, None)
    return float(conservation_loss(behavior_fn, params, batch, distance_fn))


class Auditor:
    def __init__(self, delta_current=0.02, delta_cons=1e-4):
        self.delta_current = float(delta_current)
        self.delta_cons = float(delta_cons)

    def evaluate_candidate(
        self, cand_params, prev_params, source_eval, protected_nodes, adapter
    ) -> Audit:
        """Evaluate a candidate update using train-derived validation data only.

        Reported benchmark metrics are computed outside the auditor on held-out
        test splits; source_eval is intentionally reserved for validation-gate
        decisions during training.
        """
        current_cand = _score(cand_params, source_eval, adapter)
        current_prev = _score(prev_params, source_eval, adapter)
        current_delta = current_cand - current_prev
        current_ok = current_cand >= current_prev - self.delta_current

        conservation_ok = True
        sentinel_ok = True
        regressed = []
        metrics = {
            "current_score_candidate": current_cand,
            "current_score_previous": current_prev,
            "guard_losses": {},
            "sentinel_scores": {},
        }

        for node in protected_nodes:
            prev_g = _guard_loss(prev_params, node, adapter)
            cand_g = _guard_loss(cand_params, node, adapter)
            metrics["guard_losses"][node.skill_id] = {
                "previous": prev_g,
                "candidate": cand_g,
            }
            if cand_g > prev_g + self.delta_cons:
                conservation_ok = False
                regressed.append(node.skill_id)

            score = node.sentinels.evaluate(cand_params, adapter)
            metrics["sentinel_scores"][node.skill_id] = score
            if score < node.best_score - node.allowed_regression:
                sentinel_ok = False
                if node.skill_id not in regressed:
                    regressed.append(node.skill_id)

        accept = bool(current_ok and conservation_ok and sentinel_ok)
        return Audit(
            accept=accept,
            current_delta=current_delta,
            conservation_ok=conservation_ok,
            sentinel_ok=sentinel_ok,
            regressed_nodes=regressed,
            metrics=metrics,
        )

    def skill_solved(self, params, eval_set, adapter, threshold) -> bool:
        return bool(adapter.evaluate_skill(params, eval_set) >= float(threshold))


__all__ = ["Audit", "Auditor"]
