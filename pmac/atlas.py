"""Protected Manifold Atlas graph for PMA-C."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from pmac.anchors import AnchorStore
from pmac.sentinels import SentinelStore
from pmac.tree_utils import tree_dot, tree_norm


@dataclass
class SkillNode:
    skill_id: str
    context_key: Any
    status: str
    anchors: AnchorStore
    sentinels: SentinelStore
    champion_ref: Any = None
    best_score: float = 0.0
    current_score: float = 0.0
    retention: float = 1.0
    allowed_regression: float = 0.05
    last_certified_step: int = 0
    prototype_embedding: Any = None
    local_radius: float = 1.0
    stability: float = 1.0
    guard_lambda: float = 1.0
    positive_transfer_neighbors: set = field(default_factory=set)
    interference_neighbors: set = field(default_factory=set)
    certified_impls: list = field(default_factory=list)
    redundant_impls: set = field(default_factory=set)
    expert_ref: Any = None
    slow_route_ref: Any = None

    def forgetting_risk(self) -> float:
        if self.best_score <= 0.0:
            return max(0.0, 1.0 - float(self.retention))
        drop = max(0.0, float(self.best_score) - float(self.current_score))
        return drop / max(float(self.best_score), 1e-9)

    def mark_redundant(self, impl_id) -> bool:
        if len(self.certified_impls) <= 1:
            return False
        if impl_id not in self.certified_impls:
            return False
        self.redundant_impls.add(impl_id)
        return True


class Atlas:
    def __init__(self):
        self.nodes = {}
        self.edge_scores = {}

    def create_or_update_node(self, skill_id, context_key, anchors, sentinels, **kw) -> SkillNode:
        skill_id = str(skill_id)
        if skill_id in self.nodes:
            node = self.nodes[skill_id]
            node.context_key = context_key
            node.anchors = anchors
            node.sentinels = sentinels
        else:
            node = SkillNode(
                skill_id=skill_id,
                context_key=context_key,
                status=kw.pop("status", "protected"),
                anchors=anchors,
                sentinels=sentinels,
            )
            self.nodes[skill_id] = node

        for name, value in kw.items():
            setattr(node, name, value)
        if node.champion_ref is not None and not node.certified_impls:
            impl_id = getattr(node.champion_ref, "meta", {}).get("skill_id", skill_id)
            node.certified_impls.append(impl_id)
        return node

    def protected_nodes(self) -> list[SkillNode]:
        return [node for node in self.nodes.values() if node.status == "protected"]

    def sample_protected_nodes(
        self, current_skill_id, k, strategy="interference_and_risk"
    ) -> list[SkillNode]:
        candidates = [
            node for node in self.protected_nodes() if node.skill_id != str(current_skill_id)
        ]
        if not candidates:
            return []

        def rank(node):
            risk = node.forgetting_risk()
            interference = 1.0 if str(current_skill_id) in node.interference_neighbors else 0.0
            stability_risk = max(0.0, 1.0 - float(node.stability))
            return risk + interference + 0.1 * stability_risk

        ordered = sorted(candidates, key=rank, reverse=True)
        return ordered[: min(int(k), len(ordered))]

    def insert_anchors(self, skill_id, xs, teachers, tolerances, weights, importances):
        node = self.nodes[str(skill_id)]
        node.anchors.add(
            xs,
            teachers,
            tolerances,
            weights,
            importances,
            skill_ids=[str(skill_id)] * len(xs),
        )

    def update_edge(self, i, j, g_i, g_j):
        i = str(i)
        j = str(j)
        denom = tree_norm(g_i) * tree_norm(g_j) + 1e-8
        score = float(tree_dot(g_i, g_j) / denom)
        self.edge_scores[(i, j)] = score
        self.edge_scores[(j, i)] = score

        ni = self.nodes[i]
        nj = self.nodes[j]
        if score > 0.0:
            ni.positive_transfer_neighbors.add(j)
            nj.positive_transfer_neighbors.add(i)
            ni.interference_neighbors.discard(j)
            nj.interference_neighbors.discard(i)
        elif score < 0.0:
            ni.interference_neighbors.add(j)
            nj.interference_neighbors.add(i)
            ni.positive_transfer_neighbors.discard(j)
            nj.positive_transfer_neighbors.discard(i)
        return score


__all__ = ["SkillNode", "Atlas"]
