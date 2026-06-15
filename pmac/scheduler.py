"""Risk-weighted PMA-C scheduler from spec section 15."""

from __future__ import annotations

import numpy as np


def _rng_from_key(key):
    if isinstance(key, np.random.Generator):
        return key
    if key is None:
        return np.random.default_rng()
    arr = np.asarray(key, dtype=np.uint32).reshape(-1)
    seed = 0
    for value in arr:
        seed = (1664525 * seed + int(value) + 1013904223) % (2**32)
    return np.random.default_rng(seed)


def sampling_probs(nodes, current_skill_id, betas=(1, 1, 1, 1)) -> np.ndarray:
    nodes = list(nodes)
    if not nodes:
        return np.array([], dtype=np.float64)
    b = np.asarray(betas, dtype=np.float64)
    weights = []
    for node in nodes:
        learning_need = 1.0 if str(node.skill_id) == str(current_skill_id) else 0.0
        if node.best_score > 0.0:
            forgetting_risk = max(0.0, node.best_score - node.current_score)
            forgetting_risk = forgetting_risk / max(node.best_score, 1e-9)
        else:
            forgetting_risk = max(0.0, 1.0 - node.retention)
        uncertainty = max(0.0, 1.0 - float(node.stability))
        rarity = 1.0 / max(len(node.anchors), 1)
        weights.append(
            b[0] * learning_need
            + b[1] * forgetting_risk
            + b[2] * uncertainty
            + b[3] * rarity
            + 1e-8
        )
    weights = np.asarray(weights, dtype=np.float64)
    return weights / np.sum(weights)


class Scheduler:
    def __init__(self, old_skill_sample_fraction=0.3):
        self.old_skill_sample_fraction = float(old_skill_sample_fraction)
        self.boosts = {}

    def sample_source(self, atlas, current_skill_id, key):
        rng = _rng_from_key(key)
        old_nodes = [
            node
            for node in atlas.protected_nodes()
            if str(node.skill_id) != str(current_skill_id)
        ]
        if not old_nodes or rng.random() >= self.old_skill_sample_fraction:
            return current_skill_id

        probs = sampling_probs(old_nodes, current_skill_id)
        boost = np.asarray([self.boosts.get(node.skill_id, 1.0) for node in old_nodes])
        probs = probs * boost
        probs = probs / np.sum(probs)
        idx = int(rng.choice(len(old_nodes), p=probs))
        return old_nodes[idx].skill_id

    def boost(self, regressed_node_ids):
        for node_id in regressed_node_ids:
            node_id = str(node_id)
            self.boosts[node_id] = self.boosts.get(node_id, 1.0) * 2.0


__all__ = ["sampling_probs", "Scheduler"]
