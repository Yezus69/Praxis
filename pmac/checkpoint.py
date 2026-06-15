"""Champion snapshots and safe checkpoints for PMA-C."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from pmac.conservation import conservation_loss


def _copy_leaf(x):
    if isinstance(x, np.ndarray):
        return x.copy()
    try:
        return jnp.array(x, copy=True)
    except Exception:
        try:
            return x.copy()
        except AttributeError:
            return x


def deep_copy_pytree(tree):
    return jax.tree_util.tree_map(_copy_leaf, tree)


@dataclass(frozen=True)
class Champion:
    params: Any
    route: Any
    meta: Any


class ChampionStore:
    def __init__(self):
        self.history = []
        self.by_skill = {}

    def freeze(self, params, route=None, meta=None) -> Champion:
        meta = dict(meta or {})
        champion = Champion(params=deep_copy_pytree(params), route=route, meta=meta)
        self.history.append(champion)
        skill_id = meta.get("skill_id", route if route is not None else len(self.history) - 1)
        self.by_skill[skill_id] = champion
        return champion

    def get(self, skill_id) -> Champion:
        return self.by_skill[skill_id]


class SafeCheckpoint:
    def __init__(self, params):
        self._params = deep_copy_pytree(params)

    def update_if_safe(self, params, audit):
        if bool(audit.accept):
            self._params = deep_copy_pytree(params)

    def restore(self):
        return deep_copy_pytree(self._params)


def _candidate_params(candidate_impl):
    return getattr(candidate_impl, "params", candidate_impl)


def can_archive_expert(skill_node, candidate_impl, adapter) -> bool:
    params = _candidate_params(candidate_impl)
    if len(skill_node.anchors) == 0 or skill_node.sentinels is None:
        return False
    batch = skill_node.anchors.all_batch()
    behavior_fn = lambda p, x: adapter.behavior(p, {"x": x})
    distance_fn = lambda teacher, cur: adapter.distance(cur, teacher, None)
    anchor_ok = float(conservation_loss(behavior_fn, params, batch, distance_fn)) <= 1e-8
    sentinel_ok = skill_node.sentinels.passes(
        params, adapter, skill_node.best_score, skill_node.allowed_regression
    )
    score = skill_node.sentinels.evaluate(params, adapter)
    score_ok = score >= skill_node.best_score - skill_node.allowed_regression
    return bool(anchor_ok and sentinel_ok and score_ok)


def mark_redundant(skill_node, impl_id) -> bool:
    if hasattr(skill_node, "mark_redundant"):
        return bool(skill_node.mark_redundant(impl_id))
    certified = list(getattr(skill_node, "certified_impls", []))
    if len(certified) <= 1 or impl_id not in certified:
        return False
    redundant = getattr(skill_node, "redundant_impls", set())
    redundant.add(impl_id)
    skill_node.redundant_impls = redundant
    return True


__all__ = [
    "Champion",
    "ChampionStore",
    "SafeCheckpoint",
    "can_archive_expert",
    "deep_copy_pytree",
    "mark_redundant",
]
