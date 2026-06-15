"""Anchor memory and certificate-based deletion for PMA-C."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import numpy as np

from pmac.conservation import AnchorBatch


@dataclass
class Anchor:
    x: Any
    teacher: Any
    tolerance: Any
    weight: Any
    importance: Any
    context: Any = None
    embedding: Any = None
    skill_id: Any = None
    label: Any = None


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


def _as_batch_array(value, n: int, dtype=None):
    if value is None:
        return None
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        return np.full((n,), arr.item(), dtype=arr.dtype)
    if arr.shape[0] == n:
        return arr.copy()
    if arr.shape[0] == 1:
        return np.repeat(arr, n, axis=0)
    raise ValueError(f"expected leading dimension {n}, got {arr.shape[0]}")


def _take_optional(value, indices):
    if value is None:
        return None
    return np.asarray(value)[indices]


def _get(obj, name, default=None):
    if isinstance(obj, dict):
        return obj.get(name, default)
    return getattr(obj, name, default)


class AnchorStore:
    """Fixed-capacity anchor coreset retaining highest-importance examples."""

    def __init__(self, capacity):
        self.capacity = int(capacity)
        self.x = None
        self.context = None
        self.teacher = None
        self.tolerance = None
        self.weight = None
        self.importance = None
        self.embedding = None
        self.skill_id = None
        self.label = None

    def add(
        self,
        xs,
        teachers,
        tolerances,
        weights,
        importances,
        contexts=None,
        embeddings=None,
        skill_ids=None,
        labels=None,
    ):
        xs = np.asarray(xs)
        if xs.ndim == 0:
            raise ValueError("xs must have a batch dimension")
        n = int(xs.shape[0])
        teachers = _as_batch_array(teachers, n)
        tolerances = _as_batch_array(tolerances, n, dtype=np.float32)
        weights = _as_batch_array(weights, n, dtype=np.float32)
        importances = _as_batch_array(importances, n, dtype=np.float32)
        contexts = _as_batch_array(contexts, n)
        embeddings = _as_batch_array(embeddings, n)
        skill_ids = _as_batch_array(skill_ids, n)
        labels = _as_batch_array(labels, n, dtype=np.int32)

        if self.x is None:
            self.x = xs.copy()
            self.teacher = teachers
            self.tolerance = tolerances
            self.weight = weights
            self.importance = importances
            self.context = contexts
            self.embedding = embeddings
            self.skill_id = skill_ids
            self.label = labels
        else:
            self.x = np.concatenate([self.x, xs], axis=0)
            self.teacher = np.concatenate([self.teacher, teachers], axis=0)
            self.tolerance = np.concatenate([self.tolerance, tolerances], axis=0)
            self.weight = np.concatenate([self.weight, weights], axis=0)
            self.importance = np.concatenate([self.importance, importances], axis=0)
            self.context = self._concat_optional(self.context, contexts)
            self.embedding = self._concat_optional(self.embedding, embeddings)
            self.skill_id = self._concat_optional(self.skill_id, skill_ids)
            self.label = self._concat_optional(self.label, labels)

        self._enforce_capacity()

    def _concat_optional(self, old, new):
        if old is None and new is None:
            return None
        total = len(self)
        if old is None:
            new_n = np.asarray(new).shape[0]
            old = np.full((total - new_n,), None, dtype=object)
        if new is None:
            old_n = np.asarray(old).shape[0]
            new = np.full((total - old_n,), None, dtype=object)
        return np.concatenate([np.asarray(old), np.asarray(new)], axis=0)

    def _enforce_capacity(self):
        if self.capacity <= 0:
            keep = np.array([], dtype=np.int64)
        elif len(self) <= self.capacity:
            return
        else:
            order = np.argsort(-np.asarray(self.importance), kind="mergesort")
            keep = np.sort(order[: self.capacity])
        self.x = self.x[keep]
        self.teacher = self.teacher[keep]
        self.tolerance = self.tolerance[keep]
        self.weight = self.weight[keep]
        self.importance = self.importance[keep]
        self.context = _take_optional(self.context, keep)
        self.embedding = _take_optional(self.embedding, keep)
        self.skill_id = _take_optional(self.skill_id, keep)
        self.label = _take_optional(self.label, keep)

    def sample(self, key, n) -> AnchorBatch:
        if len(self) == 0:
            raise ValueError("cannot sample from an empty AnchorStore")
        n = min(int(n), len(self))
        rng = _rng_from_key(key)
        idx = rng.choice(len(self), size=n, replace=False)
        return AnchorBatch(
            x=self.x[idx],
            context=_take_optional(self.context, idx),
            teacher=self.teacher[idx],
            tolerance=self.tolerance[idx],
            weight=self.weight[idx],
        )

    def sample_examples(self, key, n):
        if self.label is None:
            raise ValueError("AnchorStore has no replay labels")
        if len(self) == 0:
            raise ValueError("cannot sample from an empty AnchorStore")
        n = min(int(n), len(self))
        rng = _rng_from_key(key)
        idx = rng.choice(len(self), size=n, replace=False)
        return self.x[idx], self.label[idx]

    def all_batch(self) -> AnchorBatch:
        return AnchorBatch(
            x=self.x,
            context=self.context,
            teacher=self.teacher,
            tolerance=self.tolerance,
            weight=self.weight,
        )

    def __len__(self):
        if self.x is None:
            return 0
        return int(self.x.shape[0])

    def can_delete(self, idx, candidate_cover, adapter, sentinel_ok: bool) -> bool:
        if not sentinel_ok or idx < 0 or idx >= len(self):
            return False

        anchor_skill = None if self.skill_id is None else self.skill_id[idx]
        cover_skill = _get(candidate_cover, "skill_id", anchor_skill)
        same_skill = anchor_skill is None or cover_skill is None or cover_skill == anchor_skill

        anchor_embedding = None if self.embedding is None else self.embedding[idx]
        cover_embedding = _get(candidate_cover, "embedding", anchor_embedding)
        radius = float(_get(candidate_cover, "radius", _get(candidate_cover, "local_radius", 1.0)))
        if anchor_embedding is None or cover_embedding is None:
            close_in_representation = True
        else:
            dist = np.linalg.norm(np.asarray(cover_embedding) - np.asarray(anchor_embedding))
            close_in_representation = bool(dist <= radius)

        cover_teacher = _get(candidate_cover, "teacher", None)
        epsilon = float(_get(candidate_cover, "epsilon", self.tolerance[idx]))
        if cover_teacher is None:
            matching_teacher = False
        else:
            try:
                distance = adapter.distance(cover_teacher, self.teacher[idx], None)
                distance = float(np.mean(np.asarray(distance)))
            except Exception:
                diff = np.asarray(cover_teacher) - np.asarray(self.teacher[idx])
                distance = float(np.mean(diff * diff))
            matching_teacher = distance <= epsilon + 1e-12

        return bool(same_skill and close_in_representation and matching_teacher and sentinel_ok)


__all__ = ["Anchor", "AnchorStore", "AnchorBatch"]
