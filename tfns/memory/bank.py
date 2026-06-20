"""Bounded CPU sequence memory with label-free online content clusters."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

import numpy as np

from tfns.config import MemoryConfig
from tfns.memory.record import EpisodeSequence, compress, nbytes
from tfns.utils import RunningRobustStat

_EPS = 1e-8
_REDUNDANCY_THRESHOLD = 0.85


@dataclasses.dataclass(frozen=True)
class DeletionCertificate:
    behavior_covered_by_other_sentinels: bool = False
    activation_directions_covered_by_retained_bases: bool = False
    held_out_conservation_not_worsened: bool = False
    closed_loop_retention_within_threshold: bool = False


@dataclasses.dataclass
class _ClusterState:
    centroid: np.ndarray
    record_indices: list[int]


def _get_value(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _cert_bool(cert: Any, *names: str) -> bool:
    for name in names:
        value = _get_value(cert, name, None)
        if value is not None:
            return bool(value)
    return False


def can_delete_sentinel(rec: EpisodeSequence, cert: Any | None) -> bool:
    """Return whether a protected sentinel has complete deletion evidence."""

    if not rec.is_sentinel:
        return True
    if cert is None:
        return False

    behavior = _cert_bool(cert, "behavior_covered_by_other_sentinels")
    activation = _cert_bool(cert, "activation_directions_covered_by_retained_bases")

    held_out = _get_value(cert, "held_out_conservation_not_worsened", None)
    if held_out is None:
        held_out = _get_value(cert, "heldout_conservation_not_worsened", None)
    if held_out is None:
        delta = float(_get_value(cert, "held_out_conservation_delta", np.inf))
        tolerance = float(_get_value(cert, "held_out_conservation_tolerance", 0.0))
        held_out = delta <= tolerance

    closed_loop = _get_value(cert, "closed_loop_retention_within_threshold", None)
    if closed_loop is None:
        delta = float(_get_value(cert, "closed_loop_retention_delta", np.inf))
        threshold = float(_get_value(cert, "closed_loop_retention_threshold", 0.0))
        closed_loop = delta <= threshold

    return bool(behavior and activation and held_out and closed_loop)


def _normalize(vec: np.ndarray) -> np.ndarray:
    arr = np.asarray(vec, dtype=np.float32)
    norm = float(np.linalg.norm(arr))
    if norm <= _EPS:
        return np.zeros_like(arr, dtype=np.float32)
    return (arr / norm).astype(np.float32)


def sequence_signature(rec: EpisodeSequence) -> np.ndarray:
    """Return normalize(mean_t key_anchor_t)."""

    return _normalize(np.mean(np.asarray(rec.key_anchor, dtype=np.float32), axis=0))


def _cosine(a: np.ndarray, b: np.ndarray) -> float:
    denom = float(np.linalg.norm(a) * np.linalg.norm(b))
    if denom <= _EPS:
        return 0.0
    return float(np.dot(a, b) / denom)


def _softmax_mean(logits: np.ndarray) -> np.ndarray:
    x = np.asarray(logits, dtype=np.float64)
    x = x - np.max(x, axis=-1, keepdims=True)
    probs = np.exp(x)
    probs = probs / np.maximum(np.sum(probs, axis=-1, keepdims=True), _EPS)
    return np.mean(probs, axis=0).astype(np.float32)


class SequenceMemoryBank:
    """Fixed-budget CPU memory for task-free replay sequences."""

    def __init__(self, config: MemoryConfig | None = None, *, compress_frames: bool = False):
        self.config = config or MemoryConfig()
        if self.config.byte_budget <= 0:
            raise ValueError("byte_budget must be positive.")
        if self.config.max_clusters <= 0:
            raise ValueError("max_clusters must be positive.")
        if self.config.max_records <= 0:
            raise ValueError("max_records must be positive.")

        self.compress_frames = bool(compress_frames)
        self._records: list[EpisodeSequence] = []
        self._signatures: list[np.ndarray] = []
        self._added_at: list[int] = []
        self._clusters: dict[int, _ClusterState] = {}
        self._next_cluster_id = 0
        self._clock = 0
        self._bytes = 0
        self._stats = {
            "adv": RunningRobustStat(),
            "td": RunningRobustStat(),
            "causal": RunningRobustStat(),
            "surprise": RunningRobustStat(),
        }

    def __len__(self) -> int:
        return len(self._records)

    def bytes_used(self) -> int:
        return int(self._bytes)

    def records(self) -> tuple[EpisodeSequence, ...]:
        return tuple(self._records)

    def clusters(self) -> dict[int, tuple[int, ...]]:
        return {cid: tuple(state.record_indices) for cid, state in self._clusters.items()}

    def transition_importance(
        self,
        rec: EpisodeSequence,
        *,
        novelty: float | np.ndarray = 0.0,
        failure: np.ndarray | None = None,
        drift: float | np.ndarray = 0.0,
    ) -> np.ndarray:
        return self._transition_importance(rec, novelty=novelty, failure=failure, drift=drift)

    def sequence_score(
        self,
        rec: EpisodeSequence,
        *,
        novelty: float | np.ndarray = 0.0,
        failure: np.ndarray | None = None,
        drift: float | np.ndarray = 0.0,
    ) -> float:
        importance = self._transition_importance(rec, novelty=novelty, failure=failure, drift=drift)
        return self._aggregate_importance(importance)

    def add(self, rec: EpisodeSequence, cluster_risk: Any | None = None) -> bool:
        """Admit a record if it can fit without violating the fixed budget."""

        snapshot = self._snapshot()
        stored = compress(rec) if self.compress_frames else rec
        signature = sequence_signature(stored)
        novelty = self._novelty(signature)
        cluster_id = self._assign_cluster_id(signature)
        stored.cluster_id = int(cluster_id)
        drift = self._risk_for_cluster(cluster_id, cluster_risk)
        stored.seq_importance = self.sequence_score(stored, novelty=novelty, drift=drift)

        self._append(stored, signature)
        self._merge_clusters_as_needed()

        fit = self.evict_to_fit(cluster_risk=cluster_risk)
        if fit and self._contains_identity(stored):
            self._update_stats(stored)
            return True

        self._restore(snapshot)
        return False

    def evict_to_fit(
        self,
        cluster_risk: Any | None = None,
        sentinel_certs: Mapping[Any, Any] | None = None,
    ) -> bool:
        """Remove lowest-utility evictable records until byte/count caps are met.

        Utilities include O(n^2) similarity matrices, so they are computed once
        per call and used as a fixed eviction order. Sentinel certificates,
        min-per-cluster eligibility, and cheap byte/count checks are refreshed
        after each removal.
        """

        if self._within_limits():
            return True

        utilities = self._utility_array(cluster_risk=cluster_risk)
        if utilities.size == 0:
            return False

        order = np.argsort(utilities, kind="stable")
        candidates = [self._records[int(idx)] for idx in order]
        for rec in candidates:
            if self._within_limits():
                return True
            idx = self._index_of_identity(rec)
            if idx is None:
                continue
            if not self._is_evictable(idx, sentinel_certs):
                continue
            self._remove_index(idx)
        return self._within_limits()

    def eviction_utilities(self, cluster_risk: Any | None = None) -> np.ndarray:
        return self._utility_array(cluster_risk=cluster_risk)

    def age_penalty(self, rec: EpisodeSequence) -> float:
        idx = self._index_of_identity(rec)
        if idx is None:
            raise ValueError("record is not in this bank.")
        redundancy = self._redundancies()[idx]
        return self._age_penalty(idx, redundancy)

    def marginal_coverages(self) -> np.ndarray:
        return self._marginal_coverages()

    def redundancies(self) -> np.ndarray:
        return self._redundancies()

    def _snapshot(self) -> tuple[list[EpisodeSequence], list[np.ndarray], list[int], int, int, list[int], int]:
        return (
            list(self._records),
            list(self._signatures),
            list(self._added_at),
            int(self._bytes),
            int(self._next_cluster_id),
            [int(rec.cluster_id) for rec in self._records],
            int(self._clock),
        )

    def _restore(
        self,
        snapshot: tuple[list[EpisodeSequence], list[np.ndarray], list[int], int, int, list[int], int],
    ) -> None:
        records, signatures, added_at, used, next_cluster_id, cluster_ids, clock = snapshot
        self._records = records
        self._signatures = signatures
        self._added_at = added_at
        self._bytes = used
        self._next_cluster_id = next_cluster_id
        self._clock = clock
        for rec, cluster_id in zip(self._records, cluster_ids):
            rec.cluster_id = int(cluster_id)
        self._rebuild_clusters_from_records()

    def _append(self, rec: EpisodeSequence, signature: np.ndarray) -> None:
        idx = len(self._records)
        self._records.append(rec)
        self._signatures.append(signature)
        self._added_at.append(self._clock)
        self._clock += 1
        self._bytes += nbytes(rec)
        self._clusters.setdefault(rec.cluster_id, _ClusterState(signature.copy(), [])).record_indices.append(idx)
        self._recompute_cluster(rec.cluster_id)

    def _remove_index(self, idx: int) -> None:
        self._bytes -= nbytes(self._records[idx])
        del self._records[idx]
        del self._signatures[idx]
        del self._added_at[idx]
        self._rebuild_clusters_from_records()

    def _contains_identity(self, rec: EpisodeSequence) -> bool:
        return any(item is rec for item in self._records)

    def _index_of_identity(self, rec: EpisodeSequence) -> int | None:
        for idx, item in enumerate(self._records):
            if item is rec:
                return idx
        return None

    def _transition_importance(
        self,
        rec: EpisodeSequence,
        *,
        novelty: float | np.ndarray,
        failure: np.ndarray | None,
        drift: float | np.ndarray,
    ) -> np.ndarray:
        if failure is None:
            failure = np.asarray(rec.ppo_mask, dtype=np.float32)
        return (
            self.config.w_adv * self._stats["adv"].normalize(np.abs(rec.adv_mag))
            + self.config.w_td * self._stats["td"].normalize(np.abs(rec.td_mag))
            + self.config.w_causal * self._stats["causal"].normalize(rec.causal_contrib)
            + self.config.w_novelty * np.asarray(novelty, dtype=np.float64)
            + self.config.w_surprise * self._stats["surprise"].normalize(rec.surprise)
            + self.config.w_failure * np.asarray(failure, dtype=np.float64)
            + self.config.w_entropy * np.asarray(rec.teacher_entropy, dtype=np.float64)
            + self.config.w_drift * np.asarray(drift, dtype=np.float64)
        ).astype(np.float32)

    def _aggregate_importance(self, importance: np.ndarray) -> float:
        values = np.asarray(importance, dtype=np.float32)
        return float(
            self.config.score_mean_w * float(np.mean(values))
            + self.config.score_quantile_w * float(np.quantile(values, self.config.score_quantile))
            + self.config.score_max_w * float(np.max(values))
        )

    def _update_stats(self, rec: EpisodeSequence) -> None:
        self._stats["adv"] = self._stats["adv"].update(np.abs(rec.adv_mag))
        self._stats["td"] = self._stats["td"].update(np.abs(rec.td_mag))
        self._stats["causal"] = self._stats["causal"].update(rec.causal_contrib)
        self._stats["surprise"] = self._stats["surprise"].update(rec.surprise)

    def _novelty(self, signature: np.ndarray) -> float:
        if not self._clusters:
            return 1.0
        best = max(_cosine(signature, state.centroid) for state in self._clusters.values())
        return float(np.clip(1.0 - max(0.0, best), 0.0, 1.0))

    def _assign_cluster_id(self, signature: np.ndarray) -> int:
        if not self._clusters:
            return self._new_cluster_id()

        best_id = -1
        best_sim = -np.inf
        for cid, state in self._clusters.items():
            sim = _cosine(signature, state.centroid)
            if sim > best_sim:
                best_id = cid
                best_sim = sim

        if best_sim >= self.config.cluster_sim_thresh:
            return int(best_id)
        return self._new_cluster_id()

    def _new_cluster_id(self) -> int:
        cid = self._next_cluster_id
        self._next_cluster_id += 1
        self._clusters[cid] = _ClusterState(np.zeros(128, dtype=np.float32), [])
        return cid

    def _merge_clusters_as_needed(self) -> None:
        while len(self._clusters) > 1:
            pair, sim = self._closest_cluster_pair()
            if pair is None:
                return
            over_cap = len(self._clusters) > self.config.max_clusters
            if not over_cap and sim < self.config.cluster_merge_thresh:
                return
            self._merge_clusters(pair[0], pair[1])

    def _closest_cluster_pair(self) -> tuple[tuple[int, int] | None, float]:
        ids = sorted(self._clusters)
        best_pair = None
        best_sim = -np.inf
        for left_pos, left in enumerate(ids):
            for right in ids[left_pos + 1 :]:
                sim = _cosine(self._clusters[left].centroid, self._clusters[right].centroid)
                if sim > best_sim:
                    best_sim = sim
                    best_pair = (left, right)
        return best_pair, float(best_sim)

    def _merge_clusters(self, left: int, right: int) -> None:
        keep, drop = (left, right) if left < right else (right, left)
        for idx in self._clusters[drop].record_indices:
            self._records[idx].cluster_id = keep
        self._rebuild_clusters_from_records()

    def _rebuild_clusters_from_records(self) -> None:
        clusters: dict[int, _ClusterState] = {}
        for idx, rec in enumerate(self._records):
            clusters.setdefault(rec.cluster_id, _ClusterState(np.zeros_like(self._signatures[idx]), [])).record_indices.append(idx)
        self._clusters = clusters
        for cid in list(self._clusters):
            self._recompute_cluster(cid)

    def _recompute_cluster(self, cid: int) -> None:
        state = self._clusters.get(cid)
        if state is None or not state.record_indices:
            self._clusters.pop(cid, None)
            return
        mean_sig = np.mean([self._signatures[idx] for idx in state.record_indices], axis=0)
        state.centroid = _normalize(mean_sig)

    def _is_evictable(self, idx: int, sentinel_certs: Mapping[Any, Any] | None) -> bool:
        rec = self._records[idx]
        if rec.is_sentinel and not can_delete_sentinel(rec, self._cert_for(rec, sentinel_certs)):
            return False
        cluster_size = len(self._clusters[rec.cluster_id].record_indices)
        if cluster_size <= self.config.min_per_cluster:
            return False
        return True

    def _within_limits(self) -> bool:
        return (
            self._bytes <= self.config.byte_budget
            and len(self._records) <= self.config.max_records
        )

    def _cert_for(self, rec: EpisodeSequence, sentinel_certs: Mapping[Any, Any] | None) -> Any | None:
        if sentinel_certs is None:
            return None
        if id(rec) in sentinel_certs:
            return sentinel_certs[id(rec)]
        return sentinel_certs.get((rec.episode_id, rec.chunk_index))

    def _utility(self, idx: int, *, cluster_risk: Any | None) -> float:
        utilities = self._utility_array(cluster_risk=cluster_risk)
        return float(utilities[idx])

    def _utility_array(self, *, cluster_risk: Any | None) -> np.ndarray:
        n = len(self._records)
        if n == 0:
            return np.asarray([], dtype=np.float32)

        coverages = self._marginal_coverages()
        redundancies = self._redundancies()
        seq_importance = np.asarray([rec.seq_importance for rec in self._records], dtype=np.float32)
        risks = np.asarray(
            [self._risk_for_cluster(rec.cluster_id, cluster_risk) for rec in self._records],
            dtype=np.float32,
        )
        causal_values = np.asarray([self._causal_value(rec) for rec in self._records], dtype=np.float32)
        age_penalties = self._age_penalties(redundancies)
        return (
            seq_importance
            + np.float32(self.config.lam_risk) * risks
            + np.float32(self.config.lam_cover) * coverages
            + np.float32(self.config.lam_causal) * causal_values
            - np.float32(self.config.lam_red) * redundancies
            - np.float32(self.config.lam_age) * age_penalties
        ).astype(np.float32)

    def _risk_for_cluster(self, cluster_id: int, cluster_risk: Any | None) -> float:
        if cluster_risk is None:
            return 0.0
        if isinstance(cluster_risk, Mapping):
            return float(cluster_risk.get(cluster_id, 0.0))
        return float(cluster_risk)

    def _marginal_coverages(self) -> np.ndarray:
        n = len(self._signatures)
        if n == 0:
            return np.asarray([], dtype=np.float32)
        if n == 1:
            return np.ones((1,), dtype=np.float32)
        sims = self._signature_cosine_matrix()
        np.fill_diagonal(sims, -np.inf)
        max_other = np.max(sims, axis=1)
        return (1.0 - max_other).astype(np.float32)

    def _redundancies(self) -> np.ndarray:
        n = len(self._records)
        if n <= 1:
            return np.zeros((n,), dtype=np.float32)

        sig_cos = self._signature_cosine_matrix()
        policy_cos = np.maximum(self._cosine_matrix(self._policy_matrix()), 0.0)
        values = np.asarray([float(np.mean(rec.teacher_value)) for rec in self._records], dtype=np.float32)
        causals = np.asarray([float(np.mean(rec.causal_contrib)) for rec in self._records], dtype=np.float32)
        value_sim = 1.0 / (1.0 + np.abs(values[:, None] - values[None, :]))
        causal_sim = 1.0 / (1.0 + np.abs(causals[:, None] - causals[None, :]))
        combined = 0.25 * (0.5 * (sig_cos + 1.0) + policy_cos + value_sim + causal_sim)
        eligible = sig_cos >= np.float32(self.config.cluster_sim_thresh)
        np.fill_diagonal(eligible, False)
        return np.max(np.where(eligible, combined, 0.0), axis=1).astype(np.float32)

    def _signature_matrix(self) -> np.ndarray:
        if not self._signatures:
            return np.zeros((0, 0), dtype=np.float32)
        return np.stack(self._signatures, axis=0).astype(np.float32, copy=False)

    def _policy_matrix(self) -> np.ndarray:
        if not self._records:
            return np.zeros((0, 18), dtype=np.float32)
        return np.stack([_softmax_mean(rec.teacher_logits) for rec in self._records], axis=0).astype(
            np.float32,
            copy=False,
        )

    def _signature_cosine_matrix(self) -> np.ndarray:
        return self._cosine_matrix(self._signature_matrix())

    def _cosine_matrix(self, matrix: np.ndarray) -> np.ndarray:
        rows = np.asarray(matrix, dtype=np.float32)
        if rows.ndim != 2 or rows.shape[0] == 0:
            return np.zeros((0, 0), dtype=np.float32)
        norms = np.linalg.norm(rows, axis=1, keepdims=True)
        normalized = np.divide(rows, norms, out=np.zeros_like(rows, dtype=np.float32), where=norms > _EPS)
        return (normalized @ normalized.T).astype(np.float32)

    def _causal_value(self, rec: EpisodeSequence) -> float:
        values = np.abs(np.asarray(rec.causal_contrib, dtype=np.float32))
        return float(0.5 * (np.mean(values) + np.max(values)))

    def _age_penalty(self, idx: int, redundancy: float) -> float:
        if redundancy < _REDUNDANCY_THRESHOLD:
            return 0.0
        ages = np.asarray([self._clock - added for added in self._added_at], dtype=np.float32)
        max_age = float(np.max(ages)) if ages.size else 1.0
        if max_age <= 0.0:
            return 0.0
        return float(ages[idx] / max_age)

    def _age_penalties(self, redundancies: np.ndarray) -> np.ndarray:
        n = len(self._records)
        if n == 0:
            return np.asarray([], dtype=np.float32)
        ages = np.asarray([self._clock - added for added in self._added_at], dtype=np.float32)
        max_age = float(np.max(ages)) if ages.size else 1.0
        if max_age <= 0.0:
            return np.zeros((n,), dtype=np.float32)
        return np.where(redundancies >= _REDUNDANCY_THRESHOLD, ages / max_age, 0.0).astype(np.float32)


__all__ = [
    "DeletionCertificate",
    "SequenceMemoryBank",
    "can_delete_sentinel",
    "sequence_signature",
]
