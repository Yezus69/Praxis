"""Bounded compressed latent memory bank for PMA-C."""

from __future__ import annotations

from collections.abc import Mapping

import numpy as np

from pmac.envs.atari_envpool import ACT_DIM
from pmac.memory.atom import SourceFlag


EPS = 1e-8
AGE_EPS = 1e-6
DEFAULT_EVICT_WEIGHTS = {
    "sentinel": 5.0,
    "rarity": 1.0,
    "risk": 3.0,
    "teacher_conf": 0.5,
    "coverage": 3.0,
    "age": 0.5,
}


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
    arr = np.asarray(value, dtype=dtype)
    if arr.ndim == 0:
        return np.full((n,), arr.item(), dtype=arr.dtype)
    if arr.shape[0] == n:
        return arr.copy()
    if arr.shape[0] == 1:
        return np.repeat(arr, n, axis=0)
    raise ValueError(f"expected leading dimension {n}, got {arr.shape[0]}")


def _as_key_batch(value, width: int, name: str):
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2 or arr.shape[1] != width:
        raise ValueError(f"{name} must have shape [n,{width}], got {arr.shape}")
    return arr.copy()


def _as_batch_matrix(value, n: int, width: int, name: str):
    arr = np.asarray(value, dtype=np.float32)
    if arr.ndim == 0:
        return np.full((n, width), arr.item(), dtype=np.float32)
    if arr.ndim == 1:
        if arr.shape[0] != width:
            raise ValueError(f"{name} must have width {width}, got {arr.shape}")
        return np.repeat(arr[None, :], n, axis=0)
    if arr.ndim == 2 and arr.shape[1] == width:
        if arr.shape[0] == n:
            return arr.copy()
        if arr.shape[0] == 1:
            return np.repeat(arr, n, axis=0)
    raise ValueError(f"{name} must have shape [n,{width}], got {arr.shape}")


def _normalize_rows(keys):
    norms = np.linalg.norm(keys, axis=1, keepdims=True)
    return keys / (norms + EPS)  # spec §6


def _kl_policy(p, q):
    p_safe = np.maximum(np.asarray(p, dtype=np.float32), EPS)
    q_safe = np.maximum(np.asarray(q, dtype=np.float32), EPS)
    return float(np.sum(p_safe * (np.log(p_safe) - np.log(q_safe))))  # spec §22


def allocate_budgets(game_ids, u, b_total, b_min) -> dict[int, int]:
    ids = np.asarray(game_ids, dtype=np.int32).reshape(-1)
    if ids.size == 0:
        return {}

    games = list(dict.fromkeys(int(game) for game in ids))
    if isinstance(u, Mapping):
        per_game = {game: float(u.get(game, 0.0)) for game in games}
    else:
        utilities = np.asarray(u, dtype=np.float32).reshape(-1)
        if utilities.shape[0] == ids.shape[0]:
            per_game: dict[int, float] = {}
            for game, value in zip(ids, utilities):
                per_game[int(game)] = per_game.get(int(game), 0.0) + float(value)
        elif utilities.shape[0] == len(games):
            per_game = {game: float(value) for game, value in zip(games, utilities)}
        else:
            raise ValueError("u must match game_ids or unique games")
    if any(value < 0.0 for value in per_game.values()):
        raise ValueError("budget utilities must be non-negative")

    n_games = len(games)
    total = int(b_total)
    minimum = int(b_min)
    if total < 0 or minimum < 0:
        raise ValueError("budgets must be non-negative")
    if n_games * minimum > total:
        minimum = total // n_games  # spec §23: shrink B_min; global budget stays fixed.

    remaining = total - n_games * minimum
    weights = np.asarray([per_game[game] for game in games], dtype=np.float32)
    raw = minimum + remaining * weights / (float(np.sum(weights)) + EPS)  # spec §23
    caps = np.floor(raw).astype(np.int64)

    leftover = total - int(np.sum(caps))
    if leftover > 0 and np.sum(weights) > 0.0:
        fractions = raw - caps
        order = np.argsort(-fractions, kind="mergesort")
        for idx in order[:leftover]:
            caps[idx] += 1
    return {game: int(cap) for game, cap in zip(games, caps)}


class MemoryBank:
    """Fixed-budget compressed latent memory bank."""

    def __init__(
        self,
        capacity,
        d_k=128,
        d_c=16,
        act_dim=ACT_DIM,
        b_min=0,
        evict_weights=None,
        dtype=np.float16,
    ):
        self.capacity = int(capacity)
        self.d_k = int(d_k)
        self.d_c = int(d_c)
        self.act_dim = int(act_dim)
        self.b_min = int(b_min)
        if self.capacity < 0 or self.d_k <= 0 or self.d_c <= 0 or self.act_dim <= 0:
            raise ValueError("capacity and dimensions must be positive")
        if self.b_min < 0:
            raise ValueError("b_min must be non-negative")

        self.dtype = np.dtype(dtype)
        self.evict_weights = dict(DEFAULT_EVICT_WEIGHTS)
        if evict_weights is not None:
            for name, value in evict_weights.items():
                if name not in self.evict_weights:
                    raise KeyError(f"unknown eviction weight {name!r}")
                self.evict_weights[name] = float(value)

        self.size = 0
        self._rows = self.capacity
        self._next_uid = 0
        self._allocate_arrays(self._rows)

    def _allocate_arrays(self, rows):
        self.key = np.zeros((rows, self.d_k), dtype=self.dtype)
        self.context = np.zeros((rows, self.d_c), dtype=self.dtype)
        self.teacher_policy = np.zeros((rows, self.act_dim), dtype=self.dtype)
        self.teacher_value = np.zeros((rows,), dtype=self.dtype)
        self.importance = np.zeros((rows,), dtype=np.float32)
        self.game_id = np.zeros((rows,), dtype=np.int32)
        self.cluster_id = np.full((rows,), -1, dtype=np.int32)
        self.radius = np.zeros((rows,), dtype=np.float32)
        self.eps_policy = np.zeros((rows,), dtype=np.float32)
        self.eps_value = np.zeros((rows,), dtype=np.float32)
        self.rarity = np.zeros((rows,), dtype=np.float32)
        self.age = np.zeros((rows,), dtype=np.float32)
        self.count = np.ones((rows,), dtype=np.int32)
        self.source_flags = np.zeros((rows,), dtype=np.int32)
        self.model_coverage = np.zeros((rows,), dtype=bool)
        self.successor_key = np.zeros((rows, self.d_k), dtype=self.dtype)
        self.has_successor = np.zeros((rows,), dtype=bool)
        self.action = np.full((rows,), -1, dtype=np.int32)
        self.has_action = np.zeros((rows,), dtype=bool)
        self.reward = np.zeros((rows,), dtype=np.float32)
        self.has_reward = np.zeros((rows,), dtype=bool)
        self.done = np.zeros((rows,), dtype=bool)
        self.has_done = np.zeros((rows,), dtype=bool)
        self.return_trace = np.zeros((rows,), dtype=np.float32)
        self.has_return_trace = np.zeros((rows,), dtype=bool)
        self._uid = np.zeros((rows,), dtype=np.int64)

    @property
    def _array_names(self):
        return (
            "key",
            "context",
            "teacher_policy",
            "teacher_value",
            "importance",
            "game_id",
            "cluster_id",
            "radius",
            "eps_policy",
            "eps_value",
            "rarity",
            "age",
            "count",
            "source_flags",
            "model_coverage",
            "successor_key",
            "has_successor",
            "action",
            "has_action",
            "reward",
            "has_reward",
            "done",
            "has_done",
            "return_trace",
            "has_return_trace",
            "_uid",
        )

    def _ensure_rows(self, rows):
        if rows <= self._rows:
            return
        new_rows = max(rows, max(1, self._rows * 2))
        for name in self._array_names:
            old = getattr(self, name)
            fill = -1 if name == "action" else 0
            new = np.full((new_rows,) + old.shape[1:], fill, dtype=old.dtype)
            if self.size:
                new[: self.size] = old[: self.size]
            setattr(self, name, new)
        self._rows = new_rows

    def insert(
        self,
        keys,
        contexts,
        teacher_policies,
        teacher_values,
        importances,
        game_ids,
        *,
        eps_policy,
        eps_value,
        rarity=0.0,
        source_flags=0,
        successor_keys=None,
        actions=None,
        rewards=None,
        dones=None,
        return_traces=None,
        per_game_caps=None,
    ) -> np.ndarray:
        keys = _as_key_batch(keys, self.d_k, "keys")
        n = int(keys.shape[0])
        contexts = _as_batch_matrix(contexts, n, self.d_c, "contexts")
        policies = _as_batch_matrix(teacher_policies, n, self.act_dim, "teacher_policies")
        values = _as_batch_array(teacher_values, n, dtype=np.float32)
        importances = _as_batch_array(importances, n, dtype=np.float32)
        game_ids = _as_batch_array(game_ids, n, dtype=np.int32)
        eps_policy = _as_batch_array(eps_policy, n, dtype=np.float32)
        eps_value = _as_batch_array(eps_value, n, dtype=np.float32)
        rarity = _as_batch_array(rarity, n, dtype=np.float32)
        source_flags = _as_batch_array(source_flags, n, dtype=np.int32)
        successor_keys = (
            None
            if successor_keys is None
            else _as_batch_matrix(successor_keys, n, self.d_k, "successor_keys")
        )
        actions = None if actions is None else _as_batch_array(actions, n, dtype=np.int32)
        rewards = None if rewards is None else _as_batch_array(rewards, n, dtype=np.float32)
        dones = None if dones is None else _as_batch_array(dones, n, dtype=bool)
        return_traces = (
            None if return_traces is None else _as_batch_array(return_traces, n, dtype=np.float32)
        )

        old_size = self.size
        if old_size:
            self.age[:old_size] += 1.0
        self._ensure_rows(old_size + n)
        sl = slice(old_size, old_size + n)

        self.key[sl] = _normalize_rows(keys).astype(self.dtype)
        self.context[sl] = contexts.astype(self.dtype)
        self.teacher_policy[sl] = policies.astype(self.dtype)
        self.teacher_value[sl] = values.astype(self.dtype)
        self.importance[sl] = importances
        self.game_id[sl] = game_ids
        self.cluster_id[sl] = -1
        self.radius[sl] = 0.0
        self.eps_policy[sl] = eps_policy
        self.eps_value[sl] = eps_value
        self.rarity[sl] = rarity
        self.age[sl] = 0.0
        self.count[sl] = 1
        self.source_flags[sl] = source_flags
        self.model_coverage[sl] = False

        self.has_successor[sl] = successor_keys is not None
        if successor_keys is not None:
            self.successor_key[sl] = successor_keys.astype(self.dtype)
        self.has_action[sl] = actions is not None
        if actions is not None:
            self.action[sl] = actions
        self.has_reward[sl] = rewards is not None
        if rewards is not None:
            self.reward[sl] = rewards
        self.has_done[sl] = dones is not None
        if dones is not None:
            self.done[sl] = dones
        self.has_return_trace[sl] = return_traces is not None
        if return_traces is not None:
            self.return_trace[sl] = return_traces

        uids = np.arange(self._next_uid, self._next_uid + n, dtype=np.int64)
        self._next_uid += n
        self._uid[sl] = uids
        self.size += n

        if per_game_caps is not None:
            self._evict_per_game_caps(per_game_caps)
        self._evict_to(self.capacity)
        return self._indices_for_uids(uids)

    def merge_new(self, new_indices, r_merge, eps_pi_merge, eps_v_merge, lambda_count=1.0) -> int:
        indices = np.asarray(new_indices, dtype=np.int64).reshape(-1)
        indices = indices[(0 <= indices) & (indices < self.size)]
        new_uids = self._uid[indices].copy()
        merged = 0
        for uid in new_uids:
            current = np.flatnonzero(self._uid[: self.size] == uid)
            if current.size == 0:
                continue
            i = int(current[0])
            same_game = np.flatnonzero(self.game_id[: self.size] == self.game_id[i])
            same_game = same_game[same_game != i]
            if same_game.size == 0:
                continue

            sims = self.key[same_game].astype(np.float32) @ self.key[i].astype(np.float32)
            j = int(same_game[int(np.argmax(sims))])
            if float(np.max(sims)) <= 1.0 - float(r_merge):  # spec §22
                continue
            if _kl_policy(self.teacher_policy[i], self.teacher_policy[j]) >= float(eps_pi_merge):
                continue
            value_gap = abs(float(self.teacher_value[i]) - float(self.teacher_value[j]))
            if value_gap >= float(eps_v_merge):  # spec §22
                continue

            keep = min(i, j)
            drop = max(i, j)
            self._merge_pair(keep, drop, float(lambda_count))
            self._remove_indices([drop])
            merged += 1
        return merged

    def _merge_pair(self, keep, drop, lambda_count):
        n_i = int(self.count[keep])
        n_j = int(self.count[drop])
        n = n_i + n_j  # spec §22

        k = n_i * self.key[keep].astype(np.float32) + n_j * self.key[drop].astype(np.float32)
        self.key[keep] = (k / (np.linalg.norm(k) + EPS)).astype(self.dtype)  # spec §22

        p_i = self.teacher_policy[keep].astype(np.float32)
        p_j = self.teacher_policy[drop].astype(np.float32)
        self.teacher_policy[keep] = ((n_i * p_i + n_j * p_j) / n).astype(self.dtype)  # spec §22

        v_i = float(self.teacher_value[keep])
        v_j = float(self.teacher_value[drop])
        self.teacher_value[keep] = np.asarray((n_i * v_i + n_j * v_j) / n, dtype=self.dtype)  # spec §22

        self.importance[keep] = max(float(self.importance[keep]), float(self.importance[drop])) + (
            lambda_count * np.log(1.0 + n)
        )  # spec §22
        self.count[keep] = n
        self.source_flags[keep] = int(self.source_flags[keep]) | int(self.source_flags[drop])
        self.eps_policy[keep] = min(float(self.eps_policy[keep]), float(self.eps_policy[drop]))
        self.eps_value[keep] = min(float(self.eps_value[keep]), float(self.eps_value[drop]))

    def _utility(self, risk=None):
        if self.size == 0:
            return np.zeros((0,), dtype=np.float32)
        risk = {} if risk is None else risk
        risk_values = np.asarray(
            [float(risk.get(int(game), 0.0)) for game in self.game_id[: self.size]],
            dtype=np.float32,
        )
        teacher_conf = np.max(self.teacher_policy[: self.size].astype(np.float32), axis=1)  # spec §23
        max_age = float(np.max(self.age[: self.size]))
        age_penalty = self.age[: self.size] / (max_age + AGE_EPS)  # spec §23
        sentinel = (self.source_flags[: self.size] & int(SourceFlag.SENTINEL)) != 0
        coverage = self.model_coverage[: self.size].astype(np.float32)
        w = self.evict_weights
        return (
            self.importance[: self.size]
            + w["sentinel"] * sentinel.astype(np.float32)
            + w["rarity"] * self.rarity[: self.size]
            + w["risk"] * risk_values
            + w["teacher_conf"] * teacher_conf
            - w["coverage"] * coverage
            - w["age"] * age_penalty
        )  # spec §23

    def _evict_to(self, cap, risk=None):
        cap = max(0, int(cap))
        removed = []
        while self.size > cap:
            utilities = self._utility(risk=risk)
            counts = self.per_game_counts()
            remove = []
            for idx in np.argsort(utilities, kind="mergesort"):
                idx = int(idx)
                if self.size - len(remove) <= cap:
                    break
                if self._can_evict(idx, counts):
                    remove.append(idx)
                    counts[int(self.game_id[idx])] -= 1
            if not remove:
                break
            removed.extend(self._uid[remove].astype(np.int64).tolist())
            self._remove_indices(remove)
        return np.asarray(removed, dtype=np.int64)

    def _evict_per_game_caps(self, per_game_caps):
        caps = {int(game): max(0, int(cap)) for game, cap in dict(per_game_caps).items()}
        for game, cap in caps.items():
            while int(np.sum(self.game_id[: self.size] == game)) > cap:
                utilities = self._utility()
                counts = self.per_game_counts()
                candidates = np.flatnonzero(self.game_id[: self.size] == game)
                order = candidates[np.argsort(utilities[candidates], kind="mergesort")]
                chosen = [int(idx) for idx in order if self._can_evict(int(idx), counts)]
                if not chosen:
                    break
                self._remove_indices([chosen[0]])

    def _can_evict(self, idx, counts):
        sentinel = (int(self.source_flags[idx]) & int(SourceFlag.SENTINEL)) != 0
        if sentinel and not bool(self.model_coverage[idx]):
            return False
        if self.b_min > 0:
            game = int(self.game_id[idx])
            other_above_floor = any(
                other != game and count > self.b_min for other, count in counts.items()
            )
            if counts.get(game, 0) <= self.b_min and other_above_floor:
                return False
        return True

    def _remove_indices(self, indices):
        remove = np.unique(np.asarray(indices, dtype=np.int64))
        remove = remove[(0 <= remove) & (remove < self.size)]
        if remove.size == 0:
            return
        keep = np.ones((self.size,), dtype=bool)
        keep[remove] = False
        keep_idx = np.flatnonzero(keep)
        new_size = int(keep_idx.size)
        for name in self._array_names:
            arr = getattr(self, name)
            arr[:new_size] = arr[keep_idx]
        self.size = new_size

    def _indices_for_uids(self, uids):
        if self.size == 0:
            return np.zeros((0,), dtype=np.int64)
        order = {int(uid): pos for pos, uid in enumerate(uids)}
        matches = []
        for idx, uid in enumerate(self._uid[: self.size]):
            uid = int(uid)
            if uid in order:
                matches.append((order[uid], idx))
        return np.asarray([idx for _, idx in sorted(matches)], dtype=np.int64)

    def to_retrieval_arrays(self) -> dict[str, np.ndarray]:
        return self._export_indices(np.arange(self.size, dtype=np.int64))

    def _export_indices(self, indices):
        indices = np.asarray(indices, dtype=np.int64)
        return {
            "keys": np.ascontiguousarray(self.key[indices].astype(np.float32)),
            "context": np.ascontiguousarray(self.context[indices].astype(np.float32)),
            "teacher_policy": np.ascontiguousarray(
                self.teacher_policy[indices].astype(np.float32)
            ),
            "teacher_value": np.ascontiguousarray(self.teacher_value[indices].astype(np.float32)),
            "importance": np.ascontiguousarray(self.importance[indices].astype(np.float32)),
            "game_id": np.ascontiguousarray(self.game_id[indices].astype(np.int32)),
            "source_flags": np.ascontiguousarray(self.source_flags[indices].astype(np.int32)),
            "age": np.ascontiguousarray(self.age[indices].astype(np.float32)),
        }

    def sample(self, rng, n) -> dict[str, np.ndarray]:
        if self.size == 0:
            raise ValueError("cannot sample from an empty MemoryBank")
        n = min(int(n), self.size)
        rng = _rng_from_key(rng)
        # spec §7.2: warm memory reuses this bank; ANN can layer over these arrays later.
        indices = rng.choice(self.size, size=n, replace=False)
        return self._export_indices(indices)

    def per_game_counts(self) -> dict[int, int]:
        if self.size == 0:
            return {}
        ids, counts = np.unique(self.game_id[: self.size], return_counts=True)
        return {int(game): int(count) for game, count in zip(ids, counts)}

    @property
    def nbytes(self) -> int:
        total = 0
        for name in self._array_names:
            if name == "_uid":
                continue
            total += getattr(self, name)[: self.size].nbytes
        return int(total)

    def __len__(self):
        return int(self.size)


def promote(warm: MemoryBank, hot: MemoryBank, indices) -> np.ndarray:
    indices = np.asarray(indices, dtype=np.int64).reshape(-1)
    if np.any(indices < 0) or np.any(indices >= len(warm)):
        raise IndexError("promote indices out of range")
    if indices.size == 0:
        return np.zeros((0,), dtype=np.int64)

    start_uid = hot._next_uid
    inserted = hot.insert(
        warm.key[indices].astype(np.float32),
        warm.context[indices].astype(np.float32),
        warm.teacher_policy[indices].astype(np.float32),
        warm.teacher_value[indices].astype(np.float32),
        warm.importance[indices].astype(np.float32),
        warm.game_id[indices].astype(np.int32),
        eps_policy=warm.eps_policy[indices],
        eps_value=warm.eps_value[indices],
        rarity=warm.rarity[indices],
        source_flags=warm.source_flags[indices],
        successor_keys=warm.successor_key[indices].astype(np.float32),
        actions=warm.action[indices],
        rewards=warm.reward[indices],
        dones=warm.done[indices],
        return_traces=warm.return_trace[indices],
    )
    new_uids = np.arange(start_uid, start_uid + indices.size, dtype=np.int64)
    for src, uid in zip(indices, new_uids):
        dst = np.flatnonzero(hot._uid[: hot.size] == uid)
        if dst.size == 0:
            continue
        dst = int(dst[0])
        hot.has_successor[dst] = warm.has_successor[src]
        hot.has_action[dst] = warm.has_action[src]
        hot.has_reward[dst] = warm.has_reward[src]
        hot.has_done[dst] = warm.has_done[src]
        hot.has_return_trace[dst] = warm.has_return_trace[src]
    return inserted


__all__ = ["MemoryBank", "allocate_budgets", "promote"]
