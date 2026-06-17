"""Deletion certification for compressed latent memory clusters."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from pmac.behavior_distance import huber
from pmac.memory.atom import SourceFlag
from pmac.memory.losses import _kl_teacher_to_current, _latent_behavior_from_bank, _net_from_dims


def _hp_value(hp, name: str, default=None):
    if isinstance(hp, Mapping):
        return hp.get(name, default)
    return getattr(hp, name, default)


def _infer_dims(params):
    n_games, d_c = params["game_embed"]["embedding"].shape
    d_k = params["key_head"]["kernel"].shape[-1]
    d_m = params["wv"]["kernel"].shape[-1]
    act_dim = params["policy_head"]["kernel"].shape[-1]
    return {
        "n_games": int(n_games),
        "d_k": int(d_k),
        "d_c": int(d_c),
        "d_m": int(d_m),
        "act_dim": int(act_dim),
    }


def _valid_size(bank) -> int:
    if hasattr(bank, "size"):
        return int(bank.size)
    if isinstance(bank, Mapping) and "valid" in bank:
        return int(np.asarray(bank["valid"], dtype=bool).shape[0])
    if isinstance(bank, Mapping) and "keys" in bank:
        return int(np.asarray(bank["keys"]).shape[0])
    return len(bank)


def _normalize_indices(indices, size: int) -> np.ndarray:
    arr = np.asarray(indices, dtype=np.int64).reshape(-1)
    if arr.size == 0:
        return np.zeros((0,), dtype=np.int64)
    arr = np.unique(arr)
    return arr[(0 <= arr) & (arr < int(size))]


def _bank_field(bank, name: str, indices: np.ndarray):
    if isinstance(bank, Mapping):
        return np.asarray(bank[name])[indices]
    return np.asarray(getattr(bank, name))[indices]


def _atom_batch_from_indices(bank, indices: np.ndarray) -> dict[str, np.ndarray]:
    return {
        "keys": _bank_field(bank, "key", indices).astype(np.float32)
        if hasattr(bank, "key")
        else _bank_field(bank, "keys", indices).astype(np.float32),
        "game_id": _bank_field(bank, "game_id", indices).astype(np.int32),
        "teacher_policy": _bank_field(bank, "teacher_policy", indices).astype(np.float32),
        "teacher_value": _bank_field(bank, "teacher_value", indices).astype(np.float32),
        "eps_policy": _bank_field(bank, "eps_policy", indices).astype(np.float32)
        if hasattr(bank, "eps_policy") or (isinstance(bank, Mapping) and "eps_policy" in bank)
        else _bank_field(bank, "eps", indices).astype(np.float32),
        "source_flags": _bank_field(bank, "source_flags", indices).astype(np.int32)
        if hasattr(bank, "source_flags") or (isinstance(bank, Mapping) and "source_flags" in bank)
        else np.zeros((indices.shape[0],), dtype=np.int32),
    }


def _atom_batch(atoms, bank=None) -> dict[str, np.ndarray]:
    if isinstance(atoms, Mapping):
        keys_name = "keys" if "keys" in atoms else "key"
        eps_name = "eps_policy" if "eps_policy" in atoms else "eps"
        result = {
            "keys": np.asarray(atoms[keys_name], dtype=np.float32),
            "game_id": np.asarray(atoms["game_id"], dtype=np.int32),
            "teacher_policy": np.asarray(atoms["teacher_policy"], dtype=np.float32),
            "teacher_value": np.asarray(atoms["teacher_value"], dtype=np.float32),
            "eps_policy": np.asarray(atoms[eps_name], dtype=np.float32),
        }
        if "source_flags" in atoms:
            result["source_flags"] = np.asarray(atoms["source_flags"], dtype=np.int32)
        return result

    arr = np.asarray(atoms)
    if np.issubdtype(arr.dtype, np.integer):
        if bank is None:
            raise ValueError("integer atom indices require a bank")
        return _atom_batch_from_indices(bank, _normalize_indices(arr, _valid_size(bank)))

    rows = list(atoms)
    return {
        "keys": np.asarray([row.key for row in rows], dtype=np.float32),
        "game_id": np.asarray([row.game_id for row in rows], dtype=np.int32),
        "teacher_policy": np.asarray([row.teacher_policy for row in rows], dtype=np.float32),
        "teacher_value": np.asarray([row.teacher_value for row in rows], dtype=np.float32),
        "eps_policy": np.asarray([row.eps_policy for row in rows], dtype=np.float32),
        "source_flags": np.asarray([getattr(row, "source_flags", 0) for row in rows], dtype=np.int32),
    }


def _pad_array(value, rows: int, *, dtype, fill=0):
    arr = np.asarray(value, dtype=dtype)
    if arr.shape[0] >= rows:
        return arr.copy()
    out = np.full((rows,) + arr.shape[1:], fill, dtype=arr.dtype)
    out[: arr.shape[0]] = arr
    return out


def _reader_bank(bank, hp, *, exclude_indices=None) -> dict[str, jnp.ndarray]:
    top_k = int(_hp_value(hp, "top_k", 1))
    if top_k <= 0:
        raise ValueError("hp.top_k must be positive")

    if hasattr(bank, "to_retrieval_arrays"):
        size = int(bank.size)
        rows = max(int(getattr(bank, "capacity", size)), size, top_k, 1)
        valid = np.zeros((rows,), dtype=bool)
        valid[:size] = True
        if exclude_indices is not None:
            valid[_normalize_indices(exclude_indices, size)] = False
        source_flags = np.zeros((rows,), dtype=np.int32)
        source_flags[:size] = np.asarray(bank.source_flags[:size], dtype=np.int32)
        arrays = {
            "keys": _pad_array(bank.key[:size], rows, dtype=np.float32),
            "context": _pad_array(bank.context[:size], rows, dtype=np.float32),
            "teacher_policy": _pad_array(bank.teacher_policy[:size], rows, dtype=np.float32),
            "teacher_value": _pad_array(bank.teacher_value[:size], rows, dtype=np.float32),
            "importance": _pad_array(bank.importance[:size], rows, dtype=np.float32),
            "game_id": _pad_array(bank.game_id[:size], rows, dtype=np.int32),
            "source_flags": source_flags,
            "age": _pad_array(bank.age[:size], rows, dtype=np.float32),
            "valid": valid,
        }
        return {name: jnp.asarray(value) for name, value in arrays.items()}

    if not isinstance(bank, Mapping):
        raise TypeError("bank must be a MemoryBank-like object or retrieval mapping")

    base_rows = int(np.asarray(bank["keys"]).shape[0])
    rows = max(base_rows, top_k, 1)
    valid = (
        _pad_array(bank["valid"], rows, dtype=bool)
        if "valid" in bank
        else np.ones((rows,), dtype=bool)
    )
    valid[base_rows:] = False
    if exclude_indices is not None:
        valid[_normalize_indices(exclude_indices, base_rows)] = False

    arrays = {
        "keys": _pad_array(bank["keys"], rows, dtype=np.float32),
        "context": _pad_array(bank["context"], rows, dtype=np.float32),
        "teacher_policy": _pad_array(bank["teacher_policy"], rows, dtype=np.float32),
        "teacher_value": _pad_array(bank["teacher_value"], rows, dtype=np.float32),
        "importance": _pad_array(bank["importance"], rows, dtype=np.float32),
        "game_id": _pad_array(bank["game_id"], rows, dtype=np.int32),
        "age": _pad_array(bank["age"], rows, dtype=np.float32),
        "valid": valid,
    }
    if "source5" in bank:
        arrays["source5"] = _pad_array(bank["source5"], rows, dtype=np.float32)
    else:
        source = bank.get("source_flags", np.zeros((base_rows,), dtype=np.int32))
        arrays["source_flags"] = _pad_array(source, rows, dtype=np.int32)
    return {name: jnp.asarray(value) for name, value in arrays.items()}


def _cluster_token(bank, idx: int):
    game = int(bank.game_id[idx])
    cluster_id = int(bank.cluster_id[idx]) if hasattr(bank, "cluster_id") else -1
    if cluster_id >= 0:
        return game, ("cluster", cluster_id)
    if hasattr(bank, "_uid"):
        return game, ("atom", int(bank._uid[idx]))
    return game, ("atom", int(idx))


def _cluster_counts_after_removal(bank, remove: np.ndarray) -> dict[int, int]:
    remove_set = set(int(idx) for idx in remove)
    clusters: dict[int, set[tuple[str, int]]] = {}
    for idx in range(int(bank.size)):
        if idx in remove_set:
            continue
        game, token = _cluster_token(bank, idx)
        clusters.setdefault(game, set()).add(token)
    return {game: len(tokens) for game, tokens in clusters.items()}


def _min_clusters_for_game(protected_game_min_clusters, game: int) -> int:
    if protected_game_min_clusters is None:
        return 1
    if isinstance(protected_game_min_clusters, Mapping):
        return max(0, int(protected_game_min_clusters.get(int(game), 1)))
    if isinstance(protected_game_min_clusters, (set, frozenset)):
        return 1 if int(game) in protected_game_min_clusters else 1
    return max(0, int(protected_game_min_clusters))


def _preserves_last_cluster(bank, indices: np.ndarray, protected_game_min_clusters) -> bool:
    remaining = _cluster_counts_after_removal(bank, indices)
    affected_games = {int(bank.game_id[int(idx)]) for idx in indices}
    for game in affected_games:
        if remaining.get(game, 0) < _min_clusters_for_game(protected_game_min_clusters, game):
            return False
    return True


def _uids_for_indices(bank, indices: np.ndarray) -> np.ndarray:
    if hasattr(bank, "_uid"):
        return np.asarray(bank._uid[indices], dtype=np.int64)
    return np.asarray(indices, dtype=np.int64)


def _indices_for_uids(bank, uids: np.ndarray) -> np.ndarray:
    if hasattr(bank, "_uid"):
        live = np.asarray(bank._uid[: bank.size], dtype=np.int64)
        positions = []
        for uid in uids:
            match = np.flatnonzero(live == int(uid))
            if match.size:
                positions.append(int(match[0]))
        return np.asarray(positions, dtype=np.int64)
    size = _valid_size(bank)
    return _normalize_indices(uids, size)


def _candidate_specs(bank, clusters) -> list[tuple[np.ndarray, np.ndarray]]:
    specs = []
    size = _valid_size(bank)
    for cluster in clusters:
        indices = _normalize_indices(cluster, size)
        if indices.size == 0:
            continue
        specs.append((indices.copy(), _uids_for_indices(bank, indices)))
    return specs


def model_coverage(params, atoms, bank, hp, *, lambda_v=1.0) -> np.ndarray:
    """Return per-atom model coverage for spec §24 condition 1 / spec §23."""
    atom_batch = _atom_batch(atoms, bank=None if isinstance(atoms, Mapping) else bank)
    keys = jax.lax.stop_gradient(jnp.asarray(atom_batch["keys"], dtype=jnp.float32))
    game_id = jax.lax.stop_gradient(jnp.asarray(atom_batch["game_id"], dtype=jnp.int32))
    p_star = jax.lax.stop_gradient(jnp.asarray(atom_batch["teacher_policy"], dtype=jnp.float32))
    v_star = jax.lax.stop_gradient(jnp.asarray(atom_batch["teacher_value"], dtype=jnp.float32))
    eps = jax.lax.stop_gradient(jnp.asarray(atom_batch["eps_policy"], dtype=jnp.float32))

    dims = _infer_dims(params)
    net = _net_from_dims(dims)
    reader_bank = _reader_bank(bank, hp)
    logits_theta, v_theta = _latent_behavior_from_bank(net, params, keys, game_id, reader_bank, hp)
    d_pi = _kl_teacher_to_current(p_star, logits_theta)
    d_v = huber(v_theta - v_star, 1.0)
    d_i = d_pi + jnp.asarray(lambda_v, dtype=jnp.float32) * d_v
    return np.asarray(jax.device_get(d_i <= eps), dtype=bool)  # spec §23, spec §24


def certify_deletion(
    bank,
    cluster_indices,
    params,
    hp,
    *,
    protected_game_min_clusters,
    sentinel_ok=True,
    review_ok=True,
    retrieval_ok=True,
    lambda_v=1.0,
) -> bool:
    """Return True iff every local deletion-audit condition in spec §24 holds.

    ``retrieval_ok``, ``sentinel_ok``, and ``review_ok`` are caller-evaluated
    audit signals for neighboring retrieval, protected sentinel score, and
    old-game review return respectively.
    """
    indices = _normalize_indices(cluster_indices, int(bank.size))
    if indices.size == 0:
        return False

    cond5 = _preserves_last_cluster(bank, indices, protected_game_min_clusters)  # spec §24
    cond2 = bool(retrieval_ok)  # spec §24
    cond3 = bool(sentinel_ok)  # spec §24
    cond4 = bool(review_ok)  # spec §24
    if not (cond2 and cond3 and cond4 and cond5):
        return False

    atoms = _atom_batch_from_indices(bank, indices)
    coverage = model_coverage(
        params,
        atoms,
        _reader_bank(bank, hp, exclude_indices=indices),
        hp,
        lambda_v=lambda_v,
    )
    cond1 = bool(np.all(coverage))  # spec §24
    if not cond1:
        return False

    sentinel = (atoms["source_flags"] & int(SourceFlag.SENTINEL)) != 0
    if np.any(sentinel & ~coverage):
        return False
    return bool(cond1 and cond2 and cond3 and cond4 and cond5)


def certify_and_prune(
    bank,
    clusters: Sequence[Any],
    params,
    hp,
    *,
    protected_game_min_clusters,
    sentinel_ok=True,
    review_ok=True,
    retrieval_ok=True,
    lambda_v=1.0,
) -> list[int]:
    """Certify candidate clusters in order and prune those that pass spec §24."""
    pruned: list[int] = []
    for original_indices, uids in _candidate_specs(bank, clusters):
        current_indices = _indices_for_uids(bank, uids)
        if current_indices.size != uids.size:
            continue
        if not _preserves_last_cluster(bank, current_indices, protected_game_min_clusters):  # spec §24
            continue
        if not certify_deletion(
            bank,
            current_indices,
            params,
            hp,
            protected_game_min_clusters=protected_game_min_clusters,
            sentinel_ok=sentinel_ok,
            review_ok=review_ok,
            retrieval_ok=retrieval_ok,
            lambda_v=lambda_v,
        ):
            continue
        pruned.extend(int(idx) for idx in original_indices)
        bank._remove_indices(current_indices)
    return pruned


__all__ = ["certify_and_prune", "certify_deletion", "model_coverage"]
