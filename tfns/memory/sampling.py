"""Label-free replay sampling helpers for bounded sequence memory."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any, Container

import numpy as np

from tfns.memory.bank import SequenceMemoryBank
from tfns.memory.record import EpisodeSequence, reconstruct_obs, seq_len

_EPS = 1e-8


def _cfg_section(cfg: Any, name: str) -> Any:
    return getattr(cfg, name, cfg)


def _value(obj: Any, name: str, default: float = 0.0) -> float:
    if isinstance(obj, Mapping):
        return float(obj.get(name, default))
    return float(getattr(obj, name, default))


def cluster_risk(stats: Any, cfg: Any) -> float:
    """Compute section 16 risk for one internal content cluster."""

    risk_cfg = _cfg_section(cfg, "risk")
    return float(
        _value(risk_cfg, "rho_0", 0.1)
        + _value(risk_cfg, "lam_D", 1.0) * _value(stats, "behavior_violation", 0.0)
        + _value(risk_cfg, "lam_Q", 1.0) * _value(stats, "high_quantile_drift", 0.0)
        + _value(risk_cfg, "lam_R", 1.0) * _value(stats, "basis_residual", 0.0)
        + _value(risk_cfg, "lam_A", 1.0) * _value(stats, "time_since_replay", 0.0)
    )


def cluster_probs(risks: Mapping[int, float] | np.ndarray) -> Mapping[int, float] | np.ndarray:
    """Normalize cluster risks into replay probabilities."""

    if isinstance(risks, Mapping):
        if not risks:
            return {}
        keys = list(risks.keys())
        values = np.asarray([max(0.0, float(risks[key])) for key in keys], dtype=np.float64)
        probs = _normalize_probs(values)
        return {key: float(prob) for key, prob in zip(keys, probs)}

    values = np.asarray(risks, dtype=np.float64)
    return _normalize_probs(np.maximum(values, 0.0))


def replay_transition_count(on_policy_count: int, max_risk: float, cfg: Any) -> int:
    """Risk-only replay count capped at the on-policy transition count."""

    if on_policy_count <= 0:
        return 0
    replay_cfg = _cfg_section(cfg, "replay")
    start = float(getattr(replay_cfg, "replay_frac_start", 0.25))
    start = float(np.clip(start, 0.0, 1.0))
    risk = max(0.0, float(max_risk))
    risk_frac = risk / (1.0 + risk)
    frac = start + (1.0 - start) * risk_frac
    return int(min(on_policy_count, np.ceil(on_policy_count * frac)))


def sample_sequences(
    bank: SequenceMemoryBank,
    rng: np.random.Generator | np.random.RandomState | int,
    n: int,
    cluster_probs: Mapping[int, float] | None = None,
    *,
    statuses: Container[str] | None = None,
) -> list[EpisodeSequence]:
    """Sample clusters, then sequences inside those clusters, without labels."""

    if n <= 0 or len(bank) == 0:
        return []

    generator = rng if hasattr(rng, "choice") else np.random.default_rng(rng)
    clusters = bank.clusters()
    records = bank.records()
    if statuses is None:
        eligible_clusters = {cid: tuple(indices) for cid, indices in clusters.items()}
    else:
        allowed = set(statuses)
        eligible_clusters = {
            cid: tuple(
                idx
                for idx in indices
                if getattr(records[int(idx)], "status", None) in allowed
            )
            for cid, indices in clusters.items()
        }
        eligible_clusters = {
            cid: indices for cid, indices in eligible_clusters.items() if indices
        }

    cluster_ids = list(eligible_clusters.keys())
    if not cluster_ids:
        return []

    if cluster_probs is None:
        probs = np.full((len(cluster_ids),), 1.0 / len(cluster_ids), dtype=np.float64)
    else:
        raw = np.asarray([max(0.0, float(cluster_probs.get(cid, 0.0))) for cid in cluster_ids], dtype=np.float64)
        probs = _normalize_probs(raw)

    sampled: list[EpisodeSequence] = []
    for _ in range(n):
        cid = int(generator.choice(cluster_ids, p=probs))
        indices = eligible_clusters[cid]
        rec_idx = int(generator.choice(indices))
        sampled.append(records[rec_idx])
    return sampled


def burn_in_split(rec: EpisodeSequence, burn_in: int, total: int) -> tuple[slice, slice]:
    """Return burn-in and protected-loss index ranges for a replay window."""

    if burn_in < 0 or total < 0:
        raise ValueError("burn_in and total must be non-negative.")
    if burn_in > total:
        raise ValueError("burn_in cannot exceed total.")
    if total > seq_len(rec):
        raise ValueError("total cannot exceed the record length.")
    return slice(0, burn_in), slice(burn_in, total)


def split_reconstructed_obs(
    rec: EpisodeSequence, burn_in: int, total: int
) -> tuple[np.ndarray, np.ndarray]:
    """Return reconstructed observations for the burn-in and protected regions."""

    burn_slice, protected_slice = burn_in_split(rec, burn_in, total)
    obs = reconstruct_obs(rec)
    return obs[burn_slice], obs[protected_slice]


def _normalize_probs(values: np.ndarray) -> np.ndarray:
    total = float(np.sum(values))
    if total <= _EPS:
        if values.size == 0:
            return values.astype(np.float64)
        return np.full_like(values, 1.0 / values.size, dtype=np.float64)
    return (values / total).astype(np.float64)


__all__ = [
    "burn_in_split",
    "cluster_probs",
    "cluster_risk",
    "replay_transition_count",
    "sample_sequences",
    "split_reconstructed_obs",
]
