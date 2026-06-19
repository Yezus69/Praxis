"""Persistent mutable state for continual TFNS training."""

from __future__ import annotations

import copy
import dataclasses
from collections.abc import Mapping
from typing import Any

try:
    from flax.core import FrozenDict, freeze
except ImportError:  # pragma: no cover
    FrozenDict = ()  # type: ignore[assignment]
    freeze = None  # type: ignore[assignment]

import jax
import numpy as np


@dataclasses.dataclass
class ContinualState:
    """Mutable container for every continued-learning component."""

    params: Any
    opt_state: Any
    ema_params: Any
    bases: dict[str, Any]
    memory: Any
    predictor_params: Any
    predictor_opt_state: Any
    detector_state: Any
    adapter_dormant: Any
    robust_stats: dict[str, Any]
    protected_clusters: list[Any]
    rng: Any
    block_index: int = 0
    skills: dict[str, Any] = dataclasses.field(default_factory=dict)
    rollout_carry: Any = None


@dataclasses.dataclass(frozen=True)
class Snapshot:
    params: Any
    opt_state: Any
    ema_params: Any
    bases: dict[str, Any]
    memory: Any
    predictor_params: Any
    predictor_opt_state: Any
    detector_state: Any
    adapter_dormant: Any
    robust_stats: dict[str, Any]
    protected_clusters: list[Any]
    rng: Any
    block_index: int
    skills: dict[str, Any]
    rollout_carry: Any


def _is_frozen_dict(value: Any) -> bool:
    return FrozenDict != () and isinstance(value, FrozenDict)


def _is_namedtuple(value: Any) -> bool:
    return isinstance(value, tuple) and hasattr(value, "_fields")


def _is_jax_array(value: Any) -> bool:
    return hasattr(value, "__jax_array__") or type(value).__module__.startswith("jax")


def _clone(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if _is_jax_array(value):
        return value
    if _is_frozen_dict(value):
        return freeze({key: _clone(item) for key, item in value.items()})
    if isinstance(value, Mapping):
        return {copy.deepcopy(key): _clone(item) for key, item in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(
            value,
            **{field.name: _clone(getattr(value, field.name)) for field in dataclasses.fields(value)},
        )
    if _is_namedtuple(value):
        return type(value)(*(_clone(item) for item in value))
    if isinstance(value, tuple):
        return tuple(_clone(item) for item in value)
    if isinstance(value, list):
        return [_clone(item) for item in value]
    return copy.deepcopy(value)


def _to_numpy(value: Any) -> Any:
    if isinstance(value, np.ndarray):
        return value.copy()
    if _is_jax_array(value):
        return np.asarray(value)
    if _is_frozen_dict(value):
        return {key: _to_numpy(item) for key, item in value.items()}
    if isinstance(value, Mapping):
        return {copy.deepcopy(key): _to_numpy(item) for key, item in value.items()}
    if dataclasses.is_dataclass(value) and not isinstance(value, type):
        return dataclasses.replace(
            value,
            **{field.name: _to_numpy(getattr(value, field.name)) for field in dataclasses.fields(value)},
        )
    if _is_namedtuple(value):
        return type(value)(*(_to_numpy(item) for item in value))
    if isinstance(value, tuple):
        return tuple(_to_numpy(item) for item in value)
    if isinstance(value, list):
        return [_to_numpy(item) for item in value]
    return copy.deepcopy(value)


def ema_update(ema_params: Any, params: Any, decay: float) -> Any:
    """Return leafwise ``decay * ema + (1 - decay) * params``."""

    if ema_params is None:
        return _clone(params)
    decay = float(decay)
    return jax.tree_util.tree_map(lambda ema, p: decay * ema + (1.0 - decay) * p, ema_params, params)


def snapshot(state: ContinualState) -> Snapshot:
    """Deep-copy all mutable continued-learning state for rollback."""

    return Snapshot(
        params=_clone(state.params),
        opt_state=_clone(state.opt_state),
        ema_params=_clone(state.ema_params),
        bases=_clone(state.bases),
        memory=_clone(state.memory),
        predictor_params=_clone(state.predictor_params),
        predictor_opt_state=_clone(state.predictor_opt_state),
        detector_state=_clone(state.detector_state),
        adapter_dormant=_clone(state.adapter_dormant),
        robust_stats=_clone(state.robust_stats),
        protected_clusters=_clone(state.protected_clusters),
        rng=_clone(state.rng),
        block_index=int(state.block_index),
        skills=_clone(state.skills),
        rollout_carry=_clone(state.rollout_carry),
    )


def restore(state: ContinualState, snap: Snapshot) -> ContinualState:
    """Atomically restore every mutable component from ``snap``."""

    restored = {
        "params": _clone(snap.params),
        "opt_state": _clone(snap.opt_state),
        "ema_params": _clone(snap.ema_params),
        "bases": _clone(snap.bases),
        "memory": _clone(snap.memory),
        "predictor_params": _clone(snap.predictor_params),
        "predictor_opt_state": _clone(snap.predictor_opt_state),
        "detector_state": _clone(snap.detector_state),
        "adapter_dormant": _clone(snap.adapter_dormant),
        "robust_stats": _clone(snap.robust_stats),
        "protected_clusters": _clone(snap.protected_clusters),
        "rng": _clone(snap.rng),
        "block_index": int(snap.block_index),
        "skills": _clone(snap.skills),
        "rollout_carry": _clone(snap.rollout_carry),
    }
    for name, value in restored.items():
        setattr(state, name, value)
    return state


def serialize(state: ContinualState) -> dict[str, Any]:
    """Return a pickle-able NumPy/Python representation of ``state``."""

    return {
        "version": 1,
        "params": _to_numpy(state.params),
        "opt_state": _to_numpy(state.opt_state),
        "ema_params": _to_numpy(state.ema_params),
        "bases": _to_numpy(state.bases),
        "memory": _to_numpy(state.memory),
        "predictor_params": _to_numpy(state.predictor_params),
        "predictor_opt_state": _to_numpy(state.predictor_opt_state),
        "detector_state": _to_numpy(state.detector_state),
        "adapter_dormant": _to_numpy(state.adapter_dormant),
        "robust_stats": _to_numpy(state.robust_stats),
        "protected_clusters": _to_numpy(state.protected_clusters),
        "rng": _to_numpy(state.rng),
        "block_index": int(state.block_index),
        "skills": _to_numpy(state.skills),
        "rollout_carry": _to_numpy(state.rollout_carry),
    }


def deserialize(payload: Mapping[str, Any]) -> ContinualState:
    """Rebuild a ``ContinualState`` from ``serialize`` output."""

    if int(payload.get("version", 1)) != 1:
        raise ValueError(f"unsupported continual-state version {payload.get('version')!r}")
    return ContinualState(
        params=_clone(payload["params"]),
        opt_state=_clone(payload["opt_state"]),
        ema_params=_clone(payload["ema_params"]),
        bases=_clone(payload["bases"]),
        memory=_clone(payload["memory"]),
        predictor_params=_clone(payload["predictor_params"]),
        predictor_opt_state=_clone(payload["predictor_opt_state"]),
        detector_state=_clone(payload["detector_state"]),
        adapter_dormant=_clone(payload["adapter_dormant"]),
        robust_stats=_clone(payload["robust_stats"]),
        protected_clusters=_clone(payload["protected_clusters"]),
        rng=_clone(payload["rng"]),
        block_index=int(payload.get("block_index", 0)),
        skills=_clone(payload.get("skills", {})),
        rollout_carry=_clone(payload.get("rollout_carry")),
    )


__all__ = [
    "ContinualState",
    "Snapshot",
    "deserialize",
    "ema_update",
    "restore",
    "serialize",
    "snapshot",
]
