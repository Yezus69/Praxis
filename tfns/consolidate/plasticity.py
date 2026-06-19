"""Plasticity telemetry and residual-adapter activation helpers."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any

import jax.numpy as jnp
import numpy as np

from tfns.protect.bases import empty_basis, free_rank_fraction


_EPS = 1.0e-8


def _get(obj: Any, *names: str, default: Any = None) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    return default


def _adapter_cfg(cfg: Any) -> Any:
    return _get(cfg, "adapter", default=cfg)


def _finite_pos(value: Any) -> bool:
    try:
        val = float(value)
    except (TypeError, ValueError):
        return False
    return bool(np.isfinite(val) and val > 0.0)


def _module_entry(block: Any, name: str) -> Any:
    for container_name in ("modules", "per_module", "module_stats", "plasticity"):
        container = _get(block, container_name)
        if isinstance(container, Mapping) and name in container:
            return container[name]
    if isinstance(block, Mapping) and name in block and isinstance(block[name], Mapping):
        return block[name]
    return block


def _ratio_from(entry: Any) -> float | None:
    direct = _get(
        entry,
        "rho",
        "plasticity_ratio",
        "protected_update_ratio",
        "median_rho",
        default=None,
    )
    if direct is not None:
        try:
            value = float(direct)
        except (TypeError, ValueError):
            return None
        return value if np.isfinite(value) else None

    applied = _get(entry, "applied_norm", "safe_delta_norm", default=None)
    candidate = _get(entry, "candidate_delta_norm", "candidate_norm", default=None)
    if applied is None or candidate is None:
        return None
    candidate = float(candidate)
    if not np.isfinite(candidate) or candidate < 0.0:
        return None
    value = float(applied) / (candidate + _EPS)
    return value if np.isfinite(value) else None


def _history(history: Iterable[Any] | None) -> list[Any]:
    if history is None:
        return []
    return list(history)


def plasticity_report(state: Any, modules: Mapping[str, Any], telemetry_history: Iterable[Any]) -> dict[str, Any]:
    """Return per-module free rank and recent applied-update ratio telemetry."""

    blocks = _history(telemetry_history)
    module_reports: dict[str, dict[str, Any]] = {}
    aggregate_rhos: list[float] = []

    for name, module in modules.items():
        d_aug = int(module.d_aug)
        U = state.bases.get(name)
        if U is None:
            U = empty_basis(d_aug)

        ratios = []
        grad_norms = []
        dead_units = []
        for block in blocks:
            entry = _module_entry(block, name)
            ratio = _ratio_from(entry)
            if ratio is not None:
                ratios.append(ratio)
            grad_norm = _get(entry, "grad_norm", "raw_grad_norm", default=None)
            if grad_norm is not None and np.isfinite(float(grad_norm)):
                grad_norms.append(float(grad_norm))
            dead = _get(entry, "dead_unit_fraction", "dead_units", default=None)
            if dead is not None and np.isfinite(float(dead)):
                dead_units.append(float(dead))

        median_rho = float(np.median(ratios)) if ratios else float("nan")
        if np.isfinite(median_rho):
            aggregate_rhos.append(median_rho)
        module_reports[name] = {
            "free_rank": free_rank_fraction(U, d_aug),
            "rank": int(jnp.asarray(U).shape[1]),
            "d_aug": d_aug,
            "rho": median_rho,
            "dead_unit_fraction": float(np.median(dead_units)) if dead_units else None,
            "grad_norm": float(np.median(grad_norms)) if grad_norms else None,
        }

    return {
        "modules": module_reports,
        "median_rho": float(np.median(aggregate_rhos)) if aggregate_rhos else float("nan"),
    }


def _block_rho(block: Any) -> float | None:
    ratio = _ratio_from(block)
    if ratio is not None:
        return ratio

    ratios = []
    for container_name in ("modules", "per_module", "module_stats", "plasticity"):
        container = _get(block, container_name)
        if isinstance(container, Mapping):
            for entry in container.values():
                entry_ratio = _ratio_from(entry)
                if entry_ratio is not None:
                    ratios.append(entry_ratio)
    return float(np.median(ratios)) if ratios else None


def _score(block: Any) -> float | None:
    value = _get(block, "score", "current_score", "current_progress", "eval_score", default=None)
    if value is None:
        return None
    try:
        value = float(value)
    except (TypeError, ValueError):
        return None
    return value if np.isfinite(value) else None


def _active_losses(block: Any) -> bool:
    replay = _get(
        block,
        "replay_loss",
        "replay_tube_loss",
        "tube_loss",
        "L_replay",
        default=None,
    )
    cons = _get(
        block,
        "conservation_loss",
        "constraint_loss",
        "L_cons",
        "behavior_tube_loss",
        default=None,
    )
    return _finite_pos(replay) and _finite_pos(cons)


def _attributable_to_protection(block: Any, ratio_thresh: float) -> bool:
    raw_grad = _get(block, "raw_grad_norm", "grad_norm", default=None)
    if not _finite_pos(raw_grad):
        return False

    candidate = _get(block, "candidate_delta_norm", "candidate_norm", default=None)
    projected = _get(block, "projected_delta_norm", "safe_delta_norm", default=None)
    if candidate is not None and projected is not None:
        candidate = float(candidate)
        projected = float(projected)
        if not (np.isfinite(candidate) and np.isfinite(projected) and candidate > 0.0):
            return False
        return bool(projected / (candidate + _EPS) < max(float(ratio_thresh), 0.25))

    ratio = _block_rho(block)
    return bool(ratio is not None and ratio < ratio_thresh)


def should_activate_adapter(history: Iterable[Any], cfg: Any) -> bool:
    """Return whether recent blocks satisfy the section-17 adapter trigger."""

    a_cfg = _adapter_cfg(cfg)
    patience = int(_get(a_cfg, "patience_blocks", default=3))
    ratio_thresh = float(_get(a_cfg, "plasticity_ratio_thresh", default=0.1))
    min_improvement = float(_get(a_cfg, "score_improvement_min", default=1.0e-6))

    blocks = _history(history)
    if patience <= 0 or len(blocks) < patience:
        return False
    recent = blocks[-patience:]

    rhos = [_block_rho(block) for block in recent]
    if any(rho is None or not np.isfinite(float(rho)) or float(rho) >= ratio_thresh for rho in rhos):
        return False

    scores = [_score(block) for block in recent]
    if any(score is None for score in scores):
        return False
    if float(scores[-1]) > float(scores[0]) + min_improvement:
        return False

    if not all(_active_losses(block) for block in recent):
        return False
    if not all(_attributable_to_protection(block, ratio_thresh) for block in recent):
        return False
    return True


def activate_adapter(state: Any) -> tuple[Any, int | None]:
    """Activate the lowest-index dormant adapter, if any capacity remains."""

    dormant = state.adapter_dormant
    arr = np.asarray(dormant, dtype=np.bool_)
    indices = np.flatnonzero(arr)
    if indices.size == 0:
        return state, None

    idx = int(indices[0])
    if isinstance(dormant, np.ndarray):
        new_dormant = dormant.copy()
        new_dormant[idx] = False
    elif hasattr(dormant, "at"):
        new_dormant = dormant.at[idx].set(False)
    else:
        new_dormant = arr.copy()
        new_dormant[idx] = False
        if isinstance(dormant, tuple):
            new_dormant = tuple(bool(x) for x in new_dormant)
        elif isinstance(dormant, list):
            new_dormant = [bool(x) for x in new_dormant]

    state.adapter_dormant = new_dormant
    return state, idx


__all__ = [
    "activate_adapter",
    "plasticity_report",
    "should_activate_adapter",
]
