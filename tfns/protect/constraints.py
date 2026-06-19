"""Multiple non-cancelling behavior-margin constraints."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax
from jax.flatten_util import ravel_pytree
import jax.numpy as jnp
import numpy as np
from scipy.linalg import solve_triangular
from scipy.optimize import nnls

from tfns.protect.projection import ProtectedModule, project_update


class ConstraintSolverError(RuntimeError):
    """Raised when a constrained correction cannot be safely computed."""


def _nonfinite_info(active: int = 0) -> dict[str, Any]:
    return {
        "active": int(active),
        "qp_residual": np.inf,
        "lambda": None,
        "failed": True,
    }


def solve_constrained_qp(
    delta0: jnp.ndarray,
    G: jnp.ndarray,
    m: jnp.ndarray,
    ridge: float,
) -> tuple[jnp.ndarray | None, dict[str, Any]]:
    """Project ``delta0`` onto first-order constraints ``G @ delta <= m``.

    Only predicted-violated rows are kept.  The nonnegative dual is solved as
    the equivalent tiny NNLS problem from section 14.
    """

    try:
        delta0_np = np.asarray(jax.device_get(delta0), dtype=np.float64).reshape(-1)
        G_np = np.asarray(jax.device_get(G), dtype=np.float64)
        m_np = np.asarray(jax.device_get(m), dtype=np.float64).reshape(-1)
    except Exception:
        return None, _nonfinite_info()

    if G_np.ndim != 2 or delta0_np.ndim != 1 or m_np.ndim != 1:
        return None, _nonfinite_info()
    if G_np.shape[1] != delta0_np.shape[0] or G_np.shape[0] != m_np.shape[0]:
        return None, _nonfinite_info()
    if float(ridge) <= 0.0:
        return None, _nonfinite_info()
    if not (
        np.all(np.isfinite(delta0_np))
        and np.all(np.isfinite(G_np))
        and np.all(np.isfinite(m_np))
    ):
        return None, _nonfinite_info()

    predicted = G_np @ delta0_np
    active_mask = predicted > m_np
    active_count = int(np.count_nonzero(active_mask))
    dtype = jnp.asarray(delta0).dtype
    if active_count == 0:
        return jnp.asarray(delta0_np, dtype=dtype), {
            "active": 0,
            "qp_residual": float(np.max(predicted - m_np)) if m_np.size else -np.inf,
            "lambda": jnp.zeros((G_np.shape[0],), dtype=dtype),
            "failed": False,
        }

    G_active = G_np[active_mask]
    m_active = m_np[active_mask]
    b = G_active @ delta0_np - m_active
    try:
        A = G_active @ G_active.T + float(ridge) * np.eye(active_count, dtype=np.float64)
        L = np.linalg.cholesky(A)
        rhs = solve_triangular(L, b, lower=True, check_finite=True)
        lam_active, _ = nnls(L.T, rhs)
        delta_np = delta0_np - G_active.T @ lam_active
    except Exception:
        return None, _nonfinite_info(active_count)

    if not (np.all(np.isfinite(lam_active)) and np.all(np.isfinite(delta_np))):
        return None, _nonfinite_info(active_count)

    residuals = G_np @ delta_np - m_np
    if not np.all(np.isfinite(residuals)):
        return None, _nonfinite_info(active_count)

    lam_full = np.zeros((G_np.shape[0],), dtype=np.float64)
    lam_full[active_mask] = lam_active
    return jnp.asarray(delta_np, dtype=dtype), {
        "active": active_count,
        "qp_residual": float(np.max(residuals)) if residuals.size else -np.inf,
        "lambda": jnp.asarray(lam_full, dtype=dtype),
        "failed": False,
    }


def _dmax_value(d_max: Any, index: int, cluster: Any) -> float:
    if isinstance(d_max, Mapping):
        for key in (cluster, getattr(cluster, "id", None), getattr(cluster, "cluster_id", None), index):
            try:
                if key is not None and key in d_max:
                    return float(d_max[key])
            except TypeError:
                continue
        raise KeyError(f"no d_max entry for cluster index {index}")
    if isinstance(d_max, Sequence) and not isinstance(d_max, (str, bytes)):
        return float(d_max[index])
    arr = np.asarray(d_max)
    if arr.ndim > 0:
        return float(arr[index])
    return float(arr)


def make_constraint_fn(
    cluster_distance_and_grad: Any,
    clusters: Sequence[Any],
    d_max: Any,
    bases: Mapping[str, jnp.ndarray],
    modules: Mapping[str, ProtectedModule],
    *,
    ridge: float,
    max_clusters: int = 8,
) -> Any:
    """Return an optimizer ``constraint_fn`` for section-14 QP correction.

    Solver failures raise :class:`ConstraintSolverError`; they are not allowed
    to pass through as an unchanged unsafe update.
    """

    last_info: dict[str, Any] = {"active": 0, "failed": False}

    def constraint_fn(updates_safe: Any, params: Any) -> Any:
        nonlocal last_info
        delta0, unravel = ravel_pytree(updates_safe)
        update_structure = jax.tree_util.tree_structure(updates_safe)

        scored: list[tuple[float, int, Any, Any]] = []
        for index, cluster in enumerate(clusters):
            D_i, grad_tree = cluster_distance_and_grad(params, cluster)
            D_scalar = float(np.asarray(jax.device_get(D_i), dtype=np.float64))
            scored.append((D_scalar, index, cluster, grad_tree))

        scored.sort(key=lambda item: item[0], reverse=True)
        selected = scored[: int(max_clusters)]
        if not selected:
            last_info = {"active": 0, "failed": False}
            constraint_fn.last_info = last_info
            return updates_safe

        rows = []
        margins = []
        for D_scalar, index, cluster, grad_tree in selected:
            grad_safe = project_update(grad_tree, bases, modules)
            if jax.tree_util.tree_structure(grad_safe) != update_structure:
                raise ConstraintSolverError("projected gradient tree does not match update tree")
            g_flat, _ = ravel_pytree(grad_safe)
            rows.append(g_flat)
            margins.append(_dmax_value(d_max, index, cluster) - D_scalar)

        G = jnp.stack(rows, axis=0)
        m = jnp.asarray(margins, dtype=delta0.dtype)
        delta, info = solve_constrained_qp(delta0, G, m, ridge)
        last_info = dict(info)
        constraint_fn.last_info = last_info
        if delta is None:
            raise ConstraintSolverError("behavior constraint QP failed")
        return unravel(delta)

    constraint_fn.last_info = last_info
    return constraint_fn


__all__ = [
    "ConstraintSolverError",
    "make_constraint_fn",
    "solve_constrained_qp",
]
