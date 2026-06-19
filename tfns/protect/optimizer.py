"""Optimizer-safe protected TFNS update helpers."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import jax
import jax.numpy as jnp
import optax

from tfns.protect.projection import ProtectedModule, project_update
from tfns.utils import tree_add_scaled, tree_global_norm


_MISSING = object()


def _is_namedtuple(value: Any) -> bool:
    return isinstance(value, tuple) and hasattr(value, "_fields") and hasattr(value, "_replace")


def _find_named_field(tree: Any, field: str) -> Any:
    if _is_namedtuple(tree) and field in tree._fields:
        return getattr(tree, field)
    if isinstance(tree, tuple):
        for item in tree:
            found = _find_named_field(item, field)
            if found is not _MISSING:
                return found
    return _MISSING


def _replace_named_field(tree: Any, field: str, value: Any) -> tuple[Any, bool]:
    if _is_namedtuple(tree) and field in tree._fields:
        return tree._replace(**{field: value}), True
    if isinstance(tree, tuple):
        changed = False
        items = []
        for item in tree:
            if changed:
                items.append(item)
                continue
            new_item, changed = _replace_named_field(item, field, value)
            items.append(new_item)
        if changed:
            return tuple(items), True
    return tree, False


def _tree_get_mu(opt_state: optax.OptState) -> Any:
    tree_utils = getattr(optax, "tree_utils", None)
    tree_get = getattr(tree_utils, "tree_get", None)
    if tree_get is not None:
        try:
            return tree_get(opt_state, "mu")
        except (AttributeError, KeyError, TypeError, ValueError):
            pass

    mu = _find_named_field(opt_state, "mu")
    if mu is _MISSING:
        raise ValueError("could not locate Adam first-moment tree 'mu'")
    return mu


def _tree_set_mu(opt_state: optax.OptState, mu: Any) -> optax.OptState:
    tree_utils = getattr(optax, "tree_utils", None)
    tree_set = getattr(tree_utils, "tree_set", None)
    if tree_set is not None:
        try:
            return tree_set(opt_state, mu=mu)
        except (AttributeError, KeyError, TypeError, ValueError):
            pass

    updated, found = _replace_named_field(opt_state, "mu", mu)
    if not found:
        raise ValueError("could not write Adam first-moment tree 'mu'")
    return updated


def project_first_moments(
    opt_state: optax.OptState,
    bases: Mapping[str, jnp.ndarray],
    modules: Mapping[str, ProtectedModule],
) -> optax.OptState:
    """Project Adam first moments while leaving count and second moments intact."""

    mu = _tree_get_mu(opt_state)
    mu_safe = project_update(mu, bases, modules)
    return _tree_set_mu(opt_state, mu_safe)


def _tree_mul(tree: Any, scale: Any) -> Any:
    scale = jnp.asarray(scale, dtype=jnp.float32)
    return jax.tree_util.tree_map(lambda leaf: leaf * scale, tree)


def optimizer_safe_step(
    params: Any,
    opt_state: optax.OptState,
    grad: Any,
    tx: optax.GradientTransformation,
    bases: Mapping[str, jnp.ndarray],
    modules: Mapping[str, ProtectedModule],
    *,
    max_update_norm: float | None = None,
    accept_fn: Any = None,
    backtrack_scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125),
    constraint_fn: Any = None,
) -> tuple[Any, optax.OptState, dict[str, Any]]:
    """Apply the section-13 protected Adam sequence as one pure step."""

    raw_grad_norm = tree_global_norm(grad)

    g_proj = project_update(grad, bases, modules)
    updates, cand_state = tx.update(g_proj, opt_state, params)
    candidate_delta_norm = tree_global_norm(updates)

    updates_safe = project_update(updates, bases, modules)
    projected_delta_norm = tree_global_norm(updates_safe)

    cand_state = project_first_moments(cand_state, bases, modules)

    if constraint_fn is not None:
        updates_safe = constraint_fn(updates_safe, params)

    if max_update_norm is not None:
        norm = tree_global_norm(updates_safe)
        max_norm = jnp.asarray(max_update_norm, dtype=jnp.float32)
        scale = jnp.minimum(jnp.asarray(1.0, dtype=jnp.float32), max_norm / (norm + 1.0e-8))
        updates_safe = _tree_mul(updates_safe, scale)

    accepted = False
    applied_scale = 0.0
    n_backtracks = 0
    new_params = params
    new_opt_state = opt_state

    if accept_fn is None:
        applied_scale = 1.0
        new_params = tree_add_scaled(params, updates_safe, applied_scale)
        new_opt_state = cand_state
        accepted = True
    else:
        scales = tuple(float(alpha) for alpha in backtrack_scales)
        for index, alpha in enumerate(scales):
            cand_params = tree_add_scaled(params, updates_safe, alpha)
            if bool(accept_fn(cand_params)):
                applied_scale = alpha
                n_backtracks = index
                new_params = cand_params
                new_opt_state = cand_state
                accepted = True
                break
        if not accepted:
            n_backtracks = len(scales)

    if accepted:
        applied_norm = tree_global_norm(_tree_mul(updates_safe, applied_scale))
    else:
        applied_norm = jnp.asarray(0.0, dtype=jnp.float32)

    info = {
        "raw_grad_norm": raw_grad_norm,
        "candidate_delta_norm": candidate_delta_norm,
        "projected_delta_norm": projected_delta_norm,
        "applied_norm": applied_norm,
        "applied_scale": applied_scale if accepted else 0.0,
        "accepted": accepted,
        "n_backtracks": n_backtracks,
    }
    return new_params, new_opt_state, info


__all__ = [
    "optimizer_safe_step",
    "project_first_moments",
]
