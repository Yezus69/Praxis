"""Exact post-update sentinel checks for protected recurrent behavior."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp

from tfns.behavior import (
    behavior_components,
    cosine_distance,
    huber,
    key_cos_tol,
    kl_categorical,
    kl_tol,
    router_tol,
    value_tol,
)


_MISSING = object()


def _get(obj: Any, *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    if default is not _MISSING:
        return default
    raise KeyError(f"missing required sentinel field; tried {names!r}")


def _tol(obj: Any, name: str, default: float) -> float:
    if obj is None:
        return float(default)
    if isinstance(obj, Mapping):
        return float(obj.get(name, default))
    return float(getattr(obj, name, default))


def _protected(x: Any, burn_in: int) -> jnp.ndarray:
    arr = jnp.asarray(x, dtype=jnp.float32)
    return arr[int(burn_in) :]


def _max_loss(x: jnp.ndarray) -> jnp.ndarray:
    flat = jnp.reshape(jnp.asarray(x, dtype=jnp.float32), (-1,))
    if int(flat.shape[0]) == 0:
        return jnp.asarray(jnp.inf, dtype=jnp.float32)
    return jnp.max(flat)


def _all_finite(*xs: Any) -> jnp.ndarray:
    ok = jnp.asarray(True)
    for x in xs:
        if x is None:
            continue
        leaves = jax.tree_util.tree_leaves(x)
        for leaf in leaves:
            arr = jnp.asarray(leaf)
            ok = jnp.logical_and(ok, jnp.all(jnp.isfinite(arr)))
    return ok


def _router_site(router: Any, site: str) -> Any:
    if router is None:
        return None
    if isinstance(router, Mapping):
        return router.get(site)
    return getattr(router, site, None)


def _router_drift(current: Any, target: Any, burn_in: int) -> tuple[jnp.ndarray, jnp.ndarray]:
    if target is None:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return zero, zero

    l1_values = []
    lmax_values = []
    for site in ("visual", "post"):
        cur_site = _router_site(current, site)
        target_site = _router_site(target, site)
        if cur_site is None or target_site is None:
            continue
        diff = jnp.abs(_protected(cur_site, burn_in) - _protected(target_site, burn_in))
        l1_values.append(_max_loss(jnp.sum(diff, axis=-1)))
        lmax_values.append(_max_loss(jnp.max(diff, axis=-1)))

    if not l1_values:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return zero, zero
    return jnp.max(jnp.stack(l1_values)), jnp.max(jnp.stack(lmax_values))


def sentinel_cluster_metrics(
    agent: Any,
    params: Any,
    cluster_batch: Any,
    *,
    burn_in: int,
    ema_key_anchor: Any = None,
) -> dict[str, jnp.ndarray]:
    """Return max protected-region sentinel metrics for one cluster.

    The stored sequence is unrolled from its initial hidden state through the
    burn-in region; only timesteps after ``burn_in`` contribute to gates.
    """

    obs_seq = _get(cluster_batch, "obs_seq", "obs")
    act_seq = _get(cluster_batch, "act_seq", "actions", "prev_action_seq")
    rew_seq = _get(cluster_batch, "rew_seq", "rewards", "prev_reward_seq")
    reset_seq = _get(cluster_batch, "reset_seq", "resets")
    adapter_dormant = _get(cluster_batch, "adapter_dormant", default=None)
    h0 = _get(cluster_batch, "h0", "hidden0", default=None)
    if h0 is None:
        h0 = agent.init_hidden(int(jnp.asarray(obs_seq).shape[1]))

    outputs, _ = agent.unroll(
        params,
        obs_seq,
        act_seq,
        rew_seq,
        reset_seq,
        h0,
        adapter_dormant=adapter_dormant,
    )

    teacher_probs = _get(cluster_batch, "teacher_probs", "policy_probs", default=None)
    teacher_logits = _get(cluster_batch, "teacher_logits", "policy_logits", default=None)
    teacher_value = _get(cluster_batch, "teacher_value", "value_target", "value")
    teacher_key = (
        ema_key_anchor
        if ema_key_anchor is not None
        else _get(cluster_batch, "teacher_key", "key_anchor", "q_key")
    )

    burn_in = int(burn_in)
    cur_logits = _protected(outputs.logits, burn_in)
    cur_value = _protected(outputs.value, burn_in)
    cur_key = _protected(outputs.q_key, burn_in)
    target_value = _protected(teacher_value, burn_in)
    target_key = _protected(teacher_key, burn_in)

    if teacher_probs is not None:
        kl = kl_categorical(_protected(teacher_probs, burn_in), cur_logits)
        value_err = huber(cur_value - jax.lax.stop_gradient(target_value))
        key_dist = cosine_distance(cur_key, jax.lax.stop_gradient(target_key))
    else:
        comps = behavior_components(
            _protected(teacher_logits, burn_in),
            target_value,
            target_key,
            cur_logits,
            cur_value,
            cur_key,
        )
        kl = comps["kl"]
        value_err = comps["value_err"]
        key_dist = comps["key_dist"]

    target_router = _get(
        cluster_batch,
        "teacher_router_weights",
        "router_weights",
        "router_anchor",
        default=None,
    )
    router_l1, router_lmax = _router_drift(outputs.router_weights, target_router, burn_in)
    router_drift = jnp.maximum(router_l1, router_lmax)
    finite = _all_finite(
        outputs.logits,
        outputs.value,
        outputs.q_key,
        outputs.h_next,
        outputs.router_weights,
        kl,
        value_err,
        key_dist,
        router_drift,
    )

    return {
        "kl": _max_loss(kl),
        "value_err": _max_loss(value_err),
        "key_dist": _max_loss(key_dist),
        "router_drift": router_drift,
        "router_l1": router_l1,
        "router_lmax": router_lmax,
        "non_finite": jnp.logical_not(finite),
    }


def make_sentinel_acceptor(agent: Any, clusters: Sequence[Any], tols: Any) -> Any:
    """Return ``accept_fn(candidate_params)`` with correctly oriented max gates."""

    thresholds = {
        "kl": _tol(tols, "kl_tol", kl_tol),
        "value_err": _tol(tols, "value_tol", value_tol),
        "key_dist": _tol(tols, "key_cos_tol", key_cos_tol),
        "router_drift": _tol(tols, "router_tol", router_tol),
    }
    default_burn_in = int(_tol(tols, "burn_in", 0.0))
    last_metrics: list[dict[str, jnp.ndarray]] = []

    def accept_fn(candidate_params: Any) -> bool:
        nonlocal last_metrics
        last_metrics = []
        for cluster in clusters:
            burn_in = int(_get(cluster, "burn_in", default=default_burn_in))
            key_anchor = _get(cluster, "ema_key_anchor", default=None)
            metrics = sentinel_cluster_metrics(
                agent,
                candidate_params,
                cluster,
                burn_in=burn_in,
                ema_key_anchor=key_anchor,
            )
            last_metrics.append(metrics)
            if bool(metrics["non_finite"]):
                accept_fn.last_metrics = last_metrics
                return False
            if float(metrics["kl"]) > thresholds["kl"]:
                accept_fn.last_metrics = last_metrics
                return False
            if float(metrics["value_err"]) > thresholds["value_err"]:
                accept_fn.last_metrics = last_metrics
                return False
            if float(metrics["key_dist"]) > thresholds["key_dist"]:
                accept_fn.last_metrics = last_metrics
                return False
            if float(metrics["router_drift"]) > thresholds["router_drift"]:
                accept_fn.last_metrics = last_metrics
                return False
        accept_fn.last_metrics = last_metrics
        return True

    accept_fn.thresholds = thresholds
    accept_fn.last_metrics = last_metrics
    return accept_fn


__all__ = [
    "make_sentinel_acceptor",
    "sentinel_cluster_metrics",
]
