"""Rollout and probe atom labeling for the coverage CSN-PPO loop."""

import jax
import jax.numpy as jnp
import numpy as np

from agent.csn_ppo import criticality_coverage as cc
from agent.csn_ppo import mosaic_teacher
from agent.csn_ppo.memory import (
    SOURCE_RECENT_CURRENT,
    SOURCE_SYNTHETIC_PROBE,
    BehavioralMemoryBatch,
    concat_memory_batches,
)


def _teacher_is_available(teacher_normalizer, teacher_params):
    return teacher_normalizer is not None and teacher_params is not None


def _cluster_teacher(champions, cluster_id: int):
    if champions is None:
        return None, None
    try:
        return mosaic_teacher.get_cluster_teacher(
            champions,
            cluster_id,
            include_fallback=False,
        )
    except (AttributeError, IndexError):
        return None, None


def _global_teacher(global_champion, current_normalizer, current_params):
    if global_champion is None:
        return None, None
    return mosaic_teacher.teacher_snapshot(
        global_champion,
        current_normalizer,
        current_params,
        allow_current_fallback=False,
    )


def _has_any_cluster_champion(champions, num_clusters: int):
    if champions is None:
        return False
    for cluster_id in range(num_clusters):
        teacher_normalizer, teacher_params = _cluster_teacher(champions, cluster_id)
        if _teacher_is_available(teacher_normalizer, teacher_params):
            return True
    return False


def _has_global_champion(global_champion):
    if global_champion is None:
        return False
    return mosaic_teacher.has_champion(global_champion)


def _num_clusters(cfg, cluster_id_host):
    if hasattr(cfg, "num_clusters"):
        return int(cfg.num_clusters)
    if cluster_id_host.size == 0:
        return 0
    return int(np.max(cluster_id_host)) + 1


def _take_memory_batch(batch, idx):
    return BehavioralMemoryBatch(
        obs=batch.obs[idx],
        mean=batch.mean[idx],
        logstd=batch.logstd[idx],
        value=batch.value[idx],
        weight=batch.weight[idx],
        kl_budget=batch.kl_budget[idx],
        value_budget=batch.value_budget[idx],
        cluster_id=batch.cluster_id[idx],
        source_id=batch.source_id[idx],
    )


def _empty_memory_batch(obs, cfg):
    obs_dim = obs.shape[-1]
    action_dim = int(cfg.action_dim)
    return BehavioralMemoryBatch(
        obs=jnp.zeros((0, obs_dim), dtype=obs.dtype),
        mean=jnp.zeros((0, action_dim), dtype=jnp.float32),
        logstd=jnp.zeros((0, action_dim), dtype=jnp.float32),
        value=jnp.zeros((0,), dtype=jnp.float32),
        weight=jnp.zeros((0,), dtype=jnp.float32),
        kl_budget=jnp.zeros((0,), dtype=jnp.float32),
        value_budget=jnp.zeros((0,), dtype=jnp.float32),
        cluster_id=jnp.zeros((0,), dtype=jnp.int32),
        source_id=jnp.zeros((0,), dtype=jnp.int32),
    )


def _label_atoms_with_mosaic_teacher_and_indices(
    obs,
    cluster_id,
    champions,
    global_champion,
    current_params,
    current_normalizer,
    apply_policy_value,
    cfg,
):
    obs = jnp.asarray(obs)
    cluster_id = jnp.asarray(cluster_id, dtype=jnp.int32)
    cluster_id_host = np.asarray(jax.device_get(cluster_id), dtype=np.int32)
    num_clusters = _num_clusters(cfg, cluster_id_host)
    has_any_champion = (
        _has_any_cluster_champion(champions, num_clusters)
        or _has_global_champion(global_champion)
    )

    batches = []
    batch_indices = []
    for cid in range(num_clusters):
        idx = np.nonzero(cluster_id_host == cid)[0]
        if idx.size == 0:
            continue

        teacher_normalizer, teacher_params = _cluster_teacher(champions, cid)
        if not _teacher_is_available(teacher_normalizer, teacher_params):
            teacher_normalizer, teacher_params = _global_teacher(
                global_champion,
                current_normalizer,
                current_params,
            )
        if (
            not _teacher_is_available(teacher_normalizer, teacher_params)
            and not has_any_champion
        ):
            teacher_normalizer = current_normalizer
            teacher_params = current_params
        if not _teacher_is_available(teacher_normalizer, teacher_params):
            continue

        idx_jax = jnp.asarray(idx, dtype=jnp.int32)
        obs_c = obs[idx_jax]
        mean, logstd, value = apply_policy_value(
            teacher_params,
            teacher_normalizer,
            obs_c,
        )
        logstd = jnp.maximum(logstd, cfg.teacher_logstd_floor)
        criticality = jax.vmap(lambda o: cc.criticality_score(o, 0.0, cfg))(obs_c)
        batches.append(
            BehavioralMemoryBatch(
                obs=obs_c,
                mean=mean,
                logstd=logstd,
                value=value,
                weight=jax.vmap(lambda x: cc.memory_weight(x, cfg))(criticality),
                kl_budget=jax.vmap(lambda x: cc.kl_budget_from_c(x, cfg))(criticality),
                value_budget=jax.vmap(lambda x: cc.value_budget_from_c(x, cfg))(criticality),
                cluster_id=jnp.full((idx.size,), cid, dtype=jnp.int32),
                source_id=jnp.full((idx.size,), SOURCE_RECENT_CURRENT, dtype=jnp.int32),
            )
        )
        batch_indices.append(idx)

    if not batches:
        return _empty_memory_batch(obs, cfg), np.zeros((0,), dtype=np.int32)

    batch = concat_memory_batches(*batches)
    original_idx = np.concatenate(batch_indices)
    order = np.argsort(original_idx)
    return _take_memory_batch(batch, jnp.asarray(order, dtype=jnp.int32)), original_idx[order]


def label_atoms_with_mosaic_teacher(
    obs,
    cluster_id,
    champions,
    global_champion,
    current_params,
    current_normalizer,
    apply_policy_value,
    cfg,
) -> BehavioralMemoryBatch:
    """Labels atoms with per-cluster champions, then global, then early current."""
    batch, _ = _label_atoms_with_mosaic_teacher_and_indices(
        obs,
        cluster_id,
        champions,
        global_champion,
        current_params,
        current_normalizer,
        apply_policy_value,
        cfg,
    )
    return batch


def mine_atoms(
    obs_flat,
    adv_abs,
    params,
    normalizer_params,
    apply_policy_value,
    cfg,
    champions=None,
    global_champion=None,
):
    """Mines fixed-shape top-criticality atoms from current rollout observations."""
    crit = jax.vmap(lambda o, a: cc.criticality_score(o, a, cfg))(obs_flat, adv_abs)
    k = cfg.atoms_per_rollout
    _, idx = jax.lax.top_k(crit, k)
    obs = obs_flat[idx]
    c = crit[idx]

    cid = jax.vmap(cc.cluster_id_for)(obs)
    batch, kept_idx = _label_atoms_with_mosaic_teacher_and_indices(
        obs,
        cid,
        champions,
        global_champion,
        params,
        normalizer_params,
        apply_policy_value,
        cfg,
    )
    c = c[jnp.asarray(kept_idx, dtype=jnp.int32)]
    batch = batch.replace(
        weight=jax.vmap(lambda x: cc.memory_weight(x, cfg))(c),
        kl_budget=jax.vmap(lambda x: cc.kl_budget_from_c(x, cfg))(c),
        value_budget=jax.vmap(lambda x: cc.value_budget_from_c(x, cfg))(c),
        source_id=jnp.full((batch.obs.shape[0],), SOURCE_RECENT_CURRENT, dtype=jnp.int32),
    )
    slow_mask = c > cfg.slow_memory_threshold
    sorted_crit = jnp.sort(crit)
    p95_idx = jnp.asarray(0.95 * (crit.shape[0] - 1), dtype=jnp.int32)
    return batch, slow_mask, {
        "mine/crit_mean": jnp.mean(crit),
        "mine/crit_p95": sorted_crit[p95_idx],
        "mine/slow_frac": jnp.mean(slow_mask) if slow_mask.shape[0] else jnp.asarray(0.0),
    }


def label_probe_atoms(
    probe_obs,
    params,
    normalizer_params,
    apply_policy_value,
    cfg,
    champions=None,
    global_champion=None,
):
    """Labels synthetic probe atoms with the cluster-aware mosaic teacher."""
    c = jax.vmap(lambda o: cc.criticality_score(o, 0.0, cfg))(probe_obs)
    w = jax.vmap(lambda x: cc.memory_weight(x, cfg))(c)
    klb = jax.vmap(lambda x: cc.kl_budget_from_c(x, cfg))(c)
    vb = jax.vmap(lambda x: cc.value_budget_from_c(x, cfg))(c)
    cid = jax.vmap(cc.cluster_id_for)(probe_obs)
    batch, kept_idx = _label_atoms_with_mosaic_teacher_and_indices(
        probe_obs,
        cid,
        champions,
        global_champion,
        params,
        normalizer_params,
        apply_policy_value,
        cfg,
    )
    kept_idx = jnp.asarray(kept_idx, dtype=jnp.int32)
    return batch.replace(
        weight=w[kept_idx],
        kl_budget=klb[kept_idx],
        value_budget=vb[kept_idx],
        source_id=jnp.full((batch.obs.shape[0],), SOURCE_SYNTHETIC_PROBE, dtype=jnp.int32),
    )
