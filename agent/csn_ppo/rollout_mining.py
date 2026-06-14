"""Rollout and probe atom labeling for the coverage CSN-PPO loop."""

import jax
import jax.numpy as jnp

from agent.csn_ppo import criticality_coverage as cc
from agent.csn_ppo.memory import BehavioralMemoryBatch


def mine_atoms(obs_flat, adv_abs, params, normalizer_params, apply_policy_value, cfg):
    """Mines fixed-shape top-criticality atoms from current rollout observations."""
    crit = jax.vmap(lambda o, a: cc.criticality_score(o, a, cfg))(obs_flat, adv_abs)
    k = cfg.atoms_per_rollout
    _, idx = jax.lax.top_k(crit, k)
    obs = obs_flat[idx]
    c = crit[idx]

    t_mean, t_logstd, t_value = apply_policy_value(params, normalizer_params, obs)
    t_logstd = jnp.maximum(t_logstd, cfg.teacher_logstd_floor)

    w = jax.vmap(lambda x: cc.memory_weight(x, cfg))(c)
    klb = jax.vmap(lambda x: cc.kl_budget_from_c(x, cfg))(c)
    vb = jax.vmap(lambda x: cc.value_budget_from_c(x, cfg))(c)
    cid = jax.vmap(cc.cluster_id_for)(obs)
    src = jnp.zeros((k,), jnp.int32)
    batch = BehavioralMemoryBatch(
        obs=obs,
        mean=t_mean,
        logstd=t_logstd,
        value=t_value,
        weight=w,
        kl_budget=klb,
        value_budget=vb,
        cluster_id=cid,
        source_id=src,
    )
    slow_mask = c > cfg.slow_memory_threshold
    sorted_crit = jnp.sort(crit)
    p95_idx = jnp.asarray(0.95 * (crit.shape[0] - 1), dtype=jnp.int32)
    return batch, slow_mask, {
        "mine/crit_mean": jnp.mean(crit),
        "mine/crit_p95": sorted_crit[p95_idx],
        "mine/slow_frac": jnp.mean(slow_mask),
    }


def label_probe_atoms(probe_obs, params, normalizer_params, apply_policy_value, cfg):
    """Labels synthetic probe atoms with the supplied teacher snapshot."""
    t_mean, t_logstd, t_value = apply_policy_value(params, normalizer_params, probe_obs)
    t_logstd = jnp.maximum(t_logstd, cfg.teacher_logstd_floor)
    n = probe_obs.shape[0]
    c = jax.vmap(lambda o: cc.criticality_score(o, 0.0, cfg))(probe_obs)
    w = jax.vmap(lambda x: cc.memory_weight(x, cfg))(c)
    klb = jax.vmap(lambda x: cc.kl_budget_from_c(x, cfg))(c)
    vb = jax.vmap(lambda x: cc.value_budget_from_c(x, cfg))(c)
    cid = jax.vmap(cc.cluster_id_for)(probe_obs)
    src = jnp.ones((n,), jnp.int32)
    return BehavioralMemoryBatch(
        obs=probe_obs,
        mean=t_mean,
        logstd=t_logstd,
        value=t_value,
        weight=w,
        kl_budget=klb,
        value_budget=vb,
        cluster_id=cid,
        source_id=src,
    )
