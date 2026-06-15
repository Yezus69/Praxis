import jax
import jax.numpy as jnp
import numpy as np

from agent.csn_ppo.config import CSNPPOConfig, resolve_long_run_config
from agent.csn_ppo.memory import (
    SOURCE_RECENT_CURRENT,
    SOURCE_SENTINEL_FAILURE,
    SOURCE_SYNTHETIC_PROBE,
    BehavioralMemoryBatch,
    concat_memory_batches,
    init_behavioral_memory,
    insert_atoms,
    sample_memory_stratified,
    source_cluster_quotas,
)


def _atoms(cfg, start, count, source_id, cluster_id, weight=1.0):
    obs = (
        jnp.arange(start, start + count, dtype=jnp.float32)[:, None]
        * jnp.ones((count, cfg.obs_dim), dtype=jnp.float32)
    )
    mean = (
        jnp.arange(start, start + count, dtype=jnp.float32)[:, None]
        * jnp.ones((count, cfg.action_dim), dtype=jnp.float32)
    )
    return BehavioralMemoryBatch(
        obs=obs,
        mean=mean,
        logstd=jnp.zeros((count, cfg.action_dim), dtype=jnp.float32),
        value=jnp.arange(start, start + count, dtype=jnp.float32),
        weight=jnp.full((count,), weight, dtype=jnp.float32),
        kl_budget=jnp.full((count,), cfg.guard_kl_budget, dtype=jnp.float32),
        value_budget=jnp.full((count,), cfg.value_budget, dtype=jnp.float32),
        cluster_id=jnp.full((count,), cluster_id, dtype=jnp.int32),
        source_id=jnp.full((count,), source_id, dtype=jnp.int32),
    )


def test_7_1_sentinel_failures_cannot_be_evicted_by_common_current_states():
    cfg = CSNPPOConfig(
        memory_size_slow=16,
        memory_batch_size=8,
        num_clusters=4,
    )
    quotas = source_cluster_quotas(cfg.num_clusters)
    memory = init_behavioral_memory(
        cfg.memory_size_slow,
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
    )
    sentinel_atoms = concat_memory_batches(
        *[
            _atoms(
                cfg,
                start=cluster * 4,
                count=4,
                source_id=SOURCE_SENTINEL_FAILURE,
                cluster_id=cluster,
                weight=5.0,
            )
            for cluster in range(cfg.num_clusters)
        ]
    )
    memory = insert_atoms(memory, sentinel_atoms, cfg=cfg, quotas=quotas)

    for i in range(16):
        current_atoms = _atoms(
            cfg,
            start=1000 + i * 16,
            count=16,
            source_id=SOURCE_RECENT_CURRENT,
            cluster_id=0,
            weight=1.0,
        )
        memory = insert_atoms(memory, current_atoms, cfg=cfg, quotas=quotas)

    sentinel_count = int(jnp.sum(memory.source_id == SOURCE_SENTINEL_FAILURE))
    assert sentinel_count >= int(0.25 * cfg.memory_size_slow)


def test_7_2_stratified_sampling_respects_source_cluster_quotas():
    cfg = CSNPPOConfig(num_clusters=4)
    quotas = source_cluster_quotas(cfg.num_clusters)
    capacity = 960
    memory = init_behavioral_memory(
        capacity,
        obs_dim=cfg.obs_dim,
        action_dim=cfg.action_dim,
    )

    batches = []
    start = 0
    for cluster in range(cfg.num_clusters):
        batches.append(_atoms(cfg, start, 60, SOURCE_SENTINEL_FAILURE, cluster))
        start += 60
    for cluster in range(cfg.num_clusters):
        batches.append(_atoms(cfg, start, 60, SOURCE_SYNTHETIC_PROBE, cluster))
        start += 60
    for cluster in range(cfg.num_clusters):
        batches.append(_atoms(cfg, start, 120, SOURCE_RECENT_CURRENT, cluster))
        start += 120
    memory = insert_atoms(memory, concat_memory_batches(*batches))

    rng = jax.random.PRNGKey(0)
    batch_size = 64
    counts = {bucket: 0 for bucket in quotas}
    for draw in range(100):
        batch = sample_memory_stratified(
            memory,
            jax.random.fold_in(rng, draw),
            batch_size,
            quotas,
        )
        for bucket in quotas:
            source_id, cluster_id = bucket
            counts[bucket] += int(
                jnp.sum(
                    (batch.source_id == source_id)
                    & (batch.cluster_id == cluster_id)
                )
            )

    total = 100 * batch_size
    for bucket, quota in quotas.items():
        proportion = counts[bucket] / total
        np.testing.assert_allclose(proportion, quota, atol=0.03)


def test_7_3_long_run_config_sets_memory_sizes_and_sentinel():
    base_cfg = CSNPPOConfig(
        memory_size_fast=32,
        memory_size_slow=16,
        memory_batch_size=8,
        enable_sentinel=False,
    )

    cfg = resolve_long_run_config(base_cfg)

    assert cfg.memory_size_fast == 1_048_576
    assert cfg.memory_size_slow == 262_144
    assert cfg.memory_batch_size == 4096
    assert cfg.enable_sentinel is True
    assert cfg.guard_lambda_mem == 8.0
    assert cfg.synthetic_probe_batch_size == 4096
