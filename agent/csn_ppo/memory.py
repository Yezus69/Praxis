"""Fixed-size behavioral sentinel memory for CSN-PPO."""

from __future__ import annotations

from collections.abc import Mapping

import flax
import jax
import jax.numpy as jnp


SOURCE_RECENT_CURRENT = 0
SOURCE_SYNTHETIC_PROBE = 1
SOURCE_SENTINEL_FAILURE = 2
SOURCE_SENTINEL_FAILURE_UNTRUSTED = 3

_NUM_SOURCES = SOURCE_SENTINEL_FAILURE_UNTRUSTED + 1


@flax.struct.dataclass
class BehavioralMemory:
    obs: jnp.ndarray              # [N, 27]
    mean: jnp.ndarray             # [N, 2]
    logstd: jnp.ndarray           # [N, 2]
    value: jnp.ndarray            # [N]
    weight: jnp.ndarray           # [N]
    kl_budget: jnp.ndarray        # [N]
    value_budget: jnp.ndarray     # [N]
    cluster_id: jnp.ndarray       # [N]
    source_id: jnp.ndarray        # [N]
    age: jnp.ndarray              # [N]
    write_idx: jnp.ndarray        # scalar int32
    size: jnp.ndarray             # scalar int32


@flax.struct.dataclass
class BehavioralMemoryBatch:
    obs: jnp.ndarray              # [B, 27]
    mean: jnp.ndarray             # [B, 2]
    logstd: jnp.ndarray           # [B, 2]
    value: jnp.ndarray            # [B]
    weight: jnp.ndarray           # [B]
    kl_budget: jnp.ndarray        # [B]
    value_budget: jnp.ndarray     # [B]
    cluster_id: jnp.ndarray       # [B]
    source_id: jnp.ndarray        # [B]


def init_behavioral_memory(capacity, obs_dim=27, action_dim=2):
    return BehavioralMemory(
        obs=jnp.zeros((capacity, obs_dim), dtype=jnp.float32),
        mean=jnp.zeros((capacity, action_dim), dtype=jnp.float32),
        logstd=jnp.zeros((capacity, action_dim), dtype=jnp.float32),
        value=jnp.zeros((capacity,), dtype=jnp.float32),
        weight=jnp.zeros((capacity,), dtype=jnp.float32),
        kl_budget=jnp.zeros((capacity,), dtype=jnp.float32),
        value_budget=jnp.zeros((capacity,), dtype=jnp.float32),
        cluster_id=jnp.zeros((capacity,), dtype=jnp.int32),
        source_id=jnp.zeros((capacity,), dtype=jnp.int32),
        age=jnp.zeros((capacity,), dtype=jnp.int32),
        write_idx=jnp.asarray(0, dtype=jnp.int32),
        size=jnp.asarray(0, dtype=jnp.int32),
    )


def source_cluster_quotas(num_clusters: int) -> dict[tuple[int, int], float]:
    """P7 source/cluster quotas for stratified guard memory."""
    num_clusters = int(num_clusters)
    if num_clusters <= 0:
        raise ValueError("num_clusters must be positive")
    quotas = {}
    for cluster in range(num_clusters):
        quotas[(SOURCE_SENTINEL_FAILURE, cluster)] = 0.25 / num_clusters
    for cluster in range(num_clusters):
        quotas[(SOURCE_SYNTHETIC_PROBE, cluster)] = 0.25 / num_clusters
    for cluster in range(num_clusters):
        quotas[(SOURCE_RECENT_CURRENT, cluster)] = 0.50 / num_clusters
    return quotas


def _quota_items(quotas: Mapping[tuple[int, int], float] | None):
    if not quotas:
        return ()
    return tuple(
        (int(source_id), int(cluster_id), float(quota))
        for (source_id, cluster_id), quota in quotas.items()
        if float(quota) > 0.0
    )


def _num_clusters(cfg=None, quotas: Mapping[tuple[int, int], float] | None = None) -> int:
    if cfg is not None and hasattr(cfg, "num_clusters"):
        return int(cfg.num_clusters)
    items = _quota_items(quotas)
    if items:
        return max(cluster_id for _, cluster_id, _ in items) + 1
    return 4


def _cfg_value(cfg, name: str, default: float) -> float:
    if cfg is None:
        return default
    return float(getattr(cfg, name, default))


def _active_mask(memory: BehavioralMemory) -> jnp.ndarray:
    return jnp.arange(memory.obs.shape[0], dtype=jnp.int32) < memory.size


def _bucket_index(source_id, cluster_id, num_clusters: int) -> jnp.ndarray:
    source_id = jnp.clip(source_id, 0, _NUM_SOURCES - 1)
    cluster_id = jnp.clip(cluster_id, 0, num_clusters - 1)
    return source_id * int(num_clusters) + cluster_id


def replacement_score(
    memory: BehavioralMemory,
    cfg=None,
    quotas: Mapping[tuple[int, int], float] | None = None,
) -> jnp.ndarray:
    """P7 replacement score; lower active scores are evicted first.

    score(m) = criticality(m) + mem_lambda_rare * rarity(m)
             + mem_lambda_sentinel * 1[sentinel_failure]
             - mem_lambda_age * staleness(m)

    The stored guard weight is the memory atom's available criticality proxy.
    """
    num_clusters = _num_clusters(cfg, quotas)
    active = _active_mask(memory)
    active_size = jnp.maximum(memory.size.astype(jnp.float32), 1.0)
    bucket_idx = _bucket_index(memory.source_id, memory.cluster_id, num_clusters)
    bucket_idx = jnp.where(active, bucket_idx, 0)
    bucket_counts = jnp.bincount(
        bucket_idx,
        weights=active.astype(jnp.float32),
        length=_NUM_SOURCES * num_clusters,
    )
    rarity = 1.0 - bucket_counts[bucket_idx] / active_size
    sentinel_failure = (memory.source_id == SOURCE_SENTINEL_FAILURE).astype(jnp.float32)
    max_age = jnp.maximum(jnp.max(jnp.where(active, memory.age, 0)), 1)
    staleness = memory.age.astype(jnp.float32) / max_age.astype(jnp.float32)
    score = (
        memory.weight
        + _cfg_value(cfg, "mem_lambda_rare", 1.0) * rarity
        + _cfg_value(cfg, "mem_lambda_sentinel", 4.0) * sentinel_failure
        - _cfg_value(cfg, "mem_lambda_age", 0.01) * staleness
    )
    return jnp.where(active, score, jnp.inf)


def _incoming_replacement_score(
    memory: BehavioralMemory,
    atoms: BehavioralMemoryBatch,
    cfg=None,
    quotas: Mapping[tuple[int, int], float] | None = None,
) -> jnp.ndarray:
    num_clusters = _num_clusters(cfg, quotas)
    active = _active_mask(memory)
    active_size = jnp.maximum(memory.size.astype(jnp.float32), 1.0)
    memory_bucket_idx = _bucket_index(memory.source_id, memory.cluster_id, num_clusters)
    memory_bucket_idx = jnp.where(active, memory_bucket_idx, 0)
    bucket_counts = jnp.bincount(
        memory_bucket_idx,
        weights=active.astype(jnp.float32),
        length=_NUM_SOURCES * num_clusters,
    )
    atom_bucket_idx = _bucket_index(atoms.source_id, atoms.cluster_id, num_clusters)
    rarity = 1.0 - bucket_counts[atom_bucket_idx] / active_size
    sentinel_failure = (atoms.source_id == SOURCE_SENTINEL_FAILURE).astype(jnp.float32)
    return (
        atoms.weight
        + _cfg_value(cfg, "mem_lambda_rare", 1.0) * rarity
        + _cfg_value(cfg, "mem_lambda_sentinel", 4.0) * sentinel_failure
    )


def _set_atom(
    memory: BehavioralMemory,
    idx,
    atoms: BehavioralMemoryBatch,
    atom_idx: int,
    next_write_idx,
    next_size,
) -> BehavioralMemory:
    return BehavioralMemory(
        obs=memory.obs.at[idx].set(atoms.obs[atom_idx]),
        mean=memory.mean.at[idx].set(atoms.mean[atom_idx]),
        logstd=memory.logstd.at[idx].set(atoms.logstd[atom_idx]),
        value=memory.value.at[idx].set(atoms.value[atom_idx]),
        weight=memory.weight.at[idx].set(atoms.weight[atom_idx]),
        kl_budget=memory.kl_budget.at[idx].set(atoms.kl_budget[atom_idx]),
        value_budget=memory.value_budget.at[idx].set(atoms.value_budget[atom_idx]),
        cluster_id=memory.cluster_id.at[idx].set(atoms.cluster_id[atom_idx]),
        source_id=memory.source_id.at[idx].set(atoms.source_id[atom_idx]),
        age=memory.age.at[idx].set(0),
        write_idx=next_write_idx,
        size=next_size,
    )


def _insert_one_stratified(
    memory: BehavioralMemory,
    atoms: BehavioralMemoryBatch,
    atom_idx: int,
    cfg=None,
    quotas: Mapping[tuple[int, int], float] | None = None,
) -> BehavioralMemory:
    capacity = memory.obs.shape[0]
    source_id = atoms.source_id[atom_idx]
    cluster_id = atoms.cluster_id[atom_idx]

    def append_atom(_):
        idx = memory.write_idx
        return _set_atom(
            memory,
            idx,
            atoms,
            atom_idx,
            next_write_idx=(memory.write_idx + 1) % capacity,
            next_size=jnp.minimum(memory.size + 1, capacity),
        )

    def replace_atom(_):
        active = _active_mask(memory)
        same_bucket = (
            active
            & (memory.source_id == source_id)
            & (memory.cluster_id == cluster_id)
        )
        scores = replacement_score(memory, cfg, quotas)
        idx = jnp.argmin(jnp.where(same_bucket, scores, jnp.inf))
        has_bucket = jnp.any(same_bucket)

        def do_replace(__):
            return _set_atom(
                memory,
                idx,
                atoms,
                atom_idx,
                next_write_idx=(idx + 1) % capacity,
                next_size=memory.size,
            )

        return jax.lax.cond(has_bucket, do_replace, lambda __: memory, operand=None)

    return jax.lax.cond(
        memory.size < capacity,
        append_atom,
        replace_atom,
        operand=None,
    )


def insert_atoms_stratified(
    memory: BehavioralMemory,
    atoms: BehavioralMemoryBatch,
    cfg=None,
    quotas: Mapping[tuple[int, int], float] | None = None,
) -> BehavioralMemory:
    """Inserts atoms using P7 global score replacement."""
    if quotas is None:
        quotas = source_cluster_quotas(_num_clusters(cfg))
    n = atoms.obs.shape[0]
    if n == 0:
        return memory
    capacity = memory.obs.shape[0]
    if n > capacity:
        raise ValueError("stratified insert batch size must not exceed memory capacity")

    active = _active_mask(memory)
    scores = jnp.where(active, replacement_score(memory, cfg, quotas), -jnp.inf)
    evict_idx = jnp.argsort(scores, stable=True)[:n]
    slot_scores = scores[evict_idx]
    atom_scores = _incoming_replacement_score(memory, atoms, cfg, quotas)
    write_mask = atom_scores > slot_scores
    write_mask_2d = write_mask[:, None]
    max_written_idx = jnp.max(jnp.where(write_mask, evict_idx, -1))
    wrote_any = jnp.any(write_mask)

    return BehavioralMemory(
        obs=memory.obs.at[evict_idx].set(
            jnp.where(write_mask_2d, atoms.obs, memory.obs[evict_idx])
        ),
        mean=memory.mean.at[evict_idx].set(
            jnp.where(write_mask_2d, atoms.mean, memory.mean[evict_idx])
        ),
        logstd=memory.logstd.at[evict_idx].set(
            jnp.where(write_mask_2d, atoms.logstd, memory.logstd[evict_idx])
        ),
        value=memory.value.at[evict_idx].set(
            jnp.where(write_mask, atoms.value, memory.value[evict_idx])
        ),
        weight=memory.weight.at[evict_idx].set(
            jnp.where(write_mask, atoms.weight, memory.weight[evict_idx])
        ),
        kl_budget=memory.kl_budget.at[evict_idx].set(
            jnp.where(write_mask, atoms.kl_budget, memory.kl_budget[evict_idx])
        ),
        value_budget=memory.value_budget.at[evict_idx].set(
            jnp.where(write_mask, atoms.value_budget, memory.value_budget[evict_idx])
        ),
        cluster_id=memory.cluster_id.at[evict_idx].set(
            jnp.where(write_mask, atoms.cluster_id, memory.cluster_id[evict_idx])
        ),
        source_id=memory.source_id.at[evict_idx].set(
            jnp.where(write_mask, atoms.source_id, memory.source_id[evict_idx])
        ),
        age=memory.age.at[evict_idx].set(
            jnp.where(write_mask, 0, memory.age[evict_idx])
        ),
        write_idx=jnp.where(
            wrote_any,
            (max_written_idx + 1) % capacity,
            memory.write_idx,
        ),
        size=jnp.minimum(capacity, jnp.maximum(memory.size, max_written_idx + 1)),
    )


def _insert_atoms_ring(memory: BehavioralMemory, atoms: BehavioralMemoryBatch) -> BehavioralMemory:
    n = atoms.obs.shape[0]
    idx = (memory.write_idx + jnp.arange(n)) % memory.obs.shape[0]

    return BehavioralMemory(
        obs=memory.obs.at[idx].set(atoms.obs),
        mean=memory.mean.at[idx].set(atoms.mean),
        logstd=memory.logstd.at[idx].set(atoms.logstd),
        value=memory.value.at[idx].set(atoms.value),
        weight=memory.weight.at[idx].set(atoms.weight),
        kl_budget=memory.kl_budget.at[idx].set(atoms.kl_budget),
        value_budget=memory.value_budget.at[idx].set(atoms.value_budget),
        cluster_id=memory.cluster_id.at[idx].set(atoms.cluster_id),
        source_id=memory.source_id.at[idx].set(atoms.source_id),
        age=memory.age.at[idx].set(0),
        write_idx=(memory.write_idx + n) % memory.obs.shape[0],
        size=jnp.minimum(memory.size + n, memory.obs.shape[0]),
    )


def insert_atoms(
    memory: BehavioralMemory,
    atoms: BehavioralMemoryBatch,
    cfg=None,
    quotas: Mapping[tuple[int, int], float] | None = None,
) -> BehavioralMemory:
    if cfg is None and quotas is None:
        return _insert_atoms_ring(memory, atoms)
    return insert_atoms_stratified(memory, atoms, cfg=cfg, quotas=quotas)


def _gather_memory(memory: BehavioralMemory, idx) -> BehavioralMemoryBatch:
    return BehavioralMemoryBatch(
        obs=memory.obs[idx],
        mean=memory.mean[idx],
        logstd=memory.logstd[idx],
        value=memory.value[idx],
        weight=memory.weight[idx],
        kl_budget=memory.kl_budget[idx],
        value_budget=memory.value_budget[idx],
        cluster_id=memory.cluster_id[idx],
        source_id=memory.source_id[idx],
    )


def sample_memory(memory: BehavioralMemory, rng: jax.Array, batch_size: int) -> BehavioralMemoryBatch:
    max_idx = jnp.maximum(memory.size, 1)
    idx = jax.random.randint(rng, (batch_size,), minval=0, maxval=max_idx)

    return _gather_memory(memory, idx)


def _quota_counts(batch_size: int, quotas: Mapping[tuple[int, int], float]):
    items = _quota_items(quotas)
    if not items:
        return ()
    quota_total = sum(quota for _, _, quota in items)
    raw = [batch_size * quota / quota_total for _, _, quota in items]
    counts = [int(value) for value in raw]
    remainder = int(batch_size) - sum(counts)
    order = sorted(
        range(len(items)),
        key=lambda i: (raw[i] - counts[i], items[i][0], items[i][1]),
        reverse=True,
    )
    for i in order[:remainder]:
        counts[i] += 1
    return tuple(
        (source_id, cluster_id, count)
        for (source_id, cluster_id, _), count in zip(items, counts)
    )


def _sample_bucket_indices(memory: BehavioralMemory, rng, source_id: int, cluster_id: int, count: int):
    active = _active_mask(memory)
    mask = (
        active
        & (memory.source_id == jnp.asarray(source_id, dtype=jnp.int32))
        & (memory.cluster_id == jnp.asarray(cluster_id, dtype=jnp.int32))
    )
    capacity = memory.obs.shape[0]
    valid_idx = jnp.nonzero(mask, size=capacity, fill_value=0)[0]
    valid_count = jnp.maximum(jnp.sum(mask.astype(jnp.int32)), 1)
    pos = jax.random.randint(rng, (int(count),), minval=0, maxval=valid_count)
    return valid_idx[pos]


def sample_memory_stratified(
    memory: BehavioralMemory,
    rng: jax.Array,
    batch_size: int,
    quotas: Mapping[tuple[int, int], float],
) -> BehavioralMemoryBatch:
    """Samples guard memory by P7 source/cluster quotas."""
    bucket_counts = _quota_counts(int(batch_size), quotas)
    if not bucket_counts:
        return sample_memory(memory, rng, batch_size)
    keys = jax.random.split(rng, len(bucket_counts))
    sampled = [
        _sample_bucket_indices(memory, key, source_id, cluster_id, count)
        for key, (source_id, cluster_id, count) in zip(keys, bucket_counts)
        if count > 0
    ]
    if not sampled:
        return sample_memory(memory, rng, batch_size)
    idx = jnp.concatenate(sampled, axis=0)
    return _gather_memory(memory, idx)


def stratified_memory_ready(
    memory: BehavioralMemory,
    quotas: Mapping[tuple[int, int], float],
) -> jnp.ndarray:
    """Returns true once every requested source/cluster bucket has data."""
    items = _quota_items(quotas)
    if not items:
        return jnp.asarray(False)
    active = _active_mask(memory)
    ready = jnp.asarray(True)
    for source_id, cluster_id, _ in items:
        bucket_has_data = jnp.any(
            active
            & (memory.source_id == jnp.asarray(source_id, dtype=jnp.int32))
            & (memory.cluster_id == jnp.asarray(cluster_id, dtype=jnp.int32))
        )
        ready = ready & bucket_has_data
    return ready


def sample_memory_for_guard(
    memory: BehavioralMemory,
    rng: jax.Array,
    batch_size: int,
    quotas: Mapping[tuple[int, int], float],
) -> BehavioralMemoryBatch:
    """Uses stratified guard sampling once the memory can satisfy all quotas."""
    if not _quota_items(quotas):
        return sample_memory(memory, rng, batch_size)
    ready = (memory.size >= int(batch_size)) & stratified_memory_ready(memory, quotas)
    return jax.lax.cond(
        ready,
        lambda _: sample_memory_stratified(memory, rng, batch_size, quotas),
        lambda _: sample_memory(memory, rng, batch_size),
        operand=None,
    )


def concat_memory_batches(*batches: BehavioralMemoryBatch) -> BehavioralMemoryBatch:
    return BehavioralMemoryBatch(
        obs=jnp.concatenate([b.obs for b in batches], axis=0),
        mean=jnp.concatenate([b.mean for b in batches], axis=0),
        logstd=jnp.concatenate([b.logstd for b in batches], axis=0),
        value=jnp.concatenate([b.value for b in batches], axis=0),
        weight=jnp.concatenate([b.weight for b in batches], axis=0),
        kl_budget=jnp.concatenate([b.kl_budget for b in batches], axis=0),
        value_budget=jnp.concatenate([b.value_budget for b in batches], axis=0),
        cluster_id=jnp.concatenate([b.cluster_id for b in batches], axis=0),
        source_id=jnp.concatenate([b.source_id for b in batches], axis=0),
    )


def should_insert_slow_memory(criticality, threshold=3.0):
    return criticality > threshold


def age_memory(memory: BehavioralMemory) -> BehavioralMemory:
    active = jnp.arange(memory.age.shape[0]) < memory.size
    return BehavioralMemory(
        obs=memory.obs,
        mean=memory.mean,
        logstd=memory.logstd,
        value=memory.value,
        weight=memory.weight,
        kl_budget=memory.kl_budget,
        value_budget=memory.value_budget,
        cluster_id=memory.cluster_id,
        source_id=memory.source_id,
        age=jnp.where(active, memory.age + 1, memory.age),
        write_idx=memory.write_idx,
        size=memory.size,
    )
