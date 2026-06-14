"""Fixed-size behavioral sentinel memory for CSN-PPO."""

from __future__ import annotations

import flax
import jax
import jax.numpy as jnp


SOURCE_RECENT_CURRENT = 0
SOURCE_SYNTHETIC_PROBE = 1
SOURCE_SENTINEL_FAILURE = 2
SOURCE_SENTINEL_FAILURE_UNTRUSTED = 3


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


def insert_atoms(memory: BehavioralMemory, atoms: BehavioralMemoryBatch) -> BehavioralMemory:
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


def sample_memory(memory: BehavioralMemory, rng: jax.Array, batch_size: int) -> BehavioralMemoryBatch:
    max_idx = jnp.maximum(memory.size, 1)
    idx = jax.random.randint(rng, (batch_size,), minval=0, maxval=max_idx)

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
