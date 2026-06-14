import jax
import jax.numpy as jnp

from agent.csn_ppo.memory import (
    BehavioralMemoryBatch,
    init_behavioral_memory,
    insert_atoms,
    sample_memory,
)


def _atoms(start, count):
    obs = (
        jnp.arange(start, start + count, dtype=jnp.float32)[:, None]
        * jnp.ones((count, 27), dtype=jnp.float32)
    )
    mean = (
        jnp.arange(start, start + count, dtype=jnp.float32)[:, None]
        * jnp.ones((count, 2), dtype=jnp.float32)
    )
    return BehavioralMemoryBatch(
        obs=obs,
        mean=mean,
        logstd=jnp.zeros((count, 2), dtype=jnp.float32),
        value=jnp.arange(start, start + count, dtype=jnp.float32),
        weight=jnp.ones((count,), dtype=jnp.float32),
        kl_budget=jnp.full((count,), 0.02, dtype=jnp.float32),
        value_budget=jnp.full((count,), 0.25, dtype=jnp.float32),
        cluster_id=jnp.arange(start, start + count, dtype=jnp.int32),
        source_id=jnp.zeros((count,), dtype=jnp.int32),
    )


def test_insert_sample_roundtrip():
    memory = init_behavioral_memory(4)
    atoms = _atoms(10, 2)

    memory = insert_atoms(memory, atoms)

    assert int(memory.size) == 2
    assert int(memory.write_idx) == 2
    assert jnp.allclose(memory.obs[:2], atoms.obs)
    assert jnp.allclose(memory.mean[:2], atoms.mean)
    assert jnp.allclose(memory.value[:2], atoms.value)


def test_ring_buffer_wrap_caps_size_and_updates_write_idx():
    memory = init_behavioral_memory(3)

    memory = insert_atoms(memory, _atoms(0, 2))
    memory = insert_atoms(memory, _atoms(2, 2))

    assert int(memory.size) == 3
    assert int(memory.write_idx) == 1
    assert jnp.allclose(memory.obs[0], jnp.ones((27,), dtype=jnp.float32) * 3.0)
    assert jnp.allclose(memory.obs[1], jnp.ones((27,), dtype=jnp.float32) * 1.0)
    assert jnp.allclose(memory.obs[2], jnp.ones((27,), dtype=jnp.float32) * 2.0)


def test_sampled_batch_shapes_and_fields_are_correct():
    memory = init_behavioral_memory(5)
    memory = insert_atoms(memory, _atoms(0, 5))

    batch = sample_memory(memory, jax.random.PRNGKey(0), 3)

    assert batch.obs.shape == (3, 27)
    assert batch.mean.shape == (3, 2)
    assert batch.logstd.shape == (3, 2)
    assert batch.value.shape == (3,)
    assert batch.weight.shape == (3,)
    assert batch.kl_budget.shape == (3,)
    assert batch.value_budget.shape == (3,)
    assert batch.cluster_id.shape == (3,)
    assert batch.source_id.shape == (3,)
