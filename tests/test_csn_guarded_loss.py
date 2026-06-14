import jax.numpy as jnp
import pytest

from agent.csn_ppo.guarded_loss import gaussian_kl, memory_guard_loss
from agent.csn_ppo.memory import BehavioralMemoryBatch


def _memory_batch(mean, logstd, value, weight=None, kl_budget=None, value_budget=None):
    batch_size = mean.shape[0]
    if weight is None:
        weight = jnp.ones((batch_size,), dtype=jnp.float32)
    if kl_budget is None:
        kl_budget = jnp.full((batch_size,), 0.02, dtype=jnp.float32)
    if value_budget is None:
        value_budget = jnp.full((batch_size,), 0.25, dtype=jnp.float32)
    return BehavioralMemoryBatch(
        obs=jnp.zeros((batch_size, 27), dtype=jnp.float32),
        mean=mean,
        logstd=logstd,
        value=value,
        weight=weight,
        kl_budget=kl_budget,
        value_budget=value_budget,
        cluster_id=jnp.zeros((batch_size,), dtype=jnp.int32),
        source_id=jnp.zeros((batch_size,), dtype=jnp.int32),
    )


def test_gaussian_kl_zero_for_identical_distributions():
    mean = jnp.zeros((8, 2))
    logstd = jnp.zeros((8, 2))
    kl = gaussian_kl(mean, logstd, mean, logstd)
    assert jnp.allclose(kl, 0.0, atol=1e-6)


def test_guard_loss_zero_inside_budget():
    mean = jnp.zeros((4, 2), dtype=jnp.float32)
    logstd = jnp.zeros((4, 2), dtype=jnp.float32)
    value = jnp.zeros((4,), dtype=jnp.float32)
    memory_batch = _memory_batch(mean, logstd, value)

    def apply_policy_value(params, normalizer_params, obs):
        return mean, logstd, value

    loss, metrics = memory_guard_loss(None, None, memory_batch, apply_policy_value)

    assert float(loss) == pytest.approx(0.0, abs=1e-6)
    assert float(metrics["memory/policy_loss"]) == pytest.approx(0.0, abs=1e-6)
    assert float(metrics["memory/value_loss"]) == pytest.approx(0.0, abs=1e-6)


def test_guard_loss_positive_outside_budget():
    mean = jnp.zeros((4, 2), dtype=jnp.float32)
    logstd = jnp.zeros((4, 2), dtype=jnp.float32)
    value = jnp.zeros((4,), dtype=jnp.float32)
    memory_batch = _memory_batch(
        mean,
        logstd,
        value,
        kl_budget=jnp.zeros((4,), dtype=jnp.float32),
        value_budget=jnp.zeros((4,), dtype=jnp.float32),
    )

    def apply_policy_value(params, normalizer_params, obs):
        pred_mean = jnp.ones((4, 2), dtype=jnp.float32) * 3.0
        pred_logstd = jnp.zeros((4, 2), dtype=jnp.float32)
        pred_value = jnp.ones((4,), dtype=jnp.float32)
        return pred_mean, pred_logstd, pred_value

    loss, metrics = memory_guard_loss(None, None, memory_batch, apply_policy_value)

    assert float(loss) > 0.0
    assert float(metrics["memory/policy_loss"]) > 0.0
    assert float(metrics["memory/value_loss"]) > 0.0
