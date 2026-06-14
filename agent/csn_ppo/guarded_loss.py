"""Hinge-KL policy/value guard losses for CSN-PPO."""

from collections import OrderedDict

import jax
import jax.numpy as jnp

from agent.csn_ppo.memory import BehavioralMemoryBatch


CLUSTER_COLLISION_BOUNDARY = 0
CLUSTER_SUCCESSFUL_GOAL = 1
CLUSTER_DYNAMIC_OBSTACLE = 2
CLUSTER_NO_OBSTACLE_STRAIGHT_LINE = 3

SOURCE_RECENT_CURRENT = 0
SOURCE_SYNTHETIC_PROBE = 1
SOURCE_SENTINEL_FAILURE = 2

MEMORY_BUCKETS = (
    "collision_boundary",
    "successful_goal",
    "dynamic_obstacle",
    "no_obstacle_straight_line",
    "recent_current",
    "synthetic_contract_probe",
    "sentinel_regression",
)


def gaussian_kl(mean0, logstd0, mean1, logstd1):
    """KL[N(mean0, std0) || N(mean1, std1)] for diagonal Gaussians.

    Args:
        mean0: [..., action_dim]
        logstd0: [..., action_dim]
        mean1: [..., action_dim]
        logstd1: [..., action_dim]

    Returns:
        kl: [...]
    """
    var0 = jnp.exp(2.0 * logstd0)
    var1 = jnp.exp(2.0 * logstd1)

    kl_per_dim = 0.5 * (
        (var0 + (mean0 - mean1) ** 2) / (var1 + 1e-8)
        - 1.0
        + 2.0 * (logstd1 - logstd0)
    )

    return jnp.sum(kl_per_dim, axis=-1)


def _sorted_p95(values):
    sorted_values = jnp.sort(values)
    idx = jnp.asarray(0.95 * (sorted_values.shape[0] - 1), dtype=jnp.int32)
    return sorted_values[idx]


def memory_guard_loss(params, normalizer_params, memory_batch, apply_policy_value):
    pred_mean, pred_logstd, pred_value = apply_policy_value(
        params,
        normalizer_params,
        memory_batch.obs,
    )

    kl = gaussian_kl(
        memory_batch.mean,
        memory_batch.logstd,
        pred_mean,
        pred_logstd,
    )

    policy_violation = jax.nn.relu(kl - memory_batch.kl_budget)
    policy_loss = jnp.mean(
        memory_batch.weight * policy_violation ** 2
    )

    value_error = jnp.abs(pred_value - memory_batch.value)
    value_violation = jax.nn.relu(value_error - memory_batch.value_budget)
    value_loss = jnp.mean(
        memory_batch.weight * value_violation ** 2
    )

    metrics = {
        "memory/kl_mean": jnp.mean(kl),
        "memory/kl_p95": _sorted_p95(kl),
        "memory/policy_violation_frac": jnp.mean(policy_violation > 0),
        "memory/value_violation_frac": jnp.mean(value_violation > 0),
        "memory/policy_loss": policy_loss,
        "memory/value_loss": value_loss,
    }

    return policy_loss + 0.25 * value_loss, metrics


def memory_bucket_mask(memory_batch, bucket_name):
    cluster_id = memory_batch.cluster_id
    source_id = memory_batch.source_id

    if bucket_name == "collision_boundary":
        return (cluster_id == CLUSTER_COLLISION_BOUNDARY).astype(jnp.float32)
    if bucket_name == "successful_goal":
        return (cluster_id == CLUSTER_SUCCESSFUL_GOAL).astype(jnp.float32)
    if bucket_name == "dynamic_obstacle":
        return (cluster_id == CLUSTER_DYNAMIC_OBSTACLE).astype(jnp.float32)
    if bucket_name == "no_obstacle_straight_line":
        return (cluster_id == CLUSTER_NO_OBSTACLE_STRAIGHT_LINE).astype(jnp.float32)
    if bucket_name == "recent_current":
        known_cluster = (
            (cluster_id == CLUSTER_COLLISION_BOUNDARY)
            | (cluster_id == CLUSTER_SUCCESSFUL_GOAL)
            | (cluster_id == CLUSTER_DYNAMIC_OBSTACLE)
            | (cluster_id == CLUSTER_NO_OBSTACLE_STRAIGHT_LINE)
        )
        return ((source_id == SOURCE_RECENT_CURRENT) & (~known_cluster)).astype(jnp.float32)
    if bucket_name == "synthetic_contract_probe":
        return (source_id == SOURCE_SYNTHETIC_PROBE).astype(jnp.float32)
    if bucket_name == "sentinel_regression":
        return (source_id == SOURCE_SENTINEL_FAILURE).astype(jnp.float32)
    raise ValueError(f"unknown memory bucket: {bucket_name}")


def mask_memory_batch(memory_batch, mask):
    return BehavioralMemoryBatch(
        obs=memory_batch.obs,
        mean=memory_batch.mean,
        logstd=memory_batch.logstd,
        value=memory_batch.value,
        weight=memory_batch.weight * mask,
        kl_budget=memory_batch.kl_budget,
        value_budget=memory_batch.value_budget,
        cluster_id=memory_batch.cluster_id,
        source_id=memory_batch.source_id,
    )


def bucket_memory_batches(memory_batch, bucket_names=MEMORY_BUCKETS):
    return OrderedDict(
        (bucket_name, mask_memory_batch(memory_batch, memory_bucket_mask(memory_batch, bucket_name)))
        for bucket_name in bucket_names
    )


def value_and_grad_guard_loss_by_bucket(
    params,
    normalizer_params,
    memory_batch,
    apply_policy_value,
    bucket_names=MEMORY_BUCKETS,
):
    bucket_batches = bucket_memory_batches(memory_batch, bucket_names)
    loss_values = OrderedDict()
    guard_grads = []
    guard_metrics = OrderedDict()

    for bucket_name, bucket_batch in bucket_batches.items():
        def loss_fn(loss_params):
            return memory_guard_loss(
                loss_params,
                normalizer_params,
                bucket_batch,
                apply_policy_value,
            )

        (loss_value, metrics_value), grad_value = jax.value_and_grad(loss_fn, has_aux=True)(params)
        loss_values[bucket_name] = loss_value
        guard_grads.append(grad_value)
        for metric_name, metric_value in metrics_value.items():
            suffix = metric_name.removeprefix("memory/")
            guard_metrics[f"memory/{bucket_name}/{suffix}"] = metric_value

    return loss_values, guard_grads, guard_metrics
