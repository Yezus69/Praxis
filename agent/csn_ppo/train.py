"""Custom single-device CSN-PPO loop for the 28D Praxis coverage env."""

from __future__ import annotations

import functools
import math
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import optax
from brax.training import acting
from brax.training import gradients
from brax.training import types
from brax.training.acme import running_statistics
from brax.training.acme import specs
from brax.training.acting import Evaluator
from brax.training.agents.ppo import losses as ppo_losses
from brax.training.agents.ppo import networks as ppo_networks

from agent.csn_ppo import coverage_probes
from agent.csn_ppo import metrics as M
from agent.csn_ppo import mosaic_teacher
from agent.csn_ppo import rollout_mining
from agent.csn_ppo.config import CSNPPOConfig
from agent.csn_ppo.gradient_projection import (
    combine_safe_and_guard_grads,
    project_conflicting_gradient,
)
from agent.csn_ppo.guarded_loss import (
    gaussian_kl,
    memory_bucket_mask,
    memory_guard_loss,
)
from agent.csn_ppo.memory import (
    BehavioralMemoryBatch,
    init_behavioral_memory,
    insert_atoms,
    sample_memory,
)


ACTIVE_MEMORY_BUCKETS = (
    "collision_boundary",
    "successful_goal",
    "dynamic_obstacle",
    "no_obstacle_straight_line",
    "synthetic_contract_probe",
)


def make_apply_policy_value(ppo_network):
    """Returns README §8 policy/value application with Brax normalizer-first order."""

    def apply_policy_value(params, normalizer_params, obs):
        logits = ppo_network.policy_network.apply(normalizer_params, params.policy, obs)
        dist = ppo_network.parametric_action_distribution.create_dist(logits)
        value = ppo_network.value_network.apply(normalizer_params, params.value, obs)
        return dist.loc, jnp.log(dist.scale), value

    return apply_policy_value


def concat_memory_batches(*batches):
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


def _filter_memory_batch_host(batch, mask):
    idx = np.nonzero(np.asarray(mask))[0]
    if idx.size == 0:
        return None
    return _take_memory_batch(batch, idx)


def _split_train_holdout(data, rng, train_size):
    n = data.observation.shape[0]
    if train_size >= n:
        raise ValueError(
            f"holdout split needs more than train_size={train_size} rows; got {n}"
        )
    idx = jax.random.permutation(rng, n)
    train_idx = idx[:train_size]
    holdout_idx = idx[train_size:]
    train_data = jax.tree_util.tree_map(lambda x: x[train_idx], data)
    holdout_data = jax.tree_util.tree_map(lambda x: x[holdout_idx], data)
    return train_data, holdout_data


def _estimate_advantage_abs(data, params, normalizer_params, ppo_network, cfg):
    obs = data.observation.reshape((-1, cfg.obs_dim))
    next_obs = data.next_observation.reshape((-1, cfg.obs_dim))
    reward = data.reward.reshape((-1,))
    discount = data.discount.reshape((-1,))
    value = ppo_network.value_network.apply(normalizer_params, params.value, obs)
    next_value = ppo_network.value_network.apply(normalizer_params, params.value, next_obs)
    td = reward + cfg.discounting * discount * next_value - value
    return jnp.abs(td)


def _collect_total_batch_size(cfg):
    train_size = int(cfg.batch_size) * int(cfg.num_minibatches)
    keep_fraction = 1.0 - float(cfg.holdout_fraction)
    if keep_fraction <= 0.0:
        raise ValueError("holdout_fraction must be < 1.0")
    total_size = int(math.ceil(train_size / keep_fraction))
    if total_size % int(cfg.num_envs) != 0:
        total_size = int(math.ceil(total_size / int(cfg.num_envs))) * int(cfg.num_envs)
    return total_size


def _guard_bucket_losses_and_metrics(
    params,
    normalizer_params,
    memory_batch,
    apply_policy_value,
    bucket_names=ACTIVE_MEMORY_BUCKETS,
):
    pred_mean, pred_logstd, pred_value = apply_policy_value(
        params,
        normalizer_params,
        memory_batch.obs,
    )
    kl = gaussian_kl(memory_batch.mean, memory_batch.logstd, pred_mean, pred_logstd)
    policy_violation = jax.nn.relu(kl - memory_batch.kl_budget)
    value_error = jnp.abs(pred_value - memory_batch.value)
    value_violation = jax.nn.relu(value_error - memory_batch.value_budget)

    weighted_policy = memory_batch.weight * policy_violation ** 2
    weighted_value = memory_batch.weight * value_violation ** 2
    masks = jnp.stack(
        [memory_bucket_mask(memory_batch, bucket_name) for bucket_name in bucket_names],
        axis=0,
    )
    policy_losses = jnp.mean(masks * weighted_policy[None, :], axis=1)
    value_losses = jnp.mean(masks * weighted_value[None, :], axis=1)
    losses = policy_losses + 0.25 * value_losses

    sorted_kl = jnp.sort(kl)
    p95_idx = jnp.asarray(0.95 * (sorted_kl.shape[0] - 1), dtype=jnp.int32)
    policy_loss = jnp.mean(weighted_policy)
    value_loss = jnp.mean(weighted_value)
    metrics = {
        "memory/kl_mean": jnp.mean(kl),
        "memory/kl_p95": sorted_kl[p95_idx],
        "memory/policy_violation_frac": jnp.mean(policy_violation > 0),
        "memory/value_violation_frac": jnp.mean(value_violation > 0),
        "memory/policy_loss": policy_loss,
        "memory/value_loss": value_loss,
        "memory/guard_loss": policy_loss + 0.25 * value_loss,
    }
    return losses, metrics


def _unstack_guard_grads(stacked_grads, count):
    return tuple(
        jax.tree_util.tree_map(lambda x, i=i: x[i], stacked_grads)
        for i in range(count)
    )


def _zero_guard_metrics():
    zero = jnp.array(0.0, dtype=jnp.float32)
    return {
        "memory/kl_mean": zero,
        "memory/kl_p95": zero,
        "memory/policy_violation_frac": zero,
        "memory/value_violation_frac": zero,
        "memory/policy_loss": zero,
        "memory/value_loss": zero,
        "memory/guard_loss": zero,
    }


def _metric_value(metrics, key):
    return metrics[key]


def train(environment, config: CSNPPOConfig, progress_fn=None, eval_env=None):
    """Runs Phase 1b CSN-PPO over an already wrapped coverage environment."""
    rng = jax.random.PRNGKey(config.seed)
    normalize = (
        running_statistics.normalize
        if config.normalize_observations
        else (lambda x, y: x)
    )
    network_factory = functools.partial(
        ppo_networks.make_ppo_networks,
        policy_hidden_layer_sizes=config.policy_hidden_layer_sizes,
        value_hidden_layer_sizes=config.value_hidden_layer_sizes,
    )
    ppo_network = network_factory(
        config.obs_dim,
        config.action_dim,
        preprocess_observations_fn=normalize,
    )
    make_policy = ppo_networks.make_inference_fn(ppo_network, compute_value=True)
    apply_policy_value = make_apply_policy_value(ppo_network)

    rng, key_policy, key_value, key_reset, eval_key = jax.random.split(rng, 5)
    params = ppo_losses.PPONetworkParams(
        policy=ppo_network.policy_network.init(key_policy),
        value=ppo_network.value_network.init(key_value),
    )
    optimizer = optax.chain(
        optax.clip_by_global_norm(config.max_grad_norm),
        optax.adam(config.learning_rate),
    )
    opt_state = optimizer.init(params)
    normalizer_params = running_statistics.init_state(
        specs.Array((config.obs_dim,), jnp.dtype("float32")),
        std_eps=0.0,
        mode="welford",
    )

    memory_fast = init_behavioral_memory(
        config.memory_size_fast,
        config.obs_dim,
        config.action_dim,
    )
    memory_slow = init_behavioral_memory(
        config.memory_size_slow,
        config.obs_dim,
        config.action_dim,
    )
    champion = mosaic_teacher.init_champion()

    reset_keys = jax.random.split(key_reset, config.num_envs)
    env_state = jax.jit(environment.reset)(reset_keys)

    ppo_loss_fn = functools.partial(
        ppo_losses.compute_ppo_loss,
        ppo_network=ppo_network,
        entropy_cost=config.entropy_cost,
        discounting=config.discounting,
        reward_scaling=config.reward_scaling,
        gae_lambda=config.gae_lambda,
        clipping_epsilon=config.clipping_epsilon,
        normalize_advantage=config.normalize_advantage,
        vf_coefficient=config.vf_coefficient,
    )
    ppo_value_and_grad = gradients.loss_and_pgrad(
        ppo_loss_fn,
        pmap_axis_name=None,
        has_aux=True,
    )

    train_batch_size = int(config.batch_size) * int(config.num_minibatches)
    total_batch_size = _collect_total_batch_size(config)
    if total_batch_size % int(config.num_envs) != 0:
        raise ValueError("total rollout batch must be divisible by num_envs")
    num_unrolls = total_batch_size // int(config.num_envs)
    env_steps_per_update = num_unrolls * int(config.num_envs) * int(config.unroll_length)
    num_updates = max(int(config.num_timesteps) // env_steps_per_update, 1)
    eval_interval = max(num_updates // max(int(config.num_evals) - 1, 1), 1)
    champion_eval_interval = int(config.champion_eval_interval) or eval_interval

    def collect_rollout(state, rollout_rng, rollout_params, rollout_normalizer_params):
        policy = make_policy(
            (
                rollout_normalizer_params,
                rollout_params.policy,
                rollout_params.value,
            )
        )
        extra_fields = ["truncation", "episode_metrics", "episode_done"]
        if config.bootstrap_on_timeout:
            extra_fields.append("time_out")

        def collect(carry, _):
            es, key = carry
            key, step_key = jax.random.split(key)
            next_state, transition = acting.generate_unroll(
                environment,
                es,
                policy,
                step_key,
                config.unroll_length,
                extra_fields=tuple(extra_fields),
            )
            return (next_state, key), transition

        (next_state, _), data = jax.lax.scan(
            collect,
            (state, rollout_rng),
            (),
            length=num_unrolls,
        )
        data = jax.tree_util.tree_map(lambda x: jnp.swapaxes(x, 1, 2), data)
        data = jax.tree_util.tree_map(lambda x: x.reshape((-1,) + x.shape[2:]), data)
        if config.bootstrap_on_timeout:
            time_out = data.extras["state_extras"]["time_out"]
            value = data.extras["policy_extras"]["value"]
            data = types.Transition(
                observation=data.observation,
                action=data.action,
                reward=data.reward + config.discounting * time_out * value,
                discount=data.discount,
                next_observation=data.next_observation,
                extras=data.extras,
            )
        return next_state, data

    collect_rollout = jax.jit(collect_rollout)

    @functools.partial(jax.jit, static_argnames=("guard_active", "enable_projection"))
    def sgd_epoch(
        epoch_params,
        epoch_opt_state,
        epoch_normalizer_params,
        train_data,
        memory_batch,
        epoch_rng,
        guard_active: bool,
        enable_projection: bool,
    ):
        epoch_rng, perm_rng, step_rng = jax.random.split(epoch_rng, 3)

        def convert_data(x):
            x = jax.random.permutation(perm_rng, x)
            return x.reshape((config.num_minibatches, config.batch_size) + x.shape[1:])

        shuffled_data = jax.tree_util.tree_map(convert_data, train_data)

        def minibatch_step(carry, minibatch):
            step_params, step_opt_state, key = carry
            key, loss_key = jax.random.split(key)
            (ppo_loss, raw_metrics), g_ppo = ppo_value_and_grad(
                step_params,
                epoch_normalizer_params,
                minibatch,
                loss_key,
            )

            if guard_active:
                def loss_vector(loss_params):
                    losses, _ = _guard_bucket_losses_and_metrics(
                        loss_params,
                        epoch_normalizer_params,
                        memory_batch,
                        apply_policy_value,
                    )
                    return losses

                losses, guard_metrics = _guard_bucket_losses_and_metrics(
                    step_params,
                    epoch_normalizer_params,
                    memory_batch,
                    apply_policy_value,
                )
                stacked_guard_grads = jax.jacrev(loss_vector)(step_params)
                guard_grads = _unstack_guard_grads(stacked_guard_grads, losses.shape[0])
                if enable_projection:
                    g_safe = project_conflicting_gradient(
                        g_ppo,
                        guard_grads,
                        config.projection_eps,
                    )
                else:
                    g_safe = g_ppo
                coefs = (config.guard_lambda_mem,) * len(guard_grads)
                g_total = combine_safe_and_guard_grads(g_safe, guard_grads, coefs)
            else:
                g_total = g_ppo
                guard_metrics = _zero_guard_metrics()

            updates, next_opt_state = optimizer.update(g_total, step_opt_state)
            next_params = optax.apply_updates(step_params, updates)
            metrics = {
                "ppo/total_loss": ppo_loss,
                "ppo/policy_loss": _metric_value(raw_metrics, "policy_loss"),
                "ppo/v_loss": _metric_value(raw_metrics, "v_loss"),
                "ppo/entropy_loss": _metric_value(raw_metrics, "entropy_loss"),
                "ppo/kl_mean": _metric_value(raw_metrics, "kl_mean"),
                "ppo/train_surrogate": -_metric_value(raw_metrics, "policy_loss"),
                **guard_metrics,
            }
            return (next_params, next_opt_state, key), metrics

        (next_params, next_opt_state, _), metrics = jax.lax.scan(
            minibatch_step,
            (epoch_params, epoch_opt_state, step_rng),
            shuffled_data,
            length=config.num_minibatches,
        )
        metrics = jax.tree_util.tree_map(jnp.mean, metrics)
        return next_params, next_opt_state, metrics

    def eval_policy_fn(eval_params):
        return make_policy(
            eval_params,
            deterministic=config.eval_deterministic,
        )

    evaluator = Evaluator(
        eval_env or environment,
        eval_policy_fn,
        num_eval_envs=config.num_eval_envs,
        episode_length=config.episode_length,
        action_repeat=1,
        key=eval_key,
    )

    metrics = {}
    if progress_fn is not None:
        eval_metrics = evaluator.run_evaluation((normalizer_params, params.policy, params.value), {})
        champion, champion_metrics = mosaic_teacher.maybe_update_champion(
            champion,
            eval_metrics,
            normalizer_params,
            params,
            config,
        )
        metrics = M.merge_metrics(eval_metrics, champion_metrics, {"env_steps": 0})
        progress_fn(0, M.to_float_dict(metrics))

    for update in range(num_updates):
        rng, rollout_rng, split_rng, mine_rng, probe_rng, opt_rng = jax.random.split(rng, 6)
        env_state, data = collect_rollout(
            env_state,
            rollout_rng,
            params,
            normalizer_params,
        )
        normalizer_params = running_statistics.update(
            normalizer_params,
            data.observation,
            pmap_axis_name=None,
        )
        train_data, holdout_data = _split_train_holdout(
            data,
            split_rng,
            train_batch_size,
        )

        obs_flat = train_data.observation.reshape((-1, config.obs_dim))
        adv_abs = _estimate_advantage_abs(
            train_data,
            params,
            normalizer_params,
            ppo_network,
            config,
        )
        teacher_normalizer, teacher_params = mosaic_teacher.teacher_snapshot(
            champion,
            normalizer_params,
            params,
        )
        mined_batch, slow_mask, mine_metrics = rollout_mining.mine_atoms(
            obs_flat,
            adv_abs,
            teacher_params,
            teacher_normalizer,
            apply_policy_value,
            config,
        )
        memory_fast = insert_atoms(memory_fast, mined_batch)
        slow_mined_batch = _filter_memory_batch_host(mined_batch, slow_mask)
        if slow_mined_batch is not None:
            memory_slow = insert_atoms(memory_slow, slow_mined_batch)

        if update % int(config.synthetic_probe_insert_interval) == 0:
            probe_obs = coverage_probes.generate_cover_probes(
                probe_rng,
                int(config.synthetic_probe_batch_size),
            )
            probe_batch = rollout_mining.label_probe_atoms(
                probe_obs,
                teacher_params,
                teacher_normalizer,
                apply_policy_value,
                config,
            )
            slow_probe_batch = _filter_memory_batch_host(
                probe_batch,
                probe_batch.weight > config.slow_memory_threshold,
            )
            if slow_probe_batch is not None:
                memory_slow = insert_atoms(memory_slow, slow_probe_batch)

        env_steps = (update + 1) * env_steps_per_update
        guard_active = (
            bool(config.enable_guard)
            and int(np.asarray(memory_fast.size)) >= int(config.min_memory_size_before_guard)
            and env_steps >= int(config.guard_warmup_steps)
        )
        best_params = params
        best_opt_state = opt_state
        best_holdout = float("-inf")
        stopped_at = int(config.max_updates_per_batch)
        epoch_metrics = {}
        holdout_metrics = {}
        guard_eval_metrics = _zero_guard_metrics()
        memory_batch = None

        for epoch in range(int(config.max_updates_per_batch)):
            rng, mem_rng, slow_mem_rng, epoch_rng, holdout_rng = jax.random.split(rng, 5)
            mb_fast = sample_memory(memory_fast, mem_rng, config.memory_batch_size // 2)
            mb_slow = sample_memory(memory_slow, slow_mem_rng, config.memory_batch_size // 2)
            memory_batch = concat_memory_batches(mb_fast, mb_slow)
            params_candidate, opt_state_candidate, epoch_metrics = sgd_epoch(
                params,
                opt_state,
                normalizer_params,
                train_data,
                memory_batch,
                epoch_rng,
                guard_active=guard_active,
                enable_projection=bool(config.enable_gradient_projection),
            )

            _, raw_holdout_metrics = ppo_loss_fn(
                params_candidate,
                normalizer_params,
                holdout_data,
                holdout_rng,
            )
            holdout_surrogate = -raw_holdout_metrics["policy_loss"]
            if guard_active:
                _, guard_eval_metrics = memory_guard_loss(
                    params_candidate,
                    normalizer_params,
                    memory_batch,
                    apply_policy_value,
                )
                memory_kl_p95 = guard_eval_metrics["memory/kl_p95"]
            else:
                memory_kl_p95 = jnp.array(0.0, dtype=jnp.float32)

            accept = (
                float(np.asarray(holdout_surrogate)) > best_holdout
                and float(np.asarray(memory_kl_p95)) < float(config.memory_kl_limit_p95)
            )
            if accept:
                best_params = params_candidate
                best_opt_state = opt_state_candidate
                best_holdout = float(np.asarray(holdout_surrogate))

            stop = False
            if config.enable_holdout_early_stop:
                stop = M.should_stop_epoch(
                    float(np.asarray(holdout_surrogate)),
                    best_holdout,
                    float(np.asarray(memory_kl_p95)),
                    config.memory_kl_limit_p95,
                    float(np.asarray(epoch_metrics["ppo/kl_mean"])),
                    config.target_kl,
                    config.holdout_eps,
                    enable_kl_early_stop=config.enable_kl_early_stop,
                )
            params = params_candidate
            opt_state = opt_state_candidate
            holdout_metrics = {
                "ppo/holdout_surrogate": holdout_surrogate,
                "ppo/generalization_gap": M.generalization_gap(
                    epoch_metrics["ppo/train_surrogate"],
                    holdout_surrogate,
                ),
            }
            if stop:
                params = best_params
                opt_state = best_opt_state
                stopped_at = epoch + 1
                break

        do_eval = (update + 1) % eval_interval == 0 or update == num_updates - 1
        do_champion = (
            do_eval and ((update + 1) % champion_eval_interval == 0 or update == num_updates - 1)
        )
        eval_metrics = {}
        champion_metrics = {}
        if do_eval:
            eval_metrics = evaluator.run_evaluation((normalizer_params, params.policy, params.value), {})
            if do_champion and config.enable_mosaic_teacher:
                champion, champion_metrics = mosaic_teacher.maybe_update_champion(
                    champion,
                    eval_metrics,
                    normalizer_params,
                    params,
                    config,
                )

        base_memory_metrics = dict(guard_eval_metrics)
        base_memory_metrics.update(
            {
                "memory/fast_size": memory_fast.size,
                "memory/slow_size": memory_slow.size,
                "memory/guard_active": jnp.asarray(float(guard_active), dtype=jnp.float32),
            }
        )
        metrics = M.merge_metrics(
            epoch_metrics,
            holdout_metrics,
            base_memory_metrics,
            mine_metrics,
            eval_metrics,
            champion_metrics,
            {
                "epoch/stopped_at": stopped_at,
                "env_steps": env_steps,
            },
        )
        if progress_fn is not None and (update % int(config.log_interval) == 0 or do_eval):
            progress_fn(env_steps, M.to_float_dict(metrics))
        elif progress_fn is None:
            print(M.to_float_dict(metrics))

    return make_policy, params, normalizer_params, (memory_fast, memory_slow), metrics
