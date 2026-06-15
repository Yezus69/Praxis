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
from agent.csn_ppo.curriculum import (
    freeze_or_slow_curriculum,
    init_curriculum_state,
    maybe_advance_curriculum,
    sample_world_difficulties,
)
from agent.csn_ppo import metrics as M
from agent.csn_ppo import mosaic_teacher
from agent.csn_ppo import rollout_mining
from agent.csn_ppo import sentinel
from agent.csn_ppo.config import CSNPPOConfig
from agent.csn_ppo.gradient_projection import (
    combine_safe_and_guard_grads,
    project_conflicting_gradient,
)
from agent.csn_ppo.guarded_loss import (
    coefficients_for_buckets,
    condition_guard_kl_inputs,
    gaussian_kl,
    init_guard_pressure_state,
    memory_bucket_mask,
    memory_guard_loss,
    update_guard_pressure,
)
from agent.csn_ppo.memory import (
    BehavioralMemoryBatch,
    age_memory,
    concat_memory_batches,
    init_behavioral_memory,
    insert_atoms,
    sample_memory_for_guard,
    source_cluster_quotas,
)
from agent.csn_ppo.validation import (
    ValidationBank,
    create_validation_bank,
    evaluate_validation_bank,
    update_validation_best,
    validation_best,
    validation_guard_regressions,
    validation_regressed,
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
    cfg,
    bucket_names=ACTIVE_MEMORY_BUCKETS,
):
    pred_mean, pred_logstd, pred_value = apply_policy_value(
        params,
        normalizer_params,
        memory_batch.obs,
    )
    t_mean, t_logstd, p_mean, p_logstd = condition_guard_kl_inputs(
        memory_batch.mean,
        memory_batch.logstd,
        pred_mean,
        pred_logstd,
        cfg,
    )
    kl = gaussian_kl(t_mean, t_logstd, p_mean, p_logstd)
    kl = jnp.minimum(kl, jnp.asarray(cfg.max_atom_kl, dtype=kl.dtype))
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


def _sentinel_cluster_metrics(sentinel_metrics, trajectories, num_clusters):
    episode_return = jnp.sum(trajectories.reward, axis=1)
    one_hot = jax.nn.one_hot(
        trajectories.cluster_id,
        num_clusters,
        dtype=jnp.float32,
    )
    counts = jnp.maximum(jnp.sum(one_hot, axis=0), 1.0)
    mean_return = jnp.sum(one_hot * episode_return[:, None], axis=0) / counts
    return {
        "coverage": sentinel_metrics["coverage"],
        "collision_rate": sentinel_metrics["collision_rate"],
        "mean_return": mean_return,
    }


def _champion_policy_id(policy_id):
    return -1 if policy_id is None else int(policy_id)


def _sync_sentinel_bank_from_champions(sentinel_bank, champions):
    return sentinel_bank.replace(
        best_coverage=jnp.asarray(
            [c.best_coverage for c in champions.champions],
            dtype=jnp.float32,
        ),
        best_collision_rate=jnp.asarray(
            [c.best_collision_rate for c in champions.champions],
            dtype=jnp.float32,
        ),
        champion_policy_id=jnp.asarray(
            [_champion_policy_id(c.policy_id) for c in champions.champions],
            dtype=jnp.int32,
        ),
    )


def _sentinel_failure_difficulty_from_regressions(regressions, num_clusters, fallback):
    regressed = regressions["regressed"]
    has_failure = jnp.any(regressed)
    failed_cluster = jnp.argmax(regressed.astype(jnp.int32))
    denom = jnp.maximum(jnp.asarray(num_clusters - 1, dtype=jnp.float32), 1.0)
    difficulty = failed_cluster.astype(jnp.float32) / denom
    return jnp.reshape(jnp.where(has_failure, difficulty, fallback), (1,))


def _set_env_curriculum_info(
    environment,
    state,
    curriculum_state,
    sentinel_failure_difficulty,
    next_difficulty,
):
    if hasattr(environment, "set_curriculum_info"):
        return environment.set_curriculum_info(
            state,
            curriculum_state,
            sentinel_failure_difficulty,
            next_difficulty,
        )
    return state


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
    memory_quotas = source_cluster_quotas(config.num_clusters)
    champion = mosaic_teacher.init_champion()
    sentinel_bank = None
    champions = None
    num_sentinel_clusters = int(config.num_clusters)
    guard_pressure = init_guard_pressure_state(num_sentinel_clusters, config)
    curriculum_state = init_curriculum_state(config)
    # TODO(P4): replace this placeholder with a fixed-size mined sentinel-failure
    # difficulty bank once failure-world persistence exists outside the sentinel bank.
    sentinel_failure_difficulty = jnp.reshape(
        curriculum_state.current_difficulty,
        (1,),
    )
    last_safe_params = None
    last_safe_opt_state = None
    last_safe_normalizer_params = None
    if config.enable_sentinel:
        rng, sentinel_rng = jax.random.split(rng)
        sentinel_bank = sentinel.create_sentinel_bank(
            sentinel_rng,
            config.sentinel_bank_size,
            num_sentinel_clusters,
        )
        champions = mosaic_teacher.init_mosaic_champions(num_sentinel_clusters)
    validation_bank = None
    validation_regression_count = 0
    if int(config.validation_eval_interval) > 0:
        rng, validation_rng = jax.random.split(rng)
        validation_bank = create_validation_bank(validation_rng, config)

    reset_keys = jax.random.split(key_reset, config.num_envs)
    rng, initial_difficulty_rng = jax.random.split(rng)
    initial_difficulty_batch = sample_world_difficulties(
        curriculum_state,
        initial_difficulty_rng,
        config.num_envs,
        sentinel_failure_difficulty,
    )
    if hasattr(environment, "set_curriculum_info"):
        env_state = jax.jit(environment.reset)(
            reset_keys,
            initial_difficulty_batch,
            curriculum_state,
            sentinel_failure_difficulty,
        )
    else:
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

    def collect_rollout(
        state,
        rollout_rng,
        rollout_params,
        rollout_normalizer_params,
        rollout_curriculum_state,
        rollout_sentinel_failure_difficulty,
        rollout_next_difficulty,
    ):
        state = _set_env_curriculum_info(
            environment,
            state,
            rollout_curriculum_state,
            rollout_sentinel_failure_difficulty,
            rollout_next_difficulty,
        )
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
        cluster_guard_lambda,
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
                        config,
                    )
                    return losses

                losses, guard_metrics = _guard_bucket_losses_and_metrics(
                    step_params,
                    epoch_normalizer_params,
                    memory_batch,
                    apply_policy_value,
                    config,
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
                coefs = coefficients_for_buckets(
                    bucket_names=ACTIVE_MEMORY_BUCKETS,
                    cluster_guard_lambda=cluster_guard_lambda,
                    cfg=config,
                )
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
        (
            rng,
            rollout_rng,
            difficulty_rng,
            split_rng,
            mine_rng,
            probe_rng,
            opt_rng,
        ) = jax.random.split(rng, 7)
        difficulty_batch = sample_world_difficulties(
            curriculum_state,
            difficulty_rng,
            config.num_envs,
            sentinel_failure_difficulty,
        )
        env_state, data = collect_rollout(
            env_state,
            rollout_rng,
            params,
            normalizer_params,
            curriculum_state,
            sentinel_failure_difficulty,
            difficulty_batch,
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
        memory_fast = age_memory(memory_fast)
        memory_slow = age_memory(memory_slow)

        obs_flat = train_data.observation.reshape((-1, config.obs_dim))
        adv_abs = _estimate_advantage_abs(
            train_data,
            params,
            normalizer_params,
            ppo_network,
            config,
        )
        mined_batch, slow_mask, mine_metrics = rollout_mining.mine_atoms(
            obs_flat,
            adv_abs,
            params,
            normalizer_params,
            apply_policy_value,
            config,
            champions=champions,
            global_champion=champion,
        )
        memory_fast = insert_atoms(memory_fast, mined_batch, cfg=config, quotas=memory_quotas)
        slow_mined_batch = _filter_memory_batch_host(mined_batch, slow_mask)
        if slow_mined_batch is not None:
            memory_slow = insert_atoms(
                memory_slow,
                slow_mined_batch,
                cfg=config,
                quotas=memory_quotas,
            )

        if update % int(config.synthetic_probe_insert_interval) == 0:
            probe_obs = coverage_probes.generate_cover_probes(
                probe_rng,
                int(config.synthetic_probe_batch_size),
            )
            probe_batch = rollout_mining.label_probe_atoms(
                probe_obs,
                params,
                normalizer_params,
                apply_policy_value,
                config,
                champions=champions,
                global_champion=champion,
            )
            slow_probe_batch = _filter_memory_batch_host(
                probe_batch,
                probe_batch.weight > config.slow_memory_threshold,
            )
            if slow_probe_batch is not None:
                memory_slow = insert_atoms(
                    memory_slow,
                    slow_probe_batch,
                    cfg=config,
                    quotas=memory_quotas,
                )

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
            mb_fast = sample_memory_for_guard(
                memory_fast,
                mem_rng,
                config.memory_batch_size // 2,
                memory_quotas,
            )
            mb_slow = sample_memory_for_guard(
                memory_slow,
                slow_mem_rng,
                config.memory_batch_size // 2,
                memory_quotas,
            )
            memory_batch = concat_memory_batches(mb_fast, mb_slow)
            params_candidate, opt_state_candidate, epoch_metrics = sgd_epoch(
                params,
                opt_state,
                normalizer_params,
                train_data,
                memory_batch,
                guard_pressure.cluster_lambda,
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
                    cfg=config,
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

        sentinel_metrics = {}
        sentinel_regressed_now = False
        if (
            config.enable_sentinel
            and int(config.sentinel_eval_interval) > 0
            and update % int(config.sentinel_eval_interval) == 0
        ):
            raw_sentinel_metrics, sentinel_trajectories = sentinel.evaluate_sentinel_bank(
                eval_env or environment,
                sentinel_bank,
                make_policy,
                params,
                normalizer_params,
                deterministic=True,
            )
            regressions = sentinel.detect_sentinel_regressions(
                raw_sentinel_metrics,
                sentinel_bank,
                config.sentinel_success_tolerance,
                config.sentinel_collision_tolerance,
            )
            guard_pressure = update_guard_pressure(
                guard_pressure,
                regressions,
                recovered=jnp.logical_not(regressions["regressed"]),
                cfg=config,
            )
            sentinel_metrics = {
                k: v
                for k, v in M.merge_metrics(raw_sentinel_metrics, regressions).items()
                if k.startswith("sentinel/")
            }
            sentinel_metrics.update(
                {
                    f"guard/cluster_lambda/{cluster_id}": guard_pressure.cluster_lambda[cluster_id]
                    for cluster_id in range(num_sentinel_clusters)
                }
            )
            if bool(np.asarray(jnp.any(regressions["regressed"]))):
                sentinel_regressed_now = True
                # P4: freeze/slow curriculum here.
                curriculum_state = freeze_or_slow_curriculum(curriculum_state)
                sentinel_failure_difficulty = _sentinel_failure_difficulty_from_regressions(
                    regressions,
                    num_sentinel_clusters,
                    curriculum_state.current_difficulty,
                )
                failed_atoms = sentinel.label_failed_sentinel_atoms_with_best_teacher(
                    sentinel_trajectories=sentinel_trajectories,
                    regressions=regressions,
                    champions=champions,
                    global_champion=champion,
                    current_params=params,
                    current_normalizer=normalizer_params,
                    apply_policy_value=apply_policy_value,
                    cfg=config,
                )
                if failed_atoms is not None:
                    memory_slow = insert_atoms(
                        memory_slow,
                        failed_atoms,
                        cfg=config,
                        quotas=memory_quotas,
                    )
                coverage_drop = sentinel_bank.best_coverage - raw_sentinel_metrics["coverage"]
                collision_increase = (
                    raw_sentinel_metrics["collision_rate"]
                    - sentinel_bank.best_collision_rate
                )
                severe = jnp.any(
                    (coverage_drop > 2.0 * config.sentinel_success_tolerance)
                    | (collision_increase > 2.0 * config.sentinel_collision_tolerance)
                )
                if bool(np.asarray(severe)) and last_safe_params is not None:
                    params = last_safe_params
                    opt_state = last_safe_opt_state
                    normalizer_params = last_safe_normalizer_params
                    sentinel_metrics["sentinel/severe_rollback"] = jnp.asarray(
                        1.0,
                        dtype=jnp.float32,
                    )
                else:
                    sentinel_metrics["sentinel/severe_rollback"] = jnp.asarray(
                        0.0,
                        dtype=jnp.float32,
                    )
            else:
                last_safe_params = params
                last_safe_opt_state = opt_state
                last_safe_normalizer_params = normalizer_params
                curriculum_state = maybe_advance_curriculum(
                    curriculum_state,
                    {
                        # TODO(P4): split current/history once sentinel evaluation
                        # reports those slices separately.
                        "current_success_rate": jnp.mean(raw_sentinel_metrics["coverage"]),
                        "current_collision_rate": jnp.max(
                            raw_sentinel_metrics["collision_rate"]
                        ),
                        "historical_success_rate": jnp.mean(
                            raw_sentinel_metrics["coverage"]
                        ),
                        "historical_collision_rate": jnp.max(
                            raw_sentinel_metrics["collision_rate"]
                        ),
                    },
                )
                cluster_metrics = _sentinel_cluster_metrics(
                    raw_sentinel_metrics,
                    sentinel_trajectories,
                    num_sentinel_clusters,
                )
                champions = mosaic_teacher.maybe_update_champions(
                    cluster_metrics,
                    params,
                    normalizer_params,
                    champions,
                    config,
                    env_steps,
                )
                sentinel_bank = _sync_sentinel_bank_from_champions(
                    sentinel_bank,
                    champions,
                )

        validation_metrics = {}
        if (
            validation_bank is not None
            and int(config.validation_eval_interval) > 0
            and update % int(config.validation_eval_interval) == 0
        ):
            raw_validation_metrics = evaluate_validation_bank(
                eval_env or environment,
                validation_bank,
                params,
                normalizer_params,
                make_policy,
                apply_policy_value,
                config,
            )
            best_validation = validation_best(validation_bank)
            validation_regressed_now = validation_regressed(
                raw_validation_metrics,
                best_validation,
                config,
            )
            if validation_regressed_now:
                validation_regression_count += 1
            else:
                validation_regression_count = 0
            validation_patience_exhausted = (
                validation_regressed_now
                and validation_regression_count >= max(int(config.validation_patience), 1)
            )
            validation_rolled_back = (
                validation_patience_exhausted and last_safe_params is not None
            )
            if validation_rolled_back:
                params = last_safe_params
                opt_state = last_safe_opt_state
                normalizer_params = last_safe_normalizer_params
            validation_metrics = dict(raw_validation_metrics)
            validation_metrics.update(
                {
                    "validation/regressed": jnp.asarray(
                        float(validation_regressed_now),
                        dtype=jnp.float32,
                    ),
                    "validation/rollback": jnp.asarray(
                        float(validation_rolled_back),
                        dtype=jnp.float32,
                    ),
                    "validation/regression_count": jnp.asarray(
                        float(validation_regression_count),
                        dtype=jnp.float32,
                    ),
                    "validation/patience_exhausted": jnp.asarray(
                        float(validation_patience_exhausted),
                        dtype=jnp.float32,
                    ),
                    "validation/best_synthetic_kl_p95": best_validation[
                        "synthetic_kl_p95"
                    ],
                }
            )
            if validation_patience_exhausted:
                guard_pressure = update_guard_pressure(
                    guard_pressure,
                    validation_guard_regressions(num_sentinel_clusters),
                    recovered=jnp.zeros((num_sentinel_clusters,), dtype=jnp.bool_),
                    cfg=config,
                )
                validation_metrics.update(
                    {
                        f"guard/cluster_lambda/{cluster_id}": guard_pressure.cluster_lambda[
                            cluster_id
                        ]
                        for cluster_id in range(num_sentinel_clusters)
                    }
                )
                validation_regression_count = 0
            elif (not validation_regressed_now) and not sentinel_regressed_now:
                validation_bank = update_validation_best(
                    validation_bank,
                    raw_validation_metrics,
                )
                last_safe_params = params
                last_safe_opt_state = opt_state
                last_safe_normalizer_params = normalizer_params

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
            sentinel_metrics,
            validation_metrics,
            eval_metrics,
            champion_metrics,
            {
                "epoch/stopped_at": stopped_at,
                "env_steps": env_steps,
                "curriculum/current_difficulty": curriculum_state.current_difficulty,
                "curriculum/frozen": curriculum_state.frozen.astype(jnp.float32),
                "curriculum/sentinel_failure_difficulty": jnp.mean(
                    sentinel_failure_difficulty
                ),
                "curriculum/next_difficulty_mean": jnp.mean(difficulty_batch),
            },
        )
        if progress_fn is not None and (update % int(config.log_interval) == 0 or do_eval):
            progress_fn(env_steps, M.to_float_dict(metrics))
        elif progress_fn is None:
            print(M.to_float_dict(metrics))

    return make_policy, params, normalizer_params, (memory_fast, memory_slow), metrics
