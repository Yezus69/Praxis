"""Single-game PPO trainer for the memory-conditioned Atari agent."""

from __future__ import annotations

from collections import deque
from dataclasses import dataclass
from functools import partial
import time

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pmac.agents.atari_mem_net import MemAtariActorCritic, mem_apply, mem_apply_key, mem_init
from pmac.agents.ppo_atari import (
    TrainBatch,
    _categorical_entropy,
    _categorical_log_prob,
    _flatten_rollout,
    _learning_rate,
    _make_minibatches,
    _mean_or_previous,
    gae,
)
from pmac.envs.atari_envpool import ACT_DIM, EpisodeReturnTracker, make_train_env
from pmac.memory.bank import MemoryBank
from pmac.memory.reader import ema_update
from pmac.memory.runtime import RunningValueNorm, default_retrieval_hp, pad_bank
from pmac.memory.write import (
    RunningStats,
    build_insert_kwargs,
    importance,
    novelty,
    policy_entropy,
    select_writes,
    td_error,
    write_source_flags,
)


@dataclass(frozen=True)
class LMConfig:
    total_timesteps: int = 5_000_000
    num_envs: int = 64
    num_steps: int = 128
    update_epochs: int = 4
    num_minibatches: int = 4
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.1
    ent_coef: float = 0.01
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 2.5e-4
    anneal_lr: bool = True
    hot_capacity: int = 4096
    top_k: int = 16
    d_k: int = 128
    d_c: int = 16
    d_m: int = 128
    write_top_fraction: float = 0.1
    write_min_quota: int = 0
    teacher_temperature: float = 1.0
    tau_key: float = 0.005
    eps_policy: float = 0.05
    eps_value: float = 0.1
    write_every: int = 1
    memory_capacity: int = 65_536


def _validate_config(cfg: LMConfig) -> int:
    steps_per_update = int(cfg.num_envs) * int(cfg.num_steps)
    if steps_per_update <= 0:
        raise ValueError("num_envs*num_steps must be positive")
    num_updates = int(cfg.total_timesteps) // steps_per_update
    if num_updates <= 0:
        raise ValueError("total_timesteps must cover at least one PPO update")
    if int(cfg.update_epochs) <= 0:
        raise ValueError("update_epochs must be positive")
    if int(cfg.num_minibatches) <= 0:
        raise ValueError("num_minibatches must be positive")
    if steps_per_update % int(cfg.num_minibatches) != 0:
        raise ValueError("num_envs*num_steps must be divisible by num_minibatches")
    if int(cfg.hot_capacity) <= 0:
        raise ValueError("hot_capacity must be positive")
    if int(cfg.top_k) <= 0 or int(cfg.top_k) > int(cfg.hot_capacity):
        raise ValueError("top_k must be in [1, hot_capacity]")
    return num_updates


def _hp_get(hp, name: str):
    if isinstance(hp, dict):
        return hp[name]
    return getattr(hp, name)


def _hp_values(hp, bank_arrays):
    capacity = int(bank_arrays["keys"].shape[0])
    if hp is None:
        hp = default_retrieval_hp(min(16, capacity))
    top_k = int(_hp_get(hp, "top_k"))
    if top_k <= 0 or top_k > capacity:
        raise ValueError("retrieval top_k must be in [1, bank capacity]")
    return (
        float(_hp_get(hp, "tau_r")),
        float(_hp_get(hp, "beta_c")),
        float(_hp_get(hp, "beta_I")),
        float(_hp_get(hp, "beta_a")),
        float(_hp_get(hp, "w_rho")),
        float(_hp_get(hp, "w_c")),
        float(_hp_get(hp, "b0")),
        top_k,
    )


def _jit_hp(tau_r, beta_c, beta_I, beta_a, w_rho, w_c, b0, top_k):
    return {
        "tau_r": tau_r,
        "beta_c": beta_c,
        "beta_I": beta_I,
        "beta_a": beta_a,
        "top_k": int(top_k),
        "w_rho": w_rho,
        "w_c": w_c,
        "b0": b0,
    }  # spec §9


@partial(jax.jit, static_argnames=("top_k",))
def _lm_policy_step_jit(
    params,
    obs,
    game_id_vec,
    bank_arrays,
    mu_g,
    sigma_g,
    rng,
    tau_r,
    beta_c,
    beta_I,
    beta_a,
    w_rho,
    w_c,
    b0,
    top_k,
):
    hp = _jit_hp(tau_r, beta_c, beta_I, beta_a, w_rho, w_c, b0, top_k)
    rng, action_key = jax.random.split(rng)
    out = mem_apply(params, obs, game_id_vec, bank_arrays, hp, mu_g, sigma_g)
    # spec §9: explicit blend is reserved for old-game retention; current game trains on the net's own memory-conditioned policy
    logits_net = out["logits_net"]
    actions = jax.random.categorical(action_key, logits_net, axis=-1).astype(jnp.int32)
    logprobs = _categorical_log_prob(logits_net, actions)
    values = out["v_net"]
    return actions, logprobs, values, rng


def lm_policy_step(params, obs, game_id_vec, bank_arrays, mu_g, sigma_g, rng, hp=None):
    """Sample from the memory-conditioned net policy and return its old log-prob."""
    values = _hp_values(hp, bank_arrays)
    return _lm_policy_step_jit(
        params,
        obs,
        game_id_vec,
        bank_arrays,
        float(mu_g),
        float(sigma_g),
        rng,
        *values,
    )


@partial(jax.jit, static_argnames=("top_k",))
def _lm_value_step_jit(
    params,
    obs,
    game_id_vec,
    bank_arrays,
    mu_g,
    sigma_g,
    tau_r,
    beta_c,
    beta_I,
    beta_a,
    w_rho,
    w_c,
    b0,
    top_k,
):
    hp = _jit_hp(tau_r, beta_c, beta_I, beta_a, w_rho, w_c, b0, top_k)
    out = mem_apply(params, obs, game_id_vec, bank_arrays, hp, mu_g, sigma_g)
    return out["v_net"]


def lm_value_step(params, obs, game_id_vec, bank_arrays, mu_g, sigma_g, hp=None):
    values = _hp_values(hp, bank_arrays)
    return _lm_value_step_jit(
        params,
        obs,
        game_id_vec,
        bank_arrays,
        float(mu_g),
        float(sigma_g),
        *values,
    )


@partial(jax.jit, static_argnames=("top_k",))
def _lm_write_forward_jit(
    params,
    obs,
    game_id_vec,
    bank_arrays,
    mu_g,
    sigma_g,
    tau_r,
    beta_c,
    beta_I,
    beta_a,
    w_rho,
    w_c,
    b0,
    top_k,
):
    hp = _jit_hp(tau_r, beta_c, beta_I, beta_a, w_rho, w_c, b0, top_k)
    out = mem_apply(params, obs, game_id_vec, bank_arrays, hp, mu_g, sigma_g)
    return out["logits_net"]  # spec §8


def _lm_write_forward(params, obs, game_id_vec, bank_arrays, mu_g, sigma_g, hp):
    values = _hp_values(hp, bank_arrays)
    return _lm_write_forward_jit(
        params,
        obs,
        game_id_vec,
        bank_arrays,
        float(mu_g),
        float(sigma_g),
        *values,
    )


def _lm_ppo_loss(
    params,
    batch: TrainBatch,
    game_id,
    bank_arrays,
    hp,
    mu_g,
    sigma_g,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    active_mask=None,
):
    out = mem_apply(params, batch.obs, game_id, bank_arrays, hp, mu_g, sigma_g, active_mask)
    logits_net = out["logits_net"]
    new_values = out["v_net"]
    new_logprobs = _categorical_log_prob(logits_net, batch.actions)
    entropy = _categorical_entropy(logits_net)  # spec §10
    logratio = new_logprobs - batch.logprobs
    ratio = jnp.exp(logratio)  # spec §10

    advantages = (batch.advantages - jnp.mean(batch.advantages)) / (
        jnp.std(batch.advantages) + 1.0e-8
    )
    pg_loss1 = -advantages * ratio  # spec §10
    pg_loss2 = -advantages * jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)  # spec §10
    pg_loss = jnp.mean(jnp.maximum(pg_loss1, pg_loss2))  # spec §10

    v_loss_unclipped = jnp.square(new_values - batch.returns)  # spec §10
    v_clipped = batch.values + jnp.clip(new_values - batch.values, -clip_coef, clip_coef)
    v_loss_clipped = jnp.square(v_clipped - batch.returns)
    v_loss = 0.5 * jnp.mean(jnp.maximum(v_loss_unclipped, v_loss_clipped))  # spec §10

    entropy_loss = jnp.mean(entropy)
    approx_kl = jnp.mean((ratio - 1.0) - logratio)
    clipfrac = jnp.mean((jnp.abs(ratio - 1.0) > clip_coef).astype(jnp.float32))
    loss = pg_loss + vf_coef * v_loss - ent_coef * entropy_loss  # spec §10
    aux = jnp.asarray([pg_loss, v_loss, entropy_loss, approx_kl, clipfrac], dtype=jnp.float32)
    return loss, aux


def _tree_all_finite(tree):
    finite = jnp.asarray(True)
    for leaf in jax.tree_util.tree_leaves(tree):
        finite = jnp.logical_and(finite, jnp.all(jnp.isfinite(leaf)))
    return finite


def _select_tree(new_tree, old_tree, predicate):
    return jax.tree_util.tree_map(lambda new, old: jnp.where(predicate, new, old), new_tree, old_tree)


@partial(
    jax.jit,
    static_argnames=(
        "update_epochs",
        "num_minibatches",
        "minibatch_size",
        "clip_coef",
        "vf_coef",
        "ent_coef",
        "max_grad_norm",
        "top_k",
    ),
)
def _lm_update_jit(
    params,
    opt_state,
    batch: TrainBatch,
    game_id,
    bank_arrays,
    mu_g,
    sigma_g,
    rng,
    learning_rate: float,
    update_epochs: int,
    num_minibatches: int,
    minibatch_size: int,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
    tau_r,
    beta_c,
    beta_I,
    beta_a,
    w_rho,
    w_c,
    b0,
    top_k,
    active_mask,
):
    hp = _jit_hp(tau_r, beta_c, beta_I, beta_a, w_rho, w_c, b0, top_k)
    batch_size = int(num_minibatches) * int(minibatch_size)
    tx = optax.chain(
        optax.clip_by_global_norm(float(max_grad_norm)),
        optax.adam(learning_rate=learning_rate),
    )

    def epoch_step(carry, _):
        params, opt_state, rng = carry
        rng, perm_key = jax.random.split(rng)
        permutation = jax.random.permutation(perm_key, batch_size)
        minibatches = _make_minibatches(batch, permutation, int(num_minibatches), int(minibatch_size))

        def minibatch_step(carry, minibatch):
            params, opt_state = carry

            def loss_fn(p):
                return _lm_ppo_loss(
                    p,
                    minibatch,
                    game_id,
                    bank_arrays,
                    hp,
                    mu_g,
                    sigma_g,
                    clip_coef,
                    vf_coef,
                    ent_coef,
                    active_mask,
                )

            (loss, aux), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
            finite = jnp.logical_and(jnp.isfinite(loss), _tree_all_finite(grads))
            safe_grads = jax.tree_util.tree_map(
                lambda grad: jnp.where(finite, grad, jnp.zeros_like(grad)),
                grads,
            )
            updates, new_opt_state = tx.update(safe_grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            params = _select_tree(new_params, params, finite)
            opt_state = _select_tree(new_opt_state, opt_state, finite)
            safe_loss = jnp.where(jnp.isfinite(loss), loss, 0.0)
            safe_aux = jnp.where(jnp.isfinite(aux), aux, 0.0)
            metrics = jnp.concatenate(
                [jnp.asarray([safe_loss], dtype=jnp.float32), safe_aux, finite[None].astype(jnp.float32)],
                axis=0,
            )
            return (params, opt_state), metrics

        (params, opt_state), metrics = jax.lax.scan(minibatch_step, (params, opt_state), minibatches)
        return (params, opt_state, rng), jnp.mean(metrics, axis=0)

    (params, opt_state, rng), metrics = jax.lax.scan(
        epoch_step,
        (params, opt_state, rng),
        None,
        length=int(update_epochs),
    )
    return params, opt_state, rng, jnp.mean(metrics, axis=0)


def lm_update(
    params,
    opt_state,
    batch: TrainBatch,
    game_id,
    bank_arrays,
    mu_g,
    sigma_g,
    rng,
    learning_rate: float,
    update_epochs: int,
    num_minibatches: int,
    minibatch_size: int,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
    hp=None,
    active_mask=None,
):
    values = _hp_values(hp, bank_arrays)
    return _lm_update_jit(
        params,
        opt_state,
        batch,
        jnp.asarray(game_id, dtype=jnp.int32),
        bank_arrays,
        float(mu_g),
        float(sigma_g),
        rng,
        float(learning_rate),
        int(update_epochs),
        int(num_minibatches),
        int(minibatch_size),
        float(clip_coef),
        float(vf_coef),
        float(ent_coef),
        float(max_grad_norm),
        *values,
        active_mask,
    )


def _new_write_stats():
    return {
        "adv": RunningStats(),
        "delta": RunningStats(),
        "ret": RunningStats(),
    }


def _infer_dims(params):
    n_games, d_c = params["game_embed"]["embedding"].shape
    d_k = params["key_head"]["kernel"].shape[-1]
    d_m = params["wv"]["kernel"].shape[-1]
    act_dim = params["policy_head"]["kernel"].shape[-1]
    return int(n_games), int(d_k), int(d_c), int(d_m), int(act_dim)


def _context_embeddings(params, game_ids):
    n_games, d_k, d_c, d_m, act_dim = _infer_dims(params)
    net = MemAtariActorCritic(n_games=n_games, d_k=d_k, d_c=d_c, d_m=d_m, act_dim=act_dim)
    contexts = net.apply(
        {"params": params},
        jnp.asarray(game_ids, dtype=jnp.int32),
        method=MemAtariActorCritic.context,
    )
    return np.asarray(jax.device_get(contexts), dtype=np.float32)


def _write_memories(
    params,
    ema_params,
    memory_bank: MemoryBank,
    value_norm: RunningValueNorm,
    bank_arrays,
    hp,
    cfg: LMConfig,
    obs_buf,
    actions_buf,
    rewards_buf,
    dones_buf,
    values_buf,
    last_value,
    advantages,
    returns,
    game_id: int,
    mu_g: float,
    sigma_g: float,
    stats,
) -> int:
    batch_size = int(obs_buf.shape[0] * obs_buf.shape[1])
    flat_obs = np.asarray(obs_buf, dtype=np.uint8).reshape((batch_size,) + tuple(obs_buf.shape[2:]))
    flat_actions = np.asarray(actions_buf, dtype=np.int32).reshape((batch_size,))
    flat_rewards = np.asarray(rewards_buf, dtype=np.float32).reshape((batch_size,))
    flat_dones = np.asarray(dones_buf, dtype=np.float32).reshape((batch_size,))
    flat_values = np.asarray(values_buf, dtype=np.float32).reshape((batch_size,))
    flat_advantages = np.asarray(jax.device_get(advantages), dtype=np.float32).reshape((batch_size,))
    flat_returns = np.asarray(jax.device_get(returns), dtype=np.float32).reshape((batch_size,))
    last_value_np = np.asarray(jax.device_get(last_value), dtype=np.float32).reshape((1, obs_buf.shape[1]))

    next_values = np.concatenate([values_buf[1:], last_value_np], axis=0)
    next_values = np.asarray(next_values, dtype=np.float32) * (1.0 - np.asarray(dones_buf, dtype=np.float32))
    delta = td_error(flat_rewards, flat_values, next_values.reshape((batch_size,)), cfg.gamma)  # spec §8

    game_ids = np.full((batch_size,), int(game_id), dtype=np.int32)
    logits_net = np.asarray(
        jax.device_get(_lm_write_forward(params, flat_obs, game_ids, bank_arrays, mu_g, sigma_g, hp)),
        dtype=np.float32,
    )
    keys_t = np.asarray(jax.device_get(mem_apply_key(ema_params, flat_obs)), dtype=np.float32)  # spec §5

    bank = memory_bank.to_retrieval_arrays()
    bank_valid = np.ones((int(bank["keys"].shape[0]),), dtype=bool)
    novelty_t = novelty(keys_t, bank["keys"], bank_valid, bank["game_id"], game_ids)  # spec §8
    entropy_t = policy_entropy(logits_net)  # spec §8
    life_t = flat_dones.astype(np.float32)  # spec §8
    forget_t = np.zeros_like(flat_advantages, dtype=np.float32)  # spec §8

    abs_adv = np.abs(flat_advantages)
    abs_delta = np.abs(delta)
    stats["adv"].update(abs_adv)
    stats["delta"].update(abs_delta)
    stats["ret"].update(flat_returns)
    abs_adv_hat = stats["adv"].normalize(abs_adv)  # spec §8
    abs_delta_hat = stats["delta"].normalize(abs_delta)  # spec §8
    ret_hat = stats["ret"].normalize(flat_returns)  # spec §8

    scores = importance(
        abs_adv_hat,
        abs_delta_hat,
        novelty_t,
        entropy_t,
        life_t,
        ret_hat,
        forget_t,
    )  # spec §8
    selected = select_writes(
        scores,
        float(cfg.write_top_fraction),
        min_quota=int(cfg.write_min_quota),
    )  # spec §8

    inserted = 0
    if np.any(selected):
        novelty_hi = novelty_t >= float(np.quantile(novelty_t, 0.9))
        source_flags = write_source_flags(
            high_return=ret_hat > 0.0,
            near_life_loss=life_t > 0.0,
            novelty_hi=novelty_hi,
            failure_recovery=np.zeros_like(life_t, dtype=bool),
        )
        contexts = _context_embeddings(params, game_ids[selected])
        kwargs = build_insert_kwargs(
            keys_t[selected],
            contexts,
            logits_net[selected],
            flat_returns[selected],
            game_ids[selected],
            scores[selected],
            mu_g=float(mu_g),
            sigma_g=float(sigma_g),
            temperature=float(cfg.teacher_temperature),
            novelty=novelty_t[selected],
            eps_policy=float(cfg.eps_policy),
            eps_value=float(cfg.eps_value),
            source_flags=source_flags[selected],
        )  # spec §8
        memory_bank.insert(
            **kwargs,
            actions=flat_actions[selected],
            rewards=flat_rewards[selected],
            dones=flat_dones[selected].astype(bool),
            return_traces=flat_returns[selected],
        )
        inserted = int(np.sum(selected))

    value_norm.update(flat_returns)
    return inserted


def train_living_memory(
    game,
    game_id,
    n_games,
    cfg,
    seed,
    *,
    init_params=None,
    memory_bank=None,
    ema_params=None,
    value_norm=None,
) -> dict:
    """Train one Atari game with memory-conditioned rollouts, writes, and PPO."""
    cfg = LMConfig() if cfg is None else cfg
    num_updates = _validate_config(cfg)
    batch_size = int(cfg.num_envs) * int(cfg.num_steps)
    minibatch_size = batch_size // int(cfg.num_minibatches)
    hp = default_retrieval_hp(int(cfg.top_k))

    rng = jax.random.PRNGKey(int(seed))
    rng, init_key = jax.random.split(rng)
    params = init_params
    if params is None:
        params = mem_init(
            init_key,
            int(n_games),
            int(cfg.hot_capacity),
            d_k=int(cfg.d_k),
            d_c=int(cfg.d_c),
            d_m=int(cfg.d_m),
            act_dim=ACT_DIM,
            top_k=int(cfg.top_k),
        )
    if ema_params is None:
        ema_params = params
    if memory_bank is None:
        memory_bank = MemoryBank(
            capacity=max(int(cfg.memory_capacity), int(cfg.hot_capacity)),
            d_k=int(cfg.d_k),
            d_c=int(cfg.d_c),
            act_dim=ACT_DIM,
            b_min=0,
        )
    if value_norm is None:
        value_norm = RunningValueNorm()

    tx = optax.chain(
        optax.clip_by_global_norm(float(cfg.max_grad_norm)),
        optax.adam(learning_rate=float(cfg.lr)),
    )
    opt_state = tx.init(params)

    env = make_train_env(str(game), int(cfg.num_envs), int(seed))
    next_obs, _ = env.reset()
    next_obs = np.asarray(next_obs, dtype=np.uint8)
    tracker = EpisodeReturnTracker(int(cfg.num_envs))
    recent_returns = deque(maxlen=100)
    returns_curve: list[float] = []
    last_curve_value = 0.0
    write_stats = _new_write_stats()

    obs_buf = np.zeros((int(cfg.num_steps), int(cfg.num_envs), 4, 84, 84), dtype=np.uint8)
    actions_buf = np.zeros((int(cfg.num_steps), int(cfg.num_envs)), dtype=np.int32)
    logprobs_buf = np.zeros((int(cfg.num_steps), int(cfg.num_envs)), dtype=np.float32)
    rewards_buf = np.zeros((int(cfg.num_steps), int(cfg.num_envs)), dtype=np.float32)
    dones_buf = np.zeros((int(cfg.num_steps), int(cfg.num_envs)), dtype=np.float32)
    values_buf = np.zeros((int(cfg.num_steps), int(cfg.num_envs)), dtype=np.float32)

    started = time.perf_counter()
    game_id_vec = jnp.full((int(cfg.num_envs),), int(game_id), dtype=jnp.int32)
    for update in range(1, num_updates + 1):
        bank_arrays = pad_bank(
            memory_bank,
            int(cfg.hot_capacity),
            d_k=int(cfg.d_k),
            d_c=int(cfg.d_c),
            act_dim=ACT_DIM,
        )
        mu_segment = value_norm.mu()
        sigma_segment = value_norm.sigma()
        completed_this_update: list[float] = []

        for step in range(int(cfg.num_steps)):
            obs_buf[step] = next_obs
            actions, logprobs, values, rng = lm_policy_step(
                params,
                next_obs,
                game_id_vec,
                bank_arrays,
                mu_segment,
                sigma_segment,
                rng,
                hp=hp,
            )
            actions_np = np.asarray(jax.device_get(actions), dtype=np.int32)
            logprobs_buf[step] = np.asarray(jax.device_get(logprobs), dtype=np.float32)
            values_buf[step] = np.asarray(jax.device_get(values), dtype=np.float32)
            actions_buf[step] = actions_np

            next_obs, rewards, terminated, truncated, info = env.step(actions_np)
            next_obs = np.asarray(next_obs, dtype=np.uint8)
            rewards = np.asarray(rewards, dtype=np.float32)
            terminated = np.asarray(terminated, dtype=bool)
            truncated = np.asarray(truncated, dtype=bool)
            done = np.logical_or(terminated, truncated)

            rewards_buf[step] = rewards
            dones_buf[step] = done.astype(np.float32)
            completed = tracker.update(rewards, terminated, truncated, info)
            completed_this_update.extend(completed)

        if completed_this_update:
            recent_returns.extend(completed_this_update)
        last_curve_value = _mean_or_previous(recent_returns, last_curve_value)
        returns_curve.append(last_curve_value)

        last_value = lm_value_step(
            params,
            next_obs,
            game_id_vec,
            bank_arrays,
            mu_segment,
            sigma_segment,
            hp=hp,
        )
        advantages, returns = gae(
            jnp.asarray(rewards_buf),
            jnp.asarray(dones_buf),
            jnp.asarray(values_buf),
            last_value,
            float(cfg.gamma),
            float(cfg.gae_lambda),
        )  # spec §10

        if int(cfg.write_every) > 0 and update % int(cfg.write_every) == 0:
            _write_memories(
                params,
                ema_params,
                memory_bank,
                value_norm,
                bank_arrays,
                hp,
                cfg,
                obs_buf,
                actions_buf,
                rewards_buf,
                dones_buf,
                values_buf,
                last_value,
                advantages,
                returns,
                int(game_id),
                mu_segment,
                sigma_segment,
                write_stats,
            )
        else:
            value_norm.update(np.asarray(jax.device_get(returns), dtype=np.float32).reshape(-1))

        batch = _flatten_rollout(
            obs_buf,
            actions_buf,
            logprobs_buf,
            advantages,
            returns,
            values_buf,
            batch_size,
        )
        lr = _learning_rate(cfg, update, num_updates)
        params, opt_state, rng, _ = lm_update(
            params,
            opt_state,
            batch,
            int(game_id),
            bank_arrays,
            mu_segment,
            sigma_segment,
            rng,
            float(lr),
            int(cfg.update_epochs),
            int(cfg.num_minibatches),
            int(minibatch_size),
            float(cfg.clip_coef),
            float(cfg.vf_coef),
            float(cfg.ent_coef),
            float(cfg.max_grad_norm),
            hp=hp,
        )
        ema_params = ema_update(ema_params, params, float(cfg.tau_key))  # spec §5

    elapsed = max(time.perf_counter() - started, 1.0e-9)
    timesteps = int(num_updates * batch_size)
    final_return = float(returns_curve[-1]) if returns_curve else 0.0
    return {
        "params": params,
        "ema_params": ema_params,
        "memory_bank": memory_bank,
        "value_norm": value_norm,
        "returns_curve": [float(v) for v in returns_curve],
        "final_return": final_return,
        "timesteps": timesteps,
        "steps_per_sec": float(timesteps / elapsed),
        "mem_size": int(len(memory_bank)),
    }


__all__ = [
    "LMConfig",
    "lm_policy_step",
    "lm_update",
    "lm_value_step",
    "train_living_memory",
]
