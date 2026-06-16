"""Full PMA-C projected gradient updates for RL."""

from __future__ import annotations

from dataclasses import dataclass
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pmac.agents.ppo_atari import GuardBatch, TrainBatch, _guard_loss, _make_minibatches, _ppo_loss
from pmac.projection import project_conflicts, plasticity_ratio
from pmac.stability import scale_by_stability
from pmac.tree_utils import tree_add_scaled, tree_dot, tree_norm, tree_scale, tree_zeros_like


EPS = 1.0e-8

PPO_PMAC_METRIC_NAMES = (
    "ppo_loss",
    "pg_loss",
    "v_loss",
    "entropy_loss",
    "approx_kl",
    "clipfrac",
    "guard_loss",
    "g_new_norm",
    "g_projected_norm",
    "g_total_norm",
    "projection_ratio",
    "total_guard_norm",
    "clipped_guard_count",
    "nonfinite",
)


@dataclass(frozen=True)
class PMACUpdateConfig:
    guard_clip_alpha: float = 1.0
    guard_total_beta: float = 1.0
    stability_alpha: float = 10.0
    projection: bool = True
    stability: bool = True
    guard_correction: bool = True


@dataclass
class PMACUpdateMetrics:
    g_new_norm: float
    g_projected_norm: float
    g_total_norm: float
    projection_ratio: float
    total_guard_norm: float
    conflict_dots: list
    clipped_guard_count: int
    nonfinite: bool


def _tree_all_finite(tree) -> jnp.ndarray:
    finite = jnp.asarray(True)
    for leaf in jax.tree_util.tree_leaves(tree):
        finite = jnp.logical_and(finite, jnp.all(jnp.isfinite(leaf)))
    return finite


def _zero_if_nonfinite(tree, template, nonfinite):
    zeros = tree_zeros_like(template)
    return jax.tree_util.tree_map(lambda x, z: jnp.where(nonfinite, z, x), tree, zeros)


@partial(jax.jit, static_argnames=("cfg",))
def _combine_core(g_new, guard_grads, guard_lambdas, omega, cfg: PMACUpdateConfig):
    cfg = PMACUpdateConfig() if cfg is None else cfg
    k = len(guard_grads)
    g_new_norm = tree_norm(g_new)
    guard_lambdas = jnp.asarray(guard_lambdas, dtype=jnp.float32).reshape((k,))
    inputs_finite = _tree_all_finite(g_new)

    conflict_dots = []
    clipped_flags = []
    clipped_guards = []
    guard_cap = (
        float(cfg.guard_clip_alpha)
        * g_new_norm
        / jnp.sqrt(jnp.asarray(max(k, 1), dtype=jnp.float32))
    )
    for g_guard in guard_grads:
        inputs_finite = jnp.logical_and(inputs_finite, _tree_all_finite(g_guard))
        conflict_dots.append(tree_dot(g_new, g_guard))
        g_guard_norm = tree_norm(g_guard)
        scale = jnp.minimum(jnp.asarray(1.0, dtype=jnp.float32), guard_cap / (g_guard_norm + EPS))
        clipped_guards.append(tree_scale(g_guard, scale))
        clipped_flags.append((g_guard_norm > guard_cap).astype(jnp.int32))

    if bool(cfg.projection) and k > 0:
        g_projected = project_conflicts(g_new, clipped_guards, EPS)
    else:
        g_projected = g_new

    if bool(cfg.guard_correction) and k > 0:
        correction = tree_zeros_like(g_new)
        for i, g_guard in enumerate(clipped_guards):
            correction = tree_add_scaled(correction, g_guard, guard_lambdas[i])
        correction_norm = tree_norm(correction)
        correction_cap = float(cfg.guard_total_beta) * g_new_norm
        correction_scale = jnp.minimum(
            jnp.asarray(1.0, dtype=jnp.float32),
            correction_cap / (correction_norm + EPS),
        )
        correction = tree_scale(correction, correction_scale)
        total_guard_norm = tree_norm(correction)
        g_total = tree_add_scaled(g_projected, correction, 1.0)
    else:
        total_guard_norm = jnp.asarray(0.0, dtype=jnp.float32)
        g_total = g_projected

    if bool(cfg.stability):
        g_total = scale_by_stability(g_total, omega, float(cfg.stability_alpha))

    if bool(cfg.stability):
        inputs_finite = jnp.logical_and(inputs_finite, _tree_all_finite(omega))
    nonfinite = jnp.logical_not(jnp.logical_and(inputs_finite, _tree_all_finite(g_total)))
    g_total = _zero_if_nonfinite(g_total, g_new, nonfinite)
    g_projected_norm = tree_norm(g_projected)
    g_total_norm = tree_norm(g_total)
    projection_ratio = plasticity_ratio(g_projected, g_new, EPS)
    if conflict_dots:
        conflict_dots_arr = jnp.asarray(conflict_dots, dtype=jnp.float32)
        clipped_guard_count = jnp.sum(jnp.asarray(clipped_flags, dtype=jnp.int32))
    else:
        conflict_dots_arr = jnp.zeros((0,), dtype=jnp.float32)
        clipped_guard_count = jnp.asarray(0, dtype=jnp.int32)

    metrics = (
        g_new_norm,
        g_projected_norm,
        g_total_norm,
        projection_ratio,
        total_guard_norm,
        conflict_dots_arr,
        clipped_guard_count,
        nonfinite,
    )
    return g_total, metrics


def _as_float(value) -> float:
    return float(np.asarray(jax.device_get(value)))


def combine_grads(g_new, guard_grads, guard_lambdas, omega, cfg: PMACUpdateConfig):
    """Combine PPO and guard gradients using PMA-C projection geometry."""
    guard_grads = tuple(guard_grads)
    if cfg is None:
        cfg = PMACUpdateConfig()
    if omega is None:
        omega = tree_zeros_like(g_new)
    if guard_lambdas is None:
        guard_lambdas = jnp.zeros((len(guard_grads),), dtype=jnp.float32)

    g_total, raw = _combine_core(
        g_new,
        guard_grads,
        jnp.asarray(guard_lambdas, dtype=jnp.float32),
        omega,
        cfg,
    )
    (
        g_new_norm,
        g_projected_norm,
        g_total_norm,
        projection_ratio,
        total_guard_norm,
        conflict_dots,
        clipped_guard_count,
        nonfinite,
    ) = raw
    metrics = PMACUpdateMetrics(
        g_new_norm=_as_float(g_new_norm),
        g_projected_norm=_as_float(g_projected_norm),
        g_total_norm=_as_float(g_total_norm),
        projection_ratio=_as_float(projection_ratio),
        total_guard_norm=_as_float(total_guard_norm),
        conflict_dots=[float(v) for v in np.asarray(jax.device_get(conflict_dots)).reshape(-1)],
        clipped_guard_count=int(np.asarray(jax.device_get(clipped_guard_count))),
        nonfinite=bool(np.asarray(jax.device_get(nonfinite))),
    )
    return g_total, metrics


def _make_guard_batch(obs, game_onehot, teacher_logits, teacher_value) -> GuardBatch:
    return GuardBatch(
        obs=obs,
        game_onehot=game_onehot,
        teacher_logits=teacher_logits,
        teacher_value=teacher_value,
    )


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
        "guard_value_coef",
        "guard_tolerance",
        "cfg",
    ),
)
def ppo_pmac_update(
    params,
    opt_state,
    batch: TrainBatch,
    game_onehot,
    guard_obs,
    guard_game_onehot,
    guard_teacher_logits,
    guard_teacher_value,
    guard_lambdas,
    omega,
    rng,
    learning_rate: float,
    update_epochs: int,
    num_minibatches: int,
    minibatch_size: int,
    clip_coef: float,
    vf_coef: float,
    ent_coef: float,
    max_grad_norm: float,
    guard_value_coef: float,
    guard_tolerance: float,
    cfg: PMACUpdateConfig,
):
    """Run Atari PPO updates with full PMA-C projected gradient geometry."""
    cfg = PMACUpdateConfig() if cfg is None else cfg
    batch_size = int(num_minibatches) * int(minibatch_size)
    guard_k = int(guard_obs.shape[0])
    tx = optax.chain(
        optax.clip_by_global_norm(float(max_grad_norm)),
        optax.adam(learning_rate=learning_rate),
    )

    def guard_loss_fn(p, obs_k, game_k, logits_k, value_k):
        guard_batch = _make_guard_batch(obs_k, game_k, logits_k, value_k)
        return _guard_loss(p, guard_batch, guard_value_coef, guard_tolerance)

    guard_value_and_grad = jax.value_and_grad(guard_loss_fn)

    def epoch_step(carry, _):
        params, opt_state, rng = carry
        rng, perm_key = jax.random.split(rng)
        permutation = jax.random.permutation(perm_key, batch_size)
        minibatches = _make_minibatches(batch, permutation, int(num_minibatches), int(minibatch_size))

        def minibatch_step(carry, minibatch):
            params, opt_state = carry

            def ppo_loss_fn(p):
                return _ppo_loss(p, minibatch, game_onehot, clip_coef, vf_coef, ent_coef)

            (ppo_loss, ppo_aux), g_new = jax.value_and_grad(ppo_loss_fn, has_aux=True)(params)

            if guard_k > 0:
                guard_losses, guard_grads_stacked = jax.vmap(
                    lambda obs_k, game_k, logits_k, value_k: guard_value_and_grad(
                        params,
                        obs_k,
                        game_k,
                        logits_k,
                        value_k,
                    )
                )(
                    guard_obs,
                    guard_game_onehot,
                    guard_teacher_logits,
                    guard_teacher_value,
                )
                guard_grads = tuple(
                    jax.tree_util.tree_map(lambda x, i=i: x[i], guard_grads_stacked)
                    for i in range(guard_k)
                )
                guard_loss_mean = jnp.mean(guard_losses)
            else:
                guard_grads = ()
                guard_loss_mean = jnp.asarray(0.0, dtype=jnp.float32)

            g_total, combine_metrics = _combine_core(
                g_new,
                guard_grads,
                guard_lambdas,
                omega,
                cfg,
            )
            (
                g_new_norm,
                g_projected_norm,
                g_total_norm,
                projection_ratio,
                total_guard_norm,
                _conflict_dots,
                clipped_guard_count,
                nonfinite,
            ) = combine_metrics

            def apply_update(update_state):
                p, state, grads = update_state
                updates, state = tx.update(grads, state, p)
                return optax.apply_updates(p, updates), state

            def skip_update(update_state):
                p, state, _grads = update_state
                return p, state

            params, opt_state = jax.lax.cond(
                nonfinite,
                skip_update,
                apply_update,
                (params, opt_state, g_total),
            )
            metrics = jnp.concatenate(
                [
                    jnp.asarray([ppo_loss], dtype=jnp.float32),
                    ppo_aux,
                    jnp.asarray(
                        [
                            guard_loss_mean,
                            g_new_norm,
                            g_projected_norm,
                            g_total_norm,
                            projection_ratio,
                            total_guard_norm,
                            clipped_guard_count.astype(jnp.float32),
                            nonfinite.astype(jnp.float32),
                        ],
                        dtype=jnp.float32,
                    ),
                ],
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


__all__ = [
    "PMACUpdateConfig",
    "PMACUpdateMetrics",
    "PPO_PMAC_METRIC_NAMES",
    "combine_grads",
    "ppo_pmac_update",
]
