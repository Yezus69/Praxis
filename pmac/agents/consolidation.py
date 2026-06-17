"""Slow memory-to-weight consolidation for Living Memory PMA-C section 21."""

from __future__ import annotations

from collections.abc import Mapping
from functools import partial

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pmac.agents.atari_mem_net import mem_apply
from pmac.memory.losses import (
    latent_conservation_loss,
    retrieval_alignment_loss,
    visual_sentinel_loss,
)
from pmac.memory.reader import EPS, ema_update
from pmac.memory.runtime import default_retrieval_hp
from pmac.memory.sentinels_visual import VisualSentinelStore, build_align_batch


def _cfg_get(cfg, name: str, default):
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _cfg_int(cfg, name: str, default: int) -> int:
    return int(_cfg_get(cfg, name, default))


def _cfg_float(cfg, name: str, default: float) -> float:
    return float(_cfg_get(cfg, name, default))


def _infer_dims(params) -> tuple[int, int, int, int, int]:
    n_games, d_c = params["game_embed"]["embedding"].shape
    d_k = params["key_head"]["kernel"].shape[-1]
    d_m = params["wv"]["kernel"].shape[-1]
    act_dim = params["policy_head"]["kernel"].shape[-1]
    return int(n_games), int(d_k), int(d_c), int(d_m), int(act_dim)


def _hp_values(hp):
    return (
        float(hp["tau_r"]),
        float(hp["beta_c"]),
        float(hp["beta_I"]),
        float(hp["beta_a"]),
        float(hp["w_rho"]),
        float(hp["w_c"]),
        float(hp["b0"]),
        int(hp["top_k"]),
    )


def _jit_hp(tau_r, beta_c, beta_i, beta_a, w_rho, w_c, b0, top_k):
    return {
        "tau_r": tau_r,
        "beta_c": beta_c,
        "beta_I": beta_i,
        "beta_a": beta_a,
        "top_k": int(top_k),
        "w_rho": w_rho,
        "w_c": w_c,
        "b0": b0,
    }


def _tree_all_finite(tree):
    finite = jnp.asarray(True)
    for leaf in jax.tree_util.tree_leaves(tree):
        finite = jnp.logical_and(finite, jnp.all(jnp.isfinite(leaf)))
    return finite


def _select_tree(new_tree, old_tree, predicate):
    return jax.tree_util.tree_map(lambda new, old: jnp.where(predicate, new, old), new_tree, old_tree)


def _normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + float(EPS))


def _valid_rows(bank) -> np.ndarray:
    keys = np.asarray(jax.device_get(bank["keys"]), dtype=np.float32)
    valid = np.asarray(
        jax.device_get(bank.get("valid", np.ones((keys.shape[0],), dtype=bool))),
        dtype=bool,
    ).reshape(-1)
    valid = valid & np.all(np.isfinite(keys), axis=-1)
    return np.flatnonzero(valid).astype(np.int64)


def _sample_atom_batch(protected_bank, batch_size: int, seed: int, eps_policy: float) -> dict:
    keys = np.asarray(jax.device_get(protected_bank["keys"]), dtype=np.float32)
    _, d_k = keys.shape
    act_dim = int(np.asarray(protected_bank["teacher_policy"]).shape[-1])
    valid_rows = _valid_rows(protected_bank)
    rng = np.random.default_rng(int(seed))
    if valid_rows.size:
        replace = valid_rows.size < int(batch_size)
        rows = rng.choice(valid_rows, size=int(batch_size), replace=replace)
        weight = np.ones((int(batch_size),), dtype=np.float32)
        return {
            "keys": _normalize_rows(np.asarray(jax.device_get(protected_bank["keys"]))[rows]),
            "game_id": np.asarray(jax.device_get(protected_bank["game_id"]))[rows].astype(np.int32),
            "teacher_policy": np.asarray(
                jax.device_get(protected_bank["teacher_policy"])
            )[rows].astype(np.float32),
            "teacher_value": np.asarray(
                jax.device_get(protected_bank["teacher_value"])
            )[rows].astype(np.float32),
            "eps": np.full((int(batch_size),), float(eps_policy), dtype=np.float32),
            "weight": weight,
        }
    return {
        "keys": np.zeros((int(batch_size), d_k), dtype=np.float32),
        "game_id": np.zeros((int(batch_size),), dtype=np.int32),
        "teacher_policy": np.zeros((int(batch_size), act_dim), dtype=np.float32),
        "teacher_value": np.zeros((int(batch_size),), dtype=np.float32),
        "eps": np.full((int(batch_size),), float(eps_policy), dtype=np.float32),
        "weight": np.zeros((int(batch_size),), dtype=np.float32),
    }


def _empty_sent_batch(batch_size: int, d_k: int, act_dim: int) -> dict:
    return {
        "obs": np.zeros((int(batch_size), 4, 84, 84), dtype=np.uint8),
        "game_id": np.zeros((int(batch_size),), dtype=np.int32),
        "key_star": np.zeros((int(batch_size), int(d_k)), dtype=np.float32),
        "teacher_policy": np.zeros((int(batch_size), int(act_dim)), dtype=np.float32),
        "teacher_value": np.zeros((int(batch_size),), dtype=np.float32),
    }


def _empty_align_batch(batch_size: int, n_neg: int, d_k: int) -> dict:
    return {
        "obs": np.zeros((int(batch_size), 4, 84, 84), dtype=np.uint8),
        "game_id": np.zeros((int(batch_size),), dtype=np.int32),
        "pos_key": np.zeros((int(batch_size), int(d_k)), dtype=np.float32),
        "neg_keys": np.zeros((int(batch_size), int(n_neg), int(d_k)), dtype=np.float32),
        "neg_game_id": np.zeros((int(batch_size), int(n_neg)), dtype=np.int32),
    }


def _bank_has_cross_game_negatives(bank) -> bool:
    valid = _valid_rows(bank)
    if valid.size == 0:
        return False
    game_id = np.asarray(jax.device_get(bank["game_id"]), dtype=np.int32).reshape(-1)
    return int(np.unique(game_id[valid]).shape[0]) > 1


def _sample_sent_batch(visual_store, batch_size: int, seed: int, d_k: int, act_dim: int) -> tuple[dict, bool]:
    if visual_store is None or len(visual_store) == 0:
        return _empty_sent_batch(batch_size, d_k, act_dim), False
    sent_batch = visual_store.batch(int(batch_size), seed=int(seed))
    return {
        "obs": np.asarray(sent_batch["obs"], dtype=np.uint8),
        "game_id": np.asarray(sent_batch["game_id"], dtype=np.int32),
        "key_star": np.asarray(sent_batch["key_star"], dtype=np.float32),
        "teacher_policy": np.asarray(sent_batch["teacher_policy"], dtype=np.float32),
        "teacher_value": np.asarray(sent_batch["teacher_value"], dtype=np.float32),
    }, True


def _sample_align_batch(
    sent_batch,
    protected_bank,
    *,
    batch_size: int,
    n_neg: int,
    seed: int,
    d_k: int,
    enabled: bool,
) -> tuple[dict, bool]:
    if not enabled:
        return _empty_align_batch(batch_size, n_neg, d_k), False
    try:
        align_batch = build_align_batch(
            sent_batch,
            protected_bank,
            n_neg=int(n_neg),
            batch_size=int(batch_size),
            seed=int(seed),
        )
    except ValueError:
        return _empty_align_batch(batch_size, n_neg, d_k), False
    return {
        "obs": np.asarray(align_batch["obs"], dtype=np.uint8),
        "game_id": np.asarray(align_batch["game_id"], dtype=np.int32),
        "pos_key": np.asarray(align_batch["pos_key"], dtype=np.float32),
        "neg_keys": np.asarray(align_batch["neg_keys"], dtype=np.float32),
        "neg_game_id": np.asarray(align_batch["neg_game_id"], dtype=np.int32),
    }, True


def _adapter_distill_loss(params, sent_batch, bank, hp, active_mask):
    adapter_out = mem_apply(
        params,
        sent_batch["obs"],
        sent_batch["game_id"],
        bank,
        hp,
        active_mask=active_mask,
    )
    base_out = mem_apply(
        params,
        sent_batch["obs"],
        sent_batch["game_id"],
        bank,
        hp,
        active_mask=None,
    )
    p_adapter = jax.lax.stop_gradient(jax.nn.softmax(adapter_out["logits_final"], axis=-1))
    p_base = jax.nn.softmax(base_out["logits_final"], axis=-1)
    return jnp.mean(
        jnp.sum(p_adapter * (jnp.log(p_adapter + EPS) - jnp.log(p_base + EPS)), axis=-1)
    )  # spec §21


@partial(
    jax.jit,
    static_argnames=("dims", "top_k", "has_visual", "has_retr", "adapter_active"),
)
def _consolidation_step(
    params,
    opt_state,
    atom_batch,
    sent_batch,
    align_batch,
    protected_bank,
    active_mask,
    slow_lr,
    tau_r,
    beta_c,
    beta_i,
    beta_a,
    w_rho,
    w_c,
    b0,
    top_k,
    lambda_v,
    huber_delta,
    lambda_visual,
    lambda_key,
    lambda_retr,
    visual_lambda_v,
    retr_tau,
    lambda_distill,
    dims,
    has_visual: bool,
    has_retr: bool,
    adapter_active: bool,
):
    hp = _jit_hp(tau_r, beta_c, beta_i, beta_a, w_rho, w_c, b0, top_k)
    tx = optax.adam(learning_rate=slow_lr)

    def loss_fn(p):
        l_cons = latent_conservation_loss(
            p,
            atom_batch,
            protected_bank,
            hp,
            lambda_v=lambda_v,
            huber_delta=huber_delta,
            dims=dims,
        )  # spec §21
        if has_visual:
            l_key, l_visual_beh = visual_sentinel_loss(
                p,
                sent_batch,
                protected_bank,
                hp,
                lambda_v=visual_lambda_v,
                huber_delta=huber_delta,
                dims=dims,
            )  # spec §21
        else:
            l_key = jnp.asarray(0.0, dtype=jnp.float32)
            l_visual_beh = jnp.asarray(0.0, dtype=jnp.float32)
        if has_retr:
            l_retr = retrieval_alignment_loss(
                p,
                align_batch,
                tau=retr_tau,
                dims=dims,
            )  # spec §21
        else:
            l_retr = jnp.asarray(0.0, dtype=jnp.float32)
        if adapter_active:
            l_adapter = _adapter_distill_loss(p, sent_batch, protected_bank, hp, active_mask)
        else:
            l_adapter = jnp.asarray(0.0, dtype=jnp.float32)
        loss = (
            l_cons
            + jnp.asarray(lambda_visual, dtype=jnp.float32)
            * (jnp.asarray(lambda_key, dtype=jnp.float32) * l_key + l_visual_beh)
            + jnp.asarray(lambda_retr, dtype=jnp.float32) * l_retr
            + jnp.asarray(lambda_distill, dtype=jnp.float32) * l_adapter
        )  # spec §21
        metrics = jnp.asarray(
            [loss, l_cons, l_key, l_visual_beh, l_retr, l_adapter],
            dtype=jnp.float32,
        )
        return loss, metrics

    (loss, metrics), grads = jax.value_and_grad(loss_fn, has_aux=True)(params)
    finite = jnp.logical_and(jnp.isfinite(loss), jnp.all(jnp.isfinite(metrics)))
    finite = jnp.logical_and(finite, _tree_all_finite(grads))
    safe_grads = jax.tree_util.tree_map(
        lambda grad: jnp.where(finite, grad, jnp.zeros_like(grad)),
        grads,
    )
    updates, new_opt_state = tx.update(safe_grads, opt_state, params)
    new_params = optax.apply_updates(params, updates)
    params = _select_tree(new_params, params, finite)
    opt_state = _select_tree(new_opt_state, opt_state, finite)
    metrics = jnp.where(jnp.isfinite(metrics), metrics, jnp.zeros_like(metrics))
    return params, opt_state, jnp.concatenate([metrics, finite[None].astype(jnp.float32)], axis=0)


def _active_mask_arg(active_mask, params):
    if active_mask is None:
        return None
    mask = np.asarray(active_mask, dtype=np.float32).reshape(-1)
    n_adapters = int(params.get("adapter_router", {}).get("kernel", np.zeros((1, 0))).shape[-1])
    if n_adapters > 0 and int(mask.shape[0]) != n_adapters:
        raise ValueError(f"active_mask must have shape ({n_adapters},), got {mask.shape}")
    if not bool(np.any(mask > 0.0)):
        return None
    return jnp.asarray((mask > 0.0).astype(np.float32))


def _eval_score(sentinel_eval_fn, params) -> float:
    if sentinel_eval_fn is None:
        return 0.0
    return float(sentinel_eval_fn(params))


def consolidate(
    params,
    ema_params,
    protected_bank,
    visual_store: VisualSentinelStore | None,
    *,
    cfg,
    n_steps=None,
    slow_lr=None,
    active_mask=None,
    sentinel_eval_fn=None,
) -> dict:
    """Run slow §21 consolidation and accept it only when sentinels pass.

    The optional §21 review term is intentionally left out here; closed-loop
    old-game review is already handled by the §18 driver path.
    """
    n_steps = _cfg_int(cfg, "consolidate_steps", 64) if n_steps is None else int(n_steps)
    lr = _cfg_float(cfg, "lr", 2.5e-4)
    lr_frac = _cfg_float(cfg, "consolidate_lr_frac", 0.1)
    slow_lr = float(lr * lr_frac) if slow_lr is None else float(slow_lr)
    pre_score = _eval_score(sentinel_eval_fn, params)
    if n_steps <= 0 or _valid_rows(protected_bank).size == 0:
        return {
            "params": params,
            "ema_params": ema_params,
            "accepted": False,
            "pre_score": float(pre_score),
            "post_score": float(pre_score),
            "slow_lr": float(slow_lr),
            "steps": int(n_steps),
            "loss_terms": {},
            "adapter_distill_active": False,
            "reason": "no_consolidation_data",
        }

    dims = _infer_dims(params)
    _, d_k, _, _, act_dim = dims
    batch_size = _cfg_int(cfg, "guard_sample_atoms", 256)
    visual_batch = _cfg_int(cfg, "visual_sentinel_batch", 64)
    n_neg = _cfg_int(cfg, "retr_n_neg", 16)
    capacity = int(np.asarray(protected_bank["keys"]).shape[0])
    top_k = min(max(1, _cfg_int(cfg, "top_k", 1)), capacity)
    hp = default_retrieval_hp(top_k)
    hp_values = _hp_values(hp)
    lambda_v = _cfg_float(cfg, "guard_lambda_v", 1.0)
    huber_delta = _cfg_float(cfg, "huber_delta", 1.0)
    lambda_visual = _cfg_float(cfg, "lambda_visual", 0.5)
    lambda_key = _cfg_float(cfg, "lambda_key", 1.0)
    lambda_retr = _cfg_float(cfg, "lambda_retr", 0.1)
    visual_lambda_v = _cfg_float(cfg, "visual_lambda_v", 1.0)
    retr_tau = _cfg_float(cfg, "retr_tau", 0.1)
    lambda_distill = _cfg_float(cfg, "lambda_distill", 0.0)
    eps_policy = _cfg_float(cfg, "eps_policy", 0.05)
    active_arg = _active_mask_arg(active_mask, params)
    has_cross_game = _bank_has_cross_game_negatives(protected_bank)

    tx = optax.adam(learning_rate=float(slow_lr))
    opt_state = tx.init(params)
    metrics_sum = np.zeros((7,), dtype=np.float64)
    metrics_count = 0
    any_visual = False
    any_retr = False
    adapter_distill_active = False
    candidate = params
    for step in range(int(n_steps)):
        seed = 1_000_003 + int(step)
        atom_batch = _sample_atom_batch(protected_bank, batch_size, seed, eps_policy)
        sent_batch, has_visual = _sample_sent_batch(
            visual_store,
            visual_batch,
            seed + 17,
            d_k,
            act_dim,
        )
        align_batch, has_retr = _sample_align_batch(
            sent_batch,
            protected_bank,
            batch_size=visual_batch,
            n_neg=n_neg,
            seed=seed + 29,
            d_k=d_k,
            enabled=bool(has_visual and has_cross_game),
        )
        step_adapter_active = bool(active_arg is not None and has_visual)
        candidate, opt_state, metrics = _consolidation_step(
            candidate,
            opt_state,
            {name: jnp.asarray(value) for name, value in atom_batch.items()},
            {name: jnp.asarray(value) for name, value in sent_batch.items()},
            {name: jnp.asarray(value) for name, value in align_batch.items()},
            {name: jnp.asarray(value) for name, value in protected_bank.items()},
            jnp.zeros((0,), dtype=jnp.float32) if active_arg is None else active_arg,
            jnp.asarray(slow_lr, dtype=jnp.float32),
            *hp_values,
            jnp.asarray(lambda_v, dtype=jnp.float32),
            jnp.asarray(huber_delta, dtype=jnp.float32),
            jnp.asarray(lambda_visual, dtype=jnp.float32),
            jnp.asarray(lambda_key, dtype=jnp.float32),
            jnp.asarray(lambda_retr, dtype=jnp.float32),
            jnp.asarray(visual_lambda_v, dtype=jnp.float32),
            jnp.asarray(retr_tau, dtype=jnp.float32),
            jnp.asarray(lambda_distill, dtype=jnp.float32),
            dims,
            bool(has_visual),
            bool(has_retr),
            bool(step_adapter_active),
        )
        metrics_np = np.asarray(jax.device_get(metrics), dtype=np.float64).reshape(-1)
        metrics_sum += metrics_np
        metrics_count += 1
        any_visual = bool(any_visual or has_visual)
        any_retr = bool(any_retr or has_retr)
        adapter_distill_active = bool(adapter_distill_active or step_adapter_active)

    post_score = _eval_score(sentinel_eval_fn, candidate)
    tol = _cfg_float(cfg, "consolidate_tol", 0.0)
    accepted = bool(
        np.isfinite(pre_score)
        and np.isfinite(post_score)
        and float(post_score) >= float(pre_score) - float(tol)
    )  # spec §21
    if accepted:
        out_params = candidate
        out_ema = candidate if ema_params is None else ema_update(ema_params, candidate, _cfg_float(cfg, "tau_key", 0.005))
    else:
        out_params = params
        out_ema = ema_params

    mean_metrics = metrics_sum / max(metrics_count, 1)
    loss_terms = {
        "loss": float(mean_metrics[0]),
        "L_cons": float(mean_metrics[1]),
        "L_key": float(mean_metrics[2]) if any_visual else 0.0,
        "L_visual_beh": float(mean_metrics[3]) if any_visual else 0.0,
        "L_retr": float(mean_metrics[4]) if any_retr else 0.0,
        "L_adapter_distill": float(mean_metrics[5]) if adapter_distill_active else 0.0,
        "finite_step_frac": float(mean_metrics[6]),
    }
    return {
        "params": out_params,
        "ema_params": out_ema,
        "accepted": bool(accepted),
        "pre_score": float(pre_score),
        "post_score": float(post_score),
        "slow_lr": float(slow_lr),
        "steps": int(n_steps),
        "loss_terms": loss_terms,
        "adapter_distill_active": bool(adapter_distill_active),
        "reason": "accepted" if accepted else "sentinel_regression",
    }


__all__ = ["consolidate"]
