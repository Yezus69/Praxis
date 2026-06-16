"""Pure JAX memory retrieval and explicit policy/value blending for PMA-C."""

from __future__ import annotations

from typing import NamedTuple

import jax
import jax.numpy as jnp
from jax import lax


EPS = 1e-8


class RetrievalOut(NamedTuple):
    alpha: jnp.ndarray
    atom_feats: jnp.ndarray
    p_mem: jnp.ndarray
    v_mem: jnp.ndarray
    rho: jnp.ndarray
    context_match: jnp.ndarray
    b: jnp.ndarray
    valid_top: jnp.ndarray


def _hp(hp, name: str):
    if isinstance(hp, dict):
        return hp[name]
    return getattr(hp, name)


def expand_source_flags(flags) -> jnp.ndarray:
    flags = jnp.asarray(flags, dtype=jnp.int32)
    bits = jnp.asarray([1, 2, 4, 8, 16], dtype=jnp.int32)
    return ((flags[..., None] & bits) != 0).astype(jnp.float32)


def ema_update(target, online, tau):
    tau = jnp.asarray(tau, dtype=jnp.float32)
    return jax.tree_util.tree_map(
        lambda target_leaf, online_leaf: (1.0 - tau) * target_leaf + tau * online_leaf,
        target,
        online,
    )  # spec §5


def _l2_normalize(x, axis=-1):
    x = jnp.asarray(x, dtype=jnp.float32)
    return x / (jnp.linalg.norm(x, axis=axis, keepdims=True) + EPS)


def _take_rows(x, idx):
    return jnp.take(jnp.asarray(x), idx, axis=0)


def _bank_source5(bank) -> jnp.ndarray:
    if "source5" in bank:
        return jnp.asarray(bank["source5"], dtype=jnp.float32)
    return expand_source_flags(bank["source_flags"])


def retrieve(k, c_g, cur_game, bank, hp) -> RetrievalOut:
    k = jnp.asarray(k, dtype=jnp.float32)
    c_g = jnp.asarray(c_g, dtype=jnp.float32)
    cur_game = jnp.asarray(cur_game, dtype=jnp.int32)
    keys = jnp.asarray(bank["keys"], dtype=jnp.float32)
    context = jnp.asarray(bank["context"], dtype=jnp.float32)
    teacher_policy = jnp.asarray(bank["teacher_policy"], dtype=jnp.float32)
    teacher_value = jnp.asarray(bank["teacher_value"], dtype=jnp.float32)
    importance = jnp.asarray(bank["importance"], dtype=jnp.float32)
    game_id = jnp.asarray(bank["game_id"], dtype=jnp.int32)
    source5 = _bank_source5(bank)
    age = jnp.asarray(bank["age"], dtype=jnp.float32)
    valid = jnp.asarray(bank["valid"], dtype=bool)
    valid_f = valid.astype(jnp.float32)

    sim_key = k @ keys.T  # spec §9
    ctx_sim = _l2_normalize(c_g) @ _l2_normalize(context).T  # spec §9
    age_pen = age / (jnp.max(age * valid_f) + EPS)  # spec §9
    importance_for_log = jnp.where(valid, importance, 0.0)
    s = (
        sim_key / (jnp.asarray(_hp(hp, "tau_r"), dtype=jnp.float32) + EPS)
        + jnp.asarray(_hp(hp, "beta_c"), dtype=jnp.float32) * ctx_sim
        + jnp.asarray(_hp(hp, "beta_I"), dtype=jnp.float32)
        * jnp.log(importance_for_log + EPS)
        - jnp.asarray(_hp(hp, "beta_a"), dtype=jnp.float32) * age_pen[None, :]
    )  # spec §9
    s = jnp.where(valid[None, :], s, -1e30)  # spec §9

    top_k = int(_hp(hp, "top_k"))
    s_top, idx = lax.top_k(s, top_k)  # spec §9
    policy_top = _take_rows(teacher_policy, idx)
    value_top = _take_rows(teacher_value, idx)
    keys_top = _take_rows(keys, idx)
    ctx_top = _take_rows(context, idx)
    src_top = _take_rows(source5, idx)
    cos_top = jnp.take_along_axis(sim_key, idx, axis=1)
    gid_top = _take_rows(game_id, idx)
    valid_top = _take_rows(valid, idx)
    valid_top_f = valid_top.astype(jnp.float32)
    policy_top = jnp.where(valid_top[..., None], policy_top, 0.0)
    value_top = jnp.where(valid_top, value_top, 0.0)
    keys_top = jnp.where(valid_top[..., None], keys_top, 0.0)
    ctx_top = jnp.where(valid_top[..., None], ctx_top, 0.0)
    src_top = jnp.where(valid_top[..., None], src_top, 0.0)

    row_max = jnp.max(jnp.where(valid_top, s_top, -1e30), axis=-1, keepdims=True)
    exp_top = jnp.exp(s_top - row_max) * valid_top_f
    alpha = exp_top / (jnp.sum(exp_top, axis=-1, keepdims=True) + EPS)  # spec §9
    alpha = alpha * valid_top_f

    p_mem = jnp.sum(alpha[..., None] * policy_top, axis=1)  # spec §9
    v_mem = jnp.sum(alpha * value_top, axis=1)  # spec §9
    rho = jnp.max(jnp.where(valid_top, cos_top, -1.0), axis=-1)
    rho = jnp.maximum(rho, 0.0)  # spec §9
    context_match = jnp.sum(
        alpha * (gid_top == cur_game[:, None]).astype(jnp.float32), axis=-1
    )  # spec §9
    any_valid = jnp.sum(valid_top_f, axis=-1) > 0.0
    b = jax.nn.sigmoid(
        jnp.asarray(_hp(hp, "w_rho"), dtype=jnp.float32) * rho
        + jnp.asarray(_hp(hp, "w_c"), dtype=jnp.float32) * context_match
        - jnp.asarray(_hp(hp, "b0"), dtype=jnp.float32)
    )  # spec §9
    b = jnp.where(any_valid, b, 0.0)

    atom_feats = jnp.concatenate(
        [keys_top, ctx_top, policy_top, value_top[..., None], src_top],
        axis=-1,
    )  # spec §9
    return RetrievalOut(alpha, atom_feats, p_mem, v_mem, rho, context_match, b, valid_top)


def blend(p_net, v_net, p_mem, v_mem, b, mu_g=0.0, sigma_g=1.0):
    p_net = jnp.asarray(p_net, dtype=jnp.float32)
    v_net = jnp.asarray(v_net, dtype=jnp.float32)
    p_mem = jnp.asarray(p_mem, dtype=jnp.float32)
    v_mem = jnp.asarray(v_mem, dtype=jnp.float32)
    b = jnp.asarray(b, dtype=jnp.float32)
    mu_g = jnp.asarray(mu_g, dtype=jnp.float32)
    sigma_g = jnp.asarray(sigma_g, dtype=jnp.float32)

    p_final = (1.0 - b)[:, None] * p_net + b[:, None] * p_mem  # spec §9
    logits_final = jnp.log(p_final + EPS)  # spec §9
    v_final = (1.0 - b) * v_net + b * (sigma_g * v_mem + mu_g)  # spec §9
    return p_final, logits_final, v_final


__all__ = ["EPS", "RetrievalOut", "blend", "ema_update", "expand_source_flags", "retrieve"]
