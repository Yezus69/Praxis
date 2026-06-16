"""Memory loss functions from PMA-C sections 11, 12, and 13."""

from __future__ import annotations

from collections.abc import Mapping

import jax
import jax.numpy as jnp
from jax import lax

from pmac.agents.atari_mem_net import MemAtariActorCritic, mem_apply
from pmac.behavior_distance import huber
from pmac.memory import reader


EPS = 1e-8


def _field(batch, name: str):
    if isinstance(batch, Mapping):
        return batch[name]
    return getattr(batch, name)


def _dims_value(dims, name: str, index: int):
    if isinstance(dims, Mapping):
        return dims[name]
    if hasattr(dims, name):
        return getattr(dims, name)
    return dims[index]


def _net_from_dims(dims) -> MemAtariActorCritic:
    return MemAtariActorCritic(
        n_games=int(_dims_value(dims, "n_games", 0)),
        d_k=int(_dims_value(dims, "d_k", 1)),
        d_c=int(_dims_value(dims, "d_c", 2)),
        d_m=int(_dims_value(dims, "d_m", 3)),
        act_dim=int(_dims_value(dims, "act_dim", 4)),
    )


def _stop_float(x):
    return lax.stop_gradient(jnp.asarray(x, dtype=jnp.float32))


def _stop_int(x):
    return lax.stop_gradient(jnp.asarray(x, dtype=jnp.int32))


def _stop_bank(bank):
    return {name: lax.stop_gradient(jnp.asarray(value)) for name, value in bank.items()}


def _batch_game_id(game_id, batch_size: int):
    game_id = jnp.asarray(game_id, dtype=jnp.int32)
    if game_id.ndim == 0:
        return jnp.broadcast_to(game_id, (batch_size,))
    return game_id


def _latent_behavior_from_bank(net, params, k_i, game_id_i, bank, hp):
    k_i = jnp.asarray(k_i, dtype=jnp.float32)
    game_id_i = _batch_game_id(game_id_i, int(k_i.shape[0]))
    c_embed = net.apply({"params": params}, game_id_i, method=MemAtariActorCritic.context)
    out = reader.retrieve(k_i, c_embed, game_id_i, bank, hp)
    m = net.apply({"params": params}, out.atom_feats, out.alpha, method=MemAtariActorCritic.mem_summary)
    return net.apply(
        {"params": params},
        k_i,
        c_embed,
        m,
        method=MemAtariActorCritic.latent_behavior,
    )  # spec §11


def _kl_teacher_to_current(p_star, logits_theta):
    p_theta = jax.nn.softmax(logits_theta, axis=-1)
    return jnp.sum(
        p_star * (jnp.log(p_star + EPS) - jnp.log(p_theta + EPS)), axis=-1
    )  # spec §11, §12


def latent_conservation_loss(
    params,
    atom_batch,
    bank,
    hp,
    *,
    lambda_v=1.0,
    huber_delta=1.0,
    dims,
) -> jnp.ndarray:
    """Behavior-tube conservation over stored latent memory atoms."""
    net = _net_from_dims(dims)
    keys = _stop_float(_field(atom_batch, "keys"))
    game_id = _stop_int(_field(atom_batch, "game_id"))
    p_star = _stop_float(_field(atom_batch, "teacher_policy"))
    v_star = _stop_float(_field(atom_batch, "teacher_value"))
    eps = _stop_float(_field(atom_batch, "eps"))
    weight = _stop_float(_field(atom_batch, "weight"))
    bank = _stop_bank(bank)

    logits_theta, v_theta = _latent_behavior_from_bank(net, params, keys, game_id, bank, hp)
    d_pi = _kl_teacher_to_current(p_star, logits_theta)  # spec §11
    d_v = huber(v_theta - v_star, huber_delta)  # spec §11
    d = d_pi + jnp.asarray(lambda_v, dtype=jnp.float32) * d_v  # spec §11
    return jnp.mean(weight * jnp.square(jax.nn.relu(d - eps)))  # spec §11


def visual_sentinel_loss(
    params,
    sent_batch,
    bank,
    hp,
    *,
    lambda_v=1.0,
    huber_delta=1.0,
    dims,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Key-space and behavior grounding on visual sentinels."""
    net = _net_from_dims(dims)
    obs = lax.stop_gradient(jnp.asarray(_field(sent_batch, "obs")))
    game_id = _stop_int(_field(sent_batch, "game_id"))
    key_star = _stop_float(_field(sent_batch, "key_star"))
    p_star = _stop_float(_field(sent_batch, "teacher_policy"))
    v_star = _stop_float(_field(sent_batch, "teacher_value"))
    bank = _stop_bank(bank)

    _, k_theta = net.apply({"params": params}, obs, method=MemAtariActorCritic.encode)
    l_key = jnp.mean(1.0 - jnp.sum(k_theta * key_star, axis=-1))  # spec §12

    out = mem_apply(params, obs, game_id, bank, hp)
    d_pi = _kl_teacher_to_current(p_star, out["logits_net"])  # spec §12
    d_v = huber(out["v_net"] - v_star, huber_delta)  # spec §12
    l_visual_beh = jnp.mean(d_pi + jnp.asarray(lambda_v, dtype=jnp.float32) * d_v)  # spec §12
    return l_key, l_visual_beh


def retrieval_alignment_loss(
    params,
    align_batch,
    *,
    tau=0.1,
    dims,
) -> jnp.ndarray:
    """InfoNCE alignment between encoded queries and stored memory keys."""
    net = _net_from_dims(dims)
    obs = lax.stop_gradient(jnp.asarray(_field(align_batch, "obs")))
    pos_key = _stop_float(_field(align_batch, "pos_key"))
    neg_keys = _stop_float(_field(align_batch, "neg_keys"))
    tau = jnp.asarray(tau, dtype=jnp.float32)

    _, q = net.apply({"params": params}, obs, method=MemAtariActorCritic.encode)
    pos = jnp.sum(q * pos_key, axis=-1) / tau  # spec §13
    neg = jnp.sum(q[:, None, :] * neg_keys, axis=-1) / tau  # spec §13
    logits = jnp.concatenate([pos[:, None], neg], axis=-1)
    return jnp.mean(-pos + jax.nn.logsumexp(logits, axis=-1))  # spec §13


__all__ = [
    "latent_conservation_loss",
    "retrieval_alignment_loss",
    "visual_sentinel_loss",
]
