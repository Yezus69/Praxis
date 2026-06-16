"""Memory-conditioned Flax Nature-CNN actor-critic for Atari PMA-C."""

from __future__ import annotations

import math

import flax.linen as nn
import jax
import jax.numpy as jnp

from pmac.envs.atari_envpool import ACT_DIM
from pmac.memory import reader


EPS = 1e-8


def _orthogonal(scale: float):
    return nn.initializers.orthogonal(float(scale))


def _prepare_obs(obs):
    obs = jnp.asarray(obs)
    was_integer = jnp.issubdtype(obs.dtype, jnp.integer)
    if obs.ndim < 4:
        raise ValueError(f"expected batched Atari obs, got shape {obs.shape}")
    if obs.shape[-1] != 4:
        if obs.shape[-3] != 4:
            raise ValueError(f"expected NCHW or NHWC Atari obs, got shape {obs.shape}")
        obs = jnp.moveaxis(obs, -3, -1)
    obs = obs.astype(jnp.float32)
    if was_integer:
        obs = obs / 255.0
    return obs


class MemAtariActorCritic(nn.Module):
    """Nature-CNN actor-critic with PMA-C retrieval conditioning."""

    n_games: int
    d_k: int = 128
    d_c: int = 16
    d_m: int = 128
    act_dim: int = ACT_DIM

    def setup(self):
        conv_init = _orthogonal(math.sqrt(2.0))
        dense_init = _orthogonal(math.sqrt(2.0))
        self.conv1 = nn.Conv(
            features=32,
            kernel_size=(8, 8),
            strides=(4, 4),
            kernel_init=conv_init,
            bias_init=nn.initializers.zeros,
        )
        self.conv2 = nn.Conv(
            features=64,
            kernel_size=(4, 4),
            strides=(2, 2),
            kernel_init=conv_init,
            bias_init=nn.initializers.zeros,
        )
        self.conv3 = nn.Conv(
            features=64,
            kernel_size=(3, 3),
            strides=(1, 1),
            kernel_init=conv_init,
            bias_init=nn.initializers.zeros,
        )
        self.trunk_dense = nn.Dense(
            features=512,
            kernel_init=dense_init,
            bias_init=nn.initializers.zeros,
        )
        self.game_embed = nn.Embed(int(self.n_games), int(self.d_c))
        self.key_head = nn.Dense(int(self.d_k))
        self.wv = nn.Dense(int(self.d_m))
        self.key_to_h = nn.Dense(512)
        self.policy_head = nn.Dense(
            int(self.act_dim),
            kernel_init=_orthogonal(0.01),
            bias_init=nn.initializers.zeros,
        )
        self.value_head = nn.Dense(
            1,
            kernel_init=_orthogonal(1.0),
            bias_init=nn.initializers.zeros,
        )

    def trunk(self, obs):
        x = _prepare_obs(obs)
        x = self.conv1(x)
        x = nn.relu(x)
        x = self.conv2(x)
        x = nn.relu(x)
        x = self.conv3(x)
        x = nn.relu(x)
        x = x.reshape((x.shape[0], -1))
        x = self.trunk_dense(x)
        return nn.relu(x)  # spec §4

    def encode(self, obs):
        h = self.trunk(obs)
        z = self.key_head(h)
        k = z / (jnp.linalg.norm(z, axis=-1, keepdims=True) + EPS)  # spec §5
        return h, k

    def context(self, game_id):
        game_id = jnp.asarray(game_id, dtype=jnp.int32)
        return self.game_embed(game_id)

    def mem_summary(self, atom_feats, alpha):
        projected = self.wv(jnp.asarray(atom_feats, dtype=jnp.float32))
        return jnp.sum(jnp.asarray(alpha, dtype=jnp.float32)[..., None] * projected, axis=1)  # spec §9

    def policy_value(self, h, m, c_embed):
        z = jnp.concatenate([h, m, c_embed], axis=-1)
        logits = self.policy_head(z)
        value = self.value_head(z)
        return logits, jnp.squeeze(value, axis=-1)  # spec §4

    def latent_behavior(self, k_i, c_embed_i, m_i):
        h_hat = self.key_to_h(jnp.concatenate([k_i, c_embed_i], axis=-1))
        return self.policy_value(h_hat, m_i, c_embed_i)  # spec §11

    def __call__(self, obs, game_id, bank, hp, mu_g=0.0, sigma_g=1.0):
        h, k = self.encode(obs)  # spec §5
        game_id = jnp.asarray(game_id, dtype=jnp.int32)
        if game_id.ndim == 0:
            game_id = jnp.broadcast_to(game_id, (h.shape[0],))
        elif game_id.shape[0] == 1 and h.shape[0] != 1:
            game_id = jnp.broadcast_to(game_id, (h.shape[0],))
        c_embed = self.context(game_id)
        out = reader.retrieve(k, c_embed, game_id, bank, hp)  # spec §9
        m = self.mem_summary(out.atom_feats, out.alpha)  # spec §9
        logits_net, v_net = self.policy_value(h, m, c_embed)  # spec §4
        p_net = jax.nn.softmax(logits_net, axis=-1)  # spec §4
        _, logits_final, v_final = reader.blend(
            p_net, v_net, out.p_mem, out.v_mem, out.b, mu_g, sigma_g
        )  # spec §9
        return {
            "logits_final": logits_final,
            "v_final": v_final,
            "logits_net": logits_net,
            "v_net": v_net,
            "p_mem": out.p_mem,
            "v_mem": out.v_mem,
            "b": out.b,
            "rho": out.rho,
            "m": m,
            "k": k,
            "alpha": out.alpha,
        }

    def init_all(self, obs, game_id, bank, hp, k_i, m_i, mu_g=0.0, sigma_g=1.0):
        outputs = self(obs, game_id, bank, hp, mu_g, sigma_g)
        c_embed_i = self.context(game_id)
        self.latent_behavior(k_i, c_embed_i, m_i)
        return outputs


def _dummy_bank(capacity: int, d_k: int, d_c: int, act_dim: int):
    capacity = int(capacity)
    return {
        "keys": jnp.zeros((capacity, d_k), dtype=jnp.float32),
        "context": jnp.zeros((capacity, d_c), dtype=jnp.float32),
        "teacher_policy": jnp.zeros((capacity, act_dim), dtype=jnp.float32),
        "teacher_value": jnp.zeros((capacity,), dtype=jnp.float32),
        "importance": jnp.zeros((capacity,), dtype=jnp.float32),
        "game_id": jnp.zeros((capacity,), dtype=jnp.int32),
        "source5": jnp.zeros((capacity, 5), dtype=jnp.float32),
        "age": jnp.zeros((capacity,), dtype=jnp.float32),
        "valid": jnp.zeros((capacity,), dtype=bool),
    }


def _default_hp(top_k: int):
    return {
        "tau_r": 1.0,
        "beta_c": 0.0,
        "beta_I": 0.0,
        "beta_a": 0.0,
        "top_k": int(top_k),
        "w_rho": 1.0,
        "w_c": 1.0,
        "b0": 0.0,
    }


def _infer_dims(params):
    n_games, d_c = params["game_embed"]["embedding"].shape
    d_k = params["key_head"]["kernel"].shape[-1]
    d_m = params["wv"]["kernel"].shape[-1]
    act_dim = params["policy_head"]["kernel"].shape[-1]
    return int(n_games), int(d_k), int(d_c), int(d_m), int(act_dim)


def _flatten_obs(obs):
    obs = jnp.asarray(obs)
    single = obs.ndim == 3
    if single:
        obs = obs[jnp.newaxis, ...]
    lead_shape = obs.shape[:-3]
    obs_flat = obs.reshape((-1,) + obs.shape[-3:])
    return obs_flat, lead_shape, single


def _flatten_vector(value, batch: int, lead_shape, dtype):
    value = jnp.asarray(value, dtype=dtype)
    if value.ndim == 0:
        return jnp.broadcast_to(value, (batch,))
    if value.shape == lead_shape:
        return value.reshape((batch,))
    if value.shape[0] == batch:
        return value.reshape((batch,))
    return jnp.broadcast_to(value, (batch,))


def _reshape_outputs(outputs, lead_shape, single: bool):
    def reshape(value):
        value = jnp.asarray(value)
        if value.ndim == 0:
            return value
        if single:
            return value[0]
        return value.reshape(lead_shape + value.shape[1:])

    return {name: reshape(value) for name, value in outputs.items()}


def mem_init(
    key,
    n_games: int,
    capacity: int,
    *,
    d_k: int = 128,
    d_c: int = 16,
    d_m: int = 128,
    act_dim: int = ACT_DIM,
    top_k: int | None = None,
):
    if capacity <= 0:
        raise ValueError("capacity must be positive for static top_k retrieval")
    top_k = min(int(capacity), 1 if top_k is None else int(top_k))
    net = MemAtariActorCritic(
        n_games=int(n_games), d_k=int(d_k), d_c=int(d_c), d_m=int(d_m), act_dim=int(act_dim)
    )
    dummy_obs = jnp.zeros((1, 84, 84, 4), dtype=jnp.float32)
    dummy_game = jnp.zeros((1,), dtype=jnp.int32)
    dummy_bank = _dummy_bank(capacity, d_k, d_c, act_dim)
    dummy_hp = _default_hp(top_k)
    dummy_k = jnp.zeros((1, d_k), dtype=jnp.float32)
    dummy_m = jnp.zeros((1, d_m), dtype=jnp.float32)
    variables = net.init(
        key,
        dummy_obs,
        dummy_game,
        dummy_bank,
        dummy_hp,
        dummy_k,
        dummy_m,
        method=MemAtariActorCritic.init_all,
    )
    return variables["params"]


def mem_apply(params, obs, game_id, bank, hp, mu_g=0.0, sigma_g=1.0):
    n_games, d_k, d_c, d_m, act_dim = _infer_dims(params)
    net = MemAtariActorCritic(n_games=n_games, d_k=d_k, d_c=d_c, d_m=d_m, act_dim=act_dim)
    obs_flat, lead_shape, single = _flatten_obs(obs)
    batch = int(obs_flat.shape[0])
    game_flat = _flatten_vector(game_id, batch, lead_shape, jnp.int32)
    mu_flat = _flatten_vector(mu_g, batch, lead_shape, jnp.float32)
    sigma_flat = _flatten_vector(sigma_g, batch, lead_shape, jnp.float32)
    outputs = net.apply({"params": params}, obs_flat, game_flat, bank, hp, mu_flat, sigma_flat)
    return _reshape_outputs(outputs, lead_shape, single)


def mem_apply_key(params, obs):
    n_games, d_k, d_c, d_m, act_dim = _infer_dims(params)
    net = MemAtariActorCritic(n_games=n_games, d_k=d_k, d_c=d_c, d_m=d_m, act_dim=act_dim)
    obs_flat, lead_shape, single = _flatten_obs(obs)
    _, k = net.apply({"params": params}, obs_flat, method=MemAtariActorCritic.encode)
    if single:
        return k[0]
    return k.reshape(lead_shape + (k.shape[-1],))


def mem_apply_latent(params, k_i, game_id_i, m_i):
    n_games, d_k, d_c, d_m, act_dim = _infer_dims(params)
    net = MemAtariActorCritic(n_games=n_games, d_k=d_k, d_c=d_c, d_m=d_m, act_dim=act_dim)
    k_i = jnp.asarray(k_i, dtype=jnp.float32)
    single = k_i.ndim == 1
    if single:
        k_i = k_i[jnp.newaxis, :]
    lead_shape = k_i.shape[:-1]
    batch = int(k_i.reshape((-1, k_i.shape[-1])).shape[0])
    k_flat = k_i.reshape((batch, k_i.shape[-1]))
    m_i = jnp.asarray(m_i, dtype=jnp.float32)
    if m_i.ndim == 1:
        m_i = jnp.broadcast_to(m_i, lead_shape + (m_i.shape[-1],))
    m_flat = m_i.reshape((batch, m_i.shape[-1]))
    game_flat = _flatten_vector(game_id_i, batch, lead_shape, jnp.int32)
    c_embed = net.apply({"params": params}, game_flat, method=MemAtariActorCritic.context)
    logits, value = net.apply(
        {"params": params}, k_flat, c_embed, m_flat, method=MemAtariActorCritic.latent_behavior
    )
    logits = logits.reshape(lead_shape + (logits.shape[-1],))
    value = value.reshape(lead_shape)
    if single:
        return logits[0], value[0]
    return logits, value


__all__ = [
    "MemAtariActorCritic",
    "mem_apply",
    "mem_apply_key",
    "mem_apply_latent",
    "mem_init",
]
