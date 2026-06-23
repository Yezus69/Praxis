"""Integrated CASTM recurrent agent (spec sections 6.5, 8.1, 9, 30).

One live shared network whose effective weights are retrieved from compact,
content-addressed synaptic memory. The contextualized layers (spec 6.5) are the
three convolutions, the encoder dense layer, the three GRU gates, and the policy
and value heads — each an addressed :class:`SynapticMemory` with a fast LoRA
scratchpad. A dedicated recurrent context encoder produces a normalized content
query ``q_t`` used by the prototype router to retrieve the canonical address; the
context encoder has its own parameters and is not contextualized (spec 9.1).

There is no game/task/label/one-hot argument anywhere in the forward path. The
address is an internal memory coordinate (``ctx_id`` / ``k``) retrieved from
content, not an external identity.

This module assembles and exercises the mechanism (forward, gradient flow,
agent-level retention, deterministic restore). The Atari training ladder
(Stages B-F) consumes it but is run separately on GPU.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

from flax import struct
import jax
import jax.numpy as jnp

from tfns.castm import layers as L
from tfns.castm import scratch as scr
from tfns.castm import synaptic as syn


@dataclass(frozen=True)
class AgentConfig:
    obs_hw: int = 84
    frame_stack: int = 4
    conv_channels: tuple[int, int, int] = (32, 64, 64)
    conv_kernels: tuple[int, int, int] = (8, 4, 3)
    conv_strides: tuple[int, int, int] = (4, 2, 1)
    dense_dim: int = 512
    gru_hidden: int = 512
    act_dim: int = 18
    action_embed_dim: int = 32
    d_q: int = 128          # content query dimension (spec 11.3)
    d_k: int = 128          # canonical address dimension (spec 11.3)
    ctx_hidden: int = 256   # context-encoder recurrent width
    # comp_rank caps the per-context representable rank (spec 11.3 max active rank).
    comp_rank_conv: int = 64
    comp_rank_dense: int = 64
    comp_rank_head: int = 32
    n_slots: int = 8        # component pool slots per layer

    def conv_out_hw(self) -> int:
        h = self.obs_hw
        for k, s in zip(self.conv_kernels, self.conv_strides):
            h = (h - k) // s + 1
        return h

    def conv_flat_dim(self) -> int:
        h = self.conv_out_hw()
        return h * h * self.conv_channels[-1]

    def contextual_layers(self) -> dict[str, dict]:
        """Return per-layer (out, in) and conv metadata for the contextual banks."""

        c = self.conv_channels
        k = self.conv_kernels
        s = self.conv_strides
        cin = (self.frame_stack, c[0], c[1])
        layers: dict[str, dict] = {}
        for i in range(3):
            in_dim = k[i] * k[i] * cin[i]
            layers[f"conv{i+1}"] = {
                "out": c[i], "in": in_dim, "rank": self.comp_rank_conv,
                "kind": "conv", "kh": k[i], "kw": k[i], "c_in": cin[i], "stride": s[i],
            }
        layers["enc_dense"] = {"out": self.dense_dim, "in": self.conv_flat_dim(),
                               "rank": self.comp_rank_dense, "kind": "dense"}
        gate_in = self.dense_dim + self.action_embed_dim + 2 + self.gru_hidden
        for g in ("z", "r", "n"):
            layers[f"gru_{g}"] = {"out": self.gru_hidden, "in": gate_in,
                                  "rank": self.comp_rank_dense, "kind": "gru"}
        layers["policy"] = {"out": self.act_dim, "in": self.gru_hidden,
                            "rank": self.comp_rank_head, "kind": "dense"}
        layers["value"] = {"out": 1, "in": self.gru_hidden,
                           "rank": self.comp_rank_head, "kind": "dense"}
        return layers


@struct.dataclass
class AgentParams:
    """Non-contextualized shared parameters (trained, not addressed)."""

    action_embed: jnp.ndarray            # (act_dim, action_embed_dim)
    # Dedicated context encoder E_c: small Nature CNN + GRU + query projection.
    ce_conv: dict                        # conv kernels {conv1,conv2,conv3: (kh,kw,cin,cout), b}
    ce_dense_w: jnp.ndarray              # (ctx_hidden, conv_flat)
    ce_dense_b: jnp.ndarray
    ce_gru_w: jnp.ndarray                # (3*ctx_hidden, ce_in)
    ce_gru_u: jnp.ndarray                # (3*ctx_hidden, ctx_hidden)
    ce_gru_b: jnp.ndarray               # (3*ctx_hidden,)
    q_proj: jnp.ndarray                  # (d_q, ctx_hidden)
    # Self-supervised aux heads on context hidden (spec 9.1 training signals).
    aux_next_w: jnp.ndarray              # (ctx_hidden, ctx_hidden+action_embed)
    aux_reward_w: jnp.ndarray            # (3, ctx_hidden)
    aux_term_w: jnp.ndarray              # (1, ctx_hidden)


@struct.dataclass
class AgentCarry:
    main_hidden: jnp.ndarray   # (B, gru_hidden)
    ctx_hidden: jnp.ndarray    # (B, ctx_hidden)


def _orth(key, shape, scale):
    return jax.nn.initializers.orthogonal(scale)(key, shape, jnp.float32)


def init_banks(key, cfg: AgentConfig) -> dict[str, syn.SynapticMemory]:
    """Initialize the contextualized synaptic banks with orthogonal shared W0."""

    specs = cfg.contextual_layers()
    keys = jax.random.split(key, len(specs))
    banks: dict[str, syn.SynapticMemory] = {}
    for i, (name, spec) in enumerate(specs.items()):
        out_dim, in_dim = spec["out"], spec["in"]
        if name == "policy":
            scale = 0.01
        elif name == "value":
            scale = 1.0
        else:
            scale = jnp.sqrt(2.0)
        W0 = _orth(keys[i], (out_dim, in_dim), scale)
        b0 = jnp.zeros((out_dim,), jnp.float32)
        banks[name] = syn.empty_synaptic_memory(
            W0, b0, comp_rank=spec["rank"], n_slots=cfg.n_slots, d_k=cfg.d_k
        )
    return banks


def init_params(key, cfg: AgentConfig) -> AgentParams:
    keys = jax.random.split(key, 16)
    c = cfg.conv_channels
    k = cfg.conv_kernels
    cin = (cfg.frame_stack, c[0], c[1])
    ce_conv = {}
    for i in range(3):
        ce_conv[f"conv{i+1}"] = {
            "w": _orth(keys[i], (k[i] * k[i] * cin[i], c[i]), jnp.sqrt(2.0)).reshape(
                k[i], k[i], cin[i], c[i]
            ),
            "b": jnp.zeros((c[i],), jnp.float32),
        }
    conv_flat = cfg.conv_flat_dim()
    ce_in = cfg.ctx_hidden + cfg.action_embed_dim + 2  # [e_c; a_embed; reward; done]
    return AgentParams(
        action_embed=0.01 * jax.random.normal(keys[3], (cfg.act_dim, cfg.action_embed_dim)),
        ce_conv=ce_conv,
        ce_dense_w=_orth(keys[4], (cfg.ctx_hidden, conv_flat), jnp.sqrt(2.0)),
        ce_dense_b=jnp.zeros((cfg.ctx_hidden,), jnp.float32),
        ce_gru_w=_orth(keys[5], (3 * cfg.ctx_hidden, ce_in), 1.0),
        ce_gru_u=_orth(keys[6], (3 * cfg.ctx_hidden, cfg.ctx_hidden), 1.0),
        ce_gru_b=jnp.zeros((3 * cfg.ctx_hidden,), jnp.float32),
        q_proj=_orth(keys[7], (cfg.d_q, cfg.ctx_hidden), 1.0),
        aux_next_w=_orth(keys[8], (cfg.ctx_hidden, cfg.ctx_hidden + cfg.action_embed_dim), 1.0),
        aux_reward_w=_orth(keys[9], (3, cfg.ctx_hidden), 1.0),
        aux_term_w=_orth(keys[10], (1, cfg.ctx_hidden), 1.0),
    )


def init_scratch_banks(key, cfg: AgentConfig, *, scales: dict | None = None) -> dict[str, scr.ScratchDelta]:
    """Initialize a zero-delta LoRA scratch bank for every contextualized layer (spec 7)."""

    specs = cfg.contextual_layers()
    keys = jax.random.split(key, len(specs))
    scratch_rank = {"conv": 8, "dense": 16, "gru": 16, "head": 8}
    out: dict[str, scr.ScratchDelta] = {}
    for i, (name, spec) in enumerate(specs.items()):
        if spec["kind"] == "conv":
            r = scratch_rank["conv"]
        elif name in ("policy", "value"):
            r = scratch_rank["head"]
        else:
            r = scratch_rank["dense"]
        # Scratch rank can never exceed the component pool rank (spec 7/11.3).
        r = min(int(r), int(spec["rank"]), int(spec["out"]), int(spec["in"]))
        out[name] = scr.init_scratch(spec["in"], spec["out"], r, keys[i])
    return out


def init_carry(cfg: AgentConfig, batch: int) -> AgentCarry:
    return AgentCarry(
        main_hidden=jnp.zeros((int(batch), cfg.gru_hidden), jnp.float32),
        ctx_hidden=jnp.zeros((int(batch), cfg.ctx_hidden), jnp.float32),
    )


def _norm_obs(obs):
    obs = jnp.asarray(obs)
    if jnp.issubdtype(obs.dtype, jnp.integer):
        return obs.astype(jnp.float32) / 255.0
    return obs.astype(jnp.float32)


def _conv(x, kernel, stride):
    dn = jax.lax.conv_dimension_numbers(x.shape, kernel.shape, ("NHWC", "HWIO", "NHWC"))
    return jax.lax.conv_general_dilated(x, kernel, (stride, stride), "VALID", dimension_numbers=dn)


def context_query(params: AgentParams, cfg: AgentConfig, ctx_hidden, obs,
                  prev_action, prev_reward, reset) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Dedicated context encoder: returns ``(q_t, ctx_hidden')`` (spec 9.1).

    No game/task identity; consumes only obs, previous action, reward, and reset.
    """

    obs = _norm_obs(obs)
    reset = jnp.asarray(reset, dtype=bool)
    x = obs
    for i in range(3):
        cw = params.ce_conv[f"conv{i+1}"]
        x = jax.nn.relu(_conv(x, cw["w"], cfg.conv_strides[i]) + cw["b"])
    e_c = jax.nn.relu(x.reshape((x.shape[0], -1)) @ params.ce_dense_w.T + params.ce_dense_b)

    a_embed = params.action_embed[jnp.asarray(prev_action, jnp.int32)]
    r_col = jnp.asarray(prev_reward, jnp.float32)[..., None]
    d_col = reset.astype(jnp.float32)[..., None]
    h_in = jnp.where(reset[..., None], jnp.zeros_like(ctx_hidden), ctx_hidden)
    xi = jnp.concatenate([e_c, a_embed, r_col, d_col], axis=-1)

    H = cfg.ctx_hidden
    wz, wr, wn = jnp.split(params.ce_gru_w, 3, axis=0)
    uz, ur, un = jnp.split(params.ce_gru_u, 3, axis=0)
    bz, br, bn = jnp.split(params.ce_gru_b, 3, axis=0)
    z = jax.nn.sigmoid(xi @ wz.T + h_in @ uz.T + bz)
    rr = jax.nn.sigmoid(xi @ wr.T + h_in @ ur.T + br)
    n = jnp.tanh(xi @ wn.T + rr * (h_in @ un.T) + bn)
    h_c = (1.0 - z) * n + z * h_in

    q_raw = h_c @ params.q_proj.T
    q = q_raw / (jnp.linalg.norm(q_raw, axis=-1, keepdims=True) + 1e-6)
    return q, h_c


def context_aux(params: AgentParams, ctx_hidden, prev_action) -> dict[str, jnp.ndarray]:
    """Self-supervised aux predictions from context hidden (reward-sign, terminal, next-feat)."""

    a_embed = params.action_embed[jnp.asarray(prev_action, jnp.int32)]
    nf = jnp.concatenate([ctx_hidden, a_embed], axis=-1) @ params.aux_next_w.T
    rw = ctx_hidden @ params.aux_reward_w.T
    tm = jnp.squeeze(ctx_hidden @ params.aux_term_w.T, axis=-1)
    return {"next_feat": nf, "reward_logits": rw, "terminal_logit": tm}


def encode(banks, scratch, cfg: AgentConfig, obs, k, ctx_id=None, sparse=False) -> jnp.ndarray:
    """Addressed visual encoder: conv1->conv2->conv3->dense (spec 6.2/6.5)."""

    x = _norm_obs(obs)
    specs = cfg.contextual_layers()
    for i in range(3):
        name = f"conv{i+1}"
        sp = specs[name]
        s = scratch.get(name) if scratch else None
        x = L.addressed_conv_forward(
            banks[name], x, k, kh=sp["kh"], kw=sp["kw"], c_in=sp["c_in"],
            strides=(sp["stride"], sp["stride"]), padding="VALID", scratch=s,
            ctx_id=ctx_id, sparse=sparse,
        )
        x = jax.nn.relu(x)
    flat = x.reshape((x.shape[0], -1))
    sd = scratch.get("enc_dense") if scratch else None
    if sparse and ctx_id is not None:
        e = L.addressed_dense_forward_sparse(banks["enc_dense"], flat, k, int(ctx_id), sd)
    else:
        e = L.addressed_dense_forward(banks["enc_dense"], flat, k, sd)
    return jax.nn.relu(e)


def policy_step(params: AgentParams, banks, scratch, cfg: AgentConfig, main_hidden,
                obs, prev_action, prev_reward, reset, k, ctx_id=None, sparse=False):
    """Addressed policy/value forward for one step at address ``k`` (spec 6.5).

    Returns ``(logits, value, main_hidden')``.
    """

    e = encode(banks, scratch, cfg, obs, k, ctx_id=ctx_id, sparse=sparse)
    a_embed = params.action_embed[jnp.asarray(prev_action, jnp.int32)]
    r_col = jnp.asarray(prev_reward, jnp.float32)[..., None]
    reset = jnp.asarray(reset, dtype=bool)
    d_col = reset.astype(jnp.float32)[..., None]
    x_t = jnp.concatenate([e, a_embed, r_col, d_col], axis=-1)

    h_in = jnp.where(reset[..., None], jnp.zeros_like(main_hidden), main_hidden)
    gru = L.GRUMemory(z=banks["gru_z"], r=banks["gru_r"], n=banks["gru_n"])
    gscr = None
    if scratch:
        gscr = L.GRUScratch(z=scratch["gru_z"], r=scratch["gru_r"], n=scratch["gru_n"])
    if sparse and ctx_id is not None:
        h_t = L.addressed_gru_step_sparse(gru, x_t, h_in, k, int(ctx_id), jnp.zeros_like(reset), gscr)
    else:
        h_t = L.addressed_gru_step(gru, x_t, h_in, k, jnp.zeros_like(reset), gscr)

    sp = scratch.get("policy") if scratch else None
    sv = scratch.get("value") if scratch else None
    if sparse and ctx_id is not None:
        logits = L.addressed_dense_forward_sparse(banks["policy"], h_t, k, int(ctx_id), sp)
        value = L.addressed_dense_forward_sparse(banks["value"], h_t, k, int(ctx_id), sv)
    else:
        logits = L.addressed_dense_forward(banks["policy"], h_t, k, sp)
        value = L.addressed_dense_forward(banks["value"], h_t, k, sv)
    return logits, jnp.squeeze(value, axis=-1), h_t


__all__ = [
    "AgentCarry",
    "AgentConfig",
    "AgentParams",
    "context_aux",
    "context_query",
    "encode",
    "init_banks",
    "init_carry",
    "init_params",
    "init_scratch_banks",
    "policy_step",
]
