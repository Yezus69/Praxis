"""Feed-forward addressed CASTM agent for the Atari experiment ladder.

A Nature-CNN backbone whose conv1-3, encoder dense, and policy/value heads are
all addressed :class:`SynapticMemory` layers with a LoRA scratchpad. This is the
feed-forward instantiation used for the Atari pilot ladder (the recurrent GRU
variant lives in ``agent.py`` and is unit-tested; the addressed-memory mechanism
— the novel no-forgetting contribution — is identical in both).

Design (spec 8.1): the first game is learned into the shared weights ``W0``
(trained normally). ``W0`` is then frozen and later games are learned through the
LoRA scratchpad and committed to their canonical address dual, so a prior game's
decoded weights are unchanged (exact retention under sparse top-1 gather).
"""

from __future__ import annotations

from dataclasses import dataclass

import jax
import jax.numpy as jnp

from tfns.castm import layers as L
from tfns.castm import scratch as scr
from tfns.castm import synaptic as syn


FF_LAYERS = ("conv1", "conv2", "conv3", "enc_dense", "policy", "value")


@dataclass(frozen=True)
class FFConfig:
    obs_hw: int = 84
    frame_stack: int = 4
    conv_channels: tuple[int, int, int] = (32, 64, 64)
    conv_kernels: tuple[int, int, int] = (8, 4, 3)
    conv_strides: tuple[int, int, int] = (4, 2, 1)
    dense_dim: int = 512
    act_dim: int = 18
    d_k: int = 128
    comp_rank_conv: int = 64
    comp_rank_dense: int = 64
    comp_rank_head: int = 32
    n_slots: int = 8
    scratch_rank_conv: int = 8
    scratch_rank_dense: int = 16
    scratch_rank_head: int = 8

    def conv_out_hw(self) -> int:
        h = self.obs_hw
        for k, s in zip(self.conv_kernels, self.conv_strides):
            h = (h - k) // s + 1
        return h

    def conv_flat_dim(self) -> int:
        h = self.conv_out_hw()
        return h * h * self.conv_channels[-1]

    def layer_specs(self) -> dict[str, dict]:
        c, k, s = self.conv_channels, self.conv_kernels, self.conv_strides
        cin = (self.frame_stack, c[0], c[1])
        specs: dict[str, dict] = {}
        for i in range(3):
            specs[f"conv{i+1}"] = {
                "out": c[i], "in": k[i] * k[i] * cin[i], "rank": self.comp_rank_conv,
                "kind": "conv", "kh": k[i], "kw": k[i], "c_in": cin[i], "stride": s[i],
                "srank": self.scratch_rank_conv,
            }
        specs["enc_dense"] = {"out": self.dense_dim, "in": self.conv_flat_dim(),
                              "rank": self.comp_rank_dense, "kind": "dense",
                              "srank": self.scratch_rank_dense}
        specs["policy"] = {"out": self.act_dim, "in": self.dense_dim,
                           "rank": self.comp_rank_head, "kind": "dense",
                           "srank": self.scratch_rank_head}
        specs["value"] = {"out": 1, "in": self.dense_dim,
                          "rank": self.comp_rank_head, "kind": "dense",
                          "srank": self.scratch_rank_head}
        return specs


def _orth(key, shape, scale):
    return jax.nn.initializers.orthogonal(scale)(key, shape, jnp.float32)


def init_banks(key, cfg: FFConfig) -> dict[str, syn.SynapticMemory]:
    specs = cfg.layer_specs()
    keys = jax.random.split(key, len(specs))
    banks: dict[str, syn.SynapticMemory] = {}
    import math
    for i, name in enumerate(FF_LAYERS):
        sp = specs[name]
        scale = 0.01 if name == "policy" else (1.0 if name == "value" else math.sqrt(2.0))
        W0 = _orth(keys[i], (sp["out"], sp["in"]), scale)
        b0 = jnp.zeros((sp["out"],), jnp.float32)
        banks[name] = syn.empty_synaptic_memory(W0, b0, comp_rank=sp["rank"],
                                                n_slots=cfg.n_slots, d_k=cfg.d_k)
    return banks


def init_scratch(key, cfg: FFConfig) -> dict[str, scr.ScratchDelta]:
    specs = cfg.layer_specs()
    keys = jax.random.split(key, len(specs))
    out: dict[str, scr.ScratchDelta] = {}
    for i, name in enumerate(FF_LAYERS):
        sp = specs[name]
        r = min(int(sp["srank"]), int(sp["rank"]), int(sp["out"]), int(sp["in"]))
        out[name] = scr.init_scratch(sp["in"], sp["out"], r, keys[i])
    return out


def _norm_obs(obs):
    obs = jnp.asarray(obs)
    if jnp.issubdtype(obs.dtype, jnp.integer):
        return obs.astype(jnp.float32) / 255.0
    return obs.astype(jnp.float32)


def forward(banks, scratch, cfg: FFConfig, obs, k, ctx_id=None, sparse=False):
    """Return ``(logits, value)`` for the addressed FF agent at address ``k``."""

    x = _norm_obs(obs)
    specs = cfg.layer_specs()
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
    e = x.reshape((x.shape[0], -1))
    sd = scratch.get("enc_dense") if scratch else None
    if sparse and ctx_id is not None:
        e = L.addressed_dense_forward_sparse(banks["enc_dense"], e, k, int(ctx_id), sd)
    else:
        e = L.addressed_dense_forward(banks["enc_dense"], e, k, sd)
    e = jax.nn.relu(e)
    sp_ = scratch.get("policy") if scratch else None
    sv_ = scratch.get("value") if scratch else None
    if sparse and ctx_id is not None:
        logits = L.addressed_dense_forward_sparse(banks["policy"], e, k, int(ctx_id), sp_)
        value = L.addressed_dense_forward_sparse(banks["value"], e, k, int(ctx_id), sv_)
    else:
        logits = L.addressed_dense_forward(banks["policy"], e, k, sp_)
        value = L.addressed_dense_forward(banks["value"], e, k, sv_)
    return logits, jnp.squeeze(value, axis=-1)


# --- trainable/frozen split for shared-mode (game 1) optimization ---------------


def shared_trainable(banks) -> dict:
    """Extract the shared W0/b0 (the only trainable leaves in shared mode)."""

    return {name: {"W0": banks[name].W0, "b0": banks[name].b0} for name in banks}


def apply_shared_trainable(banks, trainable) -> dict:
    """Rebuild banks with updated W0/b0 from a shared-mode trainable pytree."""

    return {name: banks[name].replace(W0=trainable[name]["W0"], b0=trainable[name]["b0"])
            for name in banks}


__all__ = [
    "FFConfig",
    "FF_LAYERS",
    "apply_shared_trainable",
    "forward",
    "init_banks",
    "init_scratch",
    "shared_trainable",
]
