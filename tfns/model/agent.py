"""Task-free recurrent TFNS agent."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from flax import struct
from flax.core import FrozenDict, unfreeze
import flax.linen as nn
import jax
import jax.numpy as jnp

from tfns.config import AdapterConfig, ModelConfig
from tfns.model.adapters import ResidualAdapterBank
from tfns.model.encoder import Encoder
from tfns.model.gru import ExplicitGRU
from tfns.model.heads import (
    ContextKeyHead,
    NextFeatHead,
    PolicyHead,
    RewardCatHead,
    TerminalHead,
    ValueHead,
)


@struct.dataclass
class AuxOutput:
    next_feat: Any
    reward_cat_logits: Any
    terminal_logit: Any


@struct.dataclass
class RouterOutput:
    visual: Any
    post: Any


@struct.dataclass
class AgentOutput:
    logits: Any
    value: Any
    q_key: Any
    h_next: Any
    aux: AuxOutput
    router_weights: RouterOutput
    presyn: Any = None
    ema_features: Any = None


class RecurrentAgent(nn.Module):
    """Single live task-free Atari policy/value network.

    The forward signature has no game, task, label, or one-hot argument. If
    ``ema_encoder_params`` is supplied, it must be the slow target encoder's
    ``params["encoder"]`` subtree; the returned ``ema_features`` are
    stop-gradient features for later auxiliary targets. EMA updates are a later
    milestone and are intentionally not implemented here.
    """

    model_config: ModelConfig = ModelConfig()
    adapter_config: AdapterConfig = AdapterConfig()

    def init_hidden(self, batch: int, dtype=jnp.float32) -> jnp.ndarray:
        """Return a zero recurrent state for ``batch`` streams."""

        return jnp.zeros((int(batch), int(self.model_config.gru_hidden)), dtype=dtype)

    @nn.compact
    def __call__(
        self,
        obs: jnp.ndarray,
        prev_action: jnp.ndarray,
        prev_reward_clipped: jnp.ndarray,
        reset: jnp.ndarray,
        hidden: jnp.ndarray,
        adapter_dormant: jnp.ndarray | None = None,
        ema_encoder_params: Mapping[str, Any] | None = None,
        collect_presyn: bool = False,
    ) -> AgentOutput:
        obs = jnp.asarray(obs, dtype=jnp.float32)
        prev_action = jnp.asarray(prev_action, dtype=jnp.int32)
        prev_reward_clipped = jnp.asarray(prev_reward_clipped, dtype=jnp.float32)
        reset = jnp.asarray(reset, dtype=bool)
        hidden = jnp.asarray(hidden, dtype=jnp.float32)

        if adapter_dormant is None:
            adapter_dormant = jnp.ones((int(self.adapter_config.num_adapters),), dtype=bool)
        else:
            adapter_dormant = jnp.asarray(adapter_dormant, dtype=bool)

        hidden_in = jnp.where(reset[..., None], jnp.zeros_like(hidden), hidden)

        encoder = Encoder(
            dense_dim=int(self.model_config.dense_dim),
            activation=str(self.model_config.activation),
            frame_stack=int(self.model_config.frame_stack),
            obs_hw=int(self.model_config.obs_hw),
            conv_channels=tuple(self.model_config.conv_channels),
            conv_kernels=tuple(self.model_config.conv_kernels),
            conv_strides=tuple(self.model_config.conv_strides),
            name="encoder",
        )
        encoder_out = encoder(obs, collect_presyn=collect_presyn)
        if collect_presyn:
            e_t, encoder_presyn = encoder_out
        else:
            e_t = encoder_out
            encoder_presyn = {}

        ema_features = None
        if ema_encoder_params is not None:
            ema_encoder = Encoder(
                dense_dim=int(self.model_config.dense_dim),
                activation=str(self.model_config.activation),
                frame_stack=int(self.model_config.frame_stack),
                obs_hw=int(self.model_config.obs_hw),
                conv_channels=tuple(self.model_config.conv_channels),
                conv_kernels=tuple(self.model_config.conv_kernels),
                conv_strides=tuple(self.model_config.conv_strides),
            )
            ema_features = jax.lax.stop_gradient(
                ema_encoder.apply({"params": ema_encoder_params}, obs)
            )

        action_embed = nn.Embed(
            num_embeddings=int(self.model_config.act_dim),
            features=int(self.model_config.action_embed_dim),
            embedding_init=nn.initializers.normal(stddev=0.01),
            name="action_embed",
        )(prev_action)
        reward_col = prev_reward_clipped[..., None]
        reset_col = reset.astype(jnp.float32)[..., None]

        visual_router_input = jnp.concatenate([e_t, hidden_in, action_embed, reward_col], axis=-1)
        e_adapted, visual_weights = ResidualAdapterBank(
            num_adapters=int(self.adapter_config.num_adapters),
            rank=int(self.adapter_config.rank),
            top_k=int(self.adapter_config.top_k),
            residual_rank=int(getattr(self.adapter_config, "residual_rank", 0)),
            name="visual_adapter",
        )(e_t, visual_router_input, adapter_dormant)

        x_t = jnp.concatenate([e_adapted, action_embed, reward_col, reset_col], axis=-1)
        gru = ExplicitGRU(hidden=int(self.model_config.gru_hidden), name="gru")
        gru_out = gru(x_t, hidden_in, reset, return_presyn=collect_presyn)
        if collect_presyn:
            h_t, gru_xi = gru_out
        else:
            h_t = gru_out
            gru_xi = None

        h_adapted, post_weights = ResidualAdapterBank(
            num_adapters=int(self.adapter_config.num_adapters),
            rank=int(self.adapter_config.rank),
            top_k=int(self.adapter_config.top_k),
            residual_rank=int(getattr(self.adapter_config, "residual_rank", 0)),
            name="post_adapter",
        )(h_t, h_t, adapter_dormant)

        logits = PolicyHead(act_dim=int(self.model_config.act_dim), name="policy_head")(h_adapted)
        value = ValueHead(name="value_head")(h_adapted)
        q_key = ContextKeyHead(
            key_dim=int(self.model_config.key_dim),
            key_eps=float(self.model_config.key_eps),
            name="key_head",
        )(h_adapted)
        aux = AuxOutput(
            next_feat=NextFeatHead(
                dense_dim=int(self.model_config.dense_dim),
                name="next_feat_head",
            )(h_adapted, action_embed),
            reward_cat_logits=RewardCatHead(name="reward_cat_head")(h_adapted, action_embed),
            terminal_logit=TerminalHead(name="terminal_head")(h_adapted, action_embed),
        )

        presyn = None
        if collect_presyn:
            presyn = {
                **encoder_presyn,
                "gru_xi": gru_xi,
                "policy_head": h_adapted,
                "value_head": h_adapted,
                "key_head": h_adapted,
                "visual_adapter_input": e_t,
                "post_adapter_input": h_t,
            }

        return AgentOutput(
            logits=logits,
            value=value,
            q_key=q_key,
            h_next=h_t,
            aux=aux,
            router_weights=RouterOutput(visual=visual_weights, post=post_weights),
            presyn=presyn,
            ema_features=ema_features,
        )

    def unroll(
        self,
        params: Mapping[str, Any],
        obs_seq: jnp.ndarray,
        act_seq: jnp.ndarray,
        rew_seq: jnp.ndarray,
        reset_seq: jnp.ndarray,
        h0: jnp.ndarray,
        adapter_dormant: jnp.ndarray | None = None,
        ema_encoder_params: Mapping[str, Any] | None = None,
        collect_presyn: bool = False,
    ) -> tuple[AgentOutput, jnp.ndarray]:
        """Run a time-major recurrent unroll with ``jax.lax.scan``."""

        def step(h_prev, inputs):
            obs_t, act_t, rew_t, reset_t = inputs
            out = self.apply(
                {"params": params},
                obs_t,
                act_t,
                rew_t,
                reset_t,
                h_prev,
                adapter_dormant=adapter_dormant,
                ema_encoder_params=ema_encoder_params,
                collect_presyn=collect_presyn,
            )
            return out.h_next.astype(h_prev.dtype), out

        h_final, outputs = jax.lax.scan(step, h0, (obs_seq, act_seq, rew_seq, reset_seq))
        return outputs, h_final


def _params_root(params: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(params, FrozenDict):
        params = unfreeze(params)
    if "params" in params and isinstance(params["params"], Mapping):
        return params["params"]
    return params


def _has_path(root: Mapping[str, Any], path: tuple[str, ...]) -> bool:
    cur: Any = root
    for part in path:
        if not isinstance(cur, Mapping) or part not in cur:
            return False
        cur = cur[part]
    return True


def protected_param_paths(params: Mapping[str, Any]) -> list[tuple[str, ...]]:
    """Return Flax param paths for TFNS protected affine operators.

    Auxiliary heads are intentionally excluded. Convolution patch presynaptic
    capture is added in M2, but their kernels and biases are registered here.
    """

    root = _params_root(params)
    candidates: list[tuple[str, ...]] = []

    for module in ("conv1", "conv2", "conv3", "dense"):
        candidates.extend(
            [
                ("encoder", module, "kernel"),
                ("encoder", module, "bias"),
            ]
        )

    for gate in ("z", "r", "n"):
        candidates.extend(
            [
                ("gru", f"W_{gate}"),
                ("gru", f"U_{gate}"),
                ("gru", f"b_{gate}"),
            ]
        )

    for head in ("policy_head", "value_head", "key_head"):
        candidates.extend(
            [
                (head, "affine", "kernel"),
                (head, "affine", "bias"),
            ]
        )

    for adapter in ("visual_adapter", "post_adapter"):
        candidates.extend(
            [
                (adapter, "V"),
                (adapter, "U"),
                (adapter, "router", "kernel"),
                (adapter, "router", "bias"),
            ]
        )

    return [path for path in candidates if _has_path(root, path)]


__all__ = [
    "AgentOutput",
    "AuxOutput",
    "RecurrentAgent",
    "RouterOutput",
    "protected_param_paths",
]
