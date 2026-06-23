"""Recurrent PPO and predictive auxiliary losses."""

from __future__ import annotations

from collections.abc import Mapping
from typing import Any

from flax.core import FrozenDict, unfreeze
import jax
import jax.numpy as jnp
import optax

from tfns.config import AuxConfig, PPOConfig
from tfns.model.encoder import Encoder
from tfns.ppo.rollout import SequenceMinibatch, categorical_entropy, categorical_log_prob


EPS = 1.0e-8


def _cfg_section(cfg: Any, name: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(name, cfg)
    return getattr(cfg, name, cfg)


def _cfg_value(cfg: Any, name: str, default: float) -> float:
    if cfg is None:
        return float(default)
    if isinstance(cfg, Mapping) and name in cfg:
        return float(cfg[name])
    if hasattr(cfg, name):
        return float(getattr(cfg, name))
    return float(default)


def _time_major(x: Any) -> jnp.ndarray:
    return jnp.swapaxes(jnp.asarray(x), 0, 1)


def _mask_mean(x: Any, mask: Any) -> jnp.ndarray:
    x = jnp.asarray(x, dtype=jnp.float32)
    mask = jnp.asarray(mask, dtype=jnp.float32)
    denom = jnp.maximum(jnp.sum(mask), 1.0)
    return jnp.sum(x * mask) / denom


def _unroll_minibatch(params, agent, mb: SequenceMinibatch):
    return agent.unroll(
        params,
        _time_major(mb.obs),
        _time_major(mb.prev_action).astype(jnp.int32),
        _time_major(mb.prev_reward_clipped).astype(jnp.float32),
        _time_major(mb.reset_mask).astype(bool),
        jnp.asarray(mb.h0_chunk, dtype=jnp.float32),
    )


def _ppo_terms(
    outputs,
    mb: SequenceMinibatch,
    cfg: Any,
    *,
    ent_coef: Any | None = None,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    ppo_cfg = _cfg_section(cfg, "ppo", PPOConfig())
    clip_coef = _cfg_value(ppo_cfg, "clip_coef", PPOConfig.clip_coef)
    vf_clip = _cfg_value(ppo_cfg, "vf_clip", PPOConfig.vf_clip)
    ent_coef_value = (
        jnp.asarray(ent_coef, dtype=jnp.float32)
        if ent_coef is not None
        else jnp.asarray(_cfg_value(ppo_cfg, "ent_coef", PPOConfig.ent_coef), dtype=jnp.float32)
    )
    vf_coef = _cfg_value(ppo_cfg, "vf_coef", PPOConfig.vf_coef)

    action = _time_major(mb.action).astype(jnp.int32)
    old_logprob = _time_major(mb.old_logprob).astype(jnp.float32)
    old_value = _time_major(mb.value).astype(jnp.float32)
    adv = _time_major(mb.adv).astype(jnp.float32)
    ret = _time_major(mb.ret).astype(jnp.float32)
    valid = _time_major(mb.valid_mask).astype(jnp.float32)

    # Forced environment actions (e.g. FIRE-on-reset) were not sampled from the
    # policy, so their behavior probability is 1, not pi_theta(a). Exclude them
    # from the importance-ratio objective; value/entropy/aux still learn there.
    forced = mb.forced_mask
    if forced is None:
        policy_valid = valid
    else:
        policy_valid = valid * (1.0 - _time_major(forced).astype(jnp.float32))

    new_logprob = categorical_log_prob(outputs.logits, action)
    entropy = categorical_entropy(outputs.logits)
    logratio = new_logprob - old_logprob
    ratio = jnp.exp(logratio)

    adv_mean = _mask_mean(adv, policy_valid)
    adv_var = _mask_mean(jnp.square(adv - adv_mean), policy_valid)
    adv_norm = (adv - adv_mean) / jnp.sqrt(adv_var + EPS)

    pg_loss_unclipped = -adv_norm * ratio
    pg_loss_clipped = -adv_norm * jnp.clip(ratio, 1.0 - clip_coef, 1.0 + clip_coef)
    pg_loss = _mask_mean(jnp.maximum(pg_loss_unclipped, pg_loss_clipped), policy_valid)

    value_pred = jnp.asarray(outputs.value, dtype=jnp.float32)
    v_unclipped = jnp.square(value_pred - ret)
    value_clipped = old_value + jnp.clip(value_pred - old_value, -vf_clip, vf_clip)
    v_clipped = jnp.square(value_clipped - ret)
    v_loss = 0.5 * _mask_mean(jnp.maximum(v_unclipped, v_clipped), valid)

    entropy_loss = _mask_mean(entropy, valid)
    approx_kl = _mask_mean((ratio - 1.0) - logratio, policy_valid)
    clipfrac = _mask_mean((jnp.abs(ratio - 1.0) > clip_coef).astype(jnp.float32), policy_valid)
    loss = pg_loss + vf_coef * v_loss - ent_coef_value * entropy_loss
    aux = {
        "pg_loss": pg_loss,
        "v_loss": v_loss,
        "entropy": entropy_loss,
        "ent_coef": ent_coef_value,
        "approx_kl": approx_kl,
        "clipfrac": clipfrac,
    }
    return loss.astype(jnp.float32), aux


def ppo_loss(params, agent, mb: SequenceMinibatch, cfg: Any) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Return clipped recurrent PPO loss and diagnostics."""

    outputs, _ = _unroll_minibatch(params, agent, mb)
    return _ppo_terms(outputs, mb, cfg)


def _params_root(params: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(params, FrozenDict):
        params = unfreeze(params)
    if "params" in params and isinstance(params["params"], Mapping):
        return params["params"]
    return params


def _encoder_params(params: Mapping[str, Any]) -> Mapping[str, Any]:
    root = _params_root(params)
    if "encoder" in root and isinstance(root["encoder"], Mapping):
        return root["encoder"]
    return root


def _slow_encoder_features(agent, params, obs: jnp.ndarray) -> jnp.ndarray:
    cfg = agent.model_config
    encoder = Encoder(
        dense_dim=int(cfg.dense_dim),
        activation=str(cfg.activation),
        frame_stack=int(cfg.frame_stack),
        obs_hw=int(cfg.obs_hw),
        conv_channels=tuple(cfg.conv_channels),
        conv_kernels=tuple(cfg.conv_kernels),
        conv_strides=tuple(cfg.conv_strides),
    )
    flat_obs = obs.reshape((-1,) + tuple(obs.shape[2:]))
    flat_feat = encoder.apply({"params": _encoder_params(params)}, flat_obs)
    return flat_feat.reshape(obs.shape[:2] + flat_feat.shape[1:])


def _reward_sign_class(reward: jnp.ndarray) -> jnp.ndarray:
    reward = jnp.asarray(reward, dtype=jnp.float32)
    return jnp.where(reward < 0.0, 0, jnp.where(reward > 0.0, 2, 1)).astype(jnp.int32)


def _aux_terms(outputs, params, agent, mb: SequenceMinibatch, ema_encoder_params, cfg: Any):
    aux_cfg = _cfg_section(cfg, "aux", AuxConfig())
    aux_coef = _cfg_value(aux_cfg, "aux_coef", AuxConfig.aux_coef)
    next_coef = _cfg_value(aux_cfg, "next_feat_coef", AuxConfig.next_feat_coef)
    reward_coef = _cfg_value(aux_cfg, "reward_cat_coef", AuxConfig.reward_cat_coef)
    terminal_coef = _cfg_value(aux_cfg, "terminal_coef", AuxConfig.terminal_coef)

    valid = _time_major(mb.valid_mask).astype(jnp.float32)
    next_obs = mb.next_obs
    if next_obs is None:
        next_obs = jnp.zeros_like(mb.obs)
        next_mask = jnp.zeros_like(mb.action, dtype=bool)
    else:
        next_mask = mb.next_obs_mask
        if next_mask is None:
            next_mask = jnp.ones_like(mb.action, dtype=bool)
    next_mask = _time_major(next_mask).astype(jnp.float32) * valid
    next_obs_t = _time_major(next_obs)

    encoder_source = ema_encoder_params if ema_encoder_params is not None else params
    target_feat = jax.lax.stop_gradient(_slow_encoder_features(agent, encoder_source, next_obs_t))
    pred_feat = jnp.asarray(outputs.aux.next_feat, dtype=jnp.float32)
    pred_norm = pred_feat / (jnp.linalg.norm(pred_feat, axis=-1, keepdims=True) + EPS)
    target_norm = target_feat / (jnp.linalg.norm(target_feat, axis=-1, keepdims=True) + EPS)
    next_feat_loss = _mask_mean(1.0 - jnp.sum(pred_norm * target_norm, axis=-1), next_mask)

    reward = _time_major(mb.reward).astype(jnp.float32)
    reward_target = _reward_sign_class(reward)
    reward_ce = optax.softmax_cross_entropy_with_integer_labels(
        outputs.aux.reward_cat_logits,
        reward_target,
    )
    reward_cat_loss = _mask_mean(reward_ce, valid)

    terminal_target = mb.true_terminal if mb.true_terminal is not None else mb.reset_mask
    terminal_target = _time_major(terminal_target).astype(jnp.float32)
    terminal_bce = optax.sigmoid_binary_cross_entropy(
        outputs.aux.terminal_logit,
        terminal_target,
    )
    terminal_loss = _mask_mean(terminal_bce, valid)

    unscaled = (
        next_coef * next_feat_loss
        + reward_coef * reward_cat_loss
        + terminal_coef * terminal_loss
    )
    loss = aux_coef * unscaled
    aux = {
        "aux_loss": loss.astype(jnp.float32),
        "next_feat_loss": next_feat_loss,
        "reward_cat_loss": reward_cat_loss,
        "terminal_loss": terminal_loss,
    }
    return loss.astype(jnp.float32), aux


def aux_predictive_loss(
    params,
    agent,
    mb: SequenceMinibatch,
    ema_encoder_params,
    cfg: Any,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Return scaled predictive auxiliary loss.

    Targets are stop-gradient. The heads consume only recurrent state and
    action embeddings; no task id enters the objective.
    """

    outputs, _ = _unroll_minibatch(params, agent, mb)
    return _aux_terms(outputs, params, agent, mb, ema_encoder_params, cfg)


def total_ppo_objective(
    params,
    agent,
    mb: SequenceMinibatch,
    ema_encoder_params,
    cfg: Any,
    *,
    ent_coef: Any | None = None,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Return PPO plus predictive auxiliary losses.

    Future replay behavior-tube terms should be summed here, outside the PPO
    importance-ratio objective because replay behavior policies are stale.
    """

    outputs, _ = _unroll_minibatch(params, agent, mb)
    ppo, ppo_aux = _ppo_terms(outputs, mb, cfg, ent_coef=ent_coef)
    aux_loss, aux_aux = _aux_terms(outputs, params, agent, mb, ema_encoder_params, cfg)
    total = ppo + aux_loss
    merged = {"loss": total.astype(jnp.float32), **ppo_aux, **aux_aux}
    return total.astype(jnp.float32), merged


__all__ = [
    "aux_predictive_loss",
    "ppo_loss",
    "total_ppo_objective",
]
