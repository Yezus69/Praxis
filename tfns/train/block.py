"""One task-free continual PPO training block."""

from __future__ import annotations

import math
import dataclasses
from collections.abc import Mapping, Sequence
from functools import lru_cache, partial
from typing import Any

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np

from tfns.behavior import behavior_components, behavior_distance, combined_tol, tube_loss
from tfns.config import AuxConfig, CreditConfig, DetectConfig, PPOConfig, ReplayConfig, TFNSConfig
from tfns.consolidate.state import ContinualState, ema_update
from tfns.credit import (
    ReturnPredictor,
    causal_decomposition,
    discounted_returns,
    eligibility_trace,
    make_predictor_optimizer,
    potential_shaping,
    shaping_enabled,
    shaping_eta,
    train_step as predictor_train_step,
    unroll as predictor_unroll,
    validate,
)
from tfns.detect import PageHinkleyDetector, signature_window
from tfns.memory.record import (
    ACT_DIM,
    FRAME_STACK,
    KEY_DIM,
    OBS_HW,
    EpisodeSequence,
    frames_from_obs,
    make_record,
    seq_len,
)
from tfns.memory.sampling import cluster_probs, cluster_risk, replay_transition_count, sample_sequences
from tfns.ppo import (
    RolloutCarry,
    build_sequence_dataset,
    collect_rollout,
    compute_gae,
    iter_minibatches,
    reconstruct_hidden,
    total_ppo_objective,
)
from tfns.model.encoder import Encoder
from tfns.ppo.rollout import categorical_entropy
from tfns.protect.optimizer import optimizer_safe_step
from tfns.protect.projection import build_protected_modules
from tfns.protect.sentinel import make_sentinel_acceptor
from tfns.protect.constraints import make_constraint_fn

_EPS = 1.0e-8
_MISSING = object()


@dataclasses.dataclass(frozen=True)
class _JitPPOConfig:
    clip_coef: float
    vf_clip: float
    ent_coef: float
    vf_coef: float


@dataclasses.dataclass(frozen=True)
class _JitAuxConfig:
    aux_coef: float
    next_feat_coef: float
    reward_cat_coef: float
    terminal_coef: float


@dataclasses.dataclass(frozen=True)
class _JitReplayConfig:
    seq_len: int
    burn_in: int
    batch_size: int


@dataclasses.dataclass(frozen=True)
class _JitBehaviorConfig:
    teacher_temp: float
    kl_tol: float
    value_tol: float
    key_cos_tol: float
    lambda_v: float
    lambda_q: float
    tail_frac: float


@dataclasses.dataclass(frozen=True)
class _JitTrainConfig:
    ppo: _JitPPOConfig
    aux: _JitAuxConfig
    replay: _JitReplayConfig
    behavior: _JitBehaviorConfig
    replay_coef: float


@struct.dataclass
class ReplayBatch:
    """Fixed-shape replay batch consumed by the jitted replay tube loss."""

    obs: Any
    prev_action: Any
    prev_reward: Any
    prev_reward_clipped: Any
    reset_mask: Any
    actions: Any
    teacher_logits: Any
    teacher_value: Any
    key_anchor: Any
    burn_in: Any
    valid: Any
    seq_weight: Any


def _cfg_section(cfg: Any, name: str, default: Any) -> Any:
    if cfg is None:
        return default
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _cfg_value(obj: Any, name: str, default: Any) -> Any:
    if obj is None:
        return default
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _cfg_optional_int(obj: Any, name: str) -> int | None:
    value = _cfg_value(obj, name, None)
    if value is None:
        return None
    return int(value)


def _fixed_replay_batch_size(cfg: Any, on_policy_count: int | None = None) -> int:
    replay_cfg = _cfg_section(cfg, "replay", ReplayConfig())
    configured = _cfg_optional_int(replay_cfg, "batch_size")
    if configured is not None:
        return max(1, configured)

    ppo_cfg = _cfg_section(cfg, "ppo", PPOConfig())
    num_envs = max(1, int(_cfg_value(ppo_cfg, "num_envs", PPOConfig.num_envs)))
    if on_policy_count is None:
        rollout_len = int(_cfg_value(ppo_cfg, "rollout_len", PPOConfig.rollout_len))
        on_policy_count = rollout_len * num_envs

    seq = max(1, int(_cfg_value(replay_cfg, "seq_len", ReplayConfig.seq_len)))
    replay_frac = float(_cfg_value(replay_cfg, "replay_frac_start", ReplayConfig.replay_frac_start))
    default_rows = int(round(replay_frac * max(1, int(on_policy_count)) / seq))
    return max(1, min(num_envs, default_rows))


def _make_jit_train_config(cfg: Any, replay_batch_size: int) -> _JitTrainConfig:
    ppo_cfg = _cfg_section(cfg, "ppo", PPOConfig())
    aux_cfg = _cfg_section(cfg, "aux", AuxConfig())
    replay_cfg = _cfg_section(cfg, "replay", ReplayConfig())
    behavior_cfg = _cfg_section(cfg, "behavior", None)
    replay_coef = float(
        _cfg_value(
            cfg,
            "replay_coef",
            _cfg_value(replay_cfg, "replay_coef", 1.0),
        )
    )
    return _JitTrainConfig(
        ppo=_JitPPOConfig(
            clip_coef=float(_cfg_value(ppo_cfg, "clip_coef", PPOConfig.clip_coef)),
            vf_clip=float(_cfg_value(ppo_cfg, "vf_clip", PPOConfig.vf_clip)),
            ent_coef=float(_cfg_value(ppo_cfg, "ent_coef", PPOConfig.ent_coef)),
            vf_coef=float(_cfg_value(ppo_cfg, "vf_coef", PPOConfig.vf_coef)),
        ),
        aux=_JitAuxConfig(
            aux_coef=float(_cfg_value(aux_cfg, "aux_coef", AuxConfig.aux_coef)),
            next_feat_coef=float(_cfg_value(aux_cfg, "next_feat_coef", AuxConfig.next_feat_coef)),
            reward_cat_coef=float(_cfg_value(aux_cfg, "reward_cat_coef", AuxConfig.reward_cat_coef)),
            terminal_coef=float(_cfg_value(aux_cfg, "terminal_coef", AuxConfig.terminal_coef)),
        ),
        replay=_JitReplayConfig(
            seq_len=int(_cfg_value(replay_cfg, "seq_len", ReplayConfig.seq_len)),
            burn_in=int(_cfg_value(replay_cfg, "burn_in", ReplayConfig.burn_in)),
            batch_size=int(replay_batch_size),
        ),
        behavior=_JitBehaviorConfig(
            teacher_temp=float(_cfg_value(behavior_cfg, "teacher_temp", 1.0)),
            kl_tol=float(_cfg_value(behavior_cfg, "kl_tol", 0.01)),
            value_tol=float(_cfg_value(behavior_cfg, "value_tol", 0.1)),
            key_cos_tol=float(_cfg_value(behavior_cfg, "key_cos_tol", 0.02)),
            lambda_v=float(_cfg_value(behavior_cfg, "lambda_v", 1.0)),
            lambda_q=float(_cfg_value(behavior_cfg, "lambda_q", 1.0)),
            tail_frac=float(_cfg_value(behavior_cfg, "tail_frac", 0.10)),
        ),
        replay_coef=replay_coef,
    )


def _get(obj: Any, *names: str, default: Any = _MISSING) -> Any:
    for name in names:
        if isinstance(obj, Mapping) and name in obj:
            return obj[name]
        if hasattr(obj, name):
            return getattr(obj, name)
    if default is not _MISSING:
        return default
    raise KeyError(f"object missing any of fields: {names}")


def _as_float(value: Any) -> float:
    arr = np.asarray(jax.device_get(value), dtype=np.float64)
    if arr.size == 0:
        return 0.0
    return float(np.mean(arr))


def _mean(values: Sequence[float]) -> float:
    if not values:
        return 0.0
    return float(np.mean(np.asarray(values, dtype=np.float64)))


def _params_root(params: Mapping[str, Any]) -> Mapping[str, Any]:
    if "params" in params and isinstance(params["params"], Mapping):
        return params["params"]
    return params


def _encoder_params(params: Mapping[str, Any]) -> Mapping[str, Any]:
    root = _params_root(params)
    if "encoder" in root and isinstance(root["encoder"], Mapping):
        return root["encoder"]
    return root


def _current_obs(env_step: Any) -> Any:
    for name in ("obs", "current_obs"):
        if hasattr(env_step, name):
            value = getattr(env_step, name)
            return value() if callable(value) else value
    if hasattr(env_step, "get_obs"):
        return env_step.get_obs()
    raise ValueError("env_step must expose current observation.")


def _init_rollout_carry(agent: Any, env_step: Any) -> RolloutCarry:
    obs = jnp.asarray(_current_obs(env_step))
    if obs.ndim != 4:
        raise ValueError(f"expected batched NHWC observation, got shape {obs.shape}")
    batch = int(obs.shape[0])
    return RolloutCarry(
        hidden=agent.init_hidden(batch, dtype=jnp.float32),
        prev_action=jnp.zeros((batch,), dtype=jnp.int32),
        prev_reward_clipped=jnp.zeros((batch,), dtype=jnp.float32),
        prev_reset=jnp.ones((batch,), dtype=bool),
    )


def _split_rng(rng: Any, count: int) -> tuple[Any, list[Any]]:
    if rng is None:
        rng = jax.random.PRNGKey(0)
    keys = jax.random.split(jnp.asarray(rng), int(count) + 1)
    return keys[0], list(keys[1:])


def _np_seed(key: Any) -> int:
    return int(jax.device_get(jax.random.randint(key, (), 0, np.iinfo(np.int32).max)))


def _pad_last_dim(x: Any, dim: int) -> np.ndarray:
    arr = np.asarray(jax.device_get(x), dtype=np.float32)
    if arr.shape[-1] == dim:
        return arr
    out = np.zeros(arr.shape[:-1] + (int(dim),), dtype=np.float32)
    width = min(int(dim), int(arr.shape[-1]))
    out[..., :width] = arr[..., :width]
    return out


def _match_last_dim(x: Any, dim: int) -> jnp.ndarray:
    arr = jnp.asarray(x, dtype=jnp.float32)
    current = int(arr.shape[-1])
    if current == int(dim):
        return arr
    if current > int(dim):
        return arr[..., : int(dim)]
    pad_width = [(0, 0)] * arr.ndim
    pad_width[-1] = (0, int(dim) - current)
    return jnp.pad(arr, pad_width)


def _pad_logits(logits: Any) -> np.ndarray:
    return _pad_last_dim(logits, ACT_DIM)


def _pad_keys(keys: Any) -> np.ndarray:
    arr = _pad_last_dim(keys, KEY_DIM)
    norm = np.linalg.norm(arr, axis=-1, keepdims=True)
    return (arr / np.maximum(norm, _EPS)).astype(np.float32)


@lru_cache(maxsize=None)
def _return_predictor(act_dim: int) -> ReturnPredictor:
    return ReturnPredictor(act_dim=int(act_dim))


def _predictor_batch(
    model: ReturnPredictor,
    features: Any,
    rollout: Any,
    cfg: Any,
    time_slice: slice | None = None,
) -> dict[str, Any]:
    if time_slice is None:
        time_slice = slice(None)
    feats = jnp.asarray(features, dtype=jnp.float32)[time_slice]
    return {
        "model": model,
        "features": feats,
        "actions": jnp.asarray(rollout.action, dtype=jnp.int32)[time_slice],
        "rewards": jnp.asarray(rollout.reward, dtype=jnp.float32)[time_slice],
        "resets": jnp.asarray(rollout.reset_mask, dtype=bool)[time_slice],
        "episode_end_mask": jnp.asarray(rollout.ppo_mask, dtype=bool)[time_slice],
        "gamma": float(_cfg_value(_cfg_section(cfg, "ppo", PPOConfig()), "gamma", PPOConfig.gamma)),
        "h0": model.init_hidden(int(rollout.action.shape[1])),
    }


def _rollout_outputs(agent: Any, params: Any, rollout: Any, adapter_dormant: Any = None) -> Any:
    outputs, _ = agent.unroll(
        params,
        jnp.asarray(rollout.obs),
        jnp.asarray(rollout.prev_action, dtype=jnp.int32),
        jnp.asarray(rollout.prev_reward_clipped, dtype=jnp.float32),
        jnp.asarray(rollout.reset_mask, dtype=bool),
        jnp.asarray(rollout.h0, dtype=jnp.float32),
        adapter_dormant=adapter_dormant,
    )
    return outputs


@partial(jax.jit, static_argnames=("agent",))
def _rollout_encoder_features(agent: Any, params: Any, obs_seq: Any) -> jnp.ndarray:
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
    obs_seq = jnp.asarray(obs_seq)
    flat_obs = obs_seq.reshape((-1,) + tuple(obs_seq.shape[2:]))
    flat_features = encoder.apply({"params": _encoder_params(params)}, flat_obs)
    features = flat_features.reshape(obs_seq.shape[:2] + flat_features.shape[1:])
    return jax.lax.stop_gradient(features)


@partial(jax.jit, static_argnames=("agent",))
def _rollout_q_key_features(
    agent: Any,
    params: Any,
    obs_seq: Any,
    prev_action: Any,
    prev_reward_clipped: Any,
    reset_seq: Any,
    h0: Any,
    adapter_dormant: Any,
) -> jnp.ndarray:
    outputs, _ = agent.unroll(
        params,
        jnp.asarray(obs_seq),
        jnp.asarray(prev_action, dtype=jnp.int32),
        jnp.asarray(prev_reward_clipped, dtype=jnp.float32),
        jnp.asarray(reset_seq, dtype=bool),
        jnp.asarray(h0, dtype=jnp.float32),
        adapter_dormant=adapter_dormant,
    )
    return jax.lax.stop_gradient(outputs.q_key)


def _rollout_features(agent: Any, params: Any, ema_params: Any, rollout: Any, adapter_dormant: Any) -> Any:
    if ema_params is not None:
        return _rollout_encoder_features(agent, ema_params, rollout.obs)
    if adapter_dormant is None:
        adapter_dormant = jnp.ones((int(agent.adapter_config.num_adapters),), dtype=bool)
    return _rollout_q_key_features(
        agent,
        params,
        rollout.obs,
        rollout.prev_action,
        rollout.prev_reward_clipped,
        rollout.reset_mask,
        rollout.h0,
        adapter_dormant,
    )


def _td_residual(rollout: Any, reward: Any, cfg: Any) -> jnp.ndarray:
    ppo_cfg = _cfg_section(cfg, "ppo", PPOConfig())
    gamma = float(_cfg_value(ppo_cfg, "gamma", PPOConfig.gamma))
    reward = jnp.asarray(reward, dtype=jnp.float32)
    value = jnp.asarray(rollout.value, dtype=jnp.float32)
    next_value = jnp.concatenate([value[1:], jnp.asarray(rollout.last_value)[None, ...]], axis=0)
    done = jnp.asarray(rollout.ppo_mask, dtype=bool)
    done = done.at[-1].set(jnp.asarray(rollout.last_ppo_done, dtype=bool))
    nonterminal = 1.0 - done.astype(jnp.float32)
    return reward + gamma * next_value * nonterminal - value


def _cluster_risks(memory: Any, cfg: Any, robust_stats: Mapping[str, Any] | None = None) -> dict[int, float]:
    clusters = memory.clusters() if memory is not None and hasattr(memory, "clusters") else {}
    stored = {}
    if robust_stats is not None:
        stored = robust_stats.get("cluster_risk", {}) or {}
    return {int(cid): cluster_risk({}, cfg) + float(stored.get(int(cid), 0.0)) for cid in clusters}


def _record_inputs(rec: EpisodeSequence, total: int) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    obs = reconstruct_record_obs(rec, total)
    return (
        obs[:, None, ...],
        jnp.asarray(np.asarray(rec.actions[:total], dtype=np.int32))[:, None],
        jnp.asarray(np.asarray(rec.rewards_clipped[:total], dtype=np.float32))[:, None],
        jnp.asarray(np.asarray(rec.reset_mask[:total], dtype=np.bool_))[:, None],
    )


def reconstruct_record_obs(rec: EpisodeSequence, total: int) -> jnp.ndarray:
    from tfns.memory.record import reconstruct_obs

    return jnp.asarray(reconstruct_obs(rec)[:total], dtype=jnp.uint8)


def _tube_from_outputs(
    outputs: Any,
    teacher_logits: Any,
    teacher_value: Any,
    teacher_key: Any,
    cfg: Any,
    *,
    weight: float = 1.0,
) -> dict[str, jnp.ndarray]:
    behavior_cfg = _cfg_section(cfg, "behavior", None)
    comps = behavior_components(
        _match_last_dim(teacher_logits, outputs.logits.shape[-1]),
        jnp.asarray(teacher_value, dtype=jnp.float32),
        _match_last_dim(teacher_key, outputs.q_key.shape[-1]),
        outputs.logits,
        outputs.value,
        outputs.q_key,
        temp=float(_cfg_value(behavior_cfg, "teacher_temp", 1.0)),
    )
    dist = behavior_distance(
        comps,
        lambda_v=float(_cfg_value(behavior_cfg, "lambda_v", 1.0)),
        lambda_q=float(_cfg_value(behavior_cfg, "lambda_q", 1.0)),
    )
    return tube_loss(
        dist,
        combined_tol(behavior_cfg),
        weights=jnp.asarray(float(weight), dtype=jnp.float32),
        tail_frac=float(_cfg_value(behavior_cfg, "tail_frac", 0.10)),
    )


def _record_tube_loss(
    params: Any,
    agent: Any,
    rec: EpisodeSequence,
    cfg: Any,
    burn_default: int,
    total_default: int,
) -> dict[str, jnp.ndarray] | None:
    total = min(int(total_default), seq_len(rec))
    burn_in = min(int(burn_default), total)
    if total <= burn_in:
        return None
    obs, actions, rewards, resets = _record_inputs(rec, total)
    if burn_in > 0:
        h0 = reconstruct_hidden(
            agent,
            params,
            obs[:burn_in],
            actions[:burn_in],
            rewards[:burn_in],
            resets[:burn_in],
        )
    else:
        h0 = agent.init_hidden(1, dtype=jnp.float32)
    outputs, _ = agent.unroll(
        params,
        obs[burn_in:total],
        actions[burn_in:total],
        rewards[burn_in:total],
        resets[burn_in:total],
        h0,
    )
    return _tube_from_outputs(
        outputs,
        jnp.asarray(rec.teacher_logits[burn_in:total], dtype=jnp.float32)[:, None, :],
        jnp.asarray(rec.teacher_value[burn_in:total], dtype=jnp.float32)[:, None],
        jnp.asarray(rec.key_anchor[burn_in:total], dtype=jnp.float32)[:, None, :],
        cfg,
        weight=max(1.0, float(rec.seq_importance)),
    )


def _cluster_tube_loss(
    params: Any,
    agent: Any,
    cluster: Any,
    cfg: Any,
    burn_default: int,
    total_default: int,
) -> dict[str, jnp.ndarray] | None:
    obs = jnp.asarray(_get(cluster, "obs_seq", "obs"))
    actions = jnp.asarray(_get(cluster, "act_seq", "actions", "prev_action_seq"), dtype=jnp.int32)
    rewards = jnp.asarray(_get(cluster, "rew_seq", "rewards", "prev_reward_seq"), dtype=jnp.float32)
    resets = jnp.asarray(_get(cluster, "reset_seq", "resets"), dtype=bool)
    total = min(int(_get(cluster, "total", "seq_len", default=total_default)), int(obs.shape[0]))
    burn_in = min(int(_get(cluster, "burn_in", default=burn_default)), total)
    if total <= burn_in:
        return None
    h0 = _get(cluster, "h0", "hidden0", default=None)
    if h0 is None:
        h0 = agent.init_hidden(int(obs.shape[1]), dtype=jnp.float32)
    outputs, _ = agent.unroll(
        params,
        obs[:total],
        actions[:total],
        rewards[:total],
        resets[:total],
        jnp.asarray(h0, dtype=jnp.float32),
    )

    class _Protected:
        logits = outputs.logits[burn_in:total]
        value = outputs.value[burn_in:total]
        q_key = outputs.q_key[burn_in:total]

    return _tube_from_outputs(
        _Protected,
        jnp.asarray(_get(cluster, "teacher_logits", "policy_logits"), dtype=jnp.float32)[burn_in:total],
        jnp.asarray(_get(cluster, "teacher_value", "value_target", "value"), dtype=jnp.float32)[burn_in:total],
        jnp.asarray(_get(cluster, "teacher_key", "key_anchor", "q_key"), dtype=jnp.float32)[burn_in:total],
        cfg,
        weight=float(_get(cluster, "risk_weight", "seq_importance", default=1.0)),
    )


def stack_replay_batch(
    records: Sequence[EpisodeSequence],
    cfg: Any,
    *,
    batch_size: int | None = None,
) -> ReplayBatch:
    """Return a fixed ``(R, L, ...)`` replay pytree for jitted replay loss.

    ``R`` is the configured replay batch size and ``L`` is ``replay.seq_len``.
    Short records and unused slots are padded with zeros and ``valid=False``.
    """

    replay_cfg = _cfg_section(cfg, "replay", ReplayConfig())
    length = max(1, int(_cfg_value(replay_cfg, "seq_len", ReplayConfig.seq_len)))
    rows = int(batch_size) if batch_size is not None else _fixed_replay_batch_size(cfg)
    rows = max(1, rows)

    obs = np.zeros((rows, length, OBS_HW, OBS_HW, FRAME_STACK), dtype=np.uint8)
    prev_action = np.zeros((rows, length), dtype=np.int32)
    prev_reward = np.zeros((rows, length), dtype=np.float32)
    reset = np.zeros((rows, length), dtype=np.bool_)
    actions = np.zeros((rows, length), dtype=np.int32)
    teacher_logits = np.zeros((rows, length, ACT_DIM), dtype=np.float32)
    teacher_value = np.zeros((rows, length), dtype=np.float32)
    key_anchor = np.zeros((rows, length, KEY_DIM), dtype=np.float32)
    burn_in = np.zeros((rows,), dtype=np.int32)
    valid = np.zeros((rows, length), dtype=np.bool_)
    seq_weight = np.ones((rows,), dtype=np.float32)

    burn_default = int(_cfg_value(replay_cfg, "burn_in", ReplayConfig.burn_in))
    for row, rec in enumerate(list(records)[:rows]):
        if not isinstance(rec, EpisodeSequence):
            raise TypeError("stack_replay_batch expects EpisodeSequence records")
        total = min(length, seq_len(rec))
        rec_burn = min(max(0, burn_default), total)
        burn_in[row] = rec_burn
        if total <= 0:
            continue

        rec_obs = np.asarray(jax.device_get(reconstruct_record_obs(rec, total)), dtype=np.uint8)
        obs[row, :total] = rec_obs
        prev_action[row, :total] = np.asarray(rec.actions[:total], dtype=np.int32)
        actions[row, :total] = np.asarray(rec.actions[:total], dtype=np.int32)
        prev_reward[row, :total] = np.asarray(rec.rewards_clipped[:total], dtype=np.float32)
        reset[row, :total] = np.asarray(rec.reset_mask[:total], dtype=np.bool_)
        teacher_logits[row, :total] = np.asarray(rec.teacher_logits[:total], dtype=np.float32)
        teacher_value[row, :total] = np.asarray(rec.teacher_value[:total], dtype=np.float32)
        key_anchor[row, :total] = np.asarray(rec.key_anchor[:total], dtype=np.float32)
        valid[row, :total] = True
        seq_weight[row] = max(1.0, float(rec.seq_importance))

    prev_reward_arr = jnp.asarray(prev_reward, dtype=jnp.float32)
    return ReplayBatch(
        obs=jnp.asarray(obs, dtype=jnp.uint8),
        prev_action=jnp.asarray(prev_action, dtype=jnp.int32),
        prev_reward=prev_reward_arr,
        prev_reward_clipped=prev_reward_arr,
        reset_mask=jnp.asarray(reset, dtype=bool),
        actions=jnp.asarray(actions, dtype=jnp.int32),
        teacher_logits=jnp.asarray(teacher_logits, dtype=jnp.float32),
        teacher_value=jnp.asarray(teacher_value, dtype=jnp.float32),
        key_anchor=jnp.asarray(key_anchor, dtype=jnp.float32),
        burn_in=jnp.asarray(burn_in, dtype=jnp.int32),
        valid=jnp.asarray(valid, dtype=bool),
        seq_weight=jnp.asarray(seq_weight, dtype=jnp.float32),
    )


def _replay_batch_tube(
    outputs: Any,
    replay_batch: ReplayBatch,
    cfg: Any,
    burn_in: int,
) -> dict[str, jnp.ndarray]:
    behavior_cfg = _cfg_section(cfg, "behavior", None)
    teacher_logits = jnp.swapaxes(replay_batch.teacher_logits[:, burn_in:], 0, 1)
    teacher_value = jnp.swapaxes(replay_batch.teacher_value[:, burn_in:], 0, 1)
    teacher_key = jnp.swapaxes(replay_batch.key_anchor[:, burn_in:], 0, 1)
    comps = behavior_components(
        _match_last_dim(teacher_logits, outputs.logits.shape[-1]),
        jnp.asarray(teacher_value, dtype=jnp.float32),
        _match_last_dim(teacher_key, outputs.q_key.shape[-1]),
        outputs.logits,
        outputs.value,
        outputs.q_key,
        temp=float(_cfg_value(behavior_cfg, "teacher_temp", 1.0)),
    )
    dist = behavior_distance(
        comps,
        lambda_v=float(_cfg_value(behavior_cfg, "lambda_v", 1.0)),
        lambda_q=float(_cfg_value(behavior_cfg, "lambda_q", 1.0)),
    )
    viol = jnp.square(
        jnp.maximum(dist - jnp.asarray(combined_tol(behavior_cfg), dtype=jnp.float32), 0.0)
    )
    viol = jnp.swapaxes(viol, 0, 1)
    protected_valid = replay_batch.valid[:, burn_in:].astype(jnp.float32)
    record_count = jnp.sum(protected_valid, axis=1)
    record_active = record_count > 0.0
    safe_count = jnp.maximum(record_count, 1.0)

    weighted = replay_batch.seq_weight[:, None] * viol * protected_valid
    mean_by_record = jnp.sum(weighted, axis=1) / safe_count

    masked_viol = jnp.where(protected_valid > 0.0, viol, 0.0)
    sorted_viol = jnp.flip(jnp.sort(masked_viol, axis=1), axis=1)
    tail_frac = float(_cfg_value(behavior_cfg, "tail_frac", 0.10))
    count_int = record_count.astype(jnp.int32)
    k = jnp.maximum(1, jnp.ceil(record_count * tail_frac).astype(jnp.int32))
    k = jnp.minimum(jnp.maximum(count_int, 1), k)
    positions = jnp.arange(sorted_viol.shape[1], dtype=jnp.int32)[None, :]
    tail_mask = (positions < k[:, None]).astype(jnp.float32)
    tail_by_record = jnp.sum(sorted_viol * tail_mask, axis=1) / k.astype(jnp.float32)

    active = record_active.astype(jnp.float32)
    denom = jnp.maximum(jnp.sum(active), 1.0)
    mean = jnp.sum(jnp.where(record_active, mean_by_record, 0.0)) / denom
    tail = jnp.sum(jnp.where(record_active, tail_by_record, 0.0)) / denom
    total_by_record = mean_by_record + tail_by_record
    total = jnp.sum(jnp.where(record_active, total_by_record, 0.0)) / denom
    return {
        "mean": mean.astype(jnp.float32),
        "tail": tail.astype(jnp.float32),
        "total": total.astype(jnp.float32),
    }


def replay_tube_loss_batched(
    params: Any,
    agent: Any,
    replay_batch: ReplayBatch,
    cfg: Any,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Return behavior-tube replay loss for a fixed-shape replay batch."""

    replay_cfg = _cfg_section(cfg, "replay", ReplayConfig())
    length = int(replay_batch.obs.shape[1])
    burn_in = min(max(0, int(_cfg_value(replay_cfg, "burn_in", ReplayConfig.burn_in))), length)
    if burn_in >= length:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return zero, {"mean": zero, "tail": zero, "total": zero}

    obs = jnp.swapaxes(replay_batch.obs, 0, 1)
    actions = jnp.swapaxes(replay_batch.prev_action, 0, 1)
    rewards = jnp.swapaxes(replay_batch.prev_reward_clipped, 0, 1)
    resets = jnp.swapaxes(replay_batch.reset_mask, 0, 1)
    if burn_in > 0:
        h0 = reconstruct_hidden(
            agent,
            params,
            obs[:burn_in],
            actions[:burn_in],
            rewards[:burn_in],
            resets[:burn_in],
        )
    else:
        h0 = agent.init_hidden(int(replay_batch.obs.shape[0]), dtype=jnp.float32)
    outputs, _ = agent.unroll(
        params,
        obs[burn_in:],
        actions[burn_in:],
        rewards[burn_in:],
        resets[burn_in:],
        h0,
    )
    losses = _replay_batch_tube(outputs, replay_batch, cfg, burn_in)
    return losses["total"], losses


@partial(jax.jit, static_argnames=("cfg", "agent"))
def _grad_step(
    params: Any,
    ema_params: Any,
    mb: Any,
    replay_batch: ReplayBatch,
    cfg: _JitTrainConfig,
    agent: Any,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray], Any]:
    def loss_fn(p):
        ppo, ppo_aux = total_ppo_objective(p, agent, mb, ema_params, cfg)
        tube, tube_aux = replay_tube_loss_batched(p, agent, replay_batch, cfg)
        loss = ppo + float(cfg.replay_coef) * tube
        aux = {
            **ppo_aux,
            "ppo_loss": ppo.astype(jnp.float32),
            "replay_loss": tube.astype(jnp.float32),
            "replay_tube_mean": tube_aux["mean"],
            "replay_tube_tail": tube_aux["tail"],
            "replay_tube_total": tube_aux["total"],
        }
        return loss.astype(jnp.float32), aux

    (loss, aux), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
    return loss.astype(jnp.float32), aux, grad


@partial(jax.jit, static_argnames=("agent", "cfg"))
def _grad_step_ppo_only(
    params: Any,
    ema_params: Any,
    mb: Any,
    cfg: _JitTrainConfig,
    agent: Any,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray], Any]:
    def loss_fn(p):
        ppo, aux = total_ppo_objective(p, agent, mb, ema_params, cfg)
        return ppo, aux

    (loss, aux), grad = jax.value_and_grad(loss_fn, has_aux=True)(params)
    return loss.astype(jnp.float32), aux, grad


def replay_tube_loss(
    params: Any,
    agent: Any,
    records: Sequence[Any],
    cfg: Any,
) -> tuple[jnp.ndarray, dict[str, jnp.ndarray]]:
    """Return behavior-tube replay loss for sampled memory records."""

    replay_cfg = _cfg_section(cfg, "replay", ReplayConfig())
    burn_default = int(_cfg_value(replay_cfg, "burn_in", ReplayConfig.burn_in))
    total_default = int(_cfg_value(replay_cfg, "seq_len", ReplayConfig.seq_len))

    totals = []
    means = []
    tails = []
    for rec in records:
        if isinstance(rec, EpisodeSequence):
            losses = _record_tube_loss(params, agent, rec, cfg, burn_default, total_default)
        else:
            losses = _cluster_tube_loss(params, agent, rec, cfg, burn_default, total_default)
        if losses is None:
            continue
        totals.append(losses["total"])
        means.append(losses["mean"])
        tails.append(losses["tail"])

    if not totals:
        zero = jnp.asarray(0.0, dtype=jnp.float32)
        return zero, {"mean": zero, "tail": zero, "total": zero}
    total = jnp.mean(jnp.stack(totals))
    mean = jnp.mean(jnp.stack(means))
    tail = jnp.mean(jnp.stack(tails))
    return total.astype(jnp.float32), {"mean": mean, "tail": tail, "total": total}


def _constraint_distance_and_grad(agent: Any, cfg: Any):
    def distance(params: Any, cluster: Any):
        def loss_fn(p):
            loss, _ = replay_tube_loss(p, agent, [cluster], cfg)
            return loss

        return jax.value_and_grad(loss_fn)(params)

    return distance


def _append_predictor_history(state: ContinualState, mse_val: Any, var_g: Any) -> None:
    stats = state.robust_stats
    stats.setdefault("predictor_val_mses", [])
    stats.setdefault("predictor_val_vars", [])
    stats["predictor_val_mses"].append(_as_float(mse_val))
    stats["predictor_val_vars"].append(_as_float(var_g))


def _admit_rollout_memories(
    state: ContinualState,
    agent: Any,
    rollout: Any,
    causal: Any,
    credit_trace: Any,
    adv: Any,
    td: Any,
    surprise: Any,
    cfg: Any,
) -> int:
    current_outputs = _rollout_outputs(agent, state.params, rollout, state.adapter_dormant)
    ema_outputs = _rollout_outputs(agent, state.ema_params, rollout, state.adapter_dormant)
    entropy = categorical_entropy(current_outputs.logits)

    obs_np = np.asarray(jax.device_get(rollout.obs), dtype=np.uint8)
    action_np = np.asarray(jax.device_get(rollout.action), dtype=np.int32)
    reward_np = np.asarray(jax.device_get(rollout.reward), dtype=np.float32)
    ppo_np = np.asarray(jax.device_get(rollout.ppo_mask), dtype=np.bool_)
    reset_np = np.asarray(jax.device_get(rollout.reset_mask), dtype=np.bool_)
    logits_np = _pad_logits(current_outputs.logits)
    value_np = np.asarray(jax.device_get(current_outputs.value), dtype=np.float32)
    key_np = _pad_keys(ema_outputs.q_key)
    causal_np = np.asarray(jax.device_get(causal), dtype=np.float32)
    trace_np = np.asarray(jax.device_get(credit_trace), dtype=np.float32)
    adv_np = np.abs(np.asarray(jax.device_get(adv), dtype=np.float32))
    td_np = np.abs(np.asarray(jax.device_get(td), dtype=np.float32))
    surprise_np = np.asarray(jax.device_get(surprise), dtype=np.float32)
    entropy_np = np.asarray(jax.device_get(entropy), dtype=np.float32)

    admitted = 0
    episode_base = int(state.block_index) * 1_000_000
    for env_index in range(int(action_np.shape[1])):
        start = 0
        chunk_index = 0
        boundaries = [
            t
            for t in range(1, int(action_np.shape[0]))
            if bool(reset_np[t, env_index])
        ]
        for end in boundaries + [int(action_np.shape[0])]:
            if end > start:
                obs_seq = obs_np[start:end, env_index]
                init_stack, new_frames = frames_from_obs(obs_seq)
                sl = slice(start, end)
                rec = make_record(
                    init_stack=init_stack,
                    new_frames=new_frames,
                    actions=action_np[sl, env_index],
                    rewards_clipped=reward_np[sl, env_index],
                    rewards_raw=reward_np[sl, env_index],
                    ppo_mask=ppo_np[sl, env_index],
                    reset_mask=reset_np[sl, env_index],
                    teacher_logits=logits_np[sl, env_index],
                    teacher_value=value_np[sl, env_index],
                    key_anchor=key_np[sl, env_index],
                    causal_contrib=causal_np[sl, env_index],
                    credit_trace=trace_np[sl, env_index],
                    adv_mag=adv_np[sl, env_index],
                    td_mag=td_np[sl, env_index],
                    surprise=surprise_np[sl, env_index],
                    teacher_entropy=entropy_np[sl, env_index],
                    episode_id=episode_base + env_index,
                    chunk_index=chunk_index,
                    status="transient",
                )
                if state.memory.add(rec):
                    admitted += 1
                chunk_index += 1
            start = end
    return admitted


def train_block(
    state: ContinualState,
    agent: Any,
    tx: Any,
    env_step: Any,
    cfg: Any,
    *,
    sentinel_clusters: Sequence[Any] | None = None,
    constraint_clusters: Sequence[Any] | None = None,
    enable_shaping: bool = False,
    eval_hook: Any = None,
) -> tuple[ContinualState, dict[str, Any]]:
    """Run one continual recurrent PPO block and return telemetry."""

    cfg = cfg or TFNSConfig()
    ppo_cfg = _cfg_section(cfg, "ppo", PPOConfig())
    credit_cfg = _cfg_section(cfg, "credit", CreditConfig())
    detect_cfg = _cfg_section(cfg, "detect", DetectConfig())
    protect_cfg = _cfg_section(cfg, "protect", None)

    if state.rollout_carry is None:
        state.rollout_carry = _init_rollout_carry(agent, env_step)
    if state.detector_state is None:
        state.detector_state = PageHinkleyDetector(detect_cfg).init()
    if state.ema_params is None:
        state.ema_params = state.params
    if state.bases is None:
        state.bases = {}
    if state.protected_clusters is None:
        state.protected_clusters = []
    if state.robust_stats is None:
        state.robust_stats = {}
    if state.adapter_dormant is None:
        state.adapter_dormant = jnp.ones((int(agent.adapter_config.num_adapters),), dtype=bool)
    if state.opt_state is None:
        state.opt_state = tx.init(state.params)

    state.rng, keys = _split_rng(state.rng, 6)
    rollout_key, predictor_key, minibatch_key, replay_key, detector_key, init_key = keys
    rollout, state.rollout_carry, rollout_info = collect_rollout(
        env_step,
        agent,
        state.params,
        state.rollout_carry,
        int(_cfg_value(ppo_cfg, "rollout_len", PPOConfig.rollout_len)),
        rollout_key,
    )
    if "rng" in rollout_info:
        state.rng = rollout_info["rng"]

    features = _rollout_features(agent, state.params, state.ema_params, rollout, state.adapter_dormant)
    predictor = _return_predictor(int(agent.model_config.act_dim))
    predictor_tx = make_predictor_optimizer(credit_cfg)
    if state.predictor_params is None:
        init_batch = _predictor_batch(predictor, features, rollout, cfg)
        state.predictor_params = predictor.init(
            init_key,
            init_batch["features"],
            init_batch["actions"],
            init_batch["rewards"],
            init_batch["resets"],
            init_batch["h0"],
        )["params"]
    if state.predictor_opt_state is None:
        state.predictor_opt_state = predictor_tx.init(state.predictor_params)
    frozen_predictor_params = jax.tree_util.tree_map(lambda x: x, state.predictor_params)

    train_steps = int(_cfg_value(credit_cfg, "predictor_steps", 1))
    time_len = int(rollout.action.shape[0])
    split = max(1, time_len - 1)
    train_slice = slice(0, split)
    val_slice = slice(split, time_len) if split < time_len else slice(0, time_len)
    train_batch = _predictor_batch(predictor, features, rollout, cfg, train_slice)
    val_batch = _predictor_batch(predictor, features, rollout, cfg, val_slice)
    predictor_loss_value = jnp.asarray(0.0, dtype=jnp.float32)
    for _ in range(max(0, train_steps)):
        state.predictor_params, state.predictor_opt_state, pred_aux = predictor_train_step(
            state.predictor_params,
            state.predictor_opt_state,
            train_batch,
            predictor_tx,
        )
        predictor_loss_value = pred_aux["loss"]
    mse_val, var_g = validate(state.predictor_params, val_batch)
    _append_predictor_history(state, mse_val, var_g)

    full_pred_batch = _predictor_batch(predictor, features, rollout, cfg)
    F_seq, Phi_seq = predictor_unroll(
        predictor,
        frozen_predictor_params,
        full_pred_batch["features"],
        full_pred_batch["actions"],
        full_pred_batch["rewards"],
        full_pred_batch["resets"],
        full_pred_batch["h0"],
    )
    G0, _ = discounted_returns(
        jnp.asarray(rollout.reward, dtype=jnp.float32),
        float(_cfg_value(ppo_cfg, "gamma", PPOConfig.gamma)),
        jnp.asarray(rollout.ppo_mask, dtype=bool),
    )
    causal_parts = causal_decomposition(jax.lax.stop_gradient(F_seq), G0[0])
    causal = causal_parts["C"]
    credit_trace = eligibility_trace(
        causal,
        gamma=float(_cfg_value(ppo_cfg, "gamma", PPOConfig.gamma)),
        lambda_c=float(_cfg_value(credit_cfg, "lambda_c", CreditConfig.lambda_c)),
        episode_end_mask=rollout.ppo_mask,
    )

    eta = jnp.asarray(0.0, dtype=jnp.float32)
    reward_for_gae = rollout.reward
    if enable_shaping and shaping_enabled(
        state.robust_stats["predictor_val_mses"],
        state.robust_stats["predictor_val_vars"],
        int(_cfg_value(credit_cfg, "predictor_val_windows", CreditConfig.predictor_val_windows)),
    ):
        eta = shaping_eta(mse_val, var_g)
        eta = jnp.minimum(eta, jnp.asarray(_cfg_value(credit_cfg, "eta_max", CreditConfig.eta_max)))
        reward_for_gae = potential_shaping(
            rollout.reward,
            jax.lax.stop_gradient(Phi_seq),
            float(_cfg_value(ppo_cfg, "gamma", PPOConfig.gamma)),
            eta,
            rollout.ppo_mask,
        )

    adv, ret = compute_gae(
        reward_for_gae,
        rollout.value,
        rollout.ppo_mask,
        rollout.last_value,
        rollout.last_ppo_done,
        float(_cfg_value(ppo_cfg, "gamma", PPOConfig.gamma)),
        float(_cfg_value(ppo_cfg, "gae_lambda", PPOConfig.gae_lambda)),
    )
    td = _td_residual(rollout, reward_for_gae, cfg)
    surprise = jnp.abs(F_seq[:-1] - G0)

    modules = build_protected_modules(state.params, agent.model_config)
    sentinels = list(sentinel_clusters) if sentinel_clusters is not None else list(state.protected_clusters)
    accept_fn = make_sentinel_acceptor(agent, sentinels, _cfg_section(cfg, "behavior", None)) if sentinels else None
    constraints = list(constraint_clusters) if constraint_clusters is not None else []
    constraint_cadence = max(1, int(_cfg_value(protect_cfg, "constraint_cadence", 1)))
    max_update_norm = _cfg_value(
        cfg,
        "max_update_norm",
        _cfg_value(ppo_cfg, "max_grad_norm", PPOConfig.max_grad_norm),
    )
    backtrack_scales = tuple(_cfg_value(protect_cfg, "backtrack_scales", (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)))

    telemetry_lists: dict[str, list[float]] = {
        "loss": [],
        "pg_loss": [],
        "v_loss": [],
        "entropy": [],
        "approx_kl": [],
        "aux_loss": [],
        "replay_tube_mean": [],
        "replay_tube_tail": [],
        "replay_tube_total": [],
        "raw_grad_norm": [],
        "candidate_delta_norm": [],
        "projected_delta_norm": [],
        "applied_norm": [],
        "backtrack_scales": [],
    }
    accept_count = 0
    reject_count = 0
    update_index = 0

    epochs = int(_cfg_value(ppo_cfg, "update_epochs", PPOConfig.update_epochs))
    seq_chunk = int(_cfg_value(ppo_cfg, "seq_chunk", PPOConfig.seq_chunk))
    minibatch_size = _cfg_value(ppo_cfg, "minibatch_size", None)
    on_policy_count = int(np.prod(np.asarray(rollout.action.shape)))
    replay_batch_size = _fixed_replay_batch_size(cfg, on_policy_count)
    jit_cfg = _make_jit_train_config(cfg, replay_batch_size)
    risks = _cluster_risks(state.memory, cfg, state.robust_stats)
    probs = cluster_probs(risks) if risks else None
    max_risk = max(risks.values()) if risks else 0.0
    replay_count = replay_transition_count(on_policy_count, max_risk, cfg)
    replay_seq_len = max(
        1,
        int(_cfg_value(_cfg_section(cfg, "replay", ReplayConfig()), "seq_len", ReplayConfig.seq_len)),
    )
    replay_records_per_update = max(0, int(math.ceil(replay_count / replay_seq_len)))
    replay_records_per_update = min(replay_records_per_update, replay_batch_size)
    sequence_dataset = build_sequence_dataset(
        rollout,
        adv,
        ret,
        seq_chunk,
        agent=agent,
        params=state.params,
    )
    if minibatch_size is None:
        num_minibatches = int(_cfg_value(ppo_cfg, "num_minibatches", PPOConfig.num_minibatches))
        if num_minibatches <= 0:
            raise ValueError("num_minibatches must be positive")
        num_chunks = int(rollout.action.shape[0]) // seq_chunk
        num_envs = int(rollout.action.shape[1])
        num_seq = num_chunks * num_envs
        minibatch_size = max(1, num_seq // num_minibatches)

    for epoch in range(epochs):
        state.rng, (epoch_key,) = _split_rng(state.rng, 1)
        mb_iter = iter_minibatches(sequence_dataset, minibatch_size, epoch_key)
        for mb in mb_iter:
            replay_records = []
            if replay_records_per_update > 0 and state.memory is not None and len(state.memory) > 0:
                replay_seed = _np_seed(jax.random.fold_in(replay_key, update_index))
                replay_records = sample_sequences(state.memory, replay_seed, replay_records_per_update, probs)
            if replay_records:
                replay_batch = stack_replay_batch(replay_records, jit_cfg, batch_size=replay_batch_size)
                loss_value, aux, grad = _grad_step(
                    state.params,
                    state.ema_params,
                    mb,
                    replay_batch,
                    jit_cfg,
                    agent,
                )
            else:
                loss_value, aux, grad = _grad_step_ppo_only(
                    state.params,
                    state.ema_params,
                    mb,
                    jit_cfg,
                    agent,
                )
                zero = jnp.asarray(0.0, dtype=jnp.float32)
                aux = {
                    **aux,
                    "replay_tube_mean": zero,
                    "replay_tube_tail": zero,
                    "replay_tube_total": zero,
                }

            constraint_fn = None
            if constraints and update_index % constraint_cadence == 0:
                constraint_fn = make_constraint_fn(
                    _constraint_distance_and_grad(agent, cfg),
                    constraints,
                    combined_tol(_cfg_section(cfg, "behavior", None)),
                    state.bases,
                    modules,
                    ridge=float(_cfg_value(protect_cfg, "constraint_ridge", 1.0e-3)),
                    max_clusters=int(_cfg_value(protect_cfg, "constraint_max_clusters", 8)),
                )

            new_params, new_opt, info = optimizer_safe_step(
                state.params,
                state.opt_state,
                grad,
                tx,
                state.bases,
                modules,
                accept_fn=accept_fn,
                constraint_fn=constraint_fn,
                max_update_norm=max_update_norm,
                backtrack_scales=backtrack_scales,
            )
            if bool(info["accepted"]):
                state.params = new_params
                state.opt_state = new_opt
                state.ema_params = ema_update(
                    state.ema_params,
                    state.params,
                    float(_cfg_value(_cfg_section(cfg, "model", None), "ema_decay", 0.995)),
                )
                accept_count += 1
            else:
                reject_count += 1

            telemetry_lists["loss"].append(_as_float(loss_value))
            for name in ("pg_loss", "v_loss", "entropy", "approx_kl", "aux_loss"):
                telemetry_lists[name].append(_as_float(aux.get(name, 0.0)))
            telemetry_lists["replay_tube_mean"].append(_as_float(aux["replay_tube_mean"]))
            telemetry_lists["replay_tube_tail"].append(_as_float(aux["replay_tube_tail"]))
            telemetry_lists["replay_tube_total"].append(_as_float(aux["replay_tube_total"]))
            for name in ("raw_grad_norm", "candidate_delta_norm", "projected_delta_norm", "applied_norm"):
                telemetry_lists[name].append(_as_float(info[name]))
            telemetry_lists["backtrack_scales"].append(float(info["applied_scale"]))
            update_index += 1

    admitted = _admit_rollout_memories(
        state,
        agent,
        rollout,
        causal,
        credit_trace,
        adv,
        td,
        surprise,
        cfg,
    )

    ema_outputs = _rollout_outputs(agent, state.ema_params, rollout, state.adapter_dormant)
    signature = signature_window(np.asarray(jax.device_get(ema_outputs.q_key), dtype=np.float32))
    prev_signature = state.robust_stats.get("last_signature")
    if prev_signature is None:
        signature_distance = 0.0
    else:
        denom = float(np.linalg.norm(signature) * np.linalg.norm(prev_signature))
        signature_distance = 0.0 if denom <= _EPS else float(1.0 - np.dot(signature, prev_signature) / denom)
    state.robust_stats["last_signature"] = signature
    predictive_error = _as_float(mse_val)
    detector_input = predictive_error + signature_distance
    detector = PageHinkleyDetector(detect_cfg)
    state.detector_state, changed = detector.update(state.detector_state, detector_input)

    state.block_index += 1

    telemetry = {name: _mean(values) for name, values in telemetry_lists.items()}
    telemetry.update(
        {
            "accept_count": int(accept_count),
            "reject_count": int(reject_count),
            "predictor_loss": _as_float(predictor_loss_value),
            "predictor_val_error": predictive_error,
            "predictor_var_G": _as_float(var_g),
            "eta": _as_float(eta),
            "memory_bytes": int(state.memory.bytes_used()) if hasattr(state.memory, "bytes_used") else 0,
            "memory_count": int(len(state.memory)) if state.memory is not None else 0,
            "memory_clusters": int(len(state.memory.clusters())) if hasattr(state.memory, "clusters") else 0,
            "memory_admitted": int(admitted),
            "detector_input": float(detector_input),
            "detector_signature_distance": float(signature_distance),
            "detector_changed": bool(changed),
            "detector_cusum": float(getattr(state.detector_state, "cusum", 0.0)),
            "block_index": int(state.block_index),
        }
    )
    if eval_hook is not None:
        telemetry["eval"] = eval_hook(state, telemetry)
    return state, telemetry


__all__ = [
    "ReplayBatch",
    "replay_tube_loss",
    "replay_tube_loss_batched",
    "stack_replay_batch",
    "train_block",
]
