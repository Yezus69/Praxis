"""Consolidation lifecycle: certification, basis growth, and rollback."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import jax.numpy as jnp
import numpy as np

from tfns.consolidate.certify import (
    closed_loop_gate,
    is_learned,
    random_normalized_progress,
)
from tfns.consolidate.plasticity import activate_adapter
from tfns.consolidate.state import ema_update, restore, snapshot
from tfns.memory.record import EpisodeSequence, reconstruct_obs, seq_len
from tfns.protect.bases import empty_basis, expand_basis
from tfns.protect.optimizer import project_first_moments
from tfns.protect.projection import build_protected_modules, collect_conv_basis_columns


_EPS = 1.0e-8


def _get(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _time_batch(rec: EpisodeSequence) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    obs = jnp.asarray(reconstruct_obs(rec), dtype=jnp.float32)[:, None, ...]
    act = jnp.asarray(np.asarray(rec.actions, dtype=np.int32))[:, None]
    rew = jnp.asarray(np.asarray(rec.rewards_clipped, dtype=np.float32))[:, None]
    reset = jnp.asarray(np.asarray(rec.reset_mask, dtype=np.bool_))[:, None]
    return obs, act, rew, reset


def _importance_for_record(
    rec: EpisodeSequence,
    index: int,
    importance: Any,
) -> np.ndarray:
    raw = None
    if importance is None:
        raw = np.abs(np.asarray(rec.causal_contrib, dtype=np.float32))
    elif callable(importance):
        raw = importance(rec)
    elif isinstance(importance, Mapping):
        for key in (id(rec), (rec.episode_id, rec.chunk_index), rec.cluster_id, index):
            if key in importance:
                raw = importance[key]
                break
        if raw is None:
            raw = np.abs(np.asarray(rec.causal_contrib, dtype=np.float32))
    else:
        try:
            raw = importance[index]
        except (TypeError, IndexError, KeyError):
            raw = importance

    values = np.asarray(raw, dtype=np.float32).reshape(-1)
    t = seq_len(rec)
    if values.size != t:
        values = np.resize(values, t)
    values = np.abs(values)
    values = np.where(np.isfinite(values), values, 0.0)
    max_value = float(np.max(values)) if values.size else 0.0
    if max_value <= _EPS:
        values = np.ones((t,), dtype=np.float32)
    else:
        values = values / max_value
    return np.sqrt(values).astype(np.float32)


def _dense_columns(x: Any, weights: np.ndarray) -> jnp.ndarray:
    arr = jnp.asarray(x, dtype=jnp.float32)
    flat = arr.reshape((-1, int(arr.shape[-1])))
    ones = jnp.ones((flat.shape[0], 1), dtype=flat.dtype)
    cols = jnp.concatenate([flat, ones], axis=-1).T
    return _weight_columns(cols, weights)


def _weight_columns(cols: jnp.ndarray, weights: np.ndarray) -> jnp.ndarray:
    w_np = np.asarray(weights, dtype=np.float32).reshape(-1)
    if int(w_np.shape[0]) != int(cols.shape[1]):
        w_np = np.resize(w_np, int(cols.shape[1]))
    w = jnp.asarray(w_np, dtype=cols.dtype)
    return cols * w[None, :]


def _conv_input_key(module_name: str) -> str:
    if module_name.startswith("encoder_conv"):
        return module_name.removeprefix("encoder_") + "_in"
    return module_name + "_in"


def _weighted_conv_columns(x: Any, module: Any, weights: np.ndarray) -> jnp.ndarray:
    arr = jnp.asarray(x, dtype=jnp.float32)
    flat = arr.reshape((-1,) + tuple(arr.shape[-3:]))
    cols = collect_conv_basis_columns(
        flat,
        int(module.kh),
        int(module.kw),
        module.stride,
        int(module.c_in),
    )
    per_obs = int(cols.shape[1]) // max(1, int(flat.shape[0]))
    repeated = np.repeat(np.asarray(weights, dtype=np.float32), per_obs)
    if repeated.size != int(cols.shape[1]):
        repeated = np.resize(repeated, int(cols.shape[1]))
    return cols * jnp.asarray(repeated, dtype=cols.dtype)[None, :]


def collect_protected_activations(
    agent: Any,
    params: Any,
    records: Sequence[EpisodeSequence],
    modules: Mapping[str, Any],
    *,
    importance: Any = None,
) -> dict[str, jnp.ndarray]:
    """Collect weighted augmented activation columns for protected modules."""

    collected: dict[str, list[jnp.ndarray]] = {name: [] for name in modules}

    for rec_index, rec in enumerate(records):
        obs_seq, act_seq, rew_seq, reset_seq = _time_batch(rec)
        h0 = agent.init_hidden(1)
        outputs, _ = agent.unroll(
            params,
            obs_seq,
            act_seq,
            rew_seq,
            reset_seq,
            h0,
            collect_presyn=True,
        )
        presyn = outputs.presyn or {}
        transition_weights = _importance_for_record(rec, rec_index, importance)

        for name, module in modules.items():
            if module.kind == "conv":
                key = _conv_input_key(name)
                if key in presyn:
                    collected[name].append(_weighted_conv_columns(presyn[key], module, transition_weights))
                continue
            if module.kind == "gru_gate":
                if "gru_xi" in presyn:
                    xi = jnp.asarray(presyn["gru_xi"], dtype=jnp.float32)
                    collected[name].append(
                        _weight_columns(xi.reshape((-1, int(xi.shape[-1]))).T, transition_weights)
                    )
                continue
            if name in presyn:
                collected[name].append(_dense_columns(presyn[name], transition_weights))

    out = {}
    for name, module in modules.items():
        if collected[name]:
            out[name] = jnp.concatenate(collected[name], axis=1)
        else:
            out[name] = jnp.zeros((int(module.d_aug), 0), dtype=jnp.float32)
    return out


def expand_protected_bases(
    state: Any,
    A_by_module: Mapping[str, jnp.ndarray],
    modules: Mapping[str, Any],
    cfg: Any,
) -> tuple[Any, dict[str, Any]]:
    """Expand protected bases and project Adam first moments after growth."""

    protect_cfg = _get(cfg, "protect", cfg)
    energy = float(_get(protect_cfg, "residual_energy", 0.995))
    max_rank_frac = float(_get(protect_cfg, "max_rank_frac", 0.95))

    info: dict[str, Any] = {
        "modules": {},
        "capacity_hit": False,
        "exhausted": False,
        "activated_adapter": None,
    }

    for name, module in modules.items():
        A = A_by_module.get(name)
        if A is None:
            continue
        d_aug = int(module.d_aug)
        old_U = state.bases.get(name)
        if old_U is None:
            old_U = empty_basis(d_aug)
        old_rank = int(jnp.asarray(old_U).shape[1])
        max_rank = int(math.floor(max_rank_frac * d_aug))
        new_U, module_info = expand_basis(old_U, A, energy=energy, max_rank=max_rank)
        state.bases[name] = new_U
        module_report = dict(module_info)
        module_report["old_rank"] = old_rank
        module_report["new_rank"] = int(jnp.asarray(new_U).shape[1])
        info["modules"][name] = module_report
        if module_info.get("capacity_hit", False):
            info["capacity_hit"] = True

    if info["capacity_hit"]:
        state, idx = activate_adapter(state)
        info["activated_adapter"] = idx
        if idx is None:
            info["exhausted"] = True

    state.opt_state = project_first_moments(state.opt_state, state.bases, modules)
    return state, info


def _cluster_records(records: Sequence[EpisodeSequence], max_clusters: int = 64) -> list[list[EpisodeSequence]]:
    grouped: dict[int, list[EpisodeSequence]] = defaultdict(list)
    for rec in records:
        grouped[int(rec.cluster_id)].append(rec)
    return [grouped[cid] for cid in sorted(grouped)[:max_clusters]]


def _router_weights_for_record(rec: EpisodeSequence) -> Any:
    return getattr(rec, "teacher_router_weights", None)


def build_sentinel_clusters(
    records: Sequence[EpisodeSequence],
    agent: Any,
    ema_params: Any,
    *,
    burn_in: int,
) -> list[dict[str, Any]]:
    """Build bounded sentinel cluster batches grouped by stored cluster id."""

    selected = [
        rec for rec in records if rec.status == "protected" and rec.is_sentinel
    ] or [rec for rec in records if rec.status == "protected"] or list(records)

    clusters = []
    for recs in _cluster_records(selected):
        rec = recs[0]
        obs_seq, act_seq, rew_seq, reset_seq = _time_batch(rec)
        ema_key_anchor = np.asarray(rec.key_anchor, dtype=np.float32)[:, None, :]
        if ema_params is not None:
            h0 = agent.init_hidden(1)
            ema_outputs, _ = agent.unroll(ema_params, obs_seq, act_seq, rew_seq, reset_seq, h0)
            ema_key_anchor = np.asarray(ema_outputs.q_key, dtype=np.float32)

        clusters.append(
            {
                "cluster_id": int(rec.cluster_id),
                "record_count": len(recs),
                "obs_seq": obs_seq,
                "act_seq": act_seq,
                "rew_seq": rew_seq,
                "reset_seq": reset_seq,
                "teacher_logits": jnp.asarray(rec.teacher_logits, dtype=jnp.float32)[:, None, :],
                "teacher_value": jnp.asarray(rec.teacher_value, dtype=jnp.float32)[:, None],
                "key_anchor": jnp.asarray(rec.key_anchor, dtype=jnp.float32)[:, None, :],
                "teacher_key": jnp.asarray(rec.key_anchor, dtype=jnp.float32)[:, None, :],
                "ema_key_anchor": jnp.asarray(ema_key_anchor, dtype=jnp.float32),
                "teacher_router_weights": _router_weights_for_record(rec),
                "burn_in": int(burn_in),
            }
        )
    return clusters


def _label_records(state: Any, agent: Any, records: Sequence[EpisodeSequence]) -> None:
    for rec in records:
        obs_seq, act_seq, rew_seq, reset_seq = _time_batch(rec)
        h0 = agent.init_hidden(1)
        outputs, _ = agent.unroll(state.params, obs_seq, act_seq, rew_seq, reset_seq, h0)
        key_outputs = outputs
        if state.ema_params is not None:
            ema_outputs, _ = agent.unroll(state.ema_params, obs_seq, act_seq, rew_seq, reset_seq, h0)
            key_outputs = ema_outputs

        rec.teacher_logits = np.asarray(outputs.logits[:, 0], dtype=np.float32)
        rec.teacher_value = np.asarray(outputs.value[:, 0], dtype=np.float32)
        rec.key_anchor = np.asarray(key_outputs.q_key[:, 0], dtype=np.float32)
        rec.status = "candidate"


def _memory_records(memory: Any, fallback: Sequence[EpisodeSequence]) -> tuple[EpisodeSequence, ...]:
    if hasattr(memory, "records"):
        return tuple(memory.records())
    return tuple(fallback)


def _mark_protected(records: Sequence[EpisodeSequence]) -> None:
    seen_clusters: set[int] = set()
    for rec in records:
        rec.status = "protected"
        cluster_id = int(rec.cluster_id)
        if cluster_id not in seen_clusters:
            rec.is_sentinel = True
            seen_clusters.add(cluster_id)


def _as_retention_value(raw: Any, baseline: Any = None) -> Any:
    if isinstance(raw, Mapping):
        return raw
    if isinstance(baseline, Mapping):
        random_score = baseline.get("random", baseline.get("S_random"))
        single_score = baseline.get("single", baseline.get("S_single", baseline.get("best")))
        if random_score is not None and single_score is not None:
            return float(random_normalized_progress(raw, random_score, single_score))
    return raw


def _retention_from_eval(eval_result: Any, learned_game_keys: Any) -> tuple[dict[Any, Any], float | None]:
    if isinstance(eval_result, Mapping) and "retention_by_game" in eval_result:
        current = eval_result.get("current_progress")
        return dict(eval_result["retention_by_game"]), None if current is None else float(current)

    if isinstance(learned_game_keys, Mapping):
        keys = list(learned_game_keys.keys())
        baselines = learned_game_keys
    else:
        keys = list(learned_game_keys)
        baselines = {}

    if isinstance(eval_result, Mapping):
        if not keys:
            return dict(eval_result), None
        return {
            key: (
                _as_retention_value(eval_result[key], baselines.get(key))
                if key in eval_result
                else float("nan")
            )
            for key in keys
        }, None

    if len(keys) == 1:
        key = keys[0]
        return {key: _as_retention_value(eval_result, baselines.get(key))}, None
    raise ValueError("eval_fn must return a mapping when multiple learned games are checked")


def _slow_replay_report(tx: Any, cfg: Any) -> dict[str, Any]:
    consolidate_cfg = _get(cfg, "consolidate", cfg)
    return {
        "steps": 0,
        "tx_provided": tx is not None,
        "lr_scale": float(_get(consolidate_cfg, "slow_replay_lr_scale", 0.1)),
        "reason": "no_replay_loss_callback",
    }


def _failed_cluster_ids(records: Sequence[EpisodeSequence]) -> list[int]:
    return sorted({int(rec.cluster_id) for rec in records})


def consolidate(
    state: Any,
    agent: Any,
    tx: Any,
    eval_fn: Any,
    candidate_records: Sequence[EpisodeSequence],
    cfg: Any,
    *,
    S_random: float,
    S_single: float,
    score_windows: Any,
    learned_game_keys: Iterable[Any] | Mapping[Any, Any],
) -> tuple[Any, bool, dict[str, Any]]:
    """Run one atomic consolidation transaction."""

    records = list(candidate_records)
    snap = snapshot(state)

    learned, learn_info = is_learned(score_windows, S_random, S_single, cfg)
    if not learned:
        return state, False, {"reason": "not_learned", "learned": learn_info}

    try:
        _label_records(state, agent, records)

        modules = build_protected_modules(state.params, agent.model_config)
        A_by_module = collect_protected_activations(agent, state.params, records, modules)
        state, expansion_info = expand_protected_bases(state, A_by_module, modules, cfg)
        if expansion_info.get("exhausted", False):
            restore(state, snap)
            return (
                state,
                False,
                {
                    "reason": "capacity_exhausted",
                    "learned": learn_info,
                    "expansion": expansion_info,
                },
            )

        slow_replay = _slow_replay_report(tx, cfg)

        eval_result = eval_fn(state.params)
        retention_by_game, eval_current_progress = _retention_from_eval(eval_result, learned_game_keys)
        current_progress = (
            float(eval_current_progress)
            if eval_current_progress is not None
            else float(learn_info["recent_progress_mean"])
        )
        gate_ok, gate_info = closed_loop_gate(retention_by_game, current_progress, cfg)

        if gate_ok:
            _mark_protected(records)
            all_records = _memory_records(state.memory, records)
            burn_in = int(_get(_get(cfg, "replay", cfg), "burn_in", 0))
            protected_records = [rec for rec in all_records if rec.status == "protected"]
            seen_record_ids = {id(rec) for rec in protected_records}
            protected_records.extend(
                rec
                for rec in records
                if rec.status == "protected" and id(rec) not in seen_record_ids
            )
            state.protected_clusters = build_sentinel_clusters(
                protected_records,
                agent,
                state.ema_params,
                burn_in=burn_in,
            )
            state.ema_params = ema_update(
                state.ema_params,
                state.params,
                float(_get(_get(cfg, "model", cfg), "ema_decay", 0.995)),
            )
            return (
                state,
                True,
                {
                    "reason": "accepted",
                    "learned": learn_info,
                    "expansion": expansion_info,
                    "slow_replay": slow_replay,
                    "gate": gate_info,
                    "sentinel_clusters": len(state.protected_clusters),
                    "deletion_certification": "not_run",
                },
            )

        restore(state, snap)
        return (
            state,
            False,
            {
                "reason": "closed_loop_regression",
                "learned": learn_info,
                "expansion": expansion_info,
                "slow_replay": slow_replay,
                "gate": gate_info,
                "risk_raise_clusters": _failed_cluster_ids(records),
                "rolled_back": True,
            },
        )
    except Exception:
        restore(state, snap)
        raise


__all__ = [
    "build_sentinel_clusters",
    "collect_protected_activations",
    "consolidate",
    "expand_protected_bases",
]
