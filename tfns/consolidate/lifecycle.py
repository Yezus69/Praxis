"""Consolidation lifecycle: certification, basis growth, and rollback."""

from __future__ import annotations

import math
from collections import defaultdict
from collections.abc import Iterable, Mapping, Sequence
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np

from tfns.consolidate.certify import (
    closed_loop_gate,
    is_learned,
    random_normalized_progress,
)
from tfns.consolidate.plasticity import activate_adapter
from tfns.consolidate.state import ema_update, restore, snapshot
from tfns.memory.sampling import sample_sequences
from tfns.memory.record import EpisodeSequence, reconstruct_obs, seq_len
from tfns.ppo.losses import aux_predictive_loss
from tfns.ppo.rollout import SequenceMinibatch, reconstruct_hidden
from tfns.protect.bases import empty_basis, expand_basis
from tfns.protect.optimizer import optimizer_safe_step, project_first_moments
from tfns.protect.projection import build_protected_modules, collect_conv_basis_columns
from tfns.protect.sentinel import make_sentinel_acceptor


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
    eval_current = None
    if isinstance(eval_result, Mapping):
        raw_current = eval_result.get("eval_current_progress", eval_result.get("current_progress"))
        if raw_current is not None:
            eval_current = float(raw_current)
    if isinstance(eval_result, Mapping) and "retention_by_game" in eval_result:
        current = eval_current
        return dict(eval_result["retention_by_game"]), None if current is None else float(current)

    if isinstance(learned_game_keys, Mapping):
        keys = list(learned_game_keys.keys())
        baselines = learned_game_keys
    else:
        keys = list(learned_game_keys)
        baselines = {}

    if isinstance(eval_result, Mapping):
        if not keys:
            return {
                key: value
                for key, value in eval_result.items()
                if key not in ("current_progress", "eval_current_progress")
            }, eval_current
        return {
            key: (
                _as_retention_value(eval_result[key], baselines.get(key))
                if key in eval_result
                else float("nan")
            )
            for key in keys
        }, eval_current

    if len(keys) == 1:
        key = keys[0]
        return {key: _as_retention_value(eval_result, baselines.get(key))}, None
    raise ValueError("eval_fn must return a mapping when multiple learned games are checked")


def _as_float(value: Any) -> float:
    try:
        return float(value)
    except TypeError:
        return float(np.asarray(value))


def _sample_slow_replay_records(
    state: Any,
    candidate_records: Sequence[EpisodeSequence],
    *,
    step: int,
    count: int,
) -> list[EpisodeSequence]:
    sampled: list[EpisodeSequence] = []
    rng = np.random.default_rng(int(getattr(state, "block_index", 0)) * 1009 + int(step))
    candidates = list(candidate_records)
    if candidates:
        candidate_count = min(int(count), len(candidates))
        indices = rng.choice(len(candidates), size=candidate_count, replace=False)
        sampled.extend(candidates[int(index)] for index in np.asarray(indices).reshape(-1))

    remaining = max(0, int(count) - len(sampled))
    if remaining <= 0:
        return sampled[:count]

    memory = getattr(state, "memory", None)
    if memory is not None and hasattr(memory, "records") and len(memory) > 0:
        sampled.extend(sample_sequences(memory, rng, remaining))

    remaining = max(0, int(count) - len(sampled))
    if remaining <= 0:
        return sampled[:count]
    if not candidates:
        return sampled
    indices = rng.choice(len(candidates), size=remaining, replace=len(candidates) < remaining)
    sampled.extend(candidates[int(index)] for index in np.asarray(indices).reshape(-1))
    return sampled


def _pad_time(value: np.ndarray, length: int, *, dtype: np.dtype) -> np.ndarray:
    arr = np.asarray(value, dtype=dtype)
    if int(arr.shape[0]) == length:
        return arr
    out = np.zeros((length,) + tuple(arr.shape[1:]), dtype=dtype)
    out[: int(arr.shape[0])] = arr
    return out


def _replay_aux_minibatch(
    params: Any,
    agent: Any,
    records: Sequence[EpisodeSequence],
    cfg: Any,
) -> SequenceMinibatch | None:
    replay_cfg = _get(cfg, "replay", cfg)
    burn_default = int(_get(replay_cfg, "burn_in", 0))
    total_default = int(_get(replay_cfg, "seq_len", 64))

    rows: list[dict[str, Any]] = []
    max_len = 0
    for rec in records:
        rec_total = seq_len(rec)
        total = min(int(total_default), rec_total)
        burn_in = min(int(burn_default), total)
        length = total - burn_in
        if length <= 0:
            continue

        obs_np = reconstruct_obs(rec)
        obs = jnp.asarray(obs_np[:total], dtype=jnp.uint8)[:, None, ...]
        actions = jnp.asarray(np.asarray(rec.actions[:total], dtype=np.int32))[:, None]
        rewards = jnp.asarray(np.asarray(rec.rewards_clipped[:total], dtype=np.float32))[:, None]
        resets = jnp.asarray(np.asarray(rec.reset_mask[:total], dtype=np.bool_))[:, None]
        if burn_in > 0:
            h0 = reconstruct_hidden(agent, params, obs[:burn_in], actions[:burn_in], rewards[:burn_in], resets[:burn_in])
        else:
            h0 = agent.init_hidden(1, dtype=jnp.float32)

        next_obs = np.zeros((length,) + tuple(obs_np.shape[1:]), dtype=np.uint8)
        next_mask = np.zeros((length,), dtype=np.bool_)
        available = max(0, min(total, rec_total - 1) - burn_in)
        if available > 0:
            next_obs[:available] = obs_np[burn_in + 1 : burn_in + 1 + available]
            next_mask[:available] = True

        terminal = np.zeros((length,), dtype=np.bool_)
        reset_np = np.asarray(rec.reset_mask, dtype=np.bool_)
        shifted = max(0, min(total, rec_total - 1) - burn_in)
        if shifted > 0:
            terminal[:shifted] = reset_np[burn_in + 1 : burn_in + 1 + shifted]
        if length > shifted:
            terminal[shifted:] = np.asarray(rec.ppo_mask[burn_in + shifted : total], dtype=np.bool_)

        action_np = np.asarray(rec.actions[burn_in:total], dtype=np.int32)
        reward_np = np.asarray(rec.rewards_clipped[burn_in:total], dtype=np.float32)
        reset_cur = np.asarray(rec.reset_mask[burn_in:total], dtype=np.bool_)
        rows.append(
            {
                "obs": obs_np[burn_in:total],
                "prev_action": action_np,
                "prev_reward_clipped": reward_np,
                "action": action_np,
                "reward": reward_np,
                "ppo_mask": np.asarray(rec.ppo_mask[burn_in:total], dtype=np.bool_),
                "reset_mask": reset_cur,
                "next_obs": next_obs,
                "next_obs_mask": next_mask,
                "true_terminal": terminal,
                "h0": h0[0],
                "length": length,
            }
        )
        max_len = max(max_len, length)

    if not rows:
        return None

    valid_mask = []
    fields: dict[str, list[np.ndarray]] = defaultdict(list)
    for row in rows:
        length = int(row["length"])
        valid = np.zeros((max_len,), dtype=np.bool_)
        valid[:length] = True
        valid_mask.append(valid)
        fields["obs"].append(_pad_time(row["obs"], max_len, dtype=np.uint8))
        fields["prev_action"].append(_pad_time(row["prev_action"], max_len, dtype=np.int32))
        fields["prev_reward_clipped"].append(_pad_time(row["prev_reward_clipped"], max_len, dtype=np.float32))
        fields["action"].append(_pad_time(row["action"], max_len, dtype=np.int32))
        fields["reward"].append(_pad_time(row["reward"], max_len, dtype=np.float32))
        fields["ppo_mask"].append(_pad_time(row["ppo_mask"], max_len, dtype=np.bool_))
        fields["reset_mask"].append(_pad_time(row["reset_mask"], max_len, dtype=np.bool_))
        fields["next_obs"].append(_pad_time(row["next_obs"], max_len, dtype=np.uint8))
        fields["next_obs_mask"].append(_pad_time(row["next_obs_mask"], max_len, dtype=np.bool_))
        fields["true_terminal"].append(_pad_time(row["true_terminal"], max_len, dtype=np.bool_))

    batch_size = len(rows)
    zeros = np.zeros((batch_size, max_len), dtype=np.float32)
    return SequenceMinibatch(
        obs=jnp.asarray(np.stack(fields["obs"], axis=0), dtype=jnp.uint8),
        prev_action=jnp.asarray(np.stack(fields["prev_action"], axis=0), dtype=jnp.int32),
        prev_reward_clipped=jnp.asarray(np.stack(fields["prev_reward_clipped"], axis=0), dtype=jnp.float32),
        action=jnp.asarray(np.stack(fields["action"], axis=0), dtype=jnp.int32),
        old_logprob=jnp.asarray(zeros),
        value=jnp.asarray(zeros),
        reward=jnp.asarray(np.stack(fields["reward"], axis=0), dtype=jnp.float32),
        adv=jnp.asarray(zeros),
        ret=jnp.asarray(zeros),
        ppo_mask=jnp.asarray(np.stack(fields["ppo_mask"], axis=0), dtype=bool),
        reset_mask=jnp.asarray(np.stack(fields["reset_mask"], axis=0), dtype=bool),
        h0_chunk=jnp.stack([jnp.asarray(row["h0"], dtype=jnp.float32) for row in rows], axis=0),
        valid_mask=jnp.asarray(np.stack(valid_mask, axis=0), dtype=bool),
        next_obs=jnp.asarray(np.stack(fields["next_obs"], axis=0), dtype=jnp.uint8),
        next_obs_mask=jnp.asarray(np.stack(fields["next_obs_mask"], axis=0), dtype=bool),
        true_terminal=jnp.asarray(np.stack(fields["true_terminal"], axis=0), dtype=bool),
    )


def slow_replay(
    state: Any,
    agent: Any,
    tx: Any,
    candidate_records: Sequence[EpisodeSequence],
    modules: Mapping[str, Any],
    cfg: Any,
    accept_fn: Any,
) -> dict[str, Any]:
    consolidate_cfg = _get(cfg, "consolidate", cfg)
    steps = int(_get(consolidate_cfg, "slow_replay_steps", 4))
    max_update_norm = float(_get(consolidate_cfg, "slow_replay_max_update_norm", 0.05))
    report = {
        "steps": 0,
        "tx_provided": tx is not None,
        "max_update_norm": max_update_norm,
        "records_per_step": 0,
        "accepted": 0,
        "rejected": 0,
        "updates": [],
        "reason": "disabled" if steps <= 0 else "no_optimizer" if tx is None else "no_records",
    }

    if steps <= 0 or tx is None:
        return report

    pool_size = len(candidate_records)
    memory = getattr(state, "memory", None)
    if memory is not None and hasattr(memory, "records"):
        pool_size += len(memory.records())
    if pool_size <= 0:
        return report

    from tfns.train.block import replay_tube_loss

    count = max(1, min(pool_size, max(1, len(candidate_records) or 1)))
    report["records_per_step"] = count
    report["reason"] = "completed"

    for step in range(steps):
        records = _sample_slow_replay_records(state, candidate_records, step=step, count=count)
        if not records:
            report["updates"].append({"step": step, "accepted": False, "reason": "no_records"})
            report["rejected"] += 1
            report["steps"] += 1
            continue

        def loss_fn(params):
            replay_loss, replay_aux = replay_tube_loss(params, agent, records, cfg)
            mb = _replay_aux_minibatch(params, agent, records, cfg)
            if mb is None:
                aux_loss = jnp.asarray(0.0, dtype=jnp.float32)
                aux = {"aux_loss": aux_loss}
            else:
                aux_loss, aux = aux_predictive_loss(params, agent, mb, state.ema_params, cfg)
            loss = replay_loss + aux_loss
            return loss.astype(jnp.float32), {
                "loss": loss,
                "replay_loss": replay_loss,
                "replay_tube_mean": replay_aux["mean"],
                "replay_tube_tail": replay_aux["tail"],
                "replay_tube_total": replay_aux["total"],
                "aux_loss": aux["aux_loss"],
            }

        loss_value, loss_aux = loss_fn(state.params)
        grad = jax.grad(lambda params: loss_fn(params)[0])(state.params)
        new_params, new_opt, info = optimizer_safe_step(
            state.params,
            state.opt_state,
            grad,
            tx,
            state.bases,
            modules,
            accept_fn=accept_fn,
            max_update_norm=max_update_norm,
        )
        accepted = bool(info.get("accepted", False))
        if accepted:
            state.params = new_params
            state.opt_state = new_opt
            state.ema_params = ema_update(
                state.ema_params,
                state.params,
                float(_get(_get(cfg, "model", cfg), "ema_decay", 0.995)),
            )
            report["accepted"] += 1
        else:
            report["rejected"] += 1

        report["updates"].append(
            {
                "step": step,
                "accepted": accepted,
                "loss": _as_float(loss_value),
                "replay_loss": _as_float(loss_aux["replay_loss"]),
                "aux_loss": _as_float(loss_aux["aux_loss"]),
                "applied_norm": _as_float(info.get("applied_norm", 0.0)),
                "applied_scale": float(info.get("applied_scale", 0.0)),
            }
        )
        report["steps"] += 1
    return report


def _record_key(rec: EpisodeSequence) -> tuple[int, int]:
    return int(rec.episode_id), int(rec.chunk_index)


def _promote_failure_recovery(state: Any, failed_keys: set[tuple[int, int]]) -> int:
    promoted = 0
    memory = getattr(state, "memory", None)
    if memory is None or not hasattr(memory, "records"):
        return promoted
    for rec in memory.records():
        if _record_key(rec) in failed_keys:
            rec.status = "failure_recovery"
            promoted += 1
    return promoted


def _raise_replay_risk(state: Any, cluster_ids: Sequence[int], amount: float) -> None:
    robust_stats = state.robust_stats
    cr = robust_stats.setdefault("cluster_risk", {})
    for cid in cluster_ids:
        cid = int(cid)
        cr[cid] = float(cr.get(cid, 0.0)) + float(amount)


def _failed_cluster_ids(records: Sequence[EpisodeSequence]) -> list[int]:
    return sorted({int(rec.cluster_id) for rec in records})


def _failed_record_keys(records: Sequence[EpisodeSequence]) -> list[tuple[int, int]]:
    return [_record_key(rec) for rec in records]


def _report_record_keys(report: Mapping[str, Any]) -> set[tuple[int, int]]:
    keys: set[tuple[int, int]] = set()
    for raw in report.get("failed_records", ()) or ():
        episode_id, chunk_index = raw
        keys.add((int(episode_id), int(chunk_index)))
    return keys


def apply_rejection_feedback(state: Any, report: Mapping[str, Any], cfg: Any) -> Any:
    """Apply caller-owned retry feedback after a rejected consolidation."""

    failed_keys = _report_record_keys(report)
    if failed_keys:
        _promote_failure_recovery(state, failed_keys)

    failed_clusters = report.get("risk_raise_clusters", ()) or ()
    if failed_clusters:
        if state.robust_stats is None:
            state.robust_stats = {}
        raise_amount = float(_get(_get(cfg, "consolidate", cfg), "replay_risk_raise_on_gate_fail", 0.1))
        _raise_replay_risk(state, failed_clusters, raise_amount)
    return state


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

        sentinel_clusters = [
            cluster
            for cluster in (state.protected_clusters or [])
            if (isinstance(cluster, Mapping) and ("obs_seq" in cluster or "obs" in cluster))
            or _get(cluster, "obs_seq", None) is not None
            or _get(cluster, "obs", None) is not None
        ]
        accept_fn = (
            make_sentinel_acceptor(agent, sentinel_clusters, _get(cfg, "behavior", cfg))
            if sentinel_clusters
            else None
        )
        slow_replay_report = slow_replay(state, agent, tx, records, modules, cfg, accept_fn)

        eval_result = eval_fn(state.params)
        retention_by_game, eval_current_progress = _retention_from_eval(eval_result, learned_game_keys)
        if eval_current_progress is not None:
            current_progress = float(eval_current_progress)
            current_progress_source = "eval_fn"
        else:
            current_progress = float(learn_info["recent_progress_mean"])
            current_progress_source = "train_window"
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
                    "slow_replay": slow_replay_report,
                    "gate": gate_info,
                    "current_progress_source": current_progress_source,
                    "sentinel_clusters": len(state.protected_clusters),
                    "deletion_certification": "not_run",
                },
            )

        failed_records = _failed_record_keys(records)
        failed_clusters = _failed_cluster_ids(records)
        restore(state, snap)
        return (
            state,
            False,
            {
                "reason": "closed_loop_regression",
                "learned": learn_info,
                "expansion": expansion_info,
                "slow_replay": slow_replay_report,
                "gate": gate_info,
                "current_progress_source": current_progress_source,
                "failed_records": failed_records,
                "risk_raise_clusters": failed_clusters,
                "rolled_back": True,
            },
        )
    except Exception:
        restore(state, snap)
        raise


__all__ = [
    "apply_rejection_feedback",
    "build_sentinel_clusters",
    "collect_protected_activations",
    "consolidate",
    "expand_protected_bases",
    "slow_replay",
]
