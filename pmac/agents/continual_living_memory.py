"""Guard-aware continual Living Memory driver for PMA-C retention proofs."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from pmac.agents.atari_mem_net import MemAtariActorCritic, mem_init
from pmac.agents.living_memory_eval import (
    build_protected_bank,
    certify_protected_memories,
    eval_living_memory,
)
from pmac.agents.ppo_living_memory_fast import FastLMConfig, train_living_memory_fast
from pmac.behavior_distance import huber
from pmac.envs.atari_envpool import ACT_DIM
from pmac.evaluation import normalized_retention
from pmac.guard_schedule import allocate_guard, forgetting_risk, sample_review_games
from pmac.memory import SourceFlag
from pmac.memory.losses import retrieval_alignment_loss
from pmac.memory.reader import expand_source_flags, retrieve
from pmac.memory.runtime import RunningValueNorm, default_retrieval_hp
from pmac.memory.sentinels_visual import (
    VisualSentinelStore,
    build_align_batch,
    collect_visual_sentinels,
)
from pmac.rollback_gate import GateConfig, evaluate_gate, on_reject_actions


ABLATIONS = {
    "full",
    "no_conservation",
    "no_projection",
    "no_memory_read",
    "plain_ppo",
    "no_review",
    "no_gate",
}


def _cfg_get(cfg, name: str, default):
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _cfg_int(cfg, name: str, default: int) -> int:
    return int(_cfg_get(cfg, name, default))


def _cfg_float(cfg, name: str, default: float) -> float:
    return float(_cfg_get(cfg, name, default))


def _cfg_with_total_timesteps(cfg, total_timesteps):
    if total_timesteps is None:
        return cfg
    total_timesteps = int(total_timesteps)
    if is_dataclass(cfg):
        return replace(cfg, total_timesteps=total_timesteps)
    if isinstance(cfg, Mapping):
        out = dict(cfg)
        out["total_timesteps"] = total_timesteps
        return out
    out = copy.copy(cfg)
    setattr(out, "total_timesteps", total_timesteps)
    return out


def _configured_total_timesteps(cfg, per_game_steps) -> int:
    if per_game_steps is not None:
        return int(per_game_steps)
    return _cfg_int(cfg, "total_timesteps", FastLMConfig().total_timesteps)


def _copy_leaf(value):
    if isinstance(value, np.ndarray):
        return np.array(value, copy=True)
    if hasattr(value, "shape") and hasattr(value, "dtype"):
        try:
            return jnp.array(value)
        except (TypeError, ValueError):
            return copy.deepcopy(value)
    return copy.deepcopy(value)


def _pytree_copy(tree):
    if tree is None:
        return None
    return jax.tree_util.tree_map(_copy_leaf, tree)


def _snapshot_state(params, ema_params, value_norms, hot_bank) -> dict:
    return {
        "params": _pytree_copy(params),
        "ema_params": _pytree_copy(ema_params),
        "value_norms": _pytree_copy(value_norms),
        "hot_bank": _pytree_copy(hot_bank),
    }


def _restore_snapshot(snapshot: Mapping):
    return (
        _pytree_copy(snapshot["params"]),
        _pytree_copy(snapshot["ema_params"]),
        _pytree_copy(snapshot["value_norms"]),
        _pytree_copy(snapshot["hot_bank"]),
    )


def _params_dims(params, cfg) -> dict:
    try:
        n_games, d_c = params["game_embed"]["embedding"].shape
        d_k = params["key_head"]["kernel"].shape[-1]
        d_m = params["wv"]["kernel"].shape[-1]
        act_dim = params["policy_head"]["kernel"].shape[-1]
        return {
            "n_games": int(n_games),
            "d_k": int(d_k),
            "d_c": int(d_c),
            "d_m": int(d_m),
            "act_dim": int(act_dim),
        }
    except Exception:
        return {
            "n_games": _cfg_int(cfg, "n_games", 1),
            "d_k": _cfg_int(cfg, "d_k", 128),
            "d_c": _cfg_int(cfg, "d_c", 16),
            "d_m": _cfg_int(cfg, "d_m", 128),
            "act_dim": _cfg_int(cfg, "act_dim", ACT_DIM),
        }


def _guard_field(guard, name: str, default=None):
    if isinstance(guard, Mapping):
        return guard.get(name, default)
    return getattr(guard, name, default)


def _audit_violation_rate(params, guard, cfg) -> float:
    if params is None or guard is None:
        return 0.0
    atoms = _guard_field(guard, "atoms")
    bank = _guard_field(guard, "bank")
    if atoms is None or bank is None:
        return 0.0
    try:
        keys = jnp.asarray(atoms["keys"], dtype=jnp.float32)
        weight = jnp.asarray(atoms.get("weight", jnp.ones(keys.shape[:1])), dtype=jnp.float32)
        valid = weight > 0.0
        if int(keys.shape[0]) == 0:
            return 0.0
        dims = _params_dims(params, cfg)
        net = MemAtariActorCritic(**dims)
        game_id = jnp.asarray(atoms["game_id"], dtype=jnp.int32)
        if game_id.ndim == 0:
            game_id = jnp.broadcast_to(game_id, (int(keys.shape[0]),))
        teacher_policy = jnp.asarray(atoms["teacher_policy"], dtype=jnp.float32)
        teacher_value = jnp.asarray(atoms["teacher_value"], dtype=jnp.float32)
        eps = jnp.asarray(atoms["eps"], dtype=jnp.float32)
        hp = default_retrieval_hp(min(_cfg_int(cfg, "top_k", 1), int(bank["keys"].shape[0])))
        c_embed = net.apply({"params": params}, game_id, method=MemAtariActorCritic.context)
        retrieved = retrieve(keys, c_embed, game_id, bank, hp)
        summary = net.apply(
            {"params": params},
            retrieved.atom_feats,
            retrieved.alpha,
            method=MemAtariActorCritic.mem_summary,
        )
        logits, values = net.apply(
            {"params": params},
            keys,
            c_embed,
            summary,
            method=MemAtariActorCritic.latent_behavior,
        )
        p_theta = jax.nn.softmax(logits, axis=-1)
        d_pi = jnp.sum(
            teacher_policy * (jnp.log(teacher_policy + 1.0e-8) - jnp.log(p_theta + 1.0e-8)),
            axis=-1,
        )
        d_v = huber(values - teacher_value, _cfg_float(cfg, "huber_delta", 1.0))
        distance = d_pi + _cfg_float(cfg, "guard_lambda_v", 1.0) * d_v
        violated = jnp.logical_and(valid, distance > eps)
        denom = jnp.maximum(jnp.sum(valid.astype(jnp.float32)), 1.0)
        return float(np.asarray(jax.device_get(jnp.sum(violated.astype(jnp.float32)) / denom)))
    except Exception:
        return 1.0


def _audit_retrieval_alignment(params, aux, cfg) -> float:
    if params is None or aux is None:
        return float("inf")
    align_batch = _guard_field(aux, "align_batch")
    if align_batch is None:
        return float("inf")
    try:
        dims = _params_dims(params, cfg)
        loss = retrieval_alignment_loss(
            params,
            align_batch,
            tau=_cfg_float(cfg, "retr_tau", 0.1),
            dims=dims,
        )
        return -float(np.asarray(jax.device_get(loss)))
    except Exception:
        return float("-inf")


def _source_flags_for_failure(atom_set: Mapping, n: int) -> np.ndarray:
    flags = np.asarray(atom_set.get("source_flags", np.zeros((n,), dtype=np.int32)), dtype=np.int32)
    if flags.ndim == 0:
        flags = np.full((n,), int(flags), dtype=np.int32)
    flags = flags.reshape(-1)
    if int(flags.shape[0]) == 1 and n != 1:
        flags = np.repeat(flags, n, axis=0)
    if int(flags.shape[0]) != n:
        flags = np.resize(flags, n).astype(np.int32)
    return flags | int(SourceFlag.FAILURE_RECOVERY)


def _failure_copy(atom_set: Mapping) -> dict:
    out = {}
    for name, value in atom_set.items():
        if isinstance(value, np.ndarray):
            out[name] = np.array(value, copy=True)
        elif hasattr(value, "shape") and hasattr(value, "dtype"):
            out[name] = np.array(value)
        else:
            out[name] = copy.deepcopy(value)
    keys = np.asarray(out.get("keys", np.zeros((0, 0), dtype=np.float32)))
    if keys.ndim == 0 or keys.size == 0:
        n = 0
    elif keys.ndim == 1:
        n = 1
    else:
        n = int(keys.shape[0])
    out["source_flags"] = _source_flags_for_failure(out, n)
    if "source5" in out:
        source5 = np.asarray(out["source5"], dtype=np.float32)
        if source5.ndim == 1:
            source5 = source5.reshape((1, -1))
        if source5.shape[-1] != 5:
            source5 = np.asarray(expand_source_flags(out["source_flags"]), dtype=np.float32)
        elif source5.shape[0] == 1 and n != 1:
            source5 = np.repeat(source5, n, axis=0)
        source5 = np.array(source5, copy=True)
        if n:
            source5[:n, 4] = 1.0
        out["source5"] = source5
    out["failure_recovery"] = True
    return out


def _write_failure_memories(protected_sets: Sequence[Mapping], regressed_games: Sequence[str], cfg):
    regressed = {str(game) for game in regressed_games}
    if not regressed:
        return list(protected_sets), None, 0
    updated = list(protected_sets)
    writes = 0
    for atom_set in protected_sets:
        if atom_set.get("failure_recovery", False):
            continue
        if _first_game_key(atom_set) not in regressed:
            continue
        updated.append(_failure_copy(atom_set))
        writes += 1
    if writes == 0:
        return updated, None, 0
    bank = build_protected_bank(
        updated,
        _cfg_int(cfg, "hot_capacity", 4096),
        _cfg_int(cfg, "d_k", 128),
        _cfg_int(cfg, "d_c", 16),
        _cfg_int(cfg, "act_dim", ACT_DIM),
    )
    return updated, bank, writes


def _first_game_key(atom_set: Mapping) -> str:
    if "game" in atom_set:
        return str(atom_set["game"])
    game_id = np.asarray(atom_set.get("game_id", np.asarray([0], dtype=np.int32))).reshape(-1)
    if game_id.size == 0:
        return "0"
    return str(int(game_id[0]))


def _infer_dims(protected_sets: Sequence[Mapping], cfg) -> tuple[int, int, int]:
    d_k = _cfg_int(cfg, "d_k", 128)
    d_c = _cfg_int(cfg, "d_c", 16)
    act_dim = _cfg_int(cfg, "act_dim", ACT_DIM)
    for atom_set in protected_sets:
        keys = np.asarray(atom_set.get("keys", np.zeros((0, d_k), dtype=np.float32)))
        if keys.ndim == 1:
            keys = keys.reshape((1, -1))
        if keys.size:
            d_k = int(keys.shape[-1])
        context = np.asarray(atom_set.get("context", np.zeros((0, d_c), dtype=np.float32)))
        if context.ndim == 1:
            context = context.reshape((1, -1))
        if context.size:
            d_c = int(context.shape[-1])
        policy = np.asarray(
            atom_set.get("teacher_policy", np.zeros((0, act_dim), dtype=np.float32))
        )
        if policy.ndim == 1:
            policy = policy.reshape((1, -1))
        if policy.size:
            act_dim = int(policy.shape[-1])
    return d_k, d_c, act_dim


def _as_rows(atom_set: Mapping, name: str, n: int, width: int) -> np.ndarray:
    value = np.asarray(atom_set[name], dtype=np.float32)
    if value.ndim == 1:
        value = value.reshape((1, width))
    if value.shape == (1, width) and n != 1:
        value = np.repeat(value, n, axis=0)
    if value.shape != (n, width):
        raise ValueError(f"{name} must have shape [{n},{width}], got {value.shape}")
    return value


def _as_vector(atom_set: Mapping, name: str, n: int, dtype, default) -> np.ndarray:
    value = np.asarray(atom_set.get(name, default), dtype=dtype)
    if value.ndim == 0:
        return np.full((n,), value.item(), dtype=value.dtype)
    value = value.reshape(-1)
    if value.shape[0] == n:
        return value
    if value.shape[0] == 1:
        return np.repeat(value, n, axis=0)
    raise ValueError(f"{name} must have length {n}, got {value.shape[0]}")


def _records_for_guard(protected_sets: Sequence[Mapping], d_k: int, act_dim: int) -> list[dict]:
    records: list[dict] = []
    for set_index, atom_set in enumerate(protected_sets):
        keys = np.asarray(atom_set.get("keys", np.zeros((0, d_k), dtype=np.float32)), dtype=np.float32)
        if keys.ndim == 1:
            keys = keys.reshape((1, -1))
        if keys.shape[1:] != (int(d_k),):
            raise ValueError(f"keys must have width {int(d_k)}, got {keys.shape}")
        n = int(keys.shape[0])
        if n == 0:
            continue
        valid = _as_vector(atom_set, "valid", n, bool, np.ones((n,), dtype=bool)).astype(bool)
        game_ids = _as_vector(atom_set, "game_id", n, np.int32, np.zeros((n,), dtype=np.int32))
        teacher_policy = _as_rows(atom_set, "teacher_policy", n, act_dim)
        teacher_value = _as_vector(
            atom_set,
            "teacher_value",
            n,
            np.float32,
            np.zeros((n,), dtype=np.float32),
        )
        importance = _as_vector(
            atom_set,
            "importance",
            n,
            np.float32,
            np.ones((n,), dtype=np.float32),
        )
        game_key = _first_game_key(atom_set)
        for row in np.flatnonzero(valid):
            records.append(
                {
                    "set_index": int(set_index),
                    "row": int(row),
                    "game": game_key,
                    "key": keys[row],
                    "game_id": int(game_ids[row]),
                    "teacher_policy": teacher_policy[row],
                    "teacher_value": float(teacher_value[row]),
                    "importance": float(importance[row]),
                }
            )
    records.sort(key=lambda item: (-item["importance"], item["set_index"], item["row"]))
    return records


def _risk_for_games(games: Sequence[str], risk_scores: Mapping | None) -> dict[str, float]:
    risk_scores = {} if risk_scores is None else risk_scores
    risk = {}
    for game in games:
        value = risk_scores.get(game, risk_scores.get(str(game), 1.0))
        risk[str(game)] = max(float(value), 0.0)
    if not risk or sum(risk.values()) <= 0.0:
        return {str(game): 1.0 for game in games}
    return risk


def _review_risk_for_games(
    games: Sequence[str],
    risk_scores: Mapping[str, float],
    review_boosts: Mapping[str, float],
) -> dict[str, float]:
    combined = {
        str(game): float(risk_scores.get(str(game), 0.0)) + float(review_boosts.get(str(game), 0.0))
        for game in games
    }
    return _risk_for_games(games, combined)


def _recompute_risk(
    games: Sequence[str],
    best_scores: Mapping[str, float],
    final_scores: Mapping[str, float],
    random_scores: Mapping[str, float],
    risk_boosts: Mapping[str, float],
) -> dict[str, float]:
    out: dict[str, float] = {}
    for game in games:
        game = str(game)
        if game not in best_scores or game not in final_scores or game not in random_scores:
            continue
        out[game] = forgetting_risk(
            best_scores[game],
            final_scores[game],
            random_scores[game],
        ) + float(risk_boosts.get(game, 0.0))
    return out


def _quotas(lambda_by_game: Mapping[str, float], sample_atoms: int) -> dict[str, int]:
    games = list(lambda_by_game)
    if not games:
        return {}
    total = sum(float(lambda_by_game[game]) for game in games)
    if total <= 0.0:
        raw = {game: float(sample_atoms) / float(len(games)) for game in games}
    else:
        raw = {game: float(sample_atoms) * float(lambda_by_game[game]) / total for game in games}
    quotas = {game: int(np.floor(value)) for game, value in raw.items()}
    remaining = int(sample_atoms) - sum(quotas.values())
    order = sorted(games, key=lambda game: (-(raw[game] - quotas[game]), game))
    for game in order[:remaining]:
        quotas[game] += 1
    return quotas


def _select_records(records: Sequence[dict], lambda_by_game: Mapping[str, float], sample_atoms: int) -> list[dict]:
    quotas = _quotas(lambda_by_game, sample_atoms)
    selected_ids: set[int] = set()
    selected: list[dict] = []
    for game, quota in quotas.items():
        if quota <= 0:
            continue
        for idx, record in enumerate(records):
            if len([item for item in selected if item["game"] == game]) >= quota:
                break
            if idx in selected_ids or record["game"] != game:
                continue
            selected_ids.add(idx)
            selected.append(record)
    if len(selected) < int(sample_atoms):
        for idx, record in enumerate(records):
            if idx in selected_ids:
                continue
            selected_ids.add(idx)
            selected.append(record)
            if len(selected) >= int(sample_atoms):
                break
    selected.sort(key=lambda item: (-item["importance"], item["set_index"], item["row"]))
    return selected[: int(sample_atoms)]


def make_guard(
    protected_sets: Sequence[Mapping],
    risk_scores: Mapping | None,
    cfg,
    *,
    project=True,
    sample_atoms=256,
) -> dict:
    """Build the fixed-shape latent memory guard used by the fast trainer."""
    protected_sets = list(protected_sets)
    sample_atoms = int(sample_atoms)
    if sample_atoms <= 0:
        raise ValueError("sample_atoms must be positive")
    cfg = FastLMConfig() if cfg is None else cfg
    d_k, d_c, act_dim = _infer_dims(protected_sets, cfg)
    lambda_total = _cfg_float(cfg, "guard_lambda_total", 1.0)
    capacity = _cfg_int(cfg, "hot_capacity", 4096)

    records = _records_for_guard(protected_sets, d_k, act_dim)
    games = list(dict.fromkeys(record["game"] for record in records))
    risk = _risk_for_games(games, risk_scores)
    lambda_by_game = allocate_guard(risk, lambda_total)  # spec §16
    selected = _select_records(records, lambda_by_game, sample_atoms)

    keys = np.zeros((sample_atoms, d_k), dtype=np.float32)
    game_id = np.zeros((sample_atoms,), dtype=np.int32)
    teacher_policy = np.zeros((sample_atoms, act_dim), dtype=np.float32)
    teacher_value = np.zeros((sample_atoms,), dtype=np.float32)
    eps = np.full((sample_atoms,), _cfg_float(cfg, "eps_policy", 0.05), dtype=np.float32)
    weight = np.zeros((sample_atoms,), dtype=np.float32)
    for out_i, record in enumerate(selected):
        keys[out_i] = record["key"]
        game_id[out_i] = int(record["game_id"])
        teacher_policy[out_i] = record["teacher_policy"]
        teacher_value[out_i] = float(record["teacher_value"])
        weight[out_i] = 1.0

    atoms = {
        "keys": jnp.asarray(keys),
        "game_id": jnp.asarray(game_id),
        "teacher_policy": jnp.asarray(teacher_policy),
        "teacher_value": jnp.asarray(teacher_value),
        "eps": jnp.asarray(eps),
        "weight": jnp.asarray(weight),
    }
    bank = build_protected_bank(protected_sets, capacity, d_k, d_c, act_dim)
    return {
        "atoms": atoms,
        "bank": bank,
        "lambda_total": float(lambda_total),
        "lambda_by_game": dict(lambda_by_game),
        "kappa": _cfg_float(cfg, "guard_kappa", 0.75),
        "lambda_v": _cfg_float(cfg, "guard_lambda_v", 1.0),
        "stability_alpha": _cfg_float(cfg, "guard_stability_alpha", 10.0),
        "omega": None,
        "project": bool(project),
        "rho_omega": _cfg_float(cfg, "guard_rho_omega", 0.99),
    }


def _empty_bank(cfg):
    return build_protected_bank(
        [],
        _cfg_int(cfg, "hot_capacity", 4096),
        _cfg_int(cfg, "d_k", 128),
        _cfg_int(cfg, "d_c", 16),
        _cfg_int(cfg, "act_dim", ACT_DIM),
    )


def _make_visual_aux(store: VisualSentinelStore, protected_bank, cfg, seed: int):
    if len(store) == 0:
        return None
    batch_size = _cfg_int(cfg, "visual_sentinel_batch", 64)
    n_neg = _cfg_int(cfg, "retr_n_neg", 16)
    sent_batch = store.batch(batch_size, seed=int(seed))
    try:
        align_batch = build_align_batch(
            sent_batch,
            protected_bank,
            n_neg=n_neg,
            batch_size=batch_size,
            seed=int(seed) + 1,
        )
    except ValueError:
        return None
    return {
        "sent_batch": sent_batch,
        "align_batch": align_batch,
        "bank": protected_bank,
        "lambda_visual": _cfg_float(cfg, "lambda_visual", 0.5),
        "lambda_key": _cfg_float(cfg, "lambda_key", 1.0),
        "lambda_retr": _cfg_float(cfg, "lambda_retr", 0.1),
        "visual_lambda_v": _cfg_float(cfg, "visual_lambda_v", 1.0),
        "tau": _cfg_float(cfg, "retr_tau", 0.1),
    }


def _random_params(game_index: int, n_games: int, cfg, seed: int):
    return mem_init(
        jax.random.PRNGKey(int(seed) + 10_000 + int(game_index)),
        int(n_games),
        _cfg_int(cfg, "hot_capacity", 4096),
        d_k=_cfg_int(cfg, "d_k", 128),
        d_c=_cfg_int(cfg, "d_c", 16),
        d_m=_cfg_int(cfg, "d_m", 128),
        act_dim=_cfg_int(cfg, "act_dim", ACT_DIM),
        top_k=_cfg_int(cfg, "top_k", 16),
    )


def _blend_for_ablation(ablation: str) -> bool:
    return ablation not in {"no_memory_read", "plain_ppo"}


def continual_living_memory(
    games: Sequence[str],
    n_games,
    cfg,
    seed,
    *,
    ablation="full",
    per_game_steps=None,
) -> dict:
    """Train a game sequence and report random-normalized retention."""
    if ablation not in ABLATIONS:
        raise ValueError(f"unknown ablation {ablation!r}")
    cfg = FastLMConfig() if cfg is None else cfg
    per_game_total = _configured_total_timesteps(cfg, per_game_steps)
    n_blocks = max(1, _cfg_int(cfg, "n_blocks", 4))
    block_steps = [
        ((block_index + 1) * int(per_game_total)) // int(n_blocks)
        - (block_index * int(per_game_total)) // int(n_blocks)
        for block_index in range(int(n_blocks))
    ]
    review_steps_frac = max(0.0, _cfg_float(cfg, "review_steps_frac", 0.15))
    review_steps = [int(steps * review_steps_frac) for steps in block_steps]
    audit_every_blocks = max(1, _cfg_int(cfg, "audit_every_blocks", 1))
    gate_delta_frac = _cfg_float(cfg, "gate_delta_frac", 0.1)
    lambda_review = _cfg_float(cfg, "lambda_review", 0.5)
    games = [str(game) for game in games]
    game_to_id = {game: idx for idx, game in enumerate(games)}
    n_embed = int(n_games)
    n_seq = len(games)
    return_matrix = np.full((n_seq, n_seq), np.nan, dtype=np.float32)
    if n_seq == 0:
        return {
            "return_matrix": return_matrix,
            "best_scores": {},
            "random_scores": {},
            "final_scores": {},
            "retention": {},
            "mean_retention": 0.0,
            "worst_retention": 0.0,
            "ablation": str(ablation),
            "games": games,
            "gate_rejections": 0,
            "gate_decisions": [],
            "review_counts": {},
            "risk_scores": {},
            "failure_memory_writes": 0,
        }

    random_bank = _empty_bank(cfg)
    random_scores = {}
    for game_id, game in enumerate(games):
        random_scores[game] = float(
            eval_living_memory(
                _random_params(game_id, n_embed, cfg, int(seed)),
                game,
                game_id,
                random_bank,
                cfg=cfg,
                seed=int(seed) + 20_000 + game_id,
                blend=False,
            )
        )

    params = None
    ema_params = None
    hot_bank = None
    value_norms = {}
    protected_sets: list[dict] = []
    risk: dict[str, float] = {}
    risk_boosts: dict[str, float] = {}
    review_boosts: dict[str, float] = {}
    best_scores: dict[str, float] = {}
    final_scores: dict[str, float] = {}
    current_val_best: dict[str, float] = {}
    protected_bank = random_bank
    blend = _blend_for_ablation(str(ablation))
    visual_store = VisualSentinelStore(per_game=_cfg_int(cfg, "visual_sentinels_per_game", 64))
    gate_rejections = 0
    gate_decisions: list[dict] = []
    review_counts: dict[str, int] = {}
    failure_memory_writes = 0

    for game_id, game in enumerate(games):
        value_norms.setdefault(game, RunningValueNorm())

        def block_guard(block_index: int, seed_offset: int = 0):
            if game_id == 0 or ablation in {"no_conservation", "plain_ppo"}:
                return None, None
            guard = make_guard(
                protected_sets,
                risk,
                cfg,
                project=(ablation != "no_projection"),
                sample_atoms=_cfg_int(cfg, "guard_sample_atoms", 256),
            )
            aux = _make_visual_aux(
                visual_store,
                protected_bank,
                cfg,
                int(seed) + 30_000 + game_id * 1_000 + int(block_index) + int(seed_offset),
            )
            return guard, aux

        for block_index in range(n_blocks):
            snapshot = _snapshot_state(params, ema_params, value_norms, hot_bank)
            guard, aux = block_guard(block_index)
            block_cfg = _cfg_with_total_timesteps(cfg, block_steps[block_index])
            review_cfg = _cfg_with_total_timesteps(cfg, max(1, review_steps[block_index]))

            result = train_living_memory_fast(
                game,
                game_id,
                n_embed,
                block_cfg,
                int(seed) + 60_000 + game_id * 1_000 + block_index,
                init_params=params,
                hot_bank=hot_bank,
                ema_params=ema_params,
                value_norm=value_norms.get(game),
                guard=guard,
                aux=aux,
            )
            params = result["params"]
            ema_params = result.get("ema_params", params)
            hot_bank = result.get("hot_bank", hot_bank)
            if "value_norm" in result:
                value_norms[game] = result.get("value_norm")

            if game_id > 0 and review_steps[block_index] > 0 and ablation not in {"no_review", "plain_ppo"}:
                prior_games = games[:game_id]
                review_u = _review_risk_for_games(prior_games, risk, review_boosts)
                rng = np.random.default_rng(int(seed) + 70_000 + game_id * 1_000 + block_index)
                # spec §18
                review_games = sample_review_games(review_u, 1, rng)
                for review_game in review_games:
                    review_game = str(review_game)
                    value_norms.setdefault(review_game, RunningValueNorm())
                    review_game_id = int(game_to_id[review_game])
                    review_guard, review_aux = block_guard(block_index, seed_offset=100)
                    # spec §18
                    review_result = train_living_memory_fast(
                        review_game,
                        review_game_id,
                        n_embed,
                        review_cfg,
                        int(seed) + 80_000 + game_id * 1_000 + block_index,
                        init_params=params,
                        hot_bank=hot_bank,
                        ema_params=ema_params,
                        value_norm=value_norms.get(review_game),
                        guard=review_guard,
                        aux=review_aux,
                    )
                    params = review_result["params"]
                    ema_params = review_result.get("ema_params", params)
                    hot_bank = review_result.get("hot_bank", hot_bank)
                    if "value_norm" in review_result:
                        value_norms[review_game] = review_result.get("value_norm")
                    review_counts[review_game] = review_counts.get(review_game, 0) + 1

            if (
                game_id > 0
                and ablation not in {"no_gate", "plain_ppo"}
                and block_index % audit_every_blocks == 0
            ):
                audit_guard, audit_aux = block_guard(block_index, seed_offset=200)
                protected_audit = {}
                audited_scores = {}
                # spec §19
                for eval_id, eval_game in enumerate(games[:game_id]):
                    score = float(
                        eval_living_memory(
                            params,
                            eval_game,
                            eval_id,
                            protected_bank,
                            cfg=cfg,
                            seed=int(seed) + 90_000 + game_id * n_seq + block_index * n_seq + eval_id,
                            blend=blend,
                        )
                    )
                    audited_scores[eval_game] = score
                    protected_audit[eval_game] = {
                        "current": score,
                        "best": best_scores[eval_game],
                        "random": random_scores[eval_game],
                    }

                current_score = float(
                    eval_living_memory(
                        params,
                        game,
                        game_id,
                        protected_bank,
                        cfg=cfg,
                        seed=int(seed) + 95_000 + game_id * 1_000 + block_index,
                        blend=blend,
                    )
                )
                val_best = current_val_best.get(game)
                current_gate = {
                    "progress": current_score - random_scores[game],
                    "val_current": current_score,
                    "val_best": val_best,
                    "random": random_scores[game],
                    "best": current_score if val_best is None else val_best,
                }
                delta_abs = {
                    old_game: gate_delta_frac
                    * max(best_scores[old_game] - random_scores[old_game], 1.0e-6)
                    for old_game in games[:game_id]
                }
                violation_rate = _audit_violation_rate(params, audit_guard, cfg)
                retrieval_alignment = _audit_retrieval_alignment(params, audit_aux, cfg)
                gate_cfg = GateConfig(
                    r_min=_cfg_float(cfg, "gate_r_min", 0.9),
                    current_regress_frac=_cfg_float(cfg, "gate_current_regress_frac", gate_delta_frac),
                    delta_abs=delta_abs,
                    max_violation_rate=_cfg_float(cfg, "gate_max_violation_rate", GateConfig().max_violation_rate),
                    retrieval_floor=_cfg_float(cfg, "gate_retrieval_floor", GateConfig().retrieval_floor),
                    min_new_progress=_cfg_float(cfg, "gate_min_new_progress", GateConfig().min_new_progress),
                )
                # spec §19
                decision = evaluate_gate(
                    protected=protected_audit,
                    current=current_gate,
                    violation_rate=violation_rate,
                    retrieval_alignment=retrieval_alignment,
                    cfg=gate_cfg,
                )
                gate_decisions.append(
                    {
                        "game": game,
                        "game_id": int(game_id),
                        "block": int(block_index),
                        "accept": bool(decision.accept),
                        "regressed_games": list(decision.regressed_games),
                        "reasons": list(decision.reasons),
                        "violation_rate": float(violation_rate),
                        "retrieval_alignment": float(retrieval_alignment),
                    }
                )
                if not decision.accept:
                    params, ema_params, value_norms, hot_bank = _restore_snapshot(snapshot)
                    actions = on_reject_actions(decision)
                    for rejected_game in actions.get("increase_risk_games", []):
                        rejected_game = str(rejected_game)
                        risk_boosts[rejected_game] = risk_boosts.get(rejected_game, 0.0) + _cfg_float(
                            cfg, "gate_risk_boost", 1.0
                        )
                    for rejected_game in actions.get("increase_review_games", []):
                        rejected_game = str(rejected_game)
                        review_boosts[rejected_game] = review_boosts.get(
                            rejected_game, 0.0
                        ) + _cfg_float(cfg, "gate_review_boost", 1.0)
                    if actions.get("write_failure_memories", False):
                        updated_sets, updated_bank, writes = _write_failure_memories(
                            protected_sets,
                            decision.regressed_games,
                            cfg,
                        )
                        if writes:
                            protected_sets = updated_sets
                            protected_bank = updated_bank
                            failure_memory_writes += int(writes)
                    gate_rejections += 1
                    risk = _recompute_risk(
                        games[:game_id],
                        best_scores,
                        final_scores,
                        random_scores,
                        risk_boosts,
                    )
                else:
                    for old_game, score in audited_scores.items():
                        final_scores[old_game] = score
                        if score > best_scores[old_game]:
                            best_scores[old_game] = score
                    current_val_best[game] = max(
                        current_score,
                        current_val_best.get(game, -float("inf")),
                    )
                    risk = _recompute_risk(
                        games[:game_id],
                        best_scores,
                        final_scores,
                        random_scores,
                        risk_boosts,
                    )

        sent_set = collect_visual_sentinels(
            params,
            ema_params,
            value_norms[game],
            game,
            game_id,
            cfg=cfg,
            seed=int(seed) + 50_000 + game_id,
            n=_cfg_int(cfg, "visual_sentinels_per_game", 64),
        )
        visual_store.add_set(sent_set)

        protected = dict(
            certify_protected_memories(
                params,
                ema_params,
                value_norms[game],
                game,
                game_id,
                cfg=cfg,
                seed=int(seed) + 100 + game_id,
            )
        )
        protected["game"] = game
        protected_sets.append(protected)
        protected_bank = build_protected_bank(
            protected_sets,
            _cfg_int(cfg, "hot_capacity", 4096),
            _cfg_int(cfg, "d_k", 128),
            _cfg_int(cfg, "d_c", 16),
            _cfg_int(cfg, "act_dim", ACT_DIM),
        )

        for eval_id, eval_game in enumerate(games[: game_id + 1]):
            score = float(
                eval_living_memory(
                    params,
                    eval_game,
                    eval_id,
                    protected_bank,
                    cfg=cfg,
                    seed=int(seed) + 1_000 + game_id * n_seq + eval_id,
                    blend=blend,
                )
            )
            return_matrix[game_id, eval_id] = score
            final_scores[eval_game] = score
            if eval_game == game:
                best_scores[eval_game] = score

        risk = _recompute_risk(
            games[: game_id + 1],
            best_scores,
            final_scores,
            random_scores,
            risk_boosts,
        )

    retention = {
        game: normalized_retention(final_scores[game], best_scores[game], random_scores[game])
        for game in games
    }
    retention_values = list(retention.values())
    return {
        "return_matrix": return_matrix,
        "best_scores": dict(best_scores),
        "random_scores": dict(random_scores),
        "final_scores": dict(final_scores),
        "retention": retention,
        "mean_retention": float(np.mean(np.asarray(retention_values, dtype=np.float32))),
        "worst_retention": float(np.min(np.asarray(retention_values, dtype=np.float32))),
        "ablation": str(ablation),
        "games": games,
        "visual_sentinels": len(visual_store),
        "gate_rejections": int(gate_rejections),
        "gate_decisions": gate_decisions,
        "review_counts": dict(review_counts),
        "risk_scores": dict(risk),
        "review_boosts": dict(review_boosts),
        "failure_memory_writes": int(failure_memory_writes),
        "block_steps": [int(steps) for steps in block_steps],
        "review_steps": [int(steps) for steps in review_steps],
        "lambda_review": float(lambda_review),
    }


__all__ = ["ABLATIONS", "continual_living_memory", "make_guard"]
