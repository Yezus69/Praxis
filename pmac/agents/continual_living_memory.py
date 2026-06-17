"""Guard-aware continual Living Memory driver for PMA-C retention proofs."""

from __future__ import annotations

import copy
from collections.abc import Mapping, Sequence
from dataclasses import is_dataclass, replace

import jax
import jax.numpy as jnp
import numpy as np

from pmac.agents.atari_mem_net import mem_init
from pmac.agents.living_memory_eval import (
    build_protected_bank,
    certify_protected_memories,
    eval_living_memory,
)
from pmac.agents.ppo_living_memory_fast import FastLMConfig, train_living_memory_fast
from pmac.envs.atari_envpool import ACT_DIM
from pmac.evaluation import normalized_retention
from pmac.guard_schedule import allocate_guard, forgetting_risk
from pmac.memory.sentinels_visual import (
    VisualSentinelStore,
    build_align_batch,
    collect_visual_sentinels,
)


ABLATIONS = {"full", "no_conservation", "no_projection", "no_memory_read", "plain_ppo"}


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
    train_cfg = _cfg_with_total_timesteps(cfg, per_game_steps)
    games = [str(game) for game in games]
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
    value_norms = {}
    protected_sets: list[dict] = []
    risk: dict[str, float] = {}
    best_scores: dict[str, float] = {}
    final_scores: dict[str, float] = {}
    protected_bank = random_bank
    blend = _blend_for_ablation(str(ablation))
    visual_store = VisualSentinelStore(per_game=_cfg_int(cfg, "visual_sentinels_per_game", 64))

    for game_id, game in enumerate(games):
        if game_id == 0 or ablation in {"no_conservation", "plain_ppo"}:
            guard = None
        else:
            guard = make_guard(
                protected_sets,
                risk,
                cfg,
                project=(ablation != "no_projection"),
                sample_atoms=_cfg_int(cfg, "guard_sample_atoms", 256),
            )
        aux = None if guard is None else _make_visual_aux(
            visual_store,
            protected_bank,
            cfg,
            int(seed) + 30_000 + game_id,
        )

        result = train_living_memory_fast(
            game,
            game_id,
            n_embed,
            train_cfg,
            int(seed) + game_id,
            init_params=params,
            ema_params=ema_params,
            value_norm=value_norms.get(game),
            guard=guard,
            aux=aux,
        )
        params = result["params"]
        ema_params = result.get("ema_params", params)
        value_norms[game] = result.get("value_norm")

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

        risk = {
            old_game: forgetting_risk(
                best_scores[old_game],
                final_scores[old_game],
                random_scores[old_game],
            )
            for old_game in games[: game_id + 1]
        }

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
    }


__all__ = ["ABLATIONS", "continual_living_memory", "make_guard"]
