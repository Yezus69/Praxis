"""Protected-memory certification and deployment eval for living-memory Atari."""

from __future__ import annotations

import gc
import math
from collections.abc import Mapping, Sequence

import jax
import jax.numpy as jnp
import numpy as np

from pmac.agents.atari_mem_net import MemAtariActorCritic, mem_apply, mem_apply_key
from pmac.agents.ppo_atari import gae
from pmac.envs.atari_envpool import EpisodeReturnTracker, make_eval_env, make_train_env
from pmac.memory import SourceFlag
from pmac.memory.reader import expand_source_flags
from pmac.memory.runtime import default_retrieval_hp
from pmac.memory.write import EPS, RunningStats, teacher_targets


def _cfg_get(cfg, name: str, default):
    if isinstance(cfg, Mapping):
        return cfg.get(name, default)
    return getattr(cfg, name, default)


def _cfg_int(cfg, name: str, default: int) -> int:
    return int(_cfg_get(cfg, name, default))


def _cfg_float(cfg, name: str, default: float) -> float:
    return float(_cfg_get(cfg, name, default))


def _infer_dims(params):
    n_games, d_c = params["game_embed"]["embedding"].shape
    d_k = params["key_head"]["kernel"].shape[-1]
    d_m = params["wv"]["kernel"].shape[-1]
    act_dim = params["policy_head"]["kernel"].shape[-1]
    return int(n_games), int(d_k), int(d_c), int(d_m), int(act_dim)


def _value_norm_mu_sigma(value_norm) -> tuple[float, float]:
    if value_norm is None:
        return 0.0, 1.0
    if isinstance(value_norm, Mapping):
        mu = value_norm.get("mu", 0.0)
        sigma = value_norm.get("sigma", 1.0)
        return float(np.asarray(mu)), max(float(np.asarray(sigma)), EPS)
    if hasattr(value_norm, "mu") and hasattr(value_norm, "sigma"):
        return float(value_norm.mu()), max(float(value_norm.sigma()), EPS)
    raise TypeError("value_norm must be None, a mapping, or RunningValueNorm-like")


def _close_env(env):
    try:
        env.close()
    except Exception:
        pass
    del env
    gc.collect()


def _normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + EPS)


def _empty_bank(capacity: int, d_k: int, d_c: int, act_dim: int) -> dict:
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


def _context_embeddings(params, game_ids):
    n_games, d_k, d_c, d_m, act_dim = _infer_dims(params)
    net = MemAtariActorCritic(n_games=n_games, d_k=d_k, d_c=d_c, d_m=d_m, act_dim=act_dim)
    return net.apply(
        {"params": params},
        jnp.asarray(game_ids, dtype=jnp.int32),
        method=MemAtariActorCritic.context,
    )


def _top_return_indices(returns, n_protected: int):
    returns = np.asarray(returns, dtype=np.float32).reshape(-1)
    finite = np.isfinite(returns)
    if int(n_protected) <= 0 or not np.any(finite):
        return np.zeros((0,), dtype=np.int64), np.zeros_like(returns)

    stats = RunningStats()
    stats.update(returns[finite])
    ret_hat = stats.normalize(returns)
    scores = np.where(np.isfinite(ret_hat), ret_hat, -np.inf).astype(np.float32)
    keep_n = min(int(n_protected), int(np.sum(np.isfinite(scores))))
    order = np.argsort(-scores, kind="mergesort")[:keep_n]
    return order.astype(np.int64), scores


def _source5_from_set(atom_set: Mapping, n: int) -> np.ndarray:
    if "source5" in atom_set:
        source5 = np.asarray(atom_set["source5"], dtype=np.float32)
        if source5.ndim == 1:
            source5 = source5.reshape((1, -1))
        if source5.shape == (1, 5) and n != 1:
            source5 = np.repeat(source5, n, axis=0)
        if source5.shape != (n, 5):
            raise ValueError(f"source5 must have shape [{n},5], got {source5.shape}")
        return source5
    flags = np.asarray(atom_set.get("source_flags", np.zeros((n,), dtype=np.int32)), dtype=np.int32)
    flags = np.broadcast_to(flags.reshape(-1), (n,))
    return np.asarray(expand_source_flags(flags), dtype=np.float32)


def _as_rows(atom_set: Mapping, name: str, n: int, width: int, dtype=np.float32) -> np.ndarray:
    value = np.asarray(atom_set[name], dtype=dtype)
    if value.ndim == 1 and width > 1:
        value = value.reshape((1, width))
    if value.shape == (1, width) and n != 1:
        value = np.repeat(value, n, axis=0)
    if value.shape != (n, width):
        raise ValueError(f"{name} must have shape [{n},{width}], got {value.shape}")
    return value


def _as_vector(atom_set: Mapping, name: str, n: int, dtype, default=None) -> np.ndarray:
    if name in atom_set:
        value = np.asarray(atom_set[name], dtype=dtype)
    else:
        value = np.asarray(default, dtype=dtype)
    if value.ndim == 0:
        return np.full((n,), value.item(), dtype=value.dtype)
    value = value.reshape(-1)
    if value.shape[0] == n:
        return value
    if value.shape[0] == 1:
        return np.repeat(value, n, axis=0)
    raise ValueError(f"{name} must have length {n}, got {value.shape[0]}")


def _concat_protected_sets(
    protected_sets: Sequence[Mapping], d_k: int, d_c: int, act_dim: int
) -> dict[str, np.ndarray]:
    parts = {
        "keys": [],
        "context": [],
        "teacher_policy": [],
        "teacher_value": [],
        "importance": [],
        "game_id": [],
        "source5": [],
        "age": [],
    }
    for atom_set in protected_sets:
        keys = np.asarray(atom_set.get("keys", np.zeros((0, d_k), dtype=np.float32)), dtype=np.float32)
        if keys.ndim == 1:
            keys = keys.reshape((1, -1))
        if keys.shape[1:] != (int(d_k),):
            raise ValueError(f"keys must have width {int(d_k)}, got {keys.shape}")
        n = int(keys.shape[0])
        if n == 0:
            continue

        valid = np.asarray(atom_set.get("valid", np.ones((n,), dtype=bool)), dtype=bool).reshape(-1)
        if valid.shape[0] != n:
            raise ValueError(f"valid must have length {n}, got {valid.shape[0]}")

        rows = {
            "keys": _normalize_rows(keys),
            "context": _as_rows(atom_set, "context", n, d_c),
            "teacher_policy": _as_rows(atom_set, "teacher_policy", n, act_dim),
            "teacher_value": _as_vector(atom_set, "teacher_value", n, np.float32),
            "importance": _as_vector(atom_set, "importance", n, np.float32),
            "game_id": _as_vector(atom_set, "game_id", n, np.int32),
            "source5": _source5_from_set(atom_set, n),
            "age": _as_vector(atom_set, "age", n, np.float32, default=np.zeros((n,), dtype=np.float32)),
        }
        for name, value in rows.items():
            parts[name].append(value[valid])

    if not parts["keys"]:
        return {
            "keys": np.zeros((0, d_k), dtype=np.float32),
            "context": np.zeros((0, d_c), dtype=np.float32),
            "teacher_policy": np.zeros((0, act_dim), dtype=np.float32),
            "teacher_value": np.zeros((0,), dtype=np.float32),
            "importance": np.zeros((0,), dtype=np.float32),
            "game_id": np.zeros((0,), dtype=np.int32),
            "source5": np.zeros((0, 5), dtype=np.float32),
            "age": np.zeros((0,), dtype=np.float32),
        }
    return {name: np.concatenate(values, axis=0) for name, values in parts.items()}


def _per_game_capacity_keep(arrays: Mapping[str, np.ndarray], capacity: int) -> np.ndarray:
    n = int(arrays["game_id"].shape[0])
    if n <= int(capacity):
        return np.arange(n, dtype=np.int64)

    game_ids = np.asarray(arrays["game_id"], dtype=np.int32)
    importance = np.asarray(arrays["importance"], dtype=np.float32)
    games = list(dict.fromkeys(int(game) for game in game_ids.tolist()))
    if not games:
        return np.zeros((0,), dtype=np.int64)

    base = int(capacity) // len(games)
    budgets = {game: min(base, int(np.sum(game_ids == game))) for game in games}
    remaining = int(capacity) - int(sum(budgets.values()))
    while remaining > 0:
        best_game = None
        best_score = -np.inf
        for game in games:
            idx = np.flatnonzero(game_ids == game)
            if budgets[game] >= int(idx.shape[0]):
                continue
            order = idx[np.argsort(-importance[idx], kind="mergesort")]
            score = float(importance[order[budgets[game]]])
            if score > best_score:
                best_score = score
                best_game = game
        if best_game is None:
            break
        budgets[best_game] += 1
        remaining -= 1

    selected = np.zeros((n,), dtype=bool)
    for game in games:
        idx = np.flatnonzero(game_ids == game)
        if budgets[game] <= 0:
            continue
        order = idx[np.argsort(-importance[idx], kind="mergesort")[: budgets[game]]]
        selected[order] = True
    return np.flatnonzero(selected).astype(np.int64)


def _select_greedy_actions(mem_out: Mapping[str, jnp.ndarray], *, blend: bool):
    """Select greedy actions from the deployment blend or no-read ablation logits."""
    key = "logits_final" if bool(blend) else "logits_net"
    return jnp.argmax(jnp.asarray(mem_out[key]), axis=-1).astype(jnp.int32)


def certify_protected_memories(
    params,
    ema_params,
    value_norm,
    game,
    game_id,
    *,
    cfg,
    seed,
    n_protected=512,
    rollout_steps=4,
) -> dict:
    """Certify protected high-return teacher atoms from a trained game policy."""
    _, d_k, d_c, _, act_dim = _infer_dims(params)
    num_envs = _cfg_int(cfg, "cert_num_envs", 64)
    num_steps = _cfg_int(cfg, "num_steps", 128)
    total_steps = int(rollout_steps) * num_steps
    if num_envs <= 0:
        raise ValueError("cfg.num_envs must be positive")
    if total_steps <= 0:
        raise ValueError("rollout_steps*cfg.num_steps must be positive")

    top_k = _cfg_int(cfg, "top_k", 1)
    if top_k <= 0:
        raise ValueError("cfg.top_k must be positive")
    hp = default_retrieval_hp(top_k)
    empty_bank = _empty_bank(top_k, d_k, d_c, act_dim)
    mu_g, sigma_g = _value_norm_mu_sigma(value_norm)

    env = make_train_env(str(game), num_envs, int(seed))
    obs, _ = env.reset()
    obs = np.asarray(obs, dtype=np.uint8)
    obs_buf = np.zeros((total_steps,) + tuple(obs.shape), dtype=np.uint8)
    rewards_buf = np.zeros((total_steps, num_envs), dtype=np.float32)
    dones_buf = np.zeros((total_steps, num_envs), dtype=np.float32)
    logits_buf = np.zeros((total_steps, num_envs, act_dim), dtype=np.float32)
    values_buf = np.zeros((total_steps, num_envs), dtype=np.float32)
    game_id_vec = jnp.full((num_envs,), int(game_id), dtype=jnp.int32)

    for step in range(total_steps):
        obs_buf[step] = obs
        out = mem_apply(params, obs, game_id_vec, empty_bank, hp, mu_g, sigma_g)
        logits = np.asarray(jax.device_get(out["logits_net"]), dtype=np.float32)
        values = np.asarray(jax.device_get(out["v_net"]), dtype=np.float32)
        actions = np.argmax(logits, axis=-1).astype(np.int32)
        logits_buf[step] = logits
        values_buf[step] = values

        obs, rewards, terminated, truncated, _ = env.step(actions)
        obs = np.asarray(obs, dtype=np.uint8)
        rewards_buf[step] = np.asarray(rewards, dtype=np.float32)
        done = np.logical_or(np.asarray(terminated, dtype=bool), np.asarray(truncated, dtype=bool))
        dones_buf[step] = done.astype(np.float32)

    last_value = mem_apply(params, obs, game_id_vec, empty_bank, hp, mu_g, sigma_g)["v_net"]
    _, returns = gae(
        jnp.asarray(rewards_buf),
        jnp.asarray(dones_buf),
        jnp.asarray(values_buf),
        last_value,
        _cfg_float(cfg, "gamma", 0.99),
        _cfg_float(cfg, "gae_lambda", 0.95),
    )
    flat_returns = np.asarray(jax.device_get(returns), dtype=np.float32).reshape(-1)
    keep, ret_scores = _top_return_indices(flat_returns, int(n_protected))
    _close_env(env)
    if int(keep.shape[0]) == 0:
        return {
            "keys": np.zeros((0, d_k), dtype=np.float32),
            "context": np.zeros((0, d_c), dtype=np.float32),
            "teacher_policy": np.zeros((0, act_dim), dtype=np.float32),
            "teacher_value": np.zeros((0,), dtype=np.float32),
            "importance": np.zeros((0,), dtype=np.float32),
            "game_id": np.zeros((0,), dtype=np.int32),
            "source_flags": np.zeros((0,), dtype=np.int32),
            "age": np.zeros((0,), dtype=np.float32),
        }

    flat_obs = obs_buf.reshape((total_steps * num_envs,) + tuple(obs_buf.shape[2:]))
    flat_logits = logits_buf.reshape((total_steps * num_envs, act_dim))
    keys = np.asarray(jax.device_get(mem_apply_key(ema_params, flat_obs[keep])), dtype=np.float32)
    teacher_policy, teacher_value = teacher_targets(
        flat_logits[keep],
        flat_returns[keep],
        mu_g,
        sigma_g,
        temperature=_cfg_float(cfg, "teacher_temperature", 1.0),
    )
    game_ids = np.full((int(keep.shape[0]),), int(game_id), dtype=np.int32)
    contexts = np.asarray(jax.device_get(_context_embeddings(params, game_ids)), dtype=np.float32)

    finite_scores = ret_scores[np.isfinite(ret_scores)]
    score_floor = float(np.min(finite_scores)) if finite_scores.size else 0.0
    stored_importance = np.maximum(ret_scores[keep] - score_floor + EPS, EPS).astype(np.float32)
    flags = np.full(
        (int(keep.shape[0]),),
        int(SourceFlag.SENTINEL | SourceFlag.HIGH_RETURN),
        dtype=np.int32,
    )
    return {
        "keys": keys.astype(np.float32),
        "context": contexts.astype(np.float32),
        "teacher_policy": teacher_policy.astype(np.float32),
        "teacher_value": teacher_value.astype(np.float32),
        "importance": stored_importance,
        "game_id": game_ids,
        "source_flags": flags,
        "age": np.zeros((int(keep.shape[0]),), dtype=np.float32),
    }


def build_protected_bank(protected_sets: Sequence[Mapping], capacity, d_k, d_c, A) -> dict:
    """Build a reader-compatible fixed-capacity protected memory bank."""
    capacity = int(capacity)
    d_k = int(d_k)
    d_c = int(d_c)
    act_dim = int(A)
    if capacity <= 0:
        raise ValueError("capacity must be positive")

    arrays = _concat_protected_sets(protected_sets, d_k, d_c, act_dim)
    keep = _per_game_capacity_keep(arrays, capacity)
    padded = {
        "keys": np.zeros((capacity, d_k), dtype=np.float32),
        "context": np.zeros((capacity, d_c), dtype=np.float32),
        "teacher_policy": np.zeros((capacity, act_dim), dtype=np.float32),
        "teacher_value": np.zeros((capacity,), dtype=np.float32),
        "importance": np.zeros((capacity,), dtype=np.float32),
        "game_id": np.zeros((capacity,), dtype=np.int32),
        "source5": np.zeros((capacity, 5), dtype=np.float32),
        "age": np.zeros((capacity,), dtype=np.float32),
        "valid": np.zeros((capacity,), dtype=bool),
    }
    n = min(int(keep.shape[0]), capacity)
    if n:
        keep = keep[:n]
        padded["keys"][:n] = _normalize_rows(arrays["keys"][keep])
        padded["context"][:n] = arrays["context"][keep].astype(np.float32)
        padded["teacher_policy"][:n] = arrays["teacher_policy"][keep].astype(np.float32)
        padded["teacher_value"][:n] = arrays["teacher_value"][keep].astype(np.float32)
        padded["importance"][:n] = np.maximum(arrays["importance"][keep], EPS).astype(np.float32)
        padded["game_id"][:n] = arrays["game_id"][keep].astype(np.int32)
        padded["source5"][:n] = arrays["source5"][keep].astype(np.float32)
        padded["age"][:n] = arrays["age"][keep].astype(np.float32)
        padded["valid"][:n] = True
    return {name: jnp.asarray(value) for name, value in padded.items()}


def eval_living_memory(
    params,
    game,
    game_id,
    protected_bank,
    *,
    cfg,
    seed,
    episodes=12,
    blend=True,
) -> float:
    """Evaluate a game greedily with live model plus protected memory."""
    episodes = int(episodes)
    if episodes <= 0:
        raise ValueError("episodes must be positive")
    num_envs = _cfg_int(cfg, "eval_num_envs", _cfg_int(cfg, "eval_envs", 16))
    if num_envs <= 0:
        raise ValueError("eval_num_envs/eval_envs must be positive")

    top_k = _cfg_int(cfg, "top_k", 1)
    capacity = int(np.asarray(protected_bank["valid"]).shape[0])
    if top_k <= 0 or top_k > capacity:
        raise ValueError("cfg.top_k must be in [1, protected bank capacity]")
    hp = default_retrieval_hp(top_k)
    mu_g, sigma_g = _value_norm_mu_sigma(_cfg_get(cfg, "value_norm", None))
    max_steps_per_episode = _cfg_int(cfg, "eval_max_steps_per_episode", 30_000)
    eval_steps_cap = _cfg_int(cfg, "eval_steps_cap", 6_000)
    if max_steps_per_episode <= 0 or eval_steps_cap <= 0:
        raise ValueError("eval step limits must be positive")
    max_steps = max(
        1,
        min(
            eval_steps_cap,
            int(math.ceil(float(episodes * max_steps_per_episode) / float(num_envs))),
        ),
    )

    env = make_eval_env(str(game), num_envs, int(seed))
    obs, _ = env.reset()
    obs = np.asarray(obs, dtype=np.uint8)
    tracker = EpisodeReturnTracker(num_envs)
    completed_returns: list[float] = []
    game_id_vec = jnp.full((num_envs,), int(game_id), dtype=jnp.int32)

    for _ in range(max_steps):
        out = mem_apply(params, obs, game_id_vec, protected_bank, hp, mu_g, sigma_g)
        actions = _select_greedy_actions(out, blend=bool(blend))
        actions_np = np.asarray(jax.device_get(actions), dtype=np.int32)
        obs, rewards, terminated, truncated, info = env.step(actions_np)
        obs = np.asarray(obs, dtype=np.uint8)
        completed = tracker.update(rewards, terminated, truncated, info)
        for episode_return in completed:
            if len(completed_returns) < episodes:
                completed_returns.append(float(episode_return))
        if len(completed_returns) >= episodes:
            break

    _close_env(env)
    if not completed_returns:
        return float(np.mean(np.asarray(tracker.returns, dtype=np.float32)))
    return float(np.mean(np.asarray(completed_returns[:episodes], dtype=np.float32)))


__all__ = [
    "build_protected_bank",
    "certify_protected_memories",
    "eval_living_memory",
]
