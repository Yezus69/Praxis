"""Visual sentinel storage and alignment batches for PMA-C sections 7.3, 12, and 13."""

from __future__ import annotations

import gc
import math
from collections import defaultdict
from collections.abc import Mapping

import jax
import jax.numpy as jnp
import numpy as np

from pmac.agents.atari_mem_net import mem_apply, mem_apply_key
from pmac.agents.ppo_atari import gae
from pmac.envs.atari_envpool import make_train_env
from pmac.memory.runtime import default_retrieval_hp
from pmac.memory.write import EPS, teacher_targets


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


def _normalize_rows(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + EPS)


def _close_env(env):
    try:
        env.close()
    except Exception:
        pass
    del env
    gc.collect()


class VisualSentinelStore:
    """Bounded per-game visual sentinel memory. spec §7.3, §12"""

    def __init__(self, per_game: int = 64):
        per_game = int(per_game)
        if per_game <= 0:
            raise ValueError("per_game must be positive")
        self.per_game = per_game
        self._by_game: dict[int, list[dict]] = defaultdict(list)
        self._seq = 0
        self._d_k = 0
        self._act_dim = 0

    def __len__(self) -> int:
        return sum(len(records) for records in self._by_game.values())

    def games(self) -> tuple[int, ...]:
        return tuple(sorted(self._by_game))

    def add(self, game_id, obs, key_star, p_star, v_star):
        """Add one or more visual sentinels, keeping the newest rows per game."""
        obs = np.asarray(obs, dtype=np.uint8)
        key_star = _normalize_rows(np.asarray(key_star, dtype=np.float32))
        p_star = np.asarray(p_star, dtype=np.float32)
        v_star = np.asarray(v_star, dtype=np.float32)

        if obs.ndim == 3:
            obs = obs[None, ...]
        if key_star.ndim == 1:
            key_star = key_star[None, ...]
        if p_star.ndim == 1:
            p_star = p_star[None, ...]
        if v_star.ndim == 0:
            v_star = v_star[None]

        n = int(obs.shape[0])
        if key_star.shape[0] != n or p_star.shape[0] != n or v_star.reshape(-1).shape[0] != n:
            raise ValueError("obs, key_star, p_star, and v_star must share the same leading size")
        if obs.shape[1:] != (4, 84, 84):
            raise ValueError(f"obs must have shape [N,4,84,84], got {obs.shape}")

        game_ids = np.asarray(game_id, dtype=np.int32)
        if game_ids.ndim == 0:
            game_ids = np.full((n,), int(game_ids), dtype=np.int32)
        else:
            game_ids = game_ids.reshape(-1).astype(np.int32)
            if game_ids.shape[0] == 1 and n != 1:
                game_ids = np.repeat(game_ids, n)
        if game_ids.shape[0] != n:
            raise ValueError(f"game_id must be scalar or length {n}, got {game_ids.shape[0]}")

        self._d_k = int(key_star.shape[-1])
        self._act_dim = int(p_star.shape[-1])
        v_star = v_star.reshape(-1)
        for i in range(n):
            gid = int(game_ids[i])
            self._seq += 1
            self._by_game[gid].append(
                {
                    "seq": self._seq,
                    "obs": obs[i].copy(),
                    "game_id": gid,
                    "key_star": key_star[i].astype(np.float16),
                    "teacher_policy": p_star[i].astype(np.float16),
                    "teacher_value": np.asarray(v_star[i], dtype=np.float16),
                }
            )
            if len(self._by_game[gid]) > self.per_game:
                self._by_game[gid] = self._by_game[gid][-self.per_game :]

    def add_set(self, sent_set: Mapping):
        self.add(
            sent_set["game_id"],
            sent_set["obs"],
            sent_set["key_star"],
            sent_set["teacher_policy"],
            sent_set["teacher_value"],
        )

    def _records(self) -> list[dict]:
        records = []
        for game_id in self.games():
            records.extend(self._by_game[game_id])
        records.sort(key=lambda item: item["seq"])
        return records

    def batch(self, size: int | None = None, *, seed: int = 0) -> dict[str, np.ndarray]:
        """Return a fixed-size or full sentinel batch for ``visual_sentinel_loss``."""
        records = self._records()
        if not records:
            size = 0 if size is None else int(size)
            return {
                "obs": np.zeros((size, 4, 84, 84), dtype=np.uint8),
                "game_id": np.zeros((size,), dtype=np.int32),
                "key_star": np.zeros((size, self._d_k), dtype=np.float16),
                "teacher_policy": np.zeros((size, self._act_dim), dtype=np.float16),
                "teacher_value": np.zeros((size,), dtype=np.float16),
            }

        if size is None:
            selected = records
        else:
            size = int(size)
            rng = np.random.default_rng(int(seed))
            replace = len(records) < size
            idx = rng.choice(len(records), size=size, replace=replace)
            selected = [records[int(i)] for i in np.asarray(idx).reshape(-1)]

        return {
            "obs": np.stack([item["obs"] for item in selected], axis=0).astype(np.uint8),
            "game_id": np.asarray([item["game_id"] for item in selected], dtype=np.int32),
            "key_star": np.stack([item["key_star"] for item in selected], axis=0).astype(np.float16),
            "teacher_policy": np.stack(
                [item["teacher_policy"] for item in selected], axis=0
            ).astype(np.float16),
            "teacher_value": np.asarray(
                [item["teacher_value"] for item in selected], dtype=np.float16
            ),
        }


def collect_visual_sentinels(
    params,
    ema_params,
    value_norm,
    game,
    game_id,
    *,
    cfg,
    seed,
    n=64,
) -> dict:
    """Collect bounded raw-frame sentinels with EMA keys and teacher targets. spec §12"""
    _, d_k, d_c, _, act_dim = _infer_dims(params)
    n = int(n)
    if n <= 0:
        return {
            "obs": np.zeros((0, 4, 84, 84), dtype=np.uint8),
            "game_id": np.zeros((0,), dtype=np.int32),
            "key_star": np.zeros((0, d_k), dtype=np.float16),
            "teacher_policy": np.zeros((0, act_dim), dtype=np.float16),
            "teacher_value": np.zeros((0,), dtype=np.float16),
        }

    num_envs = min(_cfg_int(cfg, "cert_num_envs", 64), n)
    if num_envs <= 0:
        raise ValueError("cert_num_envs must be positive")
    total_steps = max(
        int(math.ceil(float(n) / float(num_envs))),
        _cfg_int(cfg, "sentinel_collect_steps", 16),
    )
    top_k = _cfg_int(cfg, "top_k", 1)
    if top_k <= 0:
        raise ValueError("cfg.top_k must be positive")
    hp = default_retrieval_hp(top_k)
    empty_bank = _empty_bank(top_k, d_k, d_c, act_dim)
    mu_g, sigma_g = _value_norm_mu_sigma(value_norm)

    env = make_train_env(str(game), num_envs, int(seed))
    try:
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
        finite = np.isfinite(flat_returns)
        if not np.any(finite):
            keep = np.zeros((0,), dtype=np.int64)
        else:
            scores = np.where(finite, flat_returns, -np.inf)
            keep = np.argsort(-scores, kind="mergesort")[:n].astype(np.int64)
        if int(keep.shape[0]) == 0:
            return {
                "obs": np.zeros((0, 4, 84, 84), dtype=np.uint8),
                "game_id": np.zeros((0,), dtype=np.int32),
                "key_star": np.zeros((0, d_k), dtype=np.float16),
                "teacher_policy": np.zeros((0, act_dim), dtype=np.float16),
                "teacher_value": np.zeros((0,), dtype=np.float16),
            }

        flat_obs = obs_buf.reshape((total_steps * num_envs,) + tuple(obs_buf.shape[2:]))
        flat_logits = logits_buf.reshape((total_steps * num_envs, act_dim))
        keys = np.asarray(jax.device_get(mem_apply_key(ema_params, flat_obs[keep])), dtype=np.float32)
        p_star, v_star = teacher_targets(
            flat_logits[keep],
            flat_returns[keep],
            mu_g,
            sigma_g,
            temperature=_cfg_float(cfg, "teacher_temperature", 1.0),
        )
        return {
            "obs": flat_obs[keep].astype(np.uint8),
            "game_id": np.full((int(keep.shape[0]),), int(game_id), dtype=np.int32),
            "key_star": _normalize_rows(keys).astype(np.float16),
            "teacher_policy": p_star.astype(np.float16),
            "teacher_value": v_star.astype(np.float16),
        }
    finally:
        _close_env(env)


def build_align_batch(
    sent_set: Mapping,
    protected_bank: Mapping,
    n_neg: int = 16,
    *,
    batch_size: int | None = None,
    seed: int = 0,
) -> dict[str, np.ndarray]:
    """Build InfoNCE pairs with different-game hard negatives. spec §13"""
    obs = np.asarray(sent_set["obs"], dtype=np.uint8)
    game_id = np.asarray(sent_set["game_id"], dtype=np.int32).reshape(-1)
    pos_key = _normalize_rows(np.asarray(sent_set["key_star"], dtype=np.float32))
    if obs.shape[0] != game_id.shape[0] or obs.shape[0] != pos_key.shape[0]:
        raise ValueError("sent_set obs, game_id, and key_star must share the same leading size")
    if int(n_neg) <= 0:
        raise ValueError("n_neg must be positive")

    bank_keys = _normalize_rows(np.asarray(protected_bank["keys"], dtype=np.float32))
    bank_game = np.asarray(protected_bank["game_id"], dtype=np.int32).reshape(-1)
    valid = np.asarray(protected_bank.get("valid", np.ones_like(bank_game, dtype=bool)), dtype=bool).reshape(-1)
    valid = valid & np.all(np.isfinite(bank_keys), axis=-1)
    valid_rows = np.flatnonzero(valid)
    if valid_rows.size == 0:
        raise ValueError("protected_bank has no valid keys for retrieval alignment")

    eligible = []
    for row, gid in enumerate(game_id):
        if np.any(valid & (bank_game != int(gid))):
            eligible.append(row)
    if not eligible:
        raise ValueError("retrieval alignment needs at least one different-game negative per row")

    rng = np.random.default_rng(int(seed))
    if batch_size is None:
        selected_rows = np.asarray(eligible, dtype=np.int64)
    else:
        batch_size = int(batch_size)
        replace = len(eligible) < batch_size
        selected_rows = rng.choice(np.asarray(eligible, dtype=np.int64), size=batch_size, replace=replace)

    neg_keys = np.zeros((int(selected_rows.shape[0]), int(n_neg), pos_key.shape[-1]), dtype=np.float32)
    neg_game_id = np.zeros((int(selected_rows.shape[0]), int(n_neg)), dtype=np.int32)
    for out_i, row in enumerate(selected_rows):
        candidates = valid_rows[bank_game[valid_rows] != int(game_id[row])]
        sims = bank_keys[candidates] @ pos_key[row]
        order = candidates[np.argsort(-sims, kind="mergesort")]
        if order.shape[0] < int(n_neg):
            repeats = int(math.ceil(float(n_neg) / float(order.shape[0])))
            order = np.tile(order, repeats)
        chosen = order[: int(n_neg)]
        neg_keys[out_i] = bank_keys[chosen]
        neg_game_id[out_i] = bank_game[chosen]

    return {
        "obs": obs[selected_rows].astype(np.uint8),
        "game_id": game_id[selected_rows].astype(np.int32),
        "pos_key": pos_key[selected_rows].astype(np.float32),
        "neg_keys": neg_keys,
        "neg_game_id": neg_game_id,
    }


__all__ = [
    "VisualSentinelStore",
    "build_align_batch",
    "collect_visual_sentinels",
]
