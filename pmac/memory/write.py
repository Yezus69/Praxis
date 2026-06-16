"""Pure NumPy memory write rule utilities for PMA-C."""

from __future__ import annotations

import numpy as np

from pmac.memory.atom import SourceFlag


EPS = 1e-8
DEFAULT_WRITE_WEIGHTS = {
    "adv": 1.0,
    "delta": 1.0,
    "novelty": 1.5,
    "entropy": 0.25,
    "life": 3.0,
    "ret": 2.0,
    "forget": 3.0,
}  # spec §8


class RunningStats:
    """EMA mean and mean absolute deviation for one game's write scores."""

    def __init__(self, momentum=0.99, eps=1e-6):
        self.momentum = float(momentum)
        self.eps = float(eps)
        self.mean = 0.0
        self.mean_abs_dev = 1.0
        self.initialized = False

    def update(self, x):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 1:
            raise ValueError(f"x must be 1-D, got {x.shape}")
        if x.size == 0:
            return self

        batch_mean = float(np.mean(x))
        batch_mad = float(np.mean(np.abs(x - batch_mean)))
        if not self.initialized:
            self.mean = batch_mean
            self.mean_abs_dev = batch_mad
            self.initialized = True
        else:
            keep = self.momentum
            update = 1.0 - keep
            self.mean = keep * self.mean + update * batch_mean
            self.mean_abs_dev = keep * self.mean_abs_dev + update * batch_mad
        return self

    def normalize(self, x):
        x = np.asarray(x, dtype=np.float32)
        return (x - self.mean) / (self.mean_abs_dev + self.eps)  # spec §8


def _log_softmax(logits):
    logits = np.asarray(logits, dtype=np.float32)
    shifted = logits - np.max(logits, axis=-1, keepdims=True)
    return shifted - np.log(np.sum(np.exp(shifted), axis=-1, keepdims=True))


def _as_rows(x, name: str):
    arr = np.asarray(x, dtype=np.float32)
    if arr.ndim == 1:
        arr = arr[None, :]
    if arr.ndim != 2:
        raise ValueError(f"{name} must be rank 1 or 2, got {arr.shape}")
    return arr


def _normalize_rows(x, eps=EPS):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + eps)


def _rng_from_key(key):
    if isinstance(key, np.random.Generator):
        return key
    arr = np.asarray(key, dtype=np.uint32).reshape(-1)
    seed = 0
    for value in arr:
        seed = (1664525 * seed + int(value) + 1013904223) % (2**32)
    return np.random.default_rng(seed)


def _rank_order(scores, rng):
    if rng is None:
        tie = np.arange(scores.shape[0], dtype=np.int64)
    else:
        tie = _rng_from_key(rng).random(scores.shape[0])
    return np.lexsort((tie, -scores))


def _optional_mask(mask, n: int, name: str):
    if mask is None:
        return np.zeros((n,), dtype=bool)
    arr = np.asarray(mask, dtype=bool).reshape(-1)
    if arr.shape[0] != n:
        raise ValueError(f"{name} must have length {n}, got {arr.shape[0]}")
    return arr


def td_error(rewards, values, next_values, gamma):
    rewards = np.asarray(rewards, dtype=np.float32)
    values = np.asarray(values, dtype=np.float32)
    next_values = np.asarray(next_values, dtype=np.float32)
    return rewards + float(gamma) * next_values - values  # spec §8


def policy_entropy(logits):
    log_probs = _log_softmax(logits)
    probs = np.exp(log_probs)
    return -np.sum(probs * log_probs, axis=-1)  # spec §8


def novelty(keys_t, bank_keys, bank_valid, bank_game_id, cur_game):
    keys = _as_rows(keys_t, "keys_t")
    bank = _as_rows(bank_keys, "bank_keys")
    if keys.shape[1] != bank.shape[1]:
        raise ValueError(f"key widths differ: {keys.shape[1]} vs {bank.shape[1]}")

    valid = np.asarray(bank_valid, dtype=bool).reshape(-1)
    game_id = np.asarray(bank_game_id, dtype=np.int32).reshape(-1)
    if valid.shape[0] != bank.shape[0] or game_id.shape[0] != bank.shape[0]:
        raise ValueError("bank_valid and bank_game_id must match bank_keys rows")

    cur = np.asarray(cur_game, dtype=np.int32)
    if cur.ndim == 0:
        games = np.full((keys.shape[0],), int(cur), dtype=np.int32)
    else:
        games = cur.reshape(-1)
        if games.shape[0] != keys.shape[0]:
            raise ValueError(f"cur_game must be scalar or length {keys.shape[0]}")

    keys_norm = _normalize_rows(keys)
    bank_norm = _normalize_rows(bank)
    out = np.ones((keys.shape[0],), dtype=np.float32)
    for row, game in enumerate(games):
        same_game = valid & (game_id == int(game))
        if np.any(same_game):
            max_cos = float(np.max(keys_norm[row : row + 1] @ bank_norm[same_game].T))
            out[row] = 1.0 - max_cos  # spec §8
    return np.clip(out, 0.0, 1.0)  # spec §8


def importance(
    abs_adv_hat,
    abs_delta_hat,
    novelty,
    entropy,
    life,
    ret_hat,
    forget_risk,
    weights=DEFAULT_WRITE_WEIGHTS,
):
    score = (
        float(weights["adv"]) * np.asarray(abs_adv_hat, dtype=np.float32)
        + float(weights["delta"]) * np.asarray(abs_delta_hat, dtype=np.float32)
        + float(weights["novelty"]) * np.asarray(novelty, dtype=np.float32)
        + float(weights["entropy"]) * np.asarray(entropy, dtype=np.float32)
        + float(weights["life"]) * np.asarray(life, dtype=np.float32)
        + float(weights["ret"]) * np.asarray(ret_hat, dtype=np.float32)
        + float(weights["forget"]) * np.asarray(forget_risk, dtype=np.float32)
    )  # spec §8
    return np.asarray(score, dtype=np.float32)


def teacher_targets(logits, values, mu_g, sigma_g, temperature=1.0, eps=EPS):
    temperature = float(temperature)
    if temperature < 1.0:
        raise ValueError("temperature must be >= 1")
    p_star = np.exp(_log_softmax(np.asarray(logits, dtype=np.float32) / temperature))  # spec §8
    v_star = (
        (np.asarray(values, dtype=np.float32) - np.asarray(mu_g, dtype=np.float32))
        / (np.asarray(sigma_g, dtype=np.float32) + float(eps))
    )  # spec §8
    return p_star.astype(np.float32), np.asarray(v_star, dtype=np.float32)


def select_writes(importance, top_fraction, *, rare_mask=None, sentinel_mask=None, min_quota=0, rng=None):
    scores = np.asarray(importance, dtype=np.float32).reshape(-1)
    n = int(scores.shape[0])
    selected = np.zeros((n,), dtype=bool)
    if n == 0:
        return selected

    fraction = float(top_fraction)
    if fraction < 0.0 or fraction > 1.0:
        raise ValueError("top_fraction must be in [0, 1]")

    order = _rank_order(scores, rng)
    top_n = 0 if fraction == 0.0 else min(n, int(np.ceil(fraction * n)))
    selected[order[:top_n]] = True
    selected |= _optional_mask(rare_mask, n, "rare_mask")
    selected |= _optional_mask(sentinel_mask, n, "sentinel_mask")  # spec §8

    quota = min(n, max(0, int(min_quota)))
    if int(np.sum(selected)) < quota:
        for idx in order:
            if not selected[idx]:
                selected[idx] = True
                if int(np.sum(selected)) >= quota:
                    break
    return selected  # spec §8


def build_insert_kwargs(
    keys,
    contexts,
    logits,
    values,
    game_ids,
    importances,
    *,
    mu_g,
    sigma_g,
    temperature,
    novelty,
    eps_policy,
    eps_value,
    source_flags=0,
):
    teacher_policies, teacher_values = teacher_targets(
        logits,
        values,
        mu_g,
        sigma_g,
        temperature=temperature,
    )
    return {
        "keys": np.asarray(keys, dtype=np.float32),
        "contexts": np.asarray(contexts, dtype=np.float32),
        "teacher_policies": teacher_policies,
        "teacher_values": teacher_values,
        "importances": np.asarray(importances, dtype=np.float32),
        "game_ids": np.asarray(game_ids, dtype=np.int32),
        "eps_policy": np.asarray(eps_policy, dtype=np.float32),
        "eps_value": np.asarray(eps_value, dtype=np.float32),
        "rarity": np.asarray(novelty, dtype=np.float32),
        "source_flags": np.asarray(source_flags, dtype=np.int32),
    }


def write_source_flags(high_return, near_life_loss, novelty_hi, failure_recovery):
    high_return, near_life_loss, novelty_hi, failure_recovery = np.broadcast_arrays(
        np.asarray(high_return, dtype=bool),
        np.asarray(near_life_loss, dtype=bool),
        np.asarray(novelty_hi, dtype=bool),
        np.asarray(failure_recovery, dtype=bool),
    )
    flags = np.zeros(high_return.shape, dtype=np.int32)
    flags |= np.where(high_return, int(SourceFlag.HIGH_RETURN), 0).astype(np.int32)
    flags |= np.where(near_life_loss, int(SourceFlag.NEAR_LIFE_LOSS), 0).astype(np.int32)
    flags |= np.where(novelty_hi, int(SourceFlag.NOVELTY), 0).astype(np.int32)
    flags |= np.where(failure_recovery, int(SourceFlag.FAILURE_RECOVERY), 0).astype(np.int32)
    return flags


__all__ = [
    "DEFAULT_WRITE_WEIGHTS",
    "EPS",
    "RunningStats",
    "build_insert_kwargs",
    "importance",
    "novelty",
    "policy_entropy",
    "select_writes",
    "td_error",
    "teacher_targets",
    "write_source_flags",
]
