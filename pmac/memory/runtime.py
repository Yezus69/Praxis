"""Runtime memory helpers for living-memory Atari PPO."""

from __future__ import annotations

import numpy as np
import jax.numpy as jnp

from pmac.memory.bank import MemoryBank
from pmac.memory.reader import expand_source_flags


def default_retrieval_hp(top_k) -> dict:
    """Default retrieval/blend hyperparameters for the hot memory bank."""
    return {
        "tau_r": 0.5,
        "beta_c": 1.0,
        "beta_I": 0.25,
        "beta_a": 0.1,
        "top_k": int(top_k),
        "w_rho": 4.0,
        "w_c": 1.0,
        "b0": 1.0,
    }  # spec §9


def _empty_arrays(capacity: int, d_k: int, d_c: int, act_dim: int):
    return {
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


def pad_bank(bank: MemoryBank, capacity, *, d_k, d_c, act_dim) -> dict:
    """Export a MemoryBank into fixed-shape device arrays for retrieval."""
    capacity = int(capacity)
    d_k = int(d_k)
    d_c = int(d_c)
    act_dim = int(act_dim)
    if capacity <= 0:
        raise ValueError("capacity must be positive")

    padded = _empty_arrays(capacity, d_k, d_c, act_dim)
    arrays = bank.to_retrieval_arrays()
    n = int(arrays["keys"].shape[0])
    if n:
        if n > capacity:
            order = np.argsort(-np.asarray(arrays["importance"], dtype=np.float32), kind="mergesort")
            keep = order[:capacity]
        else:
            keep = np.arange(n, dtype=np.int64)
        k = int(keep.shape[0])
        source5 = np.asarray(expand_source_flags(arrays["source_flags"][keep]), dtype=np.float32)
        padded["keys"][:k] = np.asarray(arrays["keys"][keep], dtype=np.float32)
        padded["context"][:k] = np.asarray(arrays["context"][keep], dtype=np.float32)
        padded["teacher_policy"][:k] = np.asarray(arrays["teacher_policy"][keep], dtype=np.float32)
        padded["teacher_value"][:k] = np.asarray(arrays["teacher_value"][keep], dtype=np.float32)
        padded["importance"][:k] = np.asarray(arrays["importance"][keep], dtype=np.float32)
        padded["game_id"][:k] = np.asarray(arrays["game_id"][keep], dtype=np.int32)
        padded["source5"][:k] = source5
        padded["age"][:k] = np.asarray(arrays["age"][keep], dtype=np.float32)
        padded["valid"][:k] = True

    return {name: jnp.asarray(value) for name, value in padded.items()}  # spec §9


class RunningValueNorm:
    """Single-game EMA return normalizer for memory value targets."""

    def __init__(self, momentum=0.99, sigma_floor=1.0e-3):
        self.momentum = float(momentum)
        self.sigma_floor = float(sigma_floor)
        self._mu = 0.0
        self._sigma = 1.0

    def update(self, returns):
        values = np.asarray(returns, dtype=np.float32).reshape(-1)
        values = values[np.isfinite(values)]
        if values.size == 0:
            return self
        keep = float(self.momentum)
        update = 1.0 - keep
        batch_mu = float(np.mean(values))
        batch_sigma = float(np.std(values))
        self._mu = keep * self._mu + update * batch_mu  # spec §8
        self._sigma = max(
            keep * self._sigma + update * batch_sigma,
            float(self.sigma_floor),
        )  # spec §8
        return self

    def mu(self) -> float:
        return float(self._mu)

    def sigma(self) -> float:
        return float(max(self._sigma, self.sigma_floor))


__all__ = ["RunningValueNorm", "default_retrieval_hp", "pad_bank"]
