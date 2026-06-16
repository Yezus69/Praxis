"""Compressed latent memory atom types for PMA-C."""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntFlag

import numpy as np


class SourceFlag(IntFlag):
    HIGH_RETURN = 1
    NEAR_LIFE_LOSS = 2
    SENTINEL = 4
    NOVELTY = 8
    FAILURE_RECOVERY = 16


@dataclass
class MemoryAtom:
    key: np.ndarray
    context: np.ndarray
    teacher_policy: np.ndarray
    teacher_value: float
    importance: float
    game_id: int
    eps_policy: float
    eps_value: float
    cluster_id: int = -1
    radius: float = 0.0
    rarity: float = 0.0
    age: float = 0.0
    count: int = 1
    source_flags: int = 0
    model_coverage: bool = False
    successor_key: np.ndarray | None = None
    action: int | None = None
    reward: float | None = None
    done: bool | None = None
    return_trace: float | None = None


@dataclass
class MemoryBatch:
    key: np.ndarray
    context: np.ndarray
    teacher_policy: np.ndarray
    teacher_value: np.ndarray
    importance: np.ndarray
    game_id: np.ndarray


__all__ = ["MemoryAtom", "MemoryBatch", "SourceFlag"]
