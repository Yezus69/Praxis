"""Robust task-free regime-change detection."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np

from tfns.config import DetectConfig
from tfns.utils import RunningRobustStat

_EPS = 1.0e-8


@dataclasses.dataclass(frozen=True)
class DetectorState:
    """Persistent Page-Hinkley detector state."""

    stat: RunningRobustStat = dataclasses.field(default_factory=RunningRobustStat)
    cusum: float = 0.0
    sustained: int = 0
    cooldown: int = 0
    n: int = 0
    last_x: float = 0.0
    last_z: float = 0.0
    changed: bool = False


def signature_window(keys: Any) -> np.ndarray:
    """Return ``normalize(mean(keys))`` for a sequence context window."""

    arr = np.asarray(keys, dtype=np.float32)
    if arr.size == 0:
        raise ValueError("keys must contain at least one vector.")
    if arr.ndim == 1:
        mean = arr
    else:
        mean = np.mean(arr.reshape((-1, arr.shape[-1])), axis=0)
    norm = float(np.linalg.norm(mean))
    if norm <= _EPS:
        return np.zeros_like(mean, dtype=np.float32)
    return (mean / norm).astype(np.float32)


class PageHinkleyDetector:
    """Outlier-resistant sustained-increase detector with cooldown."""

    def __init__(self, config: DetectConfig | None = None):
        self.config = config or DetectConfig()

    def init(self) -> DetectorState:
        return DetectorState()

    def update(self, state: DetectorState, x: float) -> tuple[DetectorState, bool]:
        x_val = float(np.asarray(x, dtype=np.float64))
        incoming_cooldown = int(state.cooldown)
        cooldown = max(0, incoming_cooldown - 1)

        if not state.stat.initialized:
            new_stat = state.stat.update(x_val)
            new_state = dataclasses.replace(
                state,
                stat=new_stat,
                n=int(state.n) + 1,
                last_x=x_val,
                last_z=0.0,
                changed=False,
                cooldown=cooldown,
            )
            return new_state, False

        z = float(np.asarray(state.stat.normalize(x_val), dtype=np.float64))
        excess = z - float(self.config.ph_delta)
        cusum = max(0.0, float(state.cusum) + excess)
        sustained = int(state.sustained) + 1 if excess > 0.0 else 0
        changed = bool(
            incoming_cooldown <= 0 and sustained >= 2 and cusum > float(self.config.ph_lambda)
        )

        new_stat = state.stat.update(x_val)
        cooldown = int(self.config.cooldown_blocks) if changed else cooldown
        if changed:
            cusum = 0.0
            sustained = 0

        new_state = DetectorState(
            stat=new_stat,
            cusum=float(cusum),
            sustained=int(sustained),
            cooldown=int(cooldown),
            n=int(state.n) + 1,
            last_x=x_val,
            last_z=z,
            changed=changed,
        )
        return new_state, changed


__all__ = [
    "DetectorState",
    "PageHinkleyDetector",
    "signature_window",
]
