"""Capacity growth trigger from PMA-C sections 10 and 25.4."""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class GrowthState:
    history: list[float] = field(default_factory=list)
    grown: int = 0


def should_grow(history, patience, min_ratio) -> bool:
    """True iff recent mean plasticity over patience steps is below min_ratio."""
    if patience <= 0:
        raise ValueError("patience must be positive")
    if len(history) < patience:
        return False
    recent = [float(x) for x in history[-patience:]]
    return (sum(recent) / float(patience)) < float(min_ratio)


class GrowthController:
    def __init__(self, patience, min_ratio):
        self.patience = int(patience)
        self.min_ratio = float(min_ratio)
        self.state = GrowthState()

    def observe(self, plasticity_ratio: float) -> None:
        self.state.history.append(float(plasticity_ratio))

    def should_grow(self) -> bool:
        return should_grow(self.state.history, self.patience, self.min_ratio)

    def reset(self) -> None:
        self.state.history.clear()
        self.state.grown += 1


__all__ = ["GrowthState", "should_grow", "GrowthController"]
