"""Continued-learning state and transactional rollback helpers."""

from tfns.consolidate.state import (
    ContinualState,
    Snapshot,
    deserialize,
    ema_update,
    restore,
    serialize,
    snapshot,
)

__all__ = [
    "ContinualState",
    "Snapshot",
    "deserialize",
    "ema_update",
    "restore",
    "serialize",
    "snapshot",
]
