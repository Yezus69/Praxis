"""Continued-learning state and transactional rollback helpers."""

from tfns.consolidate.certify import (
    closed_loop_gate,
    is_learned,
    random_normalized_progress,
)
from tfns.consolidate.lifecycle import (
    apply_rejection_feedback,
    build_sentinel_clusters,
    collect_protected_activations,
    consolidate,
    expand_protected_bases,
    slow_replay,
)
from tfns.consolidate.plasticity import (
    activate_adapter,
    plasticity_report,
    should_activate_adapter,
)
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
    "activate_adapter",
    "apply_rejection_feedback",
    "build_sentinel_clusters",
    "closed_loop_gate",
    "collect_protected_activations",
    "consolidate",
    "deserialize",
    "ema_update",
    "expand_protected_bases",
    "is_learned",
    "plasticity_report",
    "random_normalized_progress",
    "restore",
    "serialize",
    "should_activate_adapter",
    "slow_replay",
    "snapshot",
]
