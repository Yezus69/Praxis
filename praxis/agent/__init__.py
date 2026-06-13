"""praxis.agent — the trainer half (Brax PPO network factory + training entrypoint).

Lightweight package init: re-export the network factory. We intentionally do NOT
import ``train`` here (it pulls in heavy training deps and the env) so that
``from praxis.agent import make_network_factory`` stays cheap.
"""

from __future__ import annotations

from praxis.agent.networks import (
    DEFAULT_POLICY_HIDDEN_LAYER_SIZES,
    DEFAULT_VALUE_HIDDEN_LAYER_SIZES,
    make_network_factory,
)

__all__ = [
    "make_network_factory",
    "DEFAULT_POLICY_HIDDEN_LAYER_SIZES",
    "DEFAULT_VALUE_HIDDEN_LAYER_SIZES",
]
