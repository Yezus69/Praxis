"""Context-Addressed Synaptic Tensor Memory (CASTM).

A compact, content-addressed field of low-rank weight deltas that replaces the
null-space/sentinel retention mechanism. New learning is written into an address
direction algebraically orthogonal to previously stored addresses, so the full
new update is retained while previously decoded weights remain unchanged.

See ``README_CONTEXT_ADDRESSED_SYNAPTIC_MEMORY.md`` for the full specification.
"""

from __future__ import annotations

from tfns.castm import address, audit, scratch, synaptic

__all__ = ["address", "audit", "scratch", "synaptic"]
