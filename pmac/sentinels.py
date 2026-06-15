"""Sentinel evaluation store for PMA-C skills."""

from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp


@dataclass
class SentinelStore:
    x: object
    y: object
    seeds: object = None

    def as_eval_set(self):
        return {"x": jnp.asarray(self.x), "y": jnp.asarray(self.y, dtype=jnp.int32)}

    def evaluate(self, params, adapter) -> float:
        return float(adapter.evaluate_skill(params, self.as_eval_set()))

    def passes(self, params, adapter, best_score, allowed_regression) -> bool:
        score = self.evaluate(params, adapter)
        return bool(score >= float(best_score) - float(allowed_regression))


__all__ = ["SentinelStore"]
