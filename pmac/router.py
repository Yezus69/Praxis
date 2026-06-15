"""Context routing and composed logits for PMA-C."""

from __future__ import annotations

import jax.numpy as jnp


class Router:
    def __init__(self):
        self.routes = {}

    def register(self, context_key, impl_id):
        self.routes[context_key] = impl_id

    def route(self, context_key):
        return self.routes[context_key]

    def compose_logits(self, logits_list, alphas):
        if len(logits_list) != len(alphas):
            raise ValueError("logits_list and alphas must have the same length")
        out = jnp.asarray(logits_list[0]) * alphas[0]
        for logits, alpha in zip(logits_list[1:], alphas[1:]):
            out = out + jnp.asarray(logits) * alpha
        return out

    def certify_route(self, context_key, impl, skill_node, adapter) -> bool:
        selected = self.routes.get(context_key)
        impl_id = getattr(impl, "impl_id", impl)
        if selected != impl_id:
            return False
        params = getattr(impl, "params", None)
        if params is None:
            return True
        return skill_node.sentinels.passes(
            params, adapter, skill_node.best_score, skill_node.allowed_regression
        )


__all__ = ["Router"]
