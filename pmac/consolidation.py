"""Slow-core consolidation for PMA-C sections 11 and 18."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import optax

from pmac.projection import project_conflicts
from pmac.stability import scale_by_stability


def _node_teacher_behavior(node, batch, adapter):
    champion = node.champion_ref
    if champion is not None and getattr(champion, "params", None) is not None:
        return adapter.behavior(champion.params, {"x": batch.x})
    return batch.teacher


def consolidate(slow_params, atlas, adapter, cfg, omega=None, steps=50, lr=1e-3):
    """Distill protected champion behavior into a slow shared parameter tree."""

    nodes = atlas.protected_nodes()
    if not nodes:
        return slow_params, {"steps": 0, "certified": {}, "stopped_by_gate": False}

    opt = optax.sgd(float(lr))
    opt_state = opt.init(slow_params)
    params = slow_params
    stopped_by_gate = False

    def node_loss(p, node):
        batch = node.anchors.all_batch()
        teacher = _node_teacher_behavior(node, batch, adapter)
        current = adapter.behavior(p, {"x": batch.x})
        distance = adapter.distance(current, teacher, batch)
        return jnp.mean(jnp.asarray(batch.weight) * distance)

    def total_loss(p):
        loss = jnp.array(0.0)
        for node in nodes:
            loss = loss + node_loss(p, node)
        return loss / max(len(nodes), 1)

    completed = 0
    for _ in range(int(steps)):
        g = jax.grad(total_loss)(params)
        guard_grads = [jax.grad(lambda p, n=node: node_loss(p, n))(params) for node in nodes]
        g = project_conflicts(g, guard_grads)
        if omega is not None and getattr(cfg, "stability_enabled", True):
            g = scale_by_stability(g, omega, getattr(cfg, "stability_alpha", 10.0))
        updates, opt_state = opt.update(g, opt_state, params)
        candidate = optax.apply_updates(params, updates)

        if all(
            node.sentinels.passes(
                candidate, adapter, node.best_score, node.allowed_regression
            )
            for node in nodes
        ):
            params = candidate
            completed += 1
        else:
            stopped_by_gate = True
            break

    certified = {
        node.skill_id: node.sentinels.passes(
            params, adapter, node.best_score, node.allowed_regression
        )
        for node in nodes
    }
    return params, {
        "steps": completed,
        "certified": certified,
        "stopped_by_gate": stopped_by_gate,
    }


__all__ = ["consolidate"]
