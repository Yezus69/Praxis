import jax.numpy as jnp
import numpy as np

from pmac.anchors import AnchorStore
from pmac.atlas import Atlas
from pmac.checkpoint import Champion
from pmac.config import PMAConfig
from pmac.consolidation import consolidate
from pmac.sentinels import SentinelStore


class LinearAdapter:
    def behavior(self, params, batch):
        return jnp.asarray(batch["x"]) @ params["w"]

    def distance(self, cur, teacher, batch=None):
        diff = jnp.asarray(cur) - jnp.asarray(teacher)
        return jnp.sum(diff * diff, axis=-1)

    def evaluate_skill(self, params, skill_eval_set):
        logits = self.behavior(params, skill_eval_set)
        pred = jnp.argmax(logits, axis=-1)
        return float(jnp.mean(pred == skill_eval_set["y"]))


def _node(best_score=0.0, allowed=1.0):
    adapter = LinearAdapter()
    x = np.eye(2, dtype=np.float32)
    y = np.asarray([0, 1], dtype=np.int32)
    champion_params = {"w": jnp.eye(2, dtype=jnp.float32)}
    teacher = np.asarray(adapter.behavior(champion_params, {"x": x}))
    anchors = AnchorStore(capacity=2)
    anchors.add(x, teacher, np.zeros(2), np.ones(2), np.ones(2))
    atlas = Atlas()
    node = atlas.create_or_update_node(
        "s",
        "s",
        anchors,
        SentinelStore(x, y),
        champion_ref=Champion(champion_params, route="s", meta={"skill_id": "s"}),
        best_score=best_score,
        current_score=best_score,
        allowed_regression=allowed,
    )
    return atlas, node, adapter, champion_params


def _dist(params, champion_params, adapter):
    x = jnp.eye(2, dtype=jnp.float32)
    cur = adapter.behavior(params, {"x": x})
    teacher = adapter.behavior(champion_params, {"x": x})
    return float(jnp.mean(adapter.distance(cur, teacher, None)))


def test_slow_core_distillation_reduces_distance_and_does_not_mutate_champion():
    atlas, node, adapter, champion_params = _node(best_score=0.0, allowed=1.0)
    slow = {"w": jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)}
    before = _dist(slow, champion_params, adapter)
    champion_before = np.asarray(node.champion_ref.params["w"]).copy()

    new_slow, info = consolidate(slow, atlas, adapter, PMAConfig(), steps=25, lr=0.2)

    assert _dist(new_slow, champion_params, adapter) < before
    assert info["certified"]["s"]
    assert np.allclose(np.asarray(node.champion_ref.params["w"]), champion_before)


def test_consolidation_certified_flag_false_when_sentinel_does_not_pass():
    atlas, _, adapter, _ = _node(best_score=1.0, allowed=0.0)
    slow = {"w": jnp.asarray([[0.0, 1.0], [1.0, 0.0]], dtype=jnp.float32)}

    _, info = consolidate(slow, atlas, adapter, PMAConfig(), steps=5, lr=0.1)

    assert not info["certified"]["s"]
