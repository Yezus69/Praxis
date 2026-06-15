import jax.numpy as jnp
import numpy as np

from pmac.anchors import AnchorStore
from pmac.atlas import Atlas
from pmac.sentinels import SentinelStore


def _store():
    store = AnchorStore(capacity=2)
    store.add(
        np.zeros((1, 2), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.ones(1, dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )
    return store


def _sentinels():
    return SentinelStore(np.zeros((1, 2), dtype=np.float32), np.zeros(1, dtype=np.int32))


def test_create_update_and_protected_filter():
    atlas = Atlas()
    node = atlas.create_or_update_node("a", "ctx-a", _store(), _sentinels())
    atlas.create_or_update_node("b", "ctx-b", _store(), _sentinels(), status="learning")

    assert node.skill_id == "a"
    assert [n.skill_id for n in atlas.protected_nodes()] == ["a"]

    updated = atlas.create_or_update_node("a", "ctx-new", _store(), _sentinels())
    assert updated is node
    assert updated.context_key == "ctx-new"


def test_sample_protected_nodes_returns_ranked_risk_limited_to_k():
    atlas = Atlas()
    high = atlas.create_or_update_node("old-high", 0, _store(), _sentinels())
    low = atlas.create_or_update_node("old-low", 1, _store(), _sentinels())
    atlas.create_or_update_node("cur", 2, _store(), _sentinels())
    high.best_score = 1.0
    high.current_score = 0.3
    low.best_score = 1.0
    low.current_score = 0.9
    high.interference_neighbors.add("cur")

    out = atlas.sample_protected_nodes("cur", k=1)
    assert len(out) == 1
    assert out[0].skill_id == "old-high"


def test_update_edge_symmetric_and_sign_sets():
    atlas = Atlas()
    atlas.create_or_update_node("a", 0, _store(), _sentinels())
    atlas.create_or_update_node("b", 1, _store(), _sentinels())

    aligned = atlas.update_edge("a", "b", {"w": jnp.array([1.0, 0.0])}, {"w": jnp.array([2.0, 0.0])})
    assert aligned > 0.0
    assert atlas.edge_scores[("a", "b")] == atlas.edge_scores[("b", "a")]
    assert "b" in atlas.nodes["a"].positive_transfer_neighbors

    opposed = atlas.update_edge("a", "b", {"w": jnp.array([1.0, 0.0])}, {"w": jnp.array([-2.0, 0.0])})
    assert opposed < 0.0
    assert "b" in atlas.nodes["a"].interference_neighbors
    assert "a" in atlas.nodes["b"].interference_neighbors
