import jax.numpy as jnp
import numpy as np

from pmac.anchors import AnchorStore
from pmac.atlas import SkillNode
from pmac.auditor import Audit
from pmac.checkpoint import ChampionStore, SafeCheckpoint, can_archive_expert, mark_redundant
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


def _node():
    x = np.eye(2, dtype=np.float32)
    params = {"w": jnp.eye(2, dtype=jnp.float32)}
    adapter = LinearAdapter()
    teacher = np.asarray(adapter.behavior(params, {"x": x}))
    anchors = AnchorStore(capacity=2)
    anchors.add(x, teacher, np.zeros(2), np.ones(2), np.ones(2))
    sentinels = SentinelStore(x, np.asarray([0, 1], dtype=np.int32))
    return SkillNode(
        skill_id="s",
        context_key="s",
        status="protected",
        anchors=anchors,
        sentinels=sentinels,
        best_score=1.0,
        current_score=1.0,
        certified_impls=["champion"],
    )


def test_champion_params_are_independent_deep_copy():
    params = {"w": np.asarray([1.0, 2.0], dtype=np.float32)}
    store = ChampionStore()
    champion = store.freeze(params, route="s", meta={"skill_id": "s"})
    params["w"][0] = 99.0

    assert np.asarray(champion.params["w"])[0] == 1.0
    assert store.get("s") is champion


def test_safe_checkpoint_restore_returns_last_accepted():
    ckpt = SafeCheckpoint({"w": np.asarray([1.0], dtype=np.float32)})
    ckpt.update_if_safe({"w": np.asarray([2.0], dtype=np.float32)}, Audit(True, 0.0, True, True, []))
    ckpt.update_if_safe({"w": np.asarray([3.0], dtype=np.float32)}, Audit(False, 0.0, False, True, ["s"]))

    restored = ckpt.restore()
    assert np.asarray(restored["w"])[0] == 2.0


def test_can_archive_expert_false_unless_candidate_passes_certificate():
    node = _node()
    adapter = LinearAdapter()

    bad = {"w": -jnp.eye(2, dtype=jnp.float32)}
    good = {"w": jnp.eye(2, dtype=jnp.float32)}

    assert not can_archive_expert(node, bad, adapter)
    assert can_archive_expert(node, good, adapter)


def test_refuse_to_drop_last_certified_implementation():
    node = _node()
    assert not mark_redundant(node, "champion")

    node.certified_impls.append("slow")
    assert mark_redundant(node, "champion")
    assert "champion" in node.redundant_impls
