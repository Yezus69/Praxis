import jax.numpy as jnp
import numpy as np

from pmac.anchors import AnchorStore
from pmac.atlas import SkillNode
from pmac.auditor import Auditor
from pmac.sentinels import SentinelStore


class LogitAdapter:
    def behavior(self, params, batch):
        n = jnp.asarray(batch["x"]).shape[0]
        return params["logits"][:n]

    def distance(self, cur, teacher, batch=None):
        diff = jnp.asarray(cur) - jnp.asarray(teacher)
        return jnp.sum(diff * diff, axis=-1)

    def evaluate_skill(self, params, skill_eval_set):
        pred = jnp.argmax(self.behavior(params, skill_eval_set), axis=-1)
        return float(jnp.mean(pred == skill_eval_set["y"]))


def _protected_node(prev_params):
    x = np.zeros((2, 1), dtype=np.float32)
    y = np.asarray([0, 1], dtype=np.int32)
    adapter = LogitAdapter()
    teacher = np.asarray(adapter.behavior(prev_params, {"x": x}))
    anchors = AnchorStore(capacity=2)
    anchors.add(x, teacher, np.zeros(2), np.ones(2), np.ones(2))
    return SkillNode(
        skill_id="old",
        context_key="old",
        status="protected",
        anchors=anchors,
        sentinels=SentinelStore(x, y),
        best_score=1.0,
        current_score=1.0,
        allowed_regression=0.0,
    )


def test_evaluate_candidate_rejects_conservation_regression():
    adapter = LogitAdapter()
    prev = {"logits": jnp.asarray([[2.0, 0.0], [0.0, 2.0]])}
    cand = {"logits": jnp.asarray([[2.2, 0.0], [0.0, 2.2]])}
    node = _protected_node(prev)
    auditor = Auditor(delta_current=1.0, delta_cons=1e-6)

    audit = auditor.evaluate_candidate(cand, prev, {"x": jnp.zeros((2, 1)), "y": jnp.asarray([0, 1])}, [node], adapter)
    assert not audit.accept
    assert not audit.conservation_ok
    assert audit.regressed_nodes == ["old"]


def test_evaluate_candidate_rejects_sentinel_drop_and_accepts_safe_candidate():
    adapter = LogitAdapter()
    prev = {"logits": jnp.asarray([[2.0, 0.0], [0.0, 2.0]])}
    bad = {"logits": jnp.asarray([[0.0, 2.0], [2.0, 0.0]])}
    node = _protected_node(prev)
    auditor = Auditor(delta_current=1.0, delta_cons=100.0)
    source_eval = {"x": jnp.zeros((2, 1)), "y": jnp.asarray([0, 1])}

    rejected = auditor.evaluate_candidate(bad, prev, source_eval, [node], adapter)
    assert not rejected.accept
    assert not rejected.sentinel_ok

    accepted = auditor.evaluate_candidate(prev, prev, source_eval, [node], adapter)
    assert accepted.accept


def test_skill_solved_threshold_boundary():
    adapter = LogitAdapter()
    params = {"logits": jnp.asarray([[2.0, 0.0], [0.0, 2.0]])}
    eval_set = {"x": jnp.zeros((2, 1)), "y": jnp.asarray([0, 1])}

    assert Auditor().skill_solved(params, eval_set, adapter, threshold=1.0)
    assert not Auditor().skill_solved(params, eval_set, adapter, threshold=1.01)
