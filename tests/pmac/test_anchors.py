import numpy as np

from pmac.anchors import Anchor, AnchorStore


class DummyAdapter:
    def distance(self, cur, teacher, batch=None):
        diff = np.asarray(cur) - np.asarray(teacher)
        return np.sum(diff * diff, axis=-1)


def test_coreset_keeps_top_importance_over_capacity():
    store = AnchorStore(capacity=3)
    x = np.arange(5, dtype=np.float32).reshape(5, 1)
    teachers = np.zeros((5, 2), dtype=np.float32)
    store.add(
        x,
        teachers,
        tolerances=np.ones(5, dtype=np.float32),
        weights=np.ones(5, dtype=np.float32),
        importances=np.asarray([0.0, 10.0, 2.0, 7.0, 3.0], dtype=np.float32),
    )

    assert len(store) == 3
    assert set(np.asarray(store.importance).tolist()) == {10.0, 7.0, 3.0}

    store.add(
        np.asarray([[99.0]], dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        tolerances=np.ones(1, dtype=np.float32),
        weights=np.ones(1, dtype=np.float32),
        importances=np.asarray([-1.0], dtype=np.float32),
    )
    assert set(np.asarray(store.importance).tolist()) == {10.0, 7.0, 3.0}


def test_sample_and_all_batch_shapes():
    store = AnchorStore(capacity=4)
    store.add(
        np.zeros((4, 3), dtype=np.float32),
        np.zeros((4, 2), dtype=np.float32),
        tolerances=np.zeros(4, dtype=np.float32),
        weights=np.ones(4, dtype=np.float32),
        importances=np.arange(4, dtype=np.float32),
        labels=np.arange(4, dtype=np.int32),
    )

    sample = store.sample(0, 2)
    assert sample.x.shape == (2, 3)
    assert sample.teacher.shape == (2, 2)
    assert sample.tolerance.shape == (2,)

    all_batch = store.all_batch()
    assert all_batch.x.shape == (4, 3)
    replay_x, replay_y = store.sample_examples(1, 3)
    assert replay_x.shape == (3, 3)
    assert replay_y.shape == (3,)


def test_can_delete_requires_cover_teacher_embedding_skill_and_sentinel():
    store = AnchorStore(capacity=2)
    store.add(
        np.asarray([[0.0, 1.0]], dtype=np.float32),
        np.asarray([[1.0, 0.0]], dtype=np.float32),
        tolerances=np.asarray([0.05], dtype=np.float32),
        weights=np.ones(1, dtype=np.float32),
        importances=np.ones(1, dtype=np.float32),
        embeddings=np.asarray([[0.0, 0.0]], dtype=np.float32),
        skill_ids=["s0"],
    )

    cover = Anchor(
        x=None,
        teacher=np.asarray([1.0, 0.0], dtype=np.float32),
        tolerance=0.05,
        weight=1.0,
        importance=1.0,
        embedding=np.asarray([0.01, 0.0], dtype=np.float32),
        skill_id="s0",
    )

    assert store.can_delete(0, cover, DummyAdapter(), sentinel_ok=True)
    assert not store.can_delete(0, cover, DummyAdapter(), sentinel_ok=False)

    far = Anchor(**{**cover.__dict__, "embedding": np.asarray([10.0, 0.0])})
    assert not store.can_delete(0, far, DummyAdapter(), sentinel_ok=True)

    wrong_teacher = Anchor(**{**cover.__dict__, "teacher": np.asarray([0.0, 1.0])})
    assert not store.can_delete(0, wrong_teacher, DummyAdapter(), sentinel_ok=True)
