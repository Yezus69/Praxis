import numpy as np

from pmac.config import ExperimentConfig, PMAConfig
from pmac.continual import run_baseline, run_pmac
from pmac.data.streams import Task, iterate_minibatches


def _tiny_tasks(n=32, seed=0):
    rng = np.random.default_rng(seed)
    x0_train = rng.normal(size=(n, 4)).astype(np.float32)
    x0_test = rng.normal(size=(n, 4)).astype(np.float32)
    y0_train = (x0_train[:, 0] + 0.25 * x0_train[:, 1] > 0.0).astype(np.int32)
    y0_test = (x0_test[:, 0] + 0.25 * x0_test[:, 1] > 0.0).astype(np.int32)

    perm = np.asarray([2, 0, 3, 1], dtype=np.int32)
    x1_train = x0_train[:, perm]
    x1_test = x0_test[:, perm]
    y1_train = (x1_train[:, 0] - 0.2 * x1_train[:, 1] > 0.0).astype(np.int32)
    y1_test = (x1_test[:, 0] - 0.2 * x1_test[:, 1] > 0.0).astype(np.int32)

    return [
        Task("tiny0", x0_train, y0_train, x0_test, y0_test, {"task_id": 0, "source": "tiny"}),
        Task("tiny1", x1_train, y1_train, x1_test, y1_test, {"task_id": 1, "source": "tiny"}),
    ]


def _exp_cfg(use_jit):
    return ExperimentConfig(
        hidden_sizes=(16,),
        epochs_per_task=1,
        batch_size=8,
        lr=0.05,
        optimizer="sgd",
        replay_batch=8,
        num_guard_nodes=1,
        max_eval=32,
        use_jit=use_jit,
    )


def _pma_cfg():
    return PMAConfig(
        anchor_memory_per_skill=16,
        sentinel_count_per_skill=8,
        guard_lambda=2.0,
        audit_interval=0,
        consolidation_enabled=False,
        num_guard_nodes=1,
    )


def test_baseline_jit_matches_eager_on_tiny_stream():
    tasks = _tiny_tasks()
    eager = run_baseline(tasks, _exp_cfg(use_jit=False), seed=3)
    jitted = run_baseline(tasks, _exp_cfg(use_jit=True), seed=3)

    assert np.allclose(jitted.acc_matrix, eager.acc_matrix, atol=1e-3)


def test_pmac_jit_matches_eager_on_tiny_stream():
    tasks = _tiny_tasks()
    pma_cfg = _pma_cfg()
    eager = run_pmac(tasks, _exp_cfg(use_jit=False), pma_cfg, seed=3)
    jitted = run_pmac(tasks, _exp_cfg(use_jit=True), pma_cfg, seed=3)

    assert np.allclose(jitted.acc_matrix, eager.acc_matrix, atol=1e-3)


def test_iterate_minibatches_drop_last_keeps_fixed_batch_shape():
    x = np.arange(30, dtype=np.float32).reshape(10, 3)
    y = np.arange(10, dtype=np.int32)
    key = np.asarray([0, 7], dtype=np.uint32)

    default_batches = list(iterate_minibatches(key, x, y, batch_size=4))
    fixed_batches = list(iterate_minibatches(key, x, y, batch_size=4, drop_last=True))

    assert [batch_x.shape[0] for batch_x, _ in default_batches] == [4, 4, 2]
    assert [batch_x.shape[0] for batch_x, _ in fixed_batches] == [4, 4]
    assert np.array_equal(
        np.concatenate([batch_y for _, batch_y in fixed_batches]),
        np.concatenate([batch_y for _, batch_y in default_batches[:2]]),
    )
