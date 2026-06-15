import numpy as np

from pmac.config import ExperimentConfig, PMAConfig
from pmac.continual import run_baseline, run_pmac
from pmac.data.streams import Task


def _conflicting_tasks(n=80, seed=0):
    rng = np.random.default_rng(seed)
    x_train = rng.normal(size=(n, 2)).astype(np.float32)
    y_train = (x_train[:, 0] > 0).astype(np.int32)
    x_test = rng.normal(size=(n, 2)).astype(np.float32)
    y_test = (x_test[:, 0] > 0).astype(np.int32)

    task0 = Task("task0", x_train, y_train, x_test, y_test, {"task_id": 0, "source": "synthetic"})
    task1 = Task(
        "task1",
        x_train,
        1 - y_train,
        x_test,
        1 - y_test,
        {"task_id": 1, "source": "synthetic"},
    )
    return [task0, task1]


def test_matched_baseline_and_pmac_synthetic_smoke():
    tasks = _conflicting_tasks()
    exp_cfg = ExperimentConfig(
        hidden_sizes=(32,),
        epochs_per_task=4,
        batch_size=16,
        lr=0.1,
        optimizer="sgd",
        replay_batch=16,
        num_guard_nodes=1,
        max_eval=80,
    )
    pma_cfg = PMAConfig(
        anchor_memory_per_skill=80,
        sentinel_count_per_skill=40,
        guard_lambda=10.0,
        audit_interval=1,
        allowed_regression=0.0,
        consolidation_enabled=False,
    )

    baseline = run_baseline(tasks, exp_cfg, seed=0)
    pmac = run_pmac(tasks, exp_cfg, pma_cfg, seed=0)

    assert baseline.acc_matrix.shape == (2, 2)
    assert pmac.acc_matrix.shape == (2, 2)
    for key in ("ACC", "BWT", "forgetting", "mean_retention", "worst_retention"):
        assert key in baseline.metrics
        assert key in pmac.metrics
    assert pmac.metrics["mean_retention"] >= baseline.metrics["mean_retention"]
