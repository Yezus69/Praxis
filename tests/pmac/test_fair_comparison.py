import numpy as np
import jax.numpy as jnp

from pmac.auditor import Audit, Auditor
from pmac.config import ExperimentConfig, PMAConfig
from pmac.continual import (
    clip_global,
    evaluate_all_tasks,
    run_baseline,
    run_pmac,
    split_train_validation,
)
from pmac.data.streams import Task
from pmac.tree_utils import tree_norm


def _tiny_tasks(n=48, seed=0):
    rng = np.random.default_rng(seed)
    x0_train = rng.normal(size=(n, 4)).astype(np.float32)
    x0_test = (100.0 + rng.normal(size=(n, 4))).astype(np.float32)
    y0_train = (np.arange(n) % 2).astype(np.int32)
    y0_test = ((np.arange(n) + 1) % 2).astype(np.int32)

    perm = np.asarray([2, 0, 3, 1], dtype=np.int32)
    x1_train = (10.0 + rng.normal(size=(n, 4))).astype(np.float32)
    x1_test = (200.0 + rng.normal(size=(n, 4))).astype(np.float32)
    y1_train = ((np.arange(n) + 1) % 2).astype(np.int32)
    y1_test = (np.arange(n) % 2).astype(np.int32)

    return [
        Task("fair0", x0_train, y0_train, x0_test, y0_test, {"task_id": 0, "source": "fair"}),
        Task(
            "fair1",
            x1_train[:, perm],
            y1_train,
            x1_test[:, perm],
            y1_test,
            {"task_id": 1, "source": "fair"},
        ),
    ]


def _exp_cfg(**overrides):
    cfg = dict(
        hidden_sizes=(8,),
        epochs_per_task=1,
        batch_size=8,
        lr=0.05,
        optimizer="sgd",
        replay_batch=4,
        num_guard_nodes=1,
        max_eval=16,
        use_jit=False,
        val_size=5000,
        baseline_clip=True,
        max_grad_norm=0.5,
    )
    cfg.update(overrides)
    return ExperimentConfig(**cfg)


def _pma_cfg(**overrides):
    cfg = dict(
        anchor_memory_per_skill=16,
        sentinel_count_per_skill=4,
        guard_lambda=2.0,
        audit_interval=1,
        consolidation_enabled=False,
        num_guard_nodes=1,
        max_grad_norm=0.5,
    )
    cfg.update(overrides)
    return PMAConfig(**cfg)


def test_baseline_clipping_runs_and_global_clip_caps_large_grad():
    tasks = _tiny_tasks(n=40)
    result = run_baseline(tasks, _exp_cfg(), seed=7)

    assert result.acc_matrix.shape == (2, 2)
    assert result.extra["baseline_clip"] is True
    assert result.extra["max_grad_norm"] == 0.5

    clipped = clip_global({"w": jnp.asarray([300.0, 400.0])}, max_norm=0.5)
    assert float(np.asarray(tree_norm(clipped))) <= 0.5 + 1e-6


def test_replay_only_mixes_replay_without_other_protection():
    tasks = _tiny_tasks(n=48)
    pma_cfg = _pma_cfg(
        consolidation_enabled=True,
        consolidation_interval=1,
        stability_enabled=True,
        gate_enabled=True,
    )

    result = run_pmac(tasks, _exp_cfg(), pma_cfg, seed=3, ablation="replay_only")
    extra = result.extra

    assert result.mode == "pmac_replay_only"
    assert extra["replay_mixed_steps"] > 0
    assert extra["max_effective_batch"] > extra["base_batch_size"]
    assert extra["guard_enabled"] is False
    assert extra["projection_enabled"] is False
    assert extra["stability_enabled"] is False
    assert extra["gate_enabled"] is False
    assert extra["consolidation_enabled"] is False
    assert extra["guard_grad_count"] == 0
    assert extra["projection_steps"] == 0
    assert extra["stability_scaled_steps"] == 0
    assert extra["gate_audit_steps"] == 0
    assert extra["consolidation_steps"] == 0


class RecordingAdapter:
    def __init__(self):
        self.seen_x = []

    def evaluate_skill(self, params, skill_eval_set):
        self.seen_x.append(np.asarray(skill_eval_set["x"]))
        return float(np.asarray(skill_eval_set["y"]).shape[0])


def test_reported_eval_uses_test_and_training_decisions_use_validation(monkeypatch):
    tasks = _tiny_tasks(n=50)
    split0 = split_train_validation(tasks[0], val_size=5000)
    split1 = split_train_validation(tasks[1], val_size=5000)

    assert split0.val_size == 5
    assert split0.train_x.shape[0] == 45
    assert np.array_equal(split0.train_x, tasks[0].train_x[:-5])
    assert np.array_equal(split0.val_x, tasks[0].train_x[-5:])

    adapter = RecordingAdapter()
    evaluate_all_tasks(params=None, tasks=tasks, adapter=adapter, max_eval=7)
    assert np.array_equal(adapter.seen_x[0], tasks[0].test_x[:7])
    assert np.array_equal(adapter.seen_x[1], tasks[1].test_x[:7])

    gate_batches = []

    def capture_gate(self, cand_params, prev_params, source_eval, protected_nodes, adapter):
        gate_batches.append(np.asarray(source_eval["x"]))
        return Audit(True, 0.0, True, True, [], {})

    monkeypatch.setattr(Auditor, "evaluate_candidate", capture_gate)
    pma_cfg = _pma_cfg(stability_enabled=False, gate_enabled=True, audit_interval=1)
    result = run_pmac(tasks, _exp_cfg(), pma_cfg, seed=11, ablation="no_conservation")

    assert gate_batches
    assert all(np.array_equal(batch, split1.val_x) for batch in gate_batches)
    assert not any(np.array_equal(batch, tasks[1].test_x[: batch.shape[0]]) for batch in gate_batches)
    assert result.extra["train_rows_per_task"] == [split0.train_x.shape[0], split1.train_x.shape[0]]
    assert result.extra["val_rows_per_task"] == [split0.val_size, split1.val_size]
    assert result.extra["sentinel_rows_per_task"] == [4, 4]
    assert result.extra["sentinel_source"] == "validation"
    assert result.extra["gate_source"] == "validation"
    assert result.extra["eval_source"] == "test"
