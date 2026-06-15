"""Matched continual-learning runners for PMA-C supervised experiments."""

from __future__ import annotations

from dataclasses import dataclass, field

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pmac.anchors import AnchorStore
from pmac.atlas import Atlas
from pmac.auditor import Audit, Auditor
from pmac.behavior_distance import kl_categorical
from pmac.checkpoint import ChampionStore, SafeCheckpoint, deep_copy_pytree
from pmac.config import ExperimentConfig, PMAConfig
from pmac.conservation import conservation_loss
from pmac.consolidation import consolidate
from pmac.data.streams import iterate_minibatches
from pmac.memory_selector import importance_scores, select_indices
from pmac.models.mlp import init_mlp
from pmac.projection import project_conflicts
from pmac.router import Router
from pmac.scheduler import Scheduler
from pmac.sentinels import SentinelStore
from pmac.stability import scale_by_stability, update_omega, zeros_omega_like
from pmac.tree_utils import tree_add_scaled


@dataclass
class ContinualResult:
    acc_matrix: np.ndarray
    learned_acc: np.ndarray
    final_acc: np.ndarray
    peak_acc: np.ndarray
    metrics: dict
    mode: str
    source_tag: str
    extra: dict = field(default_factory=dict)


def _source_tag(tasks):
    if not tasks:
        return "unknown"
    return str(tasks[0].meta.get("source", tasks[0].name))


def _num_classes(tasks):
    max_label = 0
    for task in tasks:
        max_label = max(max_label, int(np.max(task.train_y)), int(np.max(task.test_y)))
    return max_label + 1


def _init_params(tasks, exp_cfg, seed):
    input_dim = int(tasks[0].train_x.shape[1])
    n_classes = _num_classes(tasks)
    layer_sizes = [input_dim, *tuple(exp_cfg.hidden_sizes), n_classes]
    return init_mlp(jax.random.PRNGKey(int(seed)), layer_sizes)


def _make_optimizer(exp_cfg):
    name = str(exp_cfg.optimizer).lower()
    if name == "sgd":
        return optax.sgd(float(exp_cfg.lr))
    if name == "adam":
        return optax.adam(float(exp_cfg.lr))
    raise ValueError(f"unknown optimizer: {exp_cfg.optimizer}")


def _batch(x, y):
    return {"x": jnp.asarray(x), "y": jnp.asarray(y, dtype=jnp.int32)}


def _eval_set(task, max_eval):
    n = min(int(max_eval), int(task.test_x.shape[0]))
    return _batch(task.test_x[:n], task.test_y[:n])


def evaluate_all_tasks(params, tasks, adapter, max_eval=2000) -> np.ndarray:
    scores = []
    for task in tasks:
        scores.append(adapter.evaluate_skill(params, _eval_set(task, max_eval)))
    return np.asarray(scores, dtype=np.float32)


def compute_metrics(acc_matrix) -> dict:
    acc_matrix = np.asarray(acc_matrix, dtype=np.float32)
    final = acc_matrix[-1]
    learned = np.diag(acc_matrix)
    peak = np.max(acc_matrix, axis=0)
    if acc_matrix.shape[0] > 1:
        bwt = np.mean(final[:-1] - learned[:-1])
        forgetting = np.mean(peak[:-1] - final[:-1])
    else:
        bwt = 0.0
        forgetting = 0.0
    retention = final / np.maximum(peak, 1e-9)
    return {
        "ACC": float(np.mean(final)),
        "BWT": float(bwt),
        "forgetting": float(forgetting),
        "Forgetting": float(forgetting),
        "mean_retention": float(np.mean(retention)),
        "worst_retention": float(np.min(retention)),
        "retention": retention.astype(float).tolist(),
        "learned_acc": learned.astype(float).tolist(),
        "final_acc": final.astype(float).tolist(),
        "peak_acc": peak.astype(float).tolist(),
    }


def _result(acc_matrix, mode, source_tag, extra=None):
    metrics = compute_metrics(acc_matrix)
    acc_matrix = np.asarray(acc_matrix, dtype=np.float32)
    return ContinualResult(
        acc_matrix=acc_matrix,
        learned_acc=np.diag(acc_matrix).astype(np.float32),
        final_acc=acc_matrix[-1].astype(np.float32),
        peak_acc=np.max(acc_matrix, axis=0).astype(np.float32),
        metrics=metrics,
        mode=mode,
        source_tag=source_tag,
        extra=dict(extra or {}),
    )


def run_baseline(tasks, exp_cfg, seed) -> ContinualResult:
    from pmac.adapters.supervised import SupervisedAdapter

    exp_cfg = exp_cfg or ExperimentConfig(seed=seed)
    adapter = SupervisedAdapter(temperature=exp_cfg.temperature)
    params = _init_params(tasks, exp_cfg, seed)
    opt = _make_optimizer(exp_cfg)
    opt_state = opt.init(params)
    acc_matrix = np.zeros((len(tasks), len(tasks)), dtype=np.float32)

    for task_i, task in enumerate(tasks):
        for epoch in range(int(exp_cfg.epochs_per_task)):
            key = jax.random.PRNGKey(int(seed) + 10_007 * task_i + epoch)
            for x_np, y_np in iterate_minibatches(
                key, task.train_x, task.train_y, exp_cfg.batch_size
            ):
                batch = _batch(x_np, y_np)
                grads = jax.grad(adapter.current_loss)(params, batch)
                updates, opt_state = opt.update(grads, opt_state, params)
                params = optax.apply_updates(params, updates)
        acc_matrix[task_i] = evaluate_all_tasks(
            params, tasks, adapter, max_eval=exp_cfg.max_eval
        )

    return _result(
        acc_matrix,
        mode="baseline",
        source_tag=_source_tag(tasks),
        extra={"seed": int(seed), "optimizer": exp_cfg.optimizer},
    )


def _guard_loss(params, node, adapter, temperature, key=None, n=None):
    if len(node.anchors) == 0:
        return jnp.array(0.0)
    if n is None:
        batch = node.anchors.all_batch()
    else:
        batch = node.anchors.sample(key, n)
    behavior_fn = lambda p, x: adapter.behavior(p, {"x": x})
    distance_fn = lambda teacher, cur: kl_categorical(teacher, cur, temperature)
    return conservation_loss(behavior_fn, params, batch, distance_fn)


def _sample_replay(atlas, key, n):
    nodes = [node for node in atlas.protected_nodes() if node.anchors.label is not None]
    nodes = [node for node in nodes if len(node.anchors) > 0]
    if not nodes or n <= 0:
        return None
    rng = np.random.default_rng(int(np.asarray(key, dtype=np.uint32).sum()))
    per_node = max(1, int(np.ceil(n / len(nodes))))
    xs = []
    ys = []
    for node in nodes:
        sample_key = rng.integers(0, 2**31 - 1)
        try:
            x_old, y_old = node.anchors.sample_examples(sample_key, per_node)
        except ValueError:
            continue
        xs.append(x_old)
        ys.append(y_old)
    if not xs:
        return None
    x = np.concatenate(xs, axis=0)[:n]
    y = np.concatenate(ys, axis=0)[:n]
    return x, y


def _maybe_mix_replay(x_np, y_np, atlas, step, exp_cfg, ablation):
    if ablation == "no_replay":
        return x_np, y_np
    replay = _sample_replay(atlas, step + 17, int(exp_cfg.replay_batch))
    if replay is None:
        return x_np, y_np
    x_old, y_old = replay
    x = np.concatenate([x_np, x_old], axis=0)
    y = np.concatenate([y_np, y_old], axis=0)
    return x, y


def _certify_task(
    params,
    task,
    task_i,
    atlas,
    champions,
    router,
    adapter,
    exp_cfg,
    pma_cfg,
    seed,
    ablation,
):
    train_batch = _batch(task.train_x, task.train_y)
    logits = np.asarray(adapter.behavior(params, train_batch))
    mode = "random" if ablation == "random_memory" else "importance"
    n_anchor = min(int(pma_cfg.anchor_memory_per_skill), int(task.train_x.shape[0]))
    idx = select_indices(
        logits,
        task.train_y,
        n_anchor,
        mode=mode,
        key=int(seed) + 51_001 * (task_i + 1),
    )
    scores = importance_scores(logits[idx], task.train_y[idx])
    teachers = logits[idx]
    tolerances = np.full((idx.shape[0],), pma_cfg.drift_budget_kl, dtype=np.float32)
    weights = np.ones((idx.shape[0],), dtype=np.float32)
    anchors = AnchorStore(pma_cfg.anchor_memory_per_skill)
    anchors.add(
        task.train_x[idx],
        teachers,
        tolerances,
        weights,
        scores,
        skill_ids=[task.name] * idx.shape[0],
        labels=task.train_y[idx],
    )

    n_sentinel = min(int(pma_cfg.sentinel_count_per_skill), int(idx.shape[0]))
    sentinel_idx = idx[:n_sentinel]
    sentinels = SentinelStore(
        x=task.train_x[sentinel_idx],
        y=task.train_y[sentinel_idx],
        seeds=np.arange(n_sentinel, dtype=np.int32),
    )
    eval_score = adapter.evaluate_skill(params, _eval_set(task, exp_cfg.max_eval))
    champion = champions.freeze(
        params,
        route=task.name,
        meta={"skill_id": task.name, "task_index": int(task_i)},
    )
    router.register(task.meta.get("task_id", task.name), task.name)
    node = atlas.create_or_update_node(
        task.name,
        context_key=task.meta.get("task_id", task.name),
        anchors=anchors,
        sentinels=sentinels,
        status="protected",
        champion_ref=champion,
        best_score=eval_score,
        current_score=eval_score,
        retention=1.0,
        allowed_regression=pma_cfg.allowed_regression,
        last_certified_step=task_i,
        guard_lambda=pma_cfg.guard_lambda,
        certified_impls=[task.name],
    )
    return node


def run_pmac(tasks, exp_cfg, pma_cfg, seed, ablation=None) -> ContinualResult:
    from pmac.adapters.supervised import SupervisedAdapter

    allowed = {
        None,
        "no_projection",
        "no_conservation",
        "no_replay",
        "random_memory",
        "no_stability",
        "no_gate",
    }
    if ablation not in allowed:
        raise ValueError(f"unknown PMA-C ablation: {ablation}")

    exp_cfg = exp_cfg or ExperimentConfig(seed=seed)
    pma_cfg = pma_cfg or PMAConfig()
    adapter = SupervisedAdapter(temperature=exp_cfg.temperature)
    params = _init_params(tasks, exp_cfg, seed)
    omega = zeros_omega_like(params)
    opt = _make_optimizer(exp_cfg)
    opt_state = opt.init(params)
    safe = SafeCheckpoint(params)
    safe_opt_state = deep_copy_pytree(opt_state)
    auditor = Auditor(delta_current=pma_cfg.delta_current, delta_cons=pma_cfg.delta_cons)
    atlas = Atlas()
    champions = ChampionStore()
    scheduler = Scheduler(pma_cfg.old_skill_sample_fraction)
    router = Router()
    acc_matrix = np.zeros((len(tasks), len(tasks)), dtype=np.float32)
    rollback_count = 0
    global_step = 0
    guard_loss_trace = []

    for task_i, task in enumerate(tasks):
        current_skill_id = task.name
        for epoch in range(int(exp_cfg.epochs_per_task)):
            key = jax.random.PRNGKey(int(seed) + 10_007 * task_i + epoch)
            for x_np, y_np in iterate_minibatches(
                key, task.train_x, task.train_y, exp_cfg.batch_size
            ):
                global_step += 1
                x_train, y_train = _maybe_mix_replay(
                    x_np, y_np, atlas, global_step, exp_cfg, ablation
                )
                batch = _batch(x_train, y_train)
                prev_params = params
                g_new = jax.grad(adapter.current_loss)(params, batch)

                guard_nodes = atlas.sample_protected_nodes(
                    current_skill_id,
                    getattr(exp_cfg, "num_guard_nodes", pma_cfg.num_guard_nodes),
                )
                guard_grads = []
                guard_losses = []
                if ablation != "no_conservation":
                    for guard_i, node in enumerate(guard_nodes):
                        n_guard = min(len(node.anchors), max(1, int(exp_cfg.replay_batch)))
                        loss_fn = lambda p, n=node, gi=guard_i: _guard_loss(
                            p,
                            n,
                            adapter,
                            exp_cfg.temperature,
                            key=global_step + gi * 997,
                            n=n_guard,
                        )
                        loss_value, grad_value = jax.value_and_grad(loss_fn)(params)
                        guard_losses.append(float(loss_value))
                        guard_grads.append(grad_value)

                if (
                    pma_cfg.projection_enabled
                    and ablation != "no_projection"
                    and guard_grads
                ):
                    g_total = project_conflicts(g_new, guard_grads)
                else:
                    g_total = g_new

                if ablation != "no_conservation":
                    for node, g_guard in zip(guard_nodes, guard_grads):
                        lam = min(float(node.guard_lambda), float(pma_cfg.guard_lambda_max))
                        g_total = tree_add_scaled(g_total, g_guard, lam)

                if pma_cfg.stability_enabled and ablation != "no_stability":
                    g_total = scale_by_stability(
                        g_total, omega, alpha=pma_cfg.stability_alpha
                    )

                updates, cand_opt_state = opt.update(g_total, opt_state, params)
                candidate = optax.apply_updates(params, updates)

                audit_ran = (
                    ablation != "no_gate"
                    and pma_cfg.audit_interval > 0
                    and global_step % int(pma_cfg.audit_interval) == 0
                    and bool(atlas.protected_nodes())
                )
                if audit_ran:
                    source_eval = _eval_set(task, exp_cfg.max_eval)
                    audit = auditor.evaluate_candidate(
                        candidate,
                        prev_params,
                        source_eval,
                        atlas.protected_nodes(),
                        adapter,
                    )
                    if audit.accept:
                        params = candidate
                        opt_state = cand_opt_state
                        safe.update_if_safe(params, audit)
                        safe_opt_state = deep_copy_pytree(opt_state)
                    else:
                        rollback_count += 1
                        params = safe.restore()
                        opt_state = deep_copy_pytree(safe_opt_state)
                        scheduler.boost(audit.regressed_nodes)
                else:
                    params = candidate
                    opt_state = cand_opt_state

                if guard_losses:
                    guard_loss_trace.append(float(np.mean(guard_losses)))

                if (
                    pma_cfg.consolidation_enabled
                    and pma_cfg.consolidation_interval > 0
                    and global_step % int(pma_cfg.consolidation_interval) == 0
                    and atlas.protected_nodes()
                ):
                    params, _ = consolidate(
                        params,
                        atlas,
                        adapter,
                        pma_cfg,
                        omega=omega,
                        steps=pma_cfg.consolidation_epochs,
                        lr=exp_cfg.lr * pma_cfg.slow_lr_multiplier,
                    )

        acc_matrix[task_i] = evaluate_all_tasks(
            params, tasks, adapter, max_eval=exp_cfg.max_eval
        )
        node = _certify_task(
            params,
            task,
            task_i,
            atlas,
            champions,
            router,
            adapter,
            exp_cfg,
            pma_cfg,
            seed,
            ablation,
        )
        grad_guard = jax.grad(
            lambda p, n=node: _guard_loss(
                p, n, adapter, exp_cfg.temperature, key=task_i + 1, n=None
            )
        )(params)
        omega = update_omega(omega, params, grad_guard, pma_cfg.stability_decay)
        safe = SafeCheckpoint(params)
        safe_opt_state = deep_copy_pytree(opt_state)

    mode = "pmac" if ablation is None else f"pmac_{ablation}"
    return _result(
        acc_matrix,
        mode=mode,
        source_tag=_source_tag(tasks),
        extra={
            "seed": int(seed),
            "ablation": ablation,
            "rollback_count": int(rollback_count),
            "protected_skills": list(atlas.nodes.keys()),
            "guard_loss_trace": guard_loss_trace,
            "optimizer": exp_cfg.optimizer,
        },
    )


__all__ = [
    "ContinualResult",
    "evaluate_all_tasks",
    "run_baseline",
    "run_pmac",
    "compute_metrics",
]
