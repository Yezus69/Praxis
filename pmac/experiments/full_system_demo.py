"""CPU-runnable PMA-C full-system integration demo.

This script exercises PMA-C surfaces that the headline supervised experiment
does not stress directly: champion immutability, non-deletion, no-op growth,
adapter-only plasticity, consolidation gating, and context routing.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Any

os.environ.setdefault("JAX_PLATFORM_NAME", "cpu")

import jax
import jax.numpy as jnp
import optax

from pmac.adapters.supervised import SupervisedAdapter
from pmac.anchors import AnchorStore
from pmac.atlas import Atlas
from pmac.checkpoint import (
    ChampionStore,
    can_archive_expert,
    deep_copy_pytree,
    mark_redundant,
)
from pmac.config import PMAConfig
from pmac.consolidation import consolidate
from pmac.data.streams import build_stream
from pmac.models.mlp import grow_adapter, init_mlp, mlp_apply, num_params
from pmac.router import Router
from pmac.sentinels import SentinelStore


@dataclass(frozen=True)
class DemoConfig:
    seed: int = 7
    n_features: int = 16
    n_classes: int = 4
    n_train: int = 256
    n_test: int = 128
    hidden_dim: int = 32
    anchor_count: int = 48
    sentinel_count: int = 48
    eval_count: int = 96
    train0_steps: int = 180
    train1_steps: int = 140
    train_lr: float = 0.03
    adapter_lr: float = 0.03
    preserve_weight: float = 3.0
    growth_rank: int = 16
    min_task0_acc: float = 0.75
    min_task1_gain: float = 0.02
    max_task0_drop: float = 0.05
    allowed_regression: float = 0.05
    consolidation_steps: int = 25
    consolidation_lr: float = 0.2
    slow_bias_scale: float = 0.03


@dataclass(frozen=True)
class Impl:
    impl_id: str
    params: Any


def _batch(x, y=None, count=None):
    if count is not None:
        x = x[: int(count)]
        if y is not None:
            y = y[: int(count)]
    out = {"x": jnp.asarray(x, dtype=jnp.float32)}
    if y is not None:
        out["y"] = jnp.asarray(y, dtype=jnp.int32)
    return out


def _train_batch(task, count=None):
    return _batch(task.train_x, task.train_y, count=count)


def _eval_batch(task, count=None):
    return _batch(task.test_x, task.test_y, count=count)


def _accuracy(adapter, params, task, count=None, split="test") -> float:
    batch = _train_batch(task, count) if split == "train" else _eval_batch(task, count)
    return float(adapter.evaluate_skill(params, batch))


def _tree_allclose(a, b, atol=1e-6) -> bool:
    leaves = jax.tree_util.tree_map(lambda x, y: jnp.allclose(x, y, atol=atol), a, b)
    return all(bool(v) for v in jax.tree_util.tree_leaves(leaves))


def _train_all_params(params, adapter, batch, steps: int, lr: float):
    opt = optax.adam(float(lr))
    opt_state = opt.init(params)

    @jax.jit
    def step(p, state):
        loss, grads = jax.value_and_grad(adapter.current_loss)(p, batch)
        updates, new_state = opt.update(grads, state, p)
        return optax.apply_updates(p, updates), new_state, loss

    loss = jnp.array(0.0)
    for _ in range(int(steps)):
        params, opt_state, loss = step(params, opt_state)
    return params, float(loss)


def _mask_new_adapter_grads(grads):
    zero_layers = jax.tree_util.tree_map(jnp.zeros_like, grads["layers"])
    adapters = grads.get("adapters", [])
    zero_old = [jax.tree_util.tree_map(jnp.zeros_like, adapter) for adapter in adapters[:-1]]
    return {"layers": zero_layers, "adapters": zero_old + [adapters[-1]]}


def _train_new_adapter_only(
    params,
    adapter,
    task_batch,
    preserve_batch,
    preserve_teacher,
    steps: int,
    lr: float,
    preserve_weight: float,
):
    opt = optax.adam(float(lr))
    opt_state = opt.init(params)
    preserve_weight = float(preserve_weight)

    def loss_fn(p):
        task_loss = adapter.current_loss(p, task_batch)
        cur = adapter.behavior(p, preserve_batch)
        keep_loss = jnp.mean(adapter.distance(cur, preserve_teacher, preserve_batch))
        return task_loss + preserve_weight * keep_loss

    @jax.jit
    def step(p, state):
        loss, grads = jax.value_and_grad(loss_fn)(p)
        grads = _mask_new_adapter_grads(grads)
        updates, new_state = opt.update(grads, state, p)
        return optax.apply_updates(p, updates), new_state, loss

    loss = jnp.array(0.0)
    for _ in range(int(steps)):
        params, opt_state, loss = step(params, opt_state)
    return params, float(loss)


def _make_skill_node(
    atlas,
    champions,
    adapter,
    params,
    task,
    skill_id,
    context_key,
    impl_id,
    cfg: DemoConfig,
):
    n_anchor = min(int(cfg.anchor_count), int(task.train_x.shape[0]))
    n_sentinel = min(
        int(cfg.sentinel_count),
        max(1, int(task.train_x.shape[0]) - n_anchor),
    )
    anchor_x = task.train_x[:n_anchor]
    anchor_y = task.train_y[:n_anchor]
    sentinel_x = task.train_x[n_anchor : n_anchor + n_sentinel]
    sentinel_y = task.train_y[n_anchor : n_anchor + n_sentinel]
    if sentinel_x.shape[0] == 0:
        sentinel_x = task.train_x[:n_sentinel]
        sentinel_y = task.train_y[:n_sentinel]

    teachers = adapter.behavior(params, {"x": jnp.asarray(anchor_x, dtype=jnp.float32)})
    confidence = jnp.max(jax.nn.softmax(teachers, axis=-1), axis=-1)
    anchors = AnchorStore(capacity=n_anchor)
    anchors.add(
        anchor_x,
        teachers,
        jnp.zeros((n_anchor,), dtype=jnp.float32),
        jnp.ones((n_anchor,), dtype=jnp.float32),
        confidence,
        skill_ids=[str(skill_id)] * n_anchor,
        labels=anchor_y,
    )

    sentinels = SentinelStore(
        x=sentinel_x,
        y=sentinel_y,
        seeds=jnp.arange(sentinel_x.shape[0], dtype=jnp.int32),
    )
    best_score = sentinels.evaluate(params, adapter)
    champion = champions.freeze(params, route=impl_id, meta={"skill_id": impl_id})
    return atlas.create_or_update_node(
        skill_id,
        context_key=context_key,
        anchors=anchors,
        sentinels=sentinels,
        status="protected",
        champion_ref=champion,
        best_score=best_score,
        current_score=best_score,
        retention=1.0,
        allowed_regression=cfg.allowed_regression,
        certified_impls=[impl_id],
    )


def _prepare_state(cfg: DemoConfig) -> dict[str, Any]:
    tasks, source_tag = build_stream(
        "synthetic",
        num_tasks=2,
        seed=cfg.seed,
        n_features=cfg.n_features,
        n_classes=cfg.n_classes,
        n_train=cfg.n_train,
        n_test=cfg.n_test,
    )
    adapter = SupervisedAdapter(temperature=2.0, growth_rank=cfg.growth_rank)
    key = jax.random.PRNGKey(cfg.seed)
    params = init_mlp(key, [cfg.n_features, cfg.hidden_dim, cfg.n_classes])
    params, loss = _train_all_params(
        params,
        adapter,
        _train_batch(tasks[0]),
        steps=cfg.train0_steps,
        lr=cfg.train_lr,
    )
    task0_acc = _accuracy(adapter, params, tasks[0], cfg.eval_count, split="test")
    assert task0_acc >= cfg.min_task0_acc, (
        f"task-0 accuracy {task0_acc:.3f} below {cfg.min_task0_acc:.3f}"
    )

    atlas = Atlas()
    champions = ChampionStore()
    node0 = _make_skill_node(
        atlas,
        champions,
        adapter,
        params,
        tasks[0],
        skill_id="task0",
        context_key=tasks[0].meta.get("task_id", 0),
        impl_id="task0_champion",
        cfg=cfg,
    )
    return {
        "tasks": tasks,
        "source_tag": source_tag,
        "adapter": adapter,
        "params0": params,
        "task0_loss": loss,
        "task0_acc": task0_acc,
        "atlas": atlas,
        "champions": champions,
        "node0": node0,
    }


def section_a_champions_non_deletion(state, cfg: DemoConfig) -> bool:
    node = state["node0"]
    impl_id = "task0_champion"
    champion_leaf_before = jnp.array(node.champion_ref.params["layers"][0]["w"], copy=True)
    mutated_live = deep_copy_pytree(state["params0"])
    mutated_live["layers"][0]["w"] = mutated_live["layers"][0]["w"] + 5.0

    champion_unchanged = jnp.allclose(
        node.champion_ref.params["layers"][0]["w"], champion_leaf_before
    )
    live_changed = not jnp.allclose(mutated_live["layers"][0]["w"], champion_leaf_before)
    assert bool(champion_unchanged) and bool(live_changed), "champion params changed with live mutation"

    refused_last = not mark_redundant(node, impl_id)
    assert refused_last, "mark_redundant allowed deleting the only certified impl"

    node.certified_impls.append("task0_slow_core")
    accepted_with_cover = mark_redundant(node, impl_id)
    assert accepted_with_cover, "mark_redundant refused after a second certified impl was added"

    print(
        "[A] champions/non-deletion : PASS "
        f"(task0_acc={state['task0_acc']:.3f}, certified={node.certified_impls})"
    )
    return True


def section_b_growth_plasticity(state, cfg: DemoConfig) -> bool:
    adapter = state["adapter"]
    task0, task1 = state["tasks"][:2]
    params0 = state["params0"]
    hidden_dim = int(params0["layers"][-1]["w"].shape[0])
    before_params = num_params(params0)
    grown = grow_adapter(
        jax.random.PRNGKey(cfg.seed + 101),
        params0,
        hidden_dim=hidden_dim,
        rank=cfg.growth_rank,
    )
    after_params = num_params(grown)
    probe = _train_batch(task0, cfg.eval_count)["x"]
    before_logits = mlp_apply(params0, probe)
    after_logits = mlp_apply(grown, probe)

    assert after_params > before_params, "growth did not increase parameter count"
    assert bool(jnp.allclose(before_logits, after_logits, atol=1e-5)), (
        "grow_adapter was not a no-op at initialization"
    )

    preserve_batch = _train_batch(task0, cfg.eval_count)
    preserve_teacher = adapter.behavior(grown, preserve_batch)
    task1_batch = _train_batch(task1, cfg.eval_count)
    task1_before = _accuracy(adapter, grown, task1, cfg.eval_count, split="train")
    task0_before = _accuracy(adapter, grown, task0, cfg.eval_count, split="train")
    trained, loss = _train_new_adapter_only(
        grown,
        adapter,
        task1_batch,
        preserve_batch,
        preserve_teacher,
        steps=cfg.train1_steps,
        lr=cfg.adapter_lr,
        preserve_weight=cfg.preserve_weight,
    )
    task1_after = _accuracy(adapter, trained, task1, cfg.eval_count, split="train")
    task0_after = _accuracy(adapter, trained, task0, cfg.eval_count, split="train")

    assert _tree_allclose(grown["layers"], trained["layers"]), "base MLP layers changed"
    assert task1_after >= task1_before + cfg.min_task1_gain, (
        f"task1 gain too small: before={task1_before:.3f}, after={task1_after:.3f}"
    )
    assert task0_after >= task0_before - cfg.max_task0_drop, (
        f"task0 dropped from {task0_before:.3f} to {task0_after:.3f}"
    )

    node1 = _make_skill_node(
        state["atlas"],
        state["champions"],
        adapter,
        trained,
        task1,
        skill_id="task1",
        context_key=task1.meta.get("task_id", 1),
        impl_id="task1_adapter",
        cfg=cfg,
    )
    state["params_task1"] = trained
    state["node1"] = node1

    print(
        "[B] growth/plasticity      : PASS "
        f"(params {before_params}->{after_params}, task1 {task1_before:.3f}->{task1_after:.3f}, "
        f"task0 {task0_before:.3f}->{task0_after:.3f}, loss={loss:.4f})"
    )
    return True


def _perturb_output_bias(params, scale: float):
    out = deep_copy_pytree(params)
    bias = out["layers"][-1]["b"]
    pattern = jnp.linspace(-1.0, 1.0, bias.shape[0], dtype=bias.dtype)
    out["layers"][-1]["b"] = bias + float(scale) * pattern
    return out


def _force_constant_class(params, class_id: int):
    out = deep_copy_pytree(params)
    out["layers"][-1]["w"] = jnp.zeros_like(out["layers"][-1]["w"])
    bias = jnp.full_like(out["layers"][-1]["b"], -10.0)
    out["layers"][-1]["b"] = bias.at[int(class_id)].set(10.0)
    return out


def _mean_anchor_champion_distance(params, nodes, adapter) -> float:
    total = jnp.array(0.0)
    count = 0
    for node in nodes:
        batch = node.anchors.all_batch()
        teacher = adapter.behavior(node.champion_ref.params, {"x": batch.x})
        current = adapter.behavior(params, {"x": batch.x})
        total = total + jnp.mean(adapter.distance(current, teacher, batch))
        count += 1
    return float(total / max(count, 1))


def section_c_consolidation(state, cfg: DemoConfig) -> bool:
    atlas = state["atlas"]
    adapter = state["adapter"]
    nodes = atlas.protected_nodes()
    assert len(nodes) >= 2, "consolidation demo requires at least two protected nodes"

    champion_before = {
        node.skill_id: deep_copy_pytree(node.champion_ref.params) for node in nodes
    }
    slow_params = _perturb_output_bias(state["params_task1"], cfg.slow_bias_scale)
    before = _mean_anchor_champion_distance(slow_params, nodes, adapter)
    pma_cfg = PMAConfig(stability_enabled=False, allowed_regression=cfg.allowed_regression)
    consolidated, info = consolidate(
        slow_params,
        atlas,
        adapter,
        pma_cfg,
        steps=cfg.consolidation_steps,
        lr=cfg.consolidation_lr,
    )
    after = _mean_anchor_champion_distance(consolidated, nodes, adapter)

    assert info["steps"] > 0, "consolidation gate rejected every update"
    assert after < before, f"anchor distance did not decrease: {before:.6f}->{after:.6f}"
    assert all(bool(v) for v in info["certified"].values()), (
        f"consolidated params failed certification flags: {info['certified']}"
    )

    node1 = state["node1"]
    labels = jnp.asarray(node1.sentinels.y, dtype=jnp.int32)
    counts = jnp.bincount(labels, length=cfg.n_classes)
    least_frequent_class = int(jnp.argmin(counts))
    bad_params = _force_constant_class(state["params_task1"], least_frequent_class)
    bad_score = node1.sentinels.evaluate(bad_params, adapter)
    threshold = node1.best_score - node1.allowed_regression
    assert bad_score < threshold, (
        f"bad archive probe did not regress sentinel: score={bad_score:.3f}, threshold={threshold:.3f}"
    )
    assert not can_archive_expert(node1, bad_params, adapter), (
        "regressing implementation was archive-certified"
    )

    for node in nodes:
        assert _tree_allclose(node.champion_ref.params, champion_before[node.skill_id]), (
            f"champion for {node.skill_id} mutated during consolidation"
        )

    print(
        "[C] consolidation          : PASS "
        f"(anchor_dist {before:.6f}->{after:.6f}, steps={info['steps']}, bad_score={bad_score:.3f})"
    )
    return True


def section_d_router(state, cfg: DemoConfig) -> bool:
    del cfg
    adapter = state["adapter"]
    task0, task1 = state["tasks"][:2]
    node0 = state["node0"]
    node1 = state["node1"]
    router = Router()
    ctx0 = task0.meta.get("task_id", 0)
    ctx1 = task1.meta.get("task_id", 1)
    router.register(ctx0, "task0_champion")
    router.register(ctx1, "task1_adapter")

    assert router.route(ctx0) == "task0_champion", "router returned wrong task0 impl"
    assert router.route(ctx1) == "task1_adapter", "router returned wrong task1 impl"

    probe = _train_batch(task1, 8)
    z0 = adapter.behavior(node0.champion_ref.params, probe)
    z1 = adapter.behavior(node1.champion_ref.params, probe)
    composed = router.compose_logits([z0, z1], [0.25, 0.75])
    expected = 0.25 * z0 + 0.75 * z1
    assert bool(jnp.allclose(composed, expected)), "compose_logits is not weighted sum"

    right = Impl("task0_champion", node0.champion_ref.params)
    wrong = Impl("task1_adapter", node1.champion_ref.params)
    assert router.certify_route(ctx0, right, node0, adapter), "correct route did not certify"
    assert not router.certify_route(ctx0, wrong, node0, adapter), (
        "certify_route accepted a wrong implementation"
    )

    print("[D] router                 : PASS (contexts 0/1 routed and certified)")
    return True


def run_demo(cfg: DemoConfig | None = None) -> dict[str, bool]:
    cfg = cfg or DemoConfig()
    state: dict[str, Any] = {}
    results: dict[str, bool] = {}

    sections = [
        ("A", "champions/non-deletion", section_a_champions_non_deletion),
        ("B", "growth/plasticity", section_b_growth_plasticity),
        ("C", "consolidation", section_c_consolidation),
        ("D", "router", section_d_router),
    ]

    try:
        state = _prepare_state(cfg)
    except Exception as exc:  # pragma: no cover - exercised by CLI failures.
        print(f"[setup] synthetic PMA-C setup : FAIL ({exc})")
        for key, _, _ in sections:
            results[key] = False
        _print_summary(results)
        return results

    for key, label, fn in sections:
        try:
            results[key] = bool(fn(state, cfg))
        except Exception as exc:  # pragma: no cover - exercised by CLI failures.
            results[key] = False
            print(f"[{key}] {label:<23}: FAIL ({exc})")

    _print_summary(results)
    return results


def _print_summary(results: dict[str, bool]) -> None:
    print("=== PMA-C FULL-SYSTEM DEMO ===")
    print(f"[A] champions/non-deletion : {'PASS' if results.get('A') else 'FAIL'}")
    print(f"[B] growth/plasticity      : {'PASS' if results.get('B') else 'FAIL'}")
    print(f"[C] consolidation          : {'PASS' if results.get('C') else 'FAIL'}")
    print(f"[D] router                 : {'PASS' if results.get('D') else 'FAIL'}")


def main(cfg: DemoConfig | None = None) -> dict[str, bool]:
    return run_demo(cfg)


if __name__ == "__main__":
    verdicts = main()
    raise SystemExit(0 if all(verdicts.values()) else 1)
