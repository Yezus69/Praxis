"""Matched continual-RL comparison for PMA-C."""

from __future__ import annotations

import argparse
import json
import time
from dataclasses import asdict, dataclass, field, replace
from pathlib import Path

import jax
import jax.numpy as jnp
import numpy as np
import optax

from pmac.adapters.rl import RLAdapter, init_actor_critic
from pmac.anchors import AnchorStore
from pmac.atlas import Atlas
from pmac.checkpoint import ChampionStore
from pmac.conservation import AnchorBatch, conservation_loss
from pmac.continual import clip_global, clip_guard_grad
from pmac.envs.gridworld import GridWorld, normalize_goal_cells
from pmac.projection import project_conflicts
from pmac.sentinels import SentinelStore
from pmac.stability import scale_by_stability, update_omega, zeros_omega_like
from pmac.tree_utils import tree_add_scaled


ALLOWED_RL_ABLATIONS = {None, "none", "no_conservation", "no_projection", "no_replay"}


@dataclass(frozen=True)
class RLConfig:
    grid_size: int = 5
    horizon: int = 25
    hidden_sizes: tuple[int, ...] = (64, 64)
    batch_size: int = 256
    updates_per_goal: int = 300
    lr: float = 0.01
    optimizer: str = "adam"
    gamma: float = 0.99
    value_loss_coef: float = 0.5
    entropy_coef: float = 0.01
    value_distance_coef: float = 1.0
    temperature: float = 1.0
    eval_episodes: int = 256
    anchor_memory_per_skill: int = 64
    sentinel_count_per_skill: int = 64
    guard_batch: int = 32
    replay_batch: int = 32
    num_guard_nodes: int = 4
    guard_tolerance: float = 0.005
    guard_lambda: float = 10.0
    guard_lambda_max: float = 64.0
    guard_grad_clip: float = 1.0
    replay_coef: float = 1.0
    max_grad_norm: float = 5.0
    stability_alpha: float = 10.0
    stability_decay: float = 0.99
    use_jit: bool = True
    baseline_clip: bool = True


@dataclass
class RLContinualResult:
    success_matrix: np.ndarray
    learned_success: np.ndarray
    final_success: np.ndarray
    peak_success: np.ndarray
    metrics: dict
    mode: str
    extra: dict = field(default_factory=dict)


def _parse_ints(text):
    return tuple(int(part) for part in str(text).split(",") if part)


def _parse_seeds(text):
    return [int(part) for part in str(text).split(",") if part]


def _parse_goals(text):
    if isinstance(text, int):
        return int(text)
    parts = [part.strip() for part in str(text).split(",") if part.strip()]
    if len(parts) == 1:
        return int(parts[0])
    return np.asarray([int(part) for part in parts], dtype=np.int32)


def _parse_ablations(text):
    values = [part.strip() for part in str(text).split(",") if part.strip()]
    return [None if value == "none" else value for value in values]


def _make_optimizer(cfg: RLConfig):
    name = str(cfg.optimizer).lower()
    if name == "adam":
        return optax.adam(float(cfg.lr))
    if name == "sgd":
        return optax.sgd(float(cfg.lr))
    raise ValueError(f"unknown optimizer: {cfg.optimizer}")


def _make_system(goals, cfg: RLConfig, seed: int):
    goal_cells = normalize_goal_cells(goals, cfg.grid_size)
    env = GridWorld(
        grid_size=cfg.grid_size,
        horizon=cfg.horizon,
        goal_cells=goal_cells,
    )
    adapter = RLAdapter(
        env,
        gamma=cfg.gamma,
        value_loss_coef=cfg.value_loss_coef,
        entropy_coef=cfg.entropy_coef,
        value_distance_coef=cfg.value_distance_coef,
        temperature=cfg.temperature,
        eval_episodes=cfg.eval_episodes,
    )
    params = init_actor_critic(
        jax.random.PRNGKey(int(seed)),
        env.obs_dim,
        hidden_sizes=cfg.hidden_sizes,
        num_actions=env.num_actions,
    )
    opt = _make_optimizer(cfg)
    return env, adapter, params, opt, goal_cells


def _current_value_and_grad(adapter: RLAdapter, cfg: RLConfig):
    def impl(params, key, goal_id):
        batch = adapter.rollout_batch(params, key, goal_id, cfg.batch_size)
        return jax.value_and_grad(adapter.current_loss)(params, batch)

    if cfg.use_jit:
        return jax.jit(impl)
    return impl


def _guard_loss(adapter: RLAdapter, params, x, teacher, tolerance, weight):
    batch = AnchorBatch(
        x=x,
        context=None,
        teacher=teacher,
        tolerance=tolerance,
        weight=weight,
    )
    behavior_fn = lambda p, bx: adapter.behavior(p, {"x": bx})
    return conservation_loss(behavior_fn, params, batch, adapter.anchor_distance)


def _guard_value_and_grad(adapter: RLAdapter, cfg: RLConfig):
    def impl(params, x, teacher, tolerance, weight):
        loss_fn = lambda p: _guard_loss(adapter, p, x, teacher, tolerance, weight)
        return jax.value_and_grad(loss_fn)(params)

    if cfg.use_jit:
        return jax.jit(impl)
    return impl


def _replay_loss(adapter: RLAdapter, params, x, teacher):
    current = adapter.behavior(params, {"x": x})
    return jnp.mean(adapter.anchor_distance(teacher, current))


def _replay_value_and_grad(adapter: RLAdapter, cfg: RLConfig):
    def impl(params, x, teacher):
        loss_fn = lambda p: _replay_loss(adapter, p, x, teacher)
        return jax.value_and_grad(loss_fn)(params)

    if cfg.use_jit:
        return jax.jit(impl)
    return impl


def _rng_from_int(value: int):
    return np.random.default_rng(int(value) % (2**32))


def _sample_anchor_arrays(node, key, n: int):
    if len(node.anchors) == 0 or int(n) <= 0:
        return None
    rng = _rng_from_int(key)
    replace = len(node.anchors) < int(n)
    idx = rng.choice(len(node.anchors), size=int(n), replace=replace)
    anchors = node.anchors
    return (
        jnp.asarray(anchors.x[idx]),
        jnp.asarray(anchors.teacher[idx]),
        jnp.asarray(anchors.tolerance[idx]),
        jnp.asarray(anchors.weight[idx]),
    )


def _sample_replay(atlas: Atlas, key, n: int):
    nodes = [node for node in atlas.protected_nodes() if len(node.anchors) > 0]
    if not nodes or int(n) <= 0:
        return None
    x = np.concatenate([node.anchors.x for node in nodes], axis=0)
    teacher = np.concatenate([node.anchors.teacher for node in nodes], axis=0)
    rng = _rng_from_int(key)
    replace = x.shape[0] < int(n)
    idx = rng.choice(x.shape[0], size=int(n), replace=replace)
    return jnp.asarray(x[idx]), jnp.asarray(teacher[idx])


def evaluate_all_goals(params, adapter: RLAdapter, num_goals: int, cfg: RLConfig, seed: int, step: int):
    scores = []
    for goal_id in range(int(num_goals)):
        key = jax.random.PRNGKey(int(seed) + 7919 * int(step + 1) + 101 * goal_id)
        score = adapter.evaluate_skill(
            params,
            {
                "goal_id": goal_id,
                "num_episodes": cfg.eval_episodes,
                "key": key,
                "greedy": True,
            },
        )
        scores.append(score)
    return np.asarray(scores, dtype=np.float32)


def compute_rl_metrics(success_matrix) -> dict:
    success_matrix = np.asarray(success_matrix, dtype=np.float32)
    final = success_matrix[-1]
    learned = np.diag(success_matrix)
    peak = np.max(success_matrix, axis=0)
    if success_matrix.shape[0] > 1:
        forgetting = np.mean(peak[:-1] - final[:-1])
    else:
        forgetting = 0.0
    retention = final / np.maximum(peak, 1e-9)
    return {
        "mean_final_success": float(np.mean(final)),
        "Forgetting": float(forgetting),
        "forgetting": float(forgetting),
        "retention": retention.astype(float).tolist(),
        "mean_retention": float(np.mean(retention)),
        "worst_retention": float(np.min(retention)),
        "learned_success": learned.astype(float).tolist(),
        "final_success": final.astype(float).tolist(),
        "peak_success": peak.astype(float).tolist(),
    }


def _result(success_matrix, mode: str, extra=None) -> RLContinualResult:
    success_matrix = np.asarray(success_matrix, dtype=np.float32)
    return RLContinualResult(
        success_matrix=success_matrix,
        learned_success=np.diag(success_matrix).astype(np.float32),
        final_success=success_matrix[-1].astype(np.float32),
        peak_success=np.max(success_matrix, axis=0).astype(np.float32),
        metrics=compute_rl_metrics(success_matrix),
        mode=mode,
        extra=dict(extra or {}),
    )


def _certify_goal(
    params,
    goal_id: int,
    task_i: int,
    adapter: RLAdapter,
    cfg: RLConfig,
    atlas: Atlas,
    champions: ChampionStore,
    eval_score: float,
):
    obs = adapter.env.all_observations_for_goal(goal_id)
    teacher = adapter.pack_behavior(adapter.behavior(params, {"x": obs}))
    obs_np = np.asarray(obs)
    teacher_np = np.asarray(teacher)
    n = int(obs_np.shape[0])
    confidence = np.max(np.asarray(jax.nn.softmax(teacher[..., : adapter.env.num_actions])), axis=-1)
    anchors = AnchorStore(cfg.anchor_memory_per_skill)
    anchors.add(
        obs_np,
        teacher_np,
        np.full((n,), cfg.guard_tolerance, dtype=np.float32),
        np.ones((n,), dtype=np.float32),
        np.asarray(confidence, dtype=np.float32),
        skill_ids=[f"goal_{goal_id}"] * n,
        labels=np.full((n,), int(goal_id), dtype=np.int32),
    )
    sent_n = min(int(cfg.sentinel_count_per_skill), n)
    sentinels = SentinelStore(
        x=obs_np[:sent_n],
        y=np.full((sent_n,), int(goal_id), dtype=np.int32),
        seeds=np.arange(sent_n, dtype=np.int32),
    )
    champion = champions.freeze(
        params,
        route=f"goal_{goal_id}",
        meta={"skill_id": f"goal_{goal_id}", "task_index": int(task_i)},
    )
    return atlas.create_or_update_node(
        f"goal_{goal_id}",
        context_key=int(goal_id),
        anchors=anchors,
        sentinels=sentinels,
        status="protected",
        champion_ref=champion,
        best_score=float(eval_score),
        current_score=float(eval_score),
        retention=1.0,
        allowed_regression=0.0,
        last_certified_step=int(task_i),
        guard_lambda=float(cfg.guard_lambda),
        certified_impls=[f"goal_{goal_id}"],
    )


def run_rl_baseline(goals=4, cfg: RLConfig | None = None, seed: int = 0) -> RLContinualResult:
    cfg = cfg or RLConfig()
    env, adapter, params, opt, goal_cells = _make_system(goals, cfg, seed)
    opt_state = opt.init(params)
    current_grad = _current_value_and_grad(adapter, cfg)
    success_matrix = np.zeros((env.num_goals, env.num_goals), dtype=np.float32)
    global_step = 0

    for task_i in range(env.num_goals):
        for update_i in range(int(cfg.updates_per_goal)):
            key = jax.random.PRNGKey(int(seed) + 100_003 * task_i + update_i)
            _, grads = current_grad(params, key, jnp.asarray(task_i, dtype=jnp.int32))
            if cfg.baseline_clip:
                grads = clip_global(grads, cfg.max_grad_norm)
            updates, opt_state = opt.update(grads, opt_state, params)
            params = optax.apply_updates(params, updates)
            global_step += 1
        success_matrix[task_i] = evaluate_all_goals(
            params, adapter, env.num_goals, cfg, seed, task_i
        )

    return _result(
        success_matrix,
        "baseline",
        extra={
            "seed": int(seed),
            "goal_cells": goal_cells.astype(int).tolist(),
            "updates": int(global_step),
            "config": asdict(cfg),
        },
    )


def run_rl_pmac(
    goals=4,
    cfg: RLConfig | None = None,
    seed: int = 0,
    ablation=None,
) -> RLContinualResult:
    cfg = cfg or RLConfig()
    ablation = None if ablation == "none" else ablation
    if ablation not in ALLOWED_RL_ABLATIONS:
        raise ValueError(f"unknown RL PMA-C ablation: {ablation}")

    env, adapter, params, opt, goal_cells = _make_system(goals, cfg, seed)
    opt_state = opt.init(params)
    atlas = Atlas()
    champions = ChampionStore()
    omega = zeros_omega_like(params)
    current_grad = _current_value_and_grad(adapter, cfg)
    guard_grad_fn = _guard_value_and_grad(adapter, cfg)
    replay_grad_fn = _replay_value_and_grad(adapter, cfg)
    success_matrix = np.zeros((env.num_goals, env.num_goals), dtype=np.float32)
    global_step = 0
    guard_losses = []
    replay_losses = []
    projection_steps = 0
    replay_steps = 0

    guard_enabled = ablation != "no_conservation"
    projection_enabled = ablation != "no_projection"
    replay_enabled = ablation != "no_replay"

    for task_i in range(env.num_goals):
        skill_id = f"goal_{task_i}"
        for update_i in range(int(cfg.updates_per_goal)):
            key = jax.random.PRNGKey(int(seed) + 100_003 * task_i + update_i)
            _, g_new = current_grad(params, key, jnp.asarray(task_i, dtype=jnp.int32))

            if replay_enabled:
                replay = _sample_replay(atlas, global_step + 17, cfg.replay_batch)
                if replay is not None:
                    rx, rteacher = replay
                    replay_loss, g_replay = replay_grad_fn(params, rx, rteacher)
                    g_new = tree_add_scaled(g_new, g_replay, cfg.replay_coef)
                    replay_losses.append(float(replay_loss))
                    replay_steps += 1

            guard_grads = []
            active_nodes = []
            if guard_enabled:
                nodes = atlas.sample_protected_nodes(skill_id, cfg.num_guard_nodes)
                for guard_i, node in enumerate(nodes):
                    arrays = _sample_anchor_arrays(
                        node,
                        global_step + 997 * (guard_i + 1),
                        cfg.guard_batch,
                    )
                    if arrays is None:
                        continue
                    gx, gteacher, gtol, gweight = arrays
                    guard_loss, guard_grad = guard_grad_fn(params, gx, gteacher, gtol, gweight)
                    guard_grad = clip_guard_grad(guard_grad, g_new, cfg.guard_grad_clip)
                    guard_losses.append(float(guard_loss))
                    guard_grads.append(guard_grad)
                    active_nodes.append(node)

            if projection_enabled and guard_grads:
                g_total = project_conflicts(g_new, guard_grads)
                projection_steps += 1
            else:
                g_total = g_new

            if guard_enabled:
                for node, guard_grad in zip(active_nodes, guard_grads):
                    lam = min(float(node.guard_lambda), float(cfg.guard_lambda_max))
                    g_total = tree_add_scaled(g_total, guard_grad, lam)

            g_total = scale_by_stability(g_total, omega, cfg.stability_alpha)
            g_total = clip_global(g_total, cfg.max_grad_norm)
            updates, opt_state = opt.update(g_total, opt_state, params)
            params = optax.apply_updates(params, updates)
            global_step += 1

        success_matrix[task_i] = evaluate_all_goals(
            params, adapter, env.num_goals, cfg, seed, task_i
        )
        node = _certify_goal(
            params,
            task_i,
            task_i,
            adapter,
            cfg,
            atlas,
            champions,
            eval_score=float(success_matrix[task_i, task_i]),
        )
        arrays = _sample_anchor_arrays(node, 31_337 + task_i, cfg.guard_batch)
        if arrays is not None:
            gx, gteacher, gtol, gweight = arrays
            _, grad_guard = guard_grad_fn(params, gx, gteacher, gtol, gweight)
            omega = update_omega(omega, params, grad_guard, cfg.stability_decay)

    mode = "pmac" if ablation is None else f"pmac_{ablation}"
    return _result(
        success_matrix,
        mode,
        extra={
            "seed": int(seed),
            "ablation": ablation,
            "goal_cells": goal_cells.astype(int).tolist(),
            "updates": int(global_step),
            "protected_skills": list(atlas.nodes.keys()),
            "guard_loss_trace": guard_losses,
            "replay_loss_trace": replay_losses,
            "projection_steps": int(projection_steps),
            "replay_steps": int(replay_steps),
            "guard_enabled": bool(guard_enabled),
            "projection_enabled": bool(projection_enabled),
            "replay_enabled": bool(replay_enabled),
            "config": asdict(cfg),
        },
    )


def _result_to_json(result: RLContinualResult):
    return {
        "mode": result.mode,
        "success_matrix": np.asarray(result.success_matrix).tolist(),
        "learned_success": np.asarray(result.learned_success).tolist(),
        "final_success": np.asarray(result.final_success).tolist(),
        "peak_success": np.asarray(result.peak_success).tolist(),
        "metrics": result.metrics,
        "extra": result.extra,
    }


def _aggregate(results_by_mode):
    aggregate = {}
    for mode, results in results_by_mode.items():
        keys = sorted(results[0].metrics.keys())
        stats = {}
        for key in keys:
            values = [result.metrics[key] for result in results]
            if isinstance(values[0], (list, tuple)):
                continue
            arr = np.asarray(values, dtype=np.float64)
            stats[key] = {"mean": float(np.mean(arr)), "std": float(np.std(arr))}
        aggregate[mode] = stats
    return aggregate


def _plot_results(first_seed_results, out_path):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    names = list(first_seed_results.keys())
    first = first_seed_results[names[0]]
    n_goals = int(first.success_matrix.shape[1])
    x = np.arange(n_goals)
    width = 0.8 / max(1, len(names))
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.0))
    fig.suptitle("Continual RL: PMA-C vs Baseline")

    for i, name in enumerate(names):
        result = first_seed_results[name]
        offset = (i - (len(names) - 1) / 2.0) * width
        axes[0].bar(x + offset, result.final_success, width=width, label=name)
        axes[1].plot(result.success_matrix[:, 0], marker="o", label=name)
    axes[0].set_title("Final Success by Goal")
    axes[0].set_xlabel("Goal")
    axes[0].set_ylabel("Success")
    axes[0].set_ylim(0.0, 1.05)
    axes[0].legend(fontsize=8)

    axes[1].set_title("Goal 0 Across Training")
    axes[1].set_xlabel("After Goal")
    axes[1].set_ylabel("Success")
    axes[1].set_ylim(0.0, 1.05)
    axes[1].legend(fontsize=8)
    fig.tight_layout()
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def _run_timed(seed, run_fn):
    start = time.perf_counter()
    result = run_fn()
    wall_s = time.perf_counter() - start
    result.extra = dict(result.extra)
    result.extra["wall_s"] = float(wall_s)
    learned = ", ".join(f"{v:.3f}" for v in np.asarray(result.learned_success))
    final = ", ".join(f"{v:.3f}" for v in np.asarray(result.final_success))
    print(
        f"{result.mode} seed={int(seed)} wall_s={wall_s:.3f} "
        f"learned=[{learned}] final=[{final}] "
        f"mean_final={result.metrics['mean_final_success']:.3f} "
        f"forgetting={result.metrics['forgetting']:.3f}"
    )
    return result


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument("--goals", default="4", help="goal count or comma-separated goal cell ids")
    parser.add_argument("--seeds", default="0")
    parser.add_argument("--updates-per-goal", type=int, default=300)
    parser.add_argument("--batch-size", type=int, default=256)
    parser.add_argument("--lr", type=float, default=0.01)
    parser.add_argument("--optimizer", choices=("adam", "sgd"), default="adam")
    parser.add_argument("--hidden", default="64,64")
    parser.add_argument("--grid-size", type=int, default=5)
    parser.add_argument("--horizon", type=int, default=25)
    parser.add_argument("--eval-episodes", type=int, default=256)
    parser.add_argument("--entropy-coef", type=float, default=0.01)
    parser.add_argument("--guard-lambda", type=float, default=10.0)
    parser.add_argument("--guard-tolerance", type=float, default=0.005)
    parser.add_argument("--replay-coef", type=float, default=1.0)
    parser.add_argument("--max-grad-norm", type=float, default=5.0)
    parser.add_argument("--ablations", default="none")
    parser.add_argument("--out", default="runs/pmac_rl")
    parser.add_argument("--no-jit", action="store_true")
    args = parser.parse_args(argv)

    goals = _parse_goals(args.goals)
    seeds = _parse_seeds(args.seeds)
    ablations = _parse_ablations(args.ablations)
    invalid = [value for value in ablations if value not in ALLOWED_RL_ABLATIONS]
    if invalid:
        parser.error(
            "unknown ablation(s): "
            + ", ".join(str(value) for value in invalid)
            + "; valid values are none,no_conservation,no_projection,no_replay"
        )

    cfg = RLConfig(
        grid_size=args.grid_size,
        horizon=args.horizon,
        hidden_sizes=_parse_ints(args.hidden),
        batch_size=args.batch_size,
        updates_per_goal=args.updates_per_goal,
        lr=args.lr,
        optimizer=args.optimizer,
        eval_episodes=args.eval_episodes,
        entropy_coef=args.entropy_coef,
        guard_lambda=args.guard_lambda,
        guard_tolerance=args.guard_tolerance,
        replay_coef=args.replay_coef,
        max_grad_norm=args.max_grad_norm,
        use_jit=not args.no_jit,
    )

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    raw = {
        "seeds": seeds,
        "config": asdict(cfg),
        "goals_arg": args.goals,
        "runs": {},
    }
    results_by_mode = {}
    first_seed_results = {}

    for seed in seeds:
        seed_results = {}
        baseline = _run_timed(seed, lambda seed=seed: run_rl_baseline(goals, cfg, seed))
        seed_results[baseline.mode] = baseline
        results_by_mode.setdefault(baseline.mode, []).append(baseline)

        pmac = _run_timed(seed, lambda seed=seed: run_rl_pmac(goals, cfg, seed, None))
        seed_results[pmac.mode] = pmac
        results_by_mode.setdefault(pmac.mode, []).append(pmac)

        for ablation in ablations:
            if ablation is None:
                continue
            result = _run_timed(
                seed,
                lambda seed=seed, ablation=ablation: run_rl_pmac(
                    goals, cfg, seed, ablation
                ),
            )
            seed_results[result.mode] = result
            results_by_mode.setdefault(result.mode, []).append(result)

        if not first_seed_results:
            first_seed_results = dict(seed_results)
        raw["runs"][str(seed)] = {
            "results": {mode: _result_to_json(result) for mode, result in seed_results.items()}
        }

    raw["aggregate"] = _aggregate(results_by_mode)
    results_path = out_dir / "results.json"
    with results_path.open("w", encoding="utf-8") as f:
        json.dump(raw, f, indent=2)
    plot_path = out_dir / "comparison.png"
    _plot_results(first_seed_results, plot_path)
    print(f"wrote {results_path}")
    print(f"wrote {plot_path}")


if __name__ == "__main__":
    main()


__all__ = [
    "ALLOWED_RL_ABLATIONS",
    "RLConfig",
    "RLContinualResult",
    "compute_rl_metrics",
    "evaluate_all_goals",
    "run_rl_baseline",
    "run_rl_pmac",
]
