"""praxis/train.py — Brax PPO training entrypoint for the Praxis coverage task.

CLI wiring around ``brax.training.agents.ppo.train`` against ``praxis.envs.CoverEnv``,
wrapped with ``mujoco_playground.wrapper.wrap_for_brax_training``.

Run (fast, ~2 min on one RTX 4090):
    python -m praxis.train --num-timesteps 1300000 --num-envs 2048 --num-evals 9 \
        --learning-rate 0.00015 --entropy-cost 0.005 --run-name cover

Outputs per run:
    runs/<run>/metrics.csv   — one row per eval: step, eval/episode_reward,
                               eval/success_rate, eval/collision_rate, + extra eval/* cols
    runs/<run>/tb/           — TensorBoard event files (best-effort, tensorboardX)
    ckpts/<run>/             — Orbax checkpoints (via Brax save_checkpoint_path)

The module is import-safe: heavy work is behind ``main()`` / ``if __name__ == '__main__'``.

NOTE for orchestrator: Brax/Playground API arg names are pinned to current (0.14.x /
MuJoCo Playground) usage. Spots where the exact signature may drift are flagged inline
with ``# NOTE:``.
"""

from __future__ import annotations

import argparse
import csv
import datetime
import functools
import inspect
import os
import time
from typing import Any, Callable, Dict, Optional, Tuple

# Defer heavy imports (jax/brax/playground) into main() so `import praxis.train`
# (and `python -m praxis.train --help`) is cheap and side-effect free.

from praxis import contract
from praxis.agent.networks import make_network_factory


# --------------------------------------------------------------------------- #
# Defaults (CORRECTED FACTS — first-learnable-run hyperparams)
# --------------------------------------------------------------------------- #
DEFAULTS = dict(
    num_timesteps=int(2e7),
    num_envs=2048,
    episode_length=contract.EPISODE_LENGTH,
    unroll_length=20,
    batch_size=256,
    num_minibatches=32,
    num_updates_per_batch=4,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    discounting=0.97,
    reward_scaling=1.0,
    lr_schedule="none",            # "none" | "adaptive_kl"
    desired_kl=0.01,               # target per-update KL (only used under adaptive_kl)
    lr_min=1e-5,                   # -> learning_rate_schedule_min_lr
    lr_max=1e-2,                   # -> learning_rate_schedule_max_lr
    clipping_epsilon=None,         # None => brax default (0.3); float => override
    clipping_epsilon_value=None,   # None => OFF (brax default); float => value-clip range
    normalize_advantage=None,      # None => brax default (True); bool => override
    normalize_until_count=None,    # None => off; int => freeze obs-normalizer stats after N obs
    gae_lambda=None,               # None => brax default 0.95; float => override
    vf_loss_coefficient=None,      # None => brax default 0.5; float => override
    deterministic_eval=False,      # False => current behavior (stochastic eval policy)
    normalize_observations=True,  # Brax default is False - MUST enable.
    num_evals=10,
    seed=0,
)

# --smoke preset: <=5-min DoD-2 gate on GPU.
SMOKE_OVERRIDES = dict(
    num_timesteps=int(5e6),
    num_envs=2048,
    num_evals=10,
)


# --------------------------------------------------------------------------- #
# Divisibility constraint (assert early with a clear message)
# --------------------------------------------------------------------------- #
def assert_divisibility(num_envs: int, unroll_length: int,
                        batch_size: int, num_minibatches: int) -> None:
    """(num_envs * unroll_length) MUST be divisible by (batch_size * num_minibatches).

    e.g. 2048*20 = 40960 ; 256*32 = 8192 ; 40960/8192 = 5  ✓
    """
    lhs = num_envs * unroll_length
    rhs = batch_size * num_minibatches
    if rhs == 0 or lhs % rhs != 0:
        raise ValueError(
            "Brax PPO data-shape constraint violated: "
            f"(num_envs * unroll_length) = {num_envs}*{unroll_length} = {lhs} "
            f"must be divisible by (batch_size * num_minibatches) = "
            f"{batch_size}*{num_minibatches} = {rhs}. "
            f"Got remainder {lhs % rhs if rhs else 'n/a'}. "
            "Adjust num_envs / unroll_length / batch_size / num_minibatches."
        )


# --------------------------------------------------------------------------- #
# Metric-key mapping helpers (defensive about Brax/Agent-A prefixes)
# --------------------------------------------------------------------------- #
def _find_metric(metrics: Dict[str, Any], *substrings: str) -> Optional[Tuple[str, float]]:
    """Return (key, float(value)) for the first metric whose key contains ALL substrings.

    Prefers keys under an 'eval/' prefix (Brax surfaces eval metrics there), then any.
    Used so the curves populate even if Agent-A's metric prefix differs slightly from
    the expected eval/episode_<name>.
    """
    subs = tuple(s.lower() for s in substrings)

    def _matches(kl: str) -> bool:
        return all(s in kl for s in subs)

    # Collect all coercible matches, then pick the best deterministically (dict order
    # is NOT relied upon). Ranking, lower is better:
    #   0) eval/* keys beat non-eval keys
    #   1) keys that do NOT also contain 'reward' beat ones that do — so a 'collision'
    #      lookup binds to eval/episode_collision, not eval/episode_reward_collision
    #      (unless we ARE explicitly looking up a 'reward' metric).
    #   2) shorter key wins (more specific / less compound).
    looking_for_reward = "reward" in subs
    candidates = []
    for k, v in metrics.items():
        kl = k.lower()
        if not _matches(kl):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        is_eval = 0 if kl.startswith("eval/") else 1
        has_extra_reward = 0 if (looking_for_reward or "reward" not in kl) else 1
        candidates.append(((is_eval, has_extra_reward, len(k)), k, fv))

    if not candidates:
        return None
    candidates.sort(key=lambda c: c[0])
    _, best_k, best_v = candidates[0]
    return best_k, best_v


def map_eval_metrics(metrics: Dict[str, Any]) -> Dict[str, float]:
    """Map raw Brax metrics dict -> the three canonical curves (+ all eval/* scalars).

    Returns a dict that always contains the three canonical keys (NaN when absent):
        eval/episode_reward, eval/success_rate, eval/collision_rate
    plus every scalar-coercible 'eval/' metric verbatim (so extra columns flow through).
    """
    import math

    out: Dict[str, float] = {}

    # Reward: prefer the literal Brax key, else any eval reward key.
    reward = metrics.get("eval/episode_reward")
    if reward is None:
        hit = _find_metric(metrics, "reward")
        reward = hit[1] if hit else float("nan")
    out["eval/episode_reward"] = float(reward) if reward is not None else float("nan")

    # coverage <- eval/episode_coverage (mean fraction of cells visited at episode end)
    cov = _find_metric(metrics, "coverage")
    out["eval/coverage"] = cov[1] if cov else float("nan")

    # collision_rate <- eval/episode_collision
    coll = _find_metric(metrics, "collision")
    out["eval/collision_rate"] = coll[1] if coll else float("nan")

    # Carry through every other scalar eval/* metric (e.g. reward components, std).
    for k, v in metrics.items():
        if not k.lower().startswith("eval/"):
            continue
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isnan(fv) and k not in out:
            # still record the column header existence
            out.setdefault(k, fv)
        else:
            out.setdefault(k, fv)
    return out


# --------------------------------------------------------------------------- #
# progress_fn factory (CSV + TensorBoard + print). NO time calls inside jit.
# --------------------------------------------------------------------------- #
def make_progress_fn(
    csv_path: str,
    tb_writer: Optional[Any],
    start_time: float,
) -> Callable[[int, Dict[str, Any]], None]:
    """Build progress_fn(num_steps, metrics) closing over output sinks.

    `start_time` is captured OUTSIDE any jitted code (a plain wall-clock float) and
    used only here to record elapsed seconds — we never call time.* inside a traced fn.
    """
    # CSV columns grow lazily as new eval/* keys appear; we rewrite the header set
    # by buffering rows and writing once we know all columns is over-engineering for
    # a streaming log, so we fix a stable leading column order and append extras.
    base_cols = [
        "step",
        "wall_s",
        "eval/episode_reward",
        "eval/coverage",
        "eval/collision_rate",
    ]
    state = {"header_written": False, "extra_cols": []}
    train_csv_path = os.path.join(os.path.dirname(csv_path) or ".", "train_metrics.csv")
    train_state = {"header_written": False, "metric_cols": [], "rows": []}

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    def progress_fn(num_steps: int, metrics: Dict[str, Any]) -> None:
        wall_s = time.time() - start_time

        # Only eval rounds carry the per-episode success/collision metrics we curve, so
        # write metrics.csv rows for eval rounds ONLY (keeps the DoD-2 plot clean).
        # In brax 0.14.2, log_training_metrics=True merges training/* keys into this
        # same eval callback; those are written to train_metrics.csv below.
        is_eval = any(k.startswith("eval/") for k in metrics)
        if not is_eval:
            if tb_writer is not None:
                try:
                    for k, v in metrics.items():
                        try:
                            tb_writer.add_scalar(k, float(v), int(num_steps))
                        except (TypeError, ValueError):
                            pass
                    tb_writer.flush()
                except Exception:  # noqa: BLE001
                    pass
            return

        mapped = map_eval_metrics(metrics)

        train_mapped: Dict[str, float] = {}
        for k, v in metrics.items():
            if not k.lower().startswith("training/"):
                continue
            try:
                train_mapped[k] = float(v)
            except (TypeError, ValueError):
                continue

        # Determine extra columns (sorted, stable) beyond the canonical three.
        canonical = {"eval/episode_reward", "eval/coverage", "eval/collision_rate"}
        extras = sorted(k for k in mapped.keys() if k not in canonical)
        # Lock extra column set on first write to keep CSV rectangular.
        if not state["header_written"]:
            state["extra_cols"] = extras
        cols = base_cols + state["extra_cols"]

        row = {
            "step": int(num_steps),
            "wall_s": round(wall_s, 2),
            "eval/episode_reward": mapped.get("eval/episode_reward", float("nan")),
            "eval/coverage": mapped.get("eval/coverage", float("nan")),
            "eval/collision_rate": mapped.get("eval/collision_rate", float("nan")),
        }
        for c in state["extra_cols"]:
            row[c] = mapped.get(c, float("nan"))

        # --- CSV ---
        try:
            write_header = not state["header_written"]
            with open(csv_path, "a", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=cols, extrasaction="ignore")
                if write_header:
                    writer.writeheader()
                    state["header_written"] = True
                writer.writerow(row)
        except OSError as e:
            print(f"[progress_fn] WARN: failed to write CSV row: {e}")

        # --- training CSV (merged into eval metrics by brax 0.14.2) ---
        if train_mapped:
            try:
                old_cols = list(train_state["metric_cols"])
                metric_cols = sorted(set(old_cols).union(train_mapped.keys()))
                train_state["metric_cols"] = metric_cols
                train_cols = ["step", "wall_s"] + metric_cols
                train_row = {
                    "step": int(num_steps),
                    "wall_s": round(wall_s, 2),
                }
                for c in metric_cols:
                    train_row[c] = train_mapped.get(c, float("nan"))

                rows = train_state["rows"]
                rows.append(train_row)
                header_changed = train_state["header_written"] and metric_cols != old_cols
                if header_changed:
                    with open(train_csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=train_cols, extrasaction="ignore")
                        writer.writeheader()
                        for r in rows:
                            full_row = {
                                "step": r["step"],
                                "wall_s": r["wall_s"],
                            }
                            for c in metric_cols:
                                full_row[c] = r.get(c, float("nan"))
                            writer.writerow(full_row)
                else:
                    write_header = not train_state["header_written"]
                    with open(train_csv_path, "a", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=train_cols, extrasaction="ignore")
                        if write_header:
                            writer.writeheader()
                            train_state["header_written"] = True
                        writer.writerow(train_row)
            except OSError as e:
                print(f"[progress_fn] WARN: failed to write training CSV row: {e}")

        # --- TensorBoard (best-effort) ---
        if tb_writer is not None:
            try:
                for k, v in mapped.items():
                    if v == v:  # not NaN
                        tb_writer.add_scalar(k, float(v), int(num_steps))
                tb_writer.flush()
            except Exception as e:  # noqa: BLE001 - logging must never crash training
                print(f"[progress_fn] WARN: tensorboard write failed: {e}")

        # --- human-readable line ---
        r = row["eval/episode_reward"]
        s = row["eval/coverage"]
        c = row["eval/collision_rate"]
        print(
            f"[eval] step={int(num_steps):>10d}  "
            f"reward={r:>9.3f}  coverage={s:>6.3f}  collision={c:>6.3f}  "
            f"({wall_s:6.1f}s)"
        )

    return progress_fn


# --------------------------------------------------------------------------- #
# Env construction
# --------------------------------------------------------------------------- #
def build_env(episode_length: int, reward_overrides: Optional[Dict[str, Any]] = None):
    """Construct the (unwrapped) ``praxis.envs.CoverEnv``.

    episode_length is threaded INTO the env so its timeout (and thus info['time_out']
    for value bootstrapping) fires at the SAME step the Brax EpisodeWrapper truncates.
    """
    from praxis.envs import CoverEnv, default_config  # type: ignore
    cfg = default_config()
    cfg.episode_length = int(episode_length)
    if reward_overrides:
        for key, value in reward_overrides.items():
            if value is not None:
                cfg.reward[key] = value
    print(f"[env] Using praxis.envs.CoverEnv (coverage, episode_length={int(episode_length)}).")
    return CoverEnv(cfg)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="praxis.train",
        description="Brax PPO trainer for the Praxis MJX navigation task.",
    )
    p.add_argument("--num-timesteps", type=float, default=float(DEFAULTS["num_timesteps"]),
                   help="Total env steps to train (float ok; cast to int). Default 2e7.")
    p.add_argument("--num-envs", type=int, default=DEFAULTS["num_envs"],
                   help="Parallel envs (Brax batch). Default 2048.")
    p.add_argument("--episode-length", type=int, default=DEFAULTS["episode_length"],
                   help=f"Episode horizon. Default contract.EPISODE_LENGTH="
                        f"{contract.EPISODE_LENGTH}.")
    p.add_argument("--unroll-length", type=int, default=DEFAULTS["unroll_length"])
    p.add_argument("--batch-size", type=int, default=DEFAULTS["batch_size"])
    p.add_argument("--num-minibatches", type=int, default=DEFAULTS["num_minibatches"])
    p.add_argument("--num-updates-per-batch", type=int,
                   default=DEFAULTS["num_updates_per_batch"])
    p.add_argument("--learning-rate", type=float, default=DEFAULTS["learning_rate"])
    p.add_argument("--entropy-cost", type=float, default=DEFAULTS["entropy_cost"])
    p.add_argument("--discounting", type=float, default=DEFAULTS["discounting"])
    p.add_argument("--reward-scaling", type=float, default=DEFAULTS["reward_scaling"])
    p.add_argument("--lr-schedule", type=str, choices=["none", "adaptive_kl"],
                   default=DEFAULTS["lr_schedule"],
                   help="LR controller. 'none' (default)=constant LR (current behavior). "
                        "'adaptive_kl'=brax KL-throttled adaptive LR. brax 0.14.2 has no cosine/linear schedule.")
    p.add_argument("--desired-kl", type=float, default=DEFAULTS["desired_kl"],
                   help="Target per-update KL for adaptive_kl. Inert unless --lr-schedule adaptive_kl.")
    p.add_argument("--lr-min", type=float, default=DEFAULTS["lr_min"], help="Floor LR for adaptive_kl.")
    p.add_argument("--lr-max", type=float, default=DEFAULTS["lr_max"], help="Ceiling LR for adaptive_kl.")
    p.add_argument("--clipping-epsilon", type=float, default=None,
                   help="Policy PPO clip eps. None => brax default 0.3. Lower (0.2) tightens trust region.")
    p.add_argument("--clipping-epsilon-value", type=float, default=None,
                   help="Value-function clip range. None => OFF (brax default). e.g. 0.2 enables clipped value loss.")
    p.add_argument("--normalize-advantage", action=argparse.BooleanOptionalAction, default=None,
                   help="Per-minibatch advantage standardization. None => brax default (True).")
    p.add_argument("--normalize-until-count", type=int, default=None,
                   help="Freeze obs-normalizer running stats after N observations. None => never freeze.")
    p.add_argument("--gae-lambda", type=float, default=None, help="GAE lambda. None => brax default 0.95.")
    p.add_argument("--vf-loss-coefficient", type=float, default=None,
                   help="Value loss coefficient. None => brax default 0.5.")
    p.add_argument("--deterministic-eval", action=argparse.BooleanOptionalAction, default=False,
                   help="Use the greedy (mean) policy at eval time. Default False = stochastic eval (current behavior). "
                        "Diagnostic: isolates eval-time action noise from real policy degradation.")
    p.add_argument("--k-cov", type=float, default=None)
    p.add_argument("--k-coll", type=float, default=None)
    p.add_argument("--k-time", type=float, default=None)
    p.add_argument("--k-complete", type=float, default=None)
    p.add_argument("--k-fresh", type=float, default=None)
    p.add_argument("--freshness-decay", type=float, default=None)
    p.add_argument("--collision-penalty-cap", type=float, default=None)
    p.add_argument("--terminate-on-full-coverage", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--patrol", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--num-evals", type=int, default=DEFAULTS["num_evals"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])

    p.add_argument("--run-name", type=str, default=None,
                   help="Run name; defaults to praxis-<UTC timestamp>.")
    p.add_argument("--checkpoint-dir", type=str, default=None,
                   help="Orbax checkpoint dir; defaults to ckpts/<run>.")
    p.add_argument("--restore-checkpoint-path", type=str, default=None,
                   help="Optional Orbax checkpoint to restore and continue from.")

    p.add_argument("--smoke", action="store_true",
                   help="Preset for the <=5-min DoD-2 gate "
                        "(num_timesteps=5e6, num_envs=2048, num_evals=10).")

    p.add_argument("--policy-sizes", type=int, nargs="+", default=[256, 256, 256],
                   help="Policy MLP hidden layer sizes. Default 256 256 256.")
    p.add_argument("--value-sizes", type=int, nargs="+", default=[256, 256, 256],
                   help="Value MLP hidden layer sizes. Default 256 256 256.")
    return p


def resolve_config(args: argparse.Namespace) -> Dict[str, Any]:
    """Fold --smoke preset over CLI args and produce the final hyperparam dict."""
    cfg = dict(
        num_timesteps=int(args.num_timesteps),
        num_envs=int(args.num_envs),
        episode_length=int(args.episode_length),
        unroll_length=int(args.unroll_length),
        batch_size=int(args.batch_size),
        num_minibatches=int(args.num_minibatches),
        num_updates_per_batch=int(args.num_updates_per_batch),
        learning_rate=float(args.learning_rate),
        entropy_cost=float(args.entropy_cost),
        discounting=float(args.discounting),
        reward_scaling=float(args.reward_scaling),
        num_evals=int(args.num_evals),
        seed=int(args.seed),
        normalize_observations=DEFAULTS["normalize_observations"],
        lr_schedule=str(args.lr_schedule),
        desired_kl=float(args.desired_kl),
        lr_min=float(args.lr_min),
        lr_max=float(args.lr_max),
        clipping_epsilon=(None if args.clipping_epsilon is None else float(args.clipping_epsilon)),
        clipping_epsilon_value=(None if args.clipping_epsilon_value is None else float(args.clipping_epsilon_value)),
        normalize_advantage=args.normalize_advantage,
        normalize_until_count=(None if args.normalize_until_count is None else int(args.normalize_until_count)),
        gae_lambda=(None if args.gae_lambda is None else float(args.gae_lambda)),
        vf_loss_coefficient=(None if args.vf_loss_coefficient is None else float(args.vf_loss_coefficient)),
        deterministic_eval=bool(args.deterministic_eval),
    )
    if args.smoke:
        # Only override the few smoke knobs; respect explicit user-set everything else.
        cfg["num_timesteps"] = SMOKE_OVERRIDES["num_timesteps"]
        cfg["num_envs"] = SMOKE_OVERRIDES["num_envs"]
        cfg["num_evals"] = SMOKE_OVERRIDES["num_evals"]
    return cfg


def reward_overrides_from_args(args: argparse.Namespace) -> Dict[str, Any]:
    """Return only explicit reward/env overrides, preserving no-flag defaults."""
    names = (
        "k_cov",
        "k_coll",
        "k_time",
        "k_complete",
        "k_fresh",
        "freshness_decay",
        "collision_penalty_cap",
        "terminate_on_full_coverage",
        "patrol",
    )
    return {name: getattr(args, name) for name in names if getattr(args, name) is not None}


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = resolve_config(args)
    reward_overrides = reward_overrides_from_args(args)

    # Run identity + output dirs.
    run_name = args.run_name or (
        "praxis-" + datetime.datetime.utcnow().strftime("%Y%m%d-%H%M%S")
    )
    run_dir = os.path.join("runs", run_name)
    ckpt_dir = args.checkpoint_dir or os.path.join("ckpts", run_name)
    csv_path = os.path.join(run_dir, "metrics.csv")
    tb_dir = os.path.join(run_dir, "tb")
    os.makedirs(run_dir, exist_ok=True)
    # Brax/Orbax wants an absolute checkpoint path.
    os.makedirs(ckpt_dir, exist_ok=True)
    ckpt_dir_abs = os.path.abspath(ckpt_dir)

    # Fail fast on the data-shape constraint BEFORE importing/compiling anything heavy.
    assert_divisibility(
        cfg["num_envs"], cfg["unroll_length"],
        cfg["batch_size"], cfg["num_minibatches"],
    )

    print("=" * 78)
    print(f"Praxis PPO (coverage)  run={run_name}")
    print(f"  hyperparams: {cfg}")
    if reward_overrides:
        print(f"  reward/env overrides: {reward_overrides}")
    print(f"  metrics.csv -> {os.path.abspath(csv_path)}")
    print(f"  tensorboard -> {os.path.abspath(tb_dir)}")
    print(f"  checkpoints -> {ckpt_dir_abs}")
    print("=" * 78)

    # --- Heavy imports (jax/brax/playground) happen here, inside main() ---
    from brax.training.agents.ppo import train as ppo

    # Build the UNWRAPPED env and let Brax wrap it via wrap_env_fn (the official
    # Playground+Brax pattern, verified against the installed ppo.train). Pre-wrapping +
    # the default wrap_env=True would DOUBLE-wrap and break.
    raw_env = build_env(episode_length=cfg["episode_length"], reward_overrides=reward_overrides)

    # Network factory (explicit 256/256/256 policy & value).
    network_factory = make_network_factory(
        policy_sizes=args.policy_sizes,
        value_sizes=args.value_sizes,
    )

    # TensorBoard writer (best-effort).
    tb_writer = None
    try:
        from tensorboardX import SummaryWriter
        tb_writer = SummaryWriter(logdir=tb_dir)
    except Exception as e:  # noqa: BLE001
        print(f"[tb] NOTE: tensorboardX unavailable ({e}); CSV+print only.")

    start_time = time.time()  # captured OUTSIDE jit; passed into progress_fn.
    progress_fn = make_progress_fn(csv_path, tb_writer, start_time)

    # --- Assemble ppo.train kwargs. Cast floats->int where Brax expects int. ---
    train_kwargs: Dict[str, Any] = dict(
        environment=raw_env,
        num_timesteps=int(cfg["num_timesteps"]),
        num_envs=int(cfg["num_envs"]),
        episode_length=int(cfg["episode_length"]),
        unroll_length=int(cfg["unroll_length"]),
        batch_size=int(cfg["batch_size"]),
        num_minibatches=int(cfg["num_minibatches"]),
        num_updates_per_batch=int(cfg["num_updates_per_batch"]),
        learning_rate=float(cfg["learning_rate"]),
        entropy_cost=float(cfg["entropy_cost"]),
        discounting=float(cfg["discounting"]),
        reward_scaling=float(cfg["reward_scaling"]),
        normalize_observations=bool(cfg["normalize_observations"]),
        num_evals=int(cfg["num_evals"]),
        seed=int(cfg["seed"]),
        network_factory=network_factory,
        progress_fn=progress_fn,
    )

    # log_training_metrics + checkpoint paths are supported in current Brax (0.14.x)
    # but guarded via signature inspection so an older brax doesn't hard-crash on
    # unexpected kwargs.
    train_sig_params = inspect.signature(ppo.train).parameters
    if "log_training_metrics" in train_sig_params:
        train_kwargs["log_training_metrics"] = True
    else:
        print("[ppo] NOTE: this brax has no `log_training_metrics`; skipping.")

    if "save_checkpoint_path" in train_sig_params:
        train_kwargs["save_checkpoint_path"] = ckpt_dir_abs
    else:
        print("[ppo] NOTE: this brax has no `save_checkpoint_path`; "
              "checkpoints will NOT be auto-saved by Brax.")

    if args.restore_checkpoint_path:
        if "restore_checkpoint_path" in train_sig_params:
            train_kwargs["restore_checkpoint_path"] = os.path.abspath(
                args.restore_checkpoint_path
            )
        else:
            print("[ppo] NOTE: this brax has no `restore_checkpoint_path`; "
                  "ignoring --restore-checkpoint-path.")

    # --- Env wrapping: official Playground+Brax pattern. Pass the UNWRAPPED env +
    #     wrap_env_fn so Brax wraps it (auto-reset + vmap) and threads the per-env rng
    #     into randomization_fn(model, rng). (Verified against installed ppo.train.) ---
    from mujoco_playground import wrapper as mjxp_wrapper
    if "wrap_env_fn" in train_sig_params:
        train_kwargs["wrap_env_fn"] = mjxp_wrapper.wrap_for_brax_training
    else:
        # Older brax: pre-wrap and disable brax's own wrapping to avoid double-wrap.
        train_kwargs["environment"] = mjxp_wrapper.wrap_for_brax_training(raw_env)
        if "wrap_env" in train_sig_params:
            train_kwargs["wrap_env"] = False

    # Correct truncation handling (fact #8): bootstrap value on timeout; terminate on
    # collision/success. Agent-A sets info['truncation'] accordingly.
    if "bootstrap_on_timeout" in train_sig_params:
        train_kwargs["bootstrap_on_timeout"] = True

    # Gradient clipping — the dense coverage reward can destabilize PPO late in
    # training (reward/coverage regressing after an early peak); clip to stabilize.
    if "max_grad_norm" in train_sig_params:
        train_kwargs["max_grad_norm"] = 1.0

    # LR schedule (adaptive KL trust region). Only inject when opted in.
    if cfg["lr_schedule"] != "none":
        if "learning_rate_schedule" in train_sig_params:
            train_kwargs["learning_rate_schedule"] = "ADAPTIVE_KL"
            if "desired_kl" in train_sig_params:
                train_kwargs["desired_kl"] = float(cfg["desired_kl"])
            if "learning_rate_schedule_min_lr" in train_sig_params:
                train_kwargs["learning_rate_schedule_min_lr"] = float(cfg["lr_min"])
            if "learning_rate_schedule_max_lr" in train_sig_params:
                train_kwargs["learning_rate_schedule_max_lr"] = float(cfg["lr_max"])
            print(f"[ppo] ADAPTIVE_KL LR schedule ON: desired_kl={cfg['desired_kl']} "
                  f"lr in [{cfg['lr_min']},{cfg['lr_max']}] start={cfg['learning_rate']}")
        else:
            print("[ppo] NOTE: brax has no learning_rate_schedule; --lr-schedule ignored.")

    if cfg["clipping_epsilon"] is not None:
        if "clipping_epsilon" in train_sig_params:
            train_kwargs["clipping_epsilon"] = float(cfg["clipping_epsilon"])
        else:
            print("[ppo] NOTE: brax has no clipping_epsilon; --clipping-epsilon ignored.")

    if cfg["clipping_epsilon_value"] is not None:
        if "clipping_epsilon_value" in train_sig_params:
            train_kwargs["clipping_epsilon_value"] = float(cfg["clipping_epsilon_value"])
        else:
            print("[ppo] NOTE: brax has no clipping_epsilon_value; --clipping-epsilon-value ignored.")

    if cfg["normalize_advantage"] is not None:
        if "normalize_advantage" in train_sig_params:
            train_kwargs["normalize_advantage"] = bool(cfg["normalize_advantage"])
        else:
            print("[ppo] NOTE: brax has no normalize_advantage; --[no-]normalize-advantage ignored.")

    if cfg["normalize_until_count"] is not None:
        if "normalize_until_count" in train_sig_params:
            train_kwargs["normalize_until_count"] = int(cfg["normalize_until_count"])
        else:
            print("[ppo] NOTE: brax has no normalize_until_count; ignored.")

    if cfg["gae_lambda"] is not None:
        if "gae_lambda" in train_sig_params:
            train_kwargs["gae_lambda"] = float(cfg["gae_lambda"])
        else:
            print("[ppo] NOTE: brax has no gae_lambda; ignored.")

    if cfg["vf_loss_coefficient"] is not None:
        if "vf_loss_coefficient" in train_sig_params:
            train_kwargs["vf_loss_coefficient"] = float(cfg["vf_loss_coefficient"])
        else:
            print("[ppo] NOTE: brax has no vf_loss_coefficient; ignored.")

    # Deterministic eval is a real brax kwarg but its non-default (False) IS the current behavior,
    # so only inject when the user opts in to True, and still guard on the signature.
    if cfg["deterministic_eval"]:
        if "deterministic_eval" in train_sig_params:
            train_kwargs["deterministic_eval"] = True
            print("[ppo] deterministic_eval=True (greedy/mean policy at eval).")
        else:
            print("[ppo] NOTE: brax has no deterministic_eval; --deterministic-eval ignored.")

    # --- Train. ppo.train returns a 3-tuple. ---
    print("[ppo] starting ppo.train(...) — expect 30-60s first-run JIT compile.")
    make_inference_fn, params, metrics = ppo.train(**train_kwargs)

    elapsed = time.time() - start_time
    if tb_writer is not None:
        try:
            tb_writer.close()
        except Exception:  # noqa: BLE001
            pass

    print("=" * 78)
    print(f"[done] trained {cfg['num_timesteps']} steps in {elapsed:.1f}s")
    print(f"[done] metrics.csv : {os.path.abspath(csv_path)}")
    print(f"[done] tensorboard : {os.path.abspath(tb_dir)}")
    print(f"[done] checkpoints : {ckpt_dir_abs}")
    final = map_eval_metrics(metrics) if isinstance(metrics, dict) else {}
    if final:
        print(f"[done] final eval  : reward={final.get('eval/episode_reward')}, "
              f"coverage={final.get('eval/coverage')}, "
              f"collision={final.get('eval/collision_rate')}")
    print("=" * 78)

    # make_inference_fn + params are what the rollout/eval (Agent-C) loads from ckpt.
    _ = make_inference_fn  # retained for clarity; checkpoint is the hand-off artifact.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
