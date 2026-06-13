"""praxis/train.py — Brax PPO training entrypoint for the Praxis nav task.

CLI wiring around ``brax.training.agents.ppo.train`` against ``praxis.envs.NavEnv``
(Agent-A's MJX env), wrapped with ``mujoco_playground.wrapper.wrap_for_brax_training``.

Run (once the real env + GPU container are up):
    python -m praxis.train --smoke      # <=5-min DoD-2 gate
    python -m praxis.train              # full first-learnable run (num_timesteps=2e7)
    python -m praxis.train --stub --smoke   # env-less smoke (uses praxis.agent._stub_env)

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
    normalize_observations=True,  # Brax default is False — MUST enable.
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

    # success_rate <- eval/episode_success (mean of the 'success' metric)
    succ = _find_metric(metrics, "success")
    out["eval/success_rate"] = succ[1] if succ else float("nan")

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
        "eval/success_rate",
        "eval/collision_rate",
    ]
    state = {"header_written": False, "extra_cols": []}

    os.makedirs(os.path.dirname(csv_path) or ".", exist_ok=True)

    def progress_fn(num_steps: int, metrics: Dict[str, Any]) -> None:
        wall_s = time.time() - start_time

        # Brax calls progress_fn for BOTH eval rounds (keys under 'eval/') and, when
        # log_training_metrics=True, training rounds (keys under 'training/'). Only eval
        # rounds carry the per-episode success/collision metrics we curve, so write CSV
        # rows for eval rounds ONLY; training rounds get logged to TB and skipped here
        # (keeps metrics.csv clean for the DoD-2 learning plot).
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

        # Determine extra columns (sorted, stable) beyond the canonical three.
        canonical = {"eval/episode_reward", "eval/success_rate", "eval/collision_rate"}
        extras = sorted(k for k in mapped.keys() if k not in canonical)
        # Lock extra column set on first write to keep CSV rectangular.
        if not state["header_written"]:
            state["extra_cols"] = extras
        cols = base_cols + state["extra_cols"]

        row = {
            "step": int(num_steps),
            "wall_s": round(wall_s, 2),
            "eval/episode_reward": mapped.get("eval/episode_reward", float("nan")),
            "eval/success_rate": mapped.get("eval/success_rate", float("nan")),
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
        s = row["eval/success_rate"]
        c = row["eval/collision_rate"]
        print(
            f"[eval] step={int(num_steps):>10d}  "
            f"reward={r:>9.3f}  success={s:>6.3f}  collision={c:>6.3f}  "
            f"({wall_s:6.1f}s)"
        )

    return progress_fn


# --------------------------------------------------------------------------- #
# Env construction
# --------------------------------------------------------------------------- #
def build_env(stub: bool, episode_length: int, n_obstacles: Optional[int] = None):
    """Construct the (unwrapped) environment.

    Default: real ``praxis.envs.NavEnv`` (Agent-A). ``--stub`` swaps in the dummy.
    ``n_obstacles`` (0..MAX_OBSTACLES) sets how many moving obstacles are active —
    use 0 to prove pure goal-reaching first (curriculum).
    """
    if stub:
        from praxis.agent._stub_env import make_stub_env
        print("[env] Using STUB env (praxis.agent._stub_env) — NOT the real NavEnv.")
        return make_stub_env(episode_length=episode_length)

    # Real path. Pass episode_length INTO the env so its internal timeout (and thus
    # info['time_out'] for bootstrapping) fires at the SAME step the Brax EpisodeWrapper
    # truncates. Otherwise a mismatch makes bootstrap_on_timeout incorrect.
    from praxis.envs import NavEnv, default_config  # type: ignore
    cfg = default_config()
    cfg.episode_length = int(episode_length)
    if n_obstacles is not None:
        cfg.n_active_obstacles = max(0, min(int(n_obstacles), int(contract.MAX_OBSTACLES)))
    print(f"[env] Using real praxis.envs.NavEnv (episode_length={int(episode_length)}, "
          f"n_active_obstacles={int(cfg.n_active_obstacles)}).")
    return NavEnv(cfg)


def build_randomization_fn(no_randomization: bool, stub: bool):
    """Return a randomization_fn (or None) for wrap_for_brax_training.

    Domain randomization is toggleable. It is disabled for the stub (no model) and
    when --no-randomization is set.
    """
    if no_randomization or stub:
        return None
    try:
        from praxis.envs import domain_randomize  # type: ignore
    except Exception as e:  # noqa: BLE001
        print(f"[env] NOTE: domain_randomize unavailable ({e}); training without DR.")
        return None
    # NOTE: Agent-A's domain_randomize is a Brax-style randomization_fn(model, rng)
    # -> (batched_model, in_axes). wrap_for_brax_training threads `rng` per call, so
    # we pass the function itself (optionally partial'd). No args to bind here.
    return functools.partial(domain_randomize)


def wrap_env(env, randomization_fn):
    """Wrap an MjxEnv with Playground's Brax-training wrapper (auto-reset + vmap).

    Passes randomization_fn only if the installed wrapper signature accepts it.
    """
    from mujoco_playground import wrapper  # type: ignore

    wrap = wrapper.wrap_for_brax_training
    # NOTE: wrap_for_brax_training signature varies across Playground versions; the
    # current one accepts `randomization_fn=`. We inspect to stay forward/backward safe.
    try:
        sig = inspect.signature(wrap)
        accepts_rand = "randomization_fn" in sig.parameters
    except (TypeError, ValueError):
        accepts_rand = True

    if randomization_fn is not None and accepts_rand:
        return wrap(env, randomization_fn=randomization_fn)
    if randomization_fn is not None and not accepts_rand:
        print("[env] NOTE: wrapper does not accept randomization_fn; "
              "training without domain randomization.")
    return wrap(env)


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
    p.add_argument("--num-evals", type=int, default=DEFAULTS["num_evals"])
    p.add_argument("--seed", type=int, default=DEFAULTS["seed"])

    p.add_argument("--run-name", type=str, default=None,
                   help="Run name; defaults to praxis-<UTC timestamp>.")
    p.add_argument("--checkpoint-dir", type=str, default=None,
                   help="Orbax checkpoint dir; defaults to ckpts/<run>.")
    p.add_argument("--restore-checkpoint-path", type=str, default=None,
                   help="Optional Orbax checkpoint to restore and continue from.")

    p.add_argument("--n-obstacles", type=int, default=None,
                   help="Active moving obstacles (0..4). 0 = pure goal-reaching (curriculum). "
                        "Default: env default (4).")
    p.add_argument("--no-randomization", action="store_true",
                   help="Disable domain randomization.")
    p.add_argument("--stub", action="store_true",
                   help="Use the dummy stub env (smoke import / env-less runs only).")
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
    )
    if args.smoke:
        # Only override the few smoke knobs; respect explicit user-set everything else.
        cfg["num_timesteps"] = SMOKE_OVERRIDES["num_timesteps"]
        cfg["num_envs"] = SMOKE_OVERRIDES["num_envs"]
        cfg["num_evals"] = SMOKE_OVERRIDES["num_evals"]
    return cfg


def main(argv: Optional[list] = None) -> int:
    args = build_parser().parse_args(argv)
    cfg = resolve_config(args)

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
    print(f"Praxis PPO  run={run_name}")
    print(f"  stub={args.stub}  randomization={'off' if (args.no_randomization or args.stub) else 'on'}")
    print(f"  hyperparams: {cfg}")
    print(f"  metrics.csv -> {os.path.abspath(csv_path)}")
    print(f"  tensorboard -> {os.path.abspath(tb_dir)}")
    print(f"  checkpoints -> {ckpt_dir_abs}")
    print("=" * 78)

    # --- Heavy imports (jax/brax/playground) happen here, inside main() ---
    from brax.training.agents.ppo import train as ppo

    # Build env (+ optional randomization fn). Pass the UNWRAPPED env and let Brax
    # wrap it via wrap_env_fn (the official Playground+Brax pattern, verified against
    # the installed ppo.train) so Brax threads the per-env rng into
    # randomization_fn(model, rng). Pre-wrapping + the default wrap_env=True would
    # DOUBLE-wrap the env and break. The stub is already a brax envs.Env; it is passed
    # straight through and brax default-wraps it.
    raw_env = build_env(stub=args.stub, episode_length=cfg["episode_length"],
                        n_obstacles=args.n_obstacles)
    randomization_fn = build_randomization_fn(
        no_randomization=args.no_randomization, stub=args.stub
    )

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
    if not args.stub:
        from mujoco_playground import wrapper as mjxp_wrapper
        if "wrap_env_fn" in train_sig_params:
            train_kwargs["wrap_env_fn"] = mjxp_wrapper.wrap_for_brax_training
        else:
            # Older brax: pre-wrap and disable brax's own wrapping to avoid double-wrap.
            train_kwargs["environment"] = mjxp_wrapper.wrap_for_brax_training(raw_env)
            if "wrap_env" in train_sig_params:
                train_kwargs["wrap_env"] = False
        if randomization_fn is not None and "randomization_fn" in train_sig_params:
            train_kwargs["randomization_fn"] = randomization_fn
            print("[env] domain randomization ENABLED.")

    # Correct truncation handling (fact #8): bootstrap value on timeout; terminate on
    # collision/success. Agent-A sets info['truncation'] accordingly.
    if "bootstrap_on_timeout" in train_sig_params:
        train_kwargs["bootstrap_on_timeout"] = True

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
              f"success={final.get('eval/success_rate')}, "
              f"collision={final.get('eval/collision_rate')}")
    print("=" * 78)

    # make_inference_fn + params are what the rollout/eval (Agent-C) loads from ckpt.
    _ = make_inference_fn  # retained for clarity; checkpoint is the hand-off artifact.
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
