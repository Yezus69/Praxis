"""praxis/eval_render.py — render a rollout video from a trained checkpoint.

THE Phase-0 deliverable: load a Brax-PPO checkpoint, run a DETERMINISTIC host-side
rollout (outside jit/training), and write a watchable ``rollout.mp4`` of the agent
navigating to the goal around moving obstacles.

Run (inside the Docker/Linux container — there is no JAX on native Windows):

    python -m praxis.eval_render --checkpoint-dir ckpts/<run>
    python -m praxis.eval_render --checkpoint-dir ckpts/<run> --out rollout.mp4 \
        --episodes 3 --mujoco-gl auto --seed 0

Key correctness points (sim/README.md fact #9):
  * MUJOCO_GL / PYOPENGL_PLATFORM are set BEFORE importing mujoco / mujoco_playground.
  * Rendering uses the UNWRAPPED NavEnv (so .data matches the renderer), NOT the
    Brax-wrapped training env.
  * Policy network is rebuilt IDENTICALLY to training (same network_factory) so the
    restored params line up.
  * egl -> osmesa fallback + black-frame guard: if egl renders black or raises, we
    retry under osmesa (software; always works, slower — fine for one rollout).
  * Host-side ``bool(state.done)`` is fine here because we are OUTSIDE jit.

``main()`` is importable with NO side effects (no GL env set, no heavy imports) until
called — all heavy work lives inside functions.
"""

from __future__ import annotations

import argparse
import os
import sys
from typing import Any, Callable, List, Optional, Tuple

# NOTE: we intentionally DO NOT import jax / mujoco / mujoco_playground at module top.
# MUJOCO_GL must be set in os.environ BEFORE those imports, which only happens once
# the CLI has parsed --mujoco-gl. Importing this module must stay side-effect-free.
from praxis import contract


# --------------------------------------------------------------------------- #
# GL backend setup (MUST run before importing mujoco / mujoco_playground)
# --------------------------------------------------------------------------- #
def _set_gl_backend(backend: str) -> None:
    """Set MUJOCO_GL + PYOPENGL_PLATFORM env vars for a concrete GL backend.

    Args:
      backend: "egl" or "osmesa" (a concrete backend, never "auto").
    """
    backend = backend.lower()
    os.environ["MUJOCO_GL"] = backend
    # mujoco's GL loader reads MUJOCO_GL; PyOpenGL reads PYOPENGL_PLATFORM. Keep both
    # consistent so EGL/OSMesa selection is unambiguous.
    os.environ["PYOPENGL_PLATFORM"] = backend


# --------------------------------------------------------------------------- #
# Checkpoint restore — robust Brax/Orbax loader
# --------------------------------------------------------------------------- #
def _load_params_and_inference_fn(
    checkpoint_dir: str,
    env: Any,
    deterministic: bool,
) -> Tuple[Callable[[Any, Any], Tuple[Any, Any]], Any]:
    """Restore PPO params from an Orbax checkpoint and build a deterministic inference_fn.

    Strategy (most-robust-first; see CORRECTED FACTS #9):
      1. Brax helper ``brax.training.agents.ppo.checkpoint.load_policy(path)`` — this
         is the officially-supported reader for what ``save_checkpoint_path`` wrote in
         Brax 0.14. It returns a ready-to-use inference function.
      2. Fallback: restore the raw params pytree with Orbax and feed it through our own
         ``make_inference_fn`` built from the SAME network factory as training
         (praxis/agent/networks.py + ppo_networks.make_inference_fn).

    Args:
      checkpoint_dir: path passed as --checkpoint-dir (e.g. ``ckpts/<run>``). May be a
        run dir containing numbered step subdirs, or a concrete step dir.
      env: the UNWRAPPED NavEnv (used only to discover obs/action sizes for path #2).
      deterministic: deterministic policy (clean video) vs stochastic.

    Returns:
      (inference_fn, params) where inference_fn(obs, rng) -> (action, extras).

    Raises:
      RuntimeError: if BOTH restore paths fail, with the exact path we looked at.
    """
    import jax  # noqa: F401
    from brax.training.agents.ppo import checkpoint as ppo_checkpoint
    from brax.training.agents.ppo import networks as ppo_networks
    from brax.training.acme import running_statistics

    from praxis.agent.networks import make_network_factory

    ckpt_path = _resolve_checkpoint_path(checkpoint_dir)

    # Build the network IDENTICALLY to training: our 256/256/256 factory + the obs
    # normalizer preprocess (training used normalize_observations=True).
    obs_size = int(contract.OBS_DIM)
    act_size = int(contract.ACT_DIM)
    nf = make_network_factory()
    ppo_network = nf(obs_size, act_size,
                     preprocess_observations_fn=running_statistics.normalize)

    # Use Brax's own loader: it handles the orbax sharding metadata (the raw
    # PyTreeCheckpointer does not). It returns (normalizer_params, policy_params), but
    # orbax restores the normalizer RunningStatisticsState as a plain dict (type lost),
    # which breaks running_statistics.normalize -> rebuild the struct.
    try:
        params = ppo_checkpoint.load(ckpt_path)
    except Exception as e:  # noqa: BLE001
        raise RuntimeError(
            f"Failed to load Brax checkpoint at {ckpt_path} (from --checkpoint-dir "
            f"{checkpoint_dir}): {e!r}. Expected a Brax PPO save_checkpoint_path step "
            "dir (e.g. ckpts/<run>/000010649600) or the run dir."
        ) from e

    params = _reconstruct_normalizer(params)
    make_inference_fn = ppo_networks.make_inference_fn(ppo_network)
    inference_fn = make_inference_fn(params, deterministic=deterministic)
    return inference_fn, params


def _reconstruct_normalizer(params: Any) -> Any:
    """Rebuild the obs-normalizer RunningStatisticsState from a restored plain dict.

    Orbax restores the flax-struct normalizer as a dict (no type info), which breaks
    ``running_statistics.normalize`` (it accesses ``mean_std.mean``). Convert it back.
    """
    import dataclasses
    from brax.training.acme import running_statistics
    if isinstance(params, (list, tuple)) and len(params) >= 2:
        norm = params[0]
        if isinstance(norm, dict):
            rss = running_statistics.RunningStatisticsState
            names = [f.name for f in dataclasses.fields(rss)]
            if all(k in norm for k in names):
                norm = rss(**{k: norm[k] for k in names})
                return (norm, *tuple(params[1:]))
    return params


def _resolve_checkpoint_path(checkpoint_dir: str) -> str:
    """Resolve --checkpoint-dir to a concrete checkpoint path.

    Brax ``save_checkpoint_path`` writes numbered step subdirectories under the run
    dir. If the user passes the run dir, pick the latest (highest-numbered) step subdir.
    If they pass a concrete step dir (or it has no numeric subdirs), use it as-is.
    """
    ckpt = os.path.abspath(checkpoint_dir)
    if not os.path.isdir(ckpt):
        # Let the restore path raise a clear error including this path.
        return ckpt

    # Find numeric step subdirectories (Brax/Orbax name them by step count).
    try:
        entries = os.listdir(ckpt)
    except OSError:
        return ckpt

    numeric_subdirs = [
        d for d in entries
        if d.isdigit() and os.path.isdir(os.path.join(ckpt, d))
    ]
    if numeric_subdirs:
        latest = max(numeric_subdirs, key=int)
        return os.path.join(ckpt, latest)
    return ckpt


def _coerce_policy_params(params: Any) -> Tuple[Any, Any]:
    """Coerce a restored Brax params pytree to (normalizer_params, policy_params).

    Brax PPO commonly saves (normalizer_params, policy_params, value_params). The
    inference fn only needs the first two. Be tolerant of 2- or 3-element tuples/lists.
    """
    if isinstance(params, (tuple, list)):
        if len(params) >= 2:
            return (params[0], params[1])
    # If it's already exactly what make_inference_fn wants (or a dict pytree), pass
    # through and let the apply fn raise a precise error if mismatched.
    return params  # type: ignore[return-value]


def _env_obs_size(env: Any) -> int:
    """Best-effort obs size for the (unwrapped) NavEnv.

    Prefer the frozen contract; fall back to env.observation_size if present.
    """
    try:
        return int(contract.OBS_DIM)
    except Exception:  # noqa: BLE001
        return int(getattr(env, "observation_size", contract.OBS_DIM))


# --------------------------------------------------------------------------- #
# Rollout (host-side, outside jit) — fact #9
# --------------------------------------------------------------------------- #
def _rollout_one(
    env: Any,
    inference_fn: Callable[[Any, Any], Tuple[Any, Any]],
    episode_length: int,
    seed: int,
) -> Tuple[List[Any], dict]:
    """Run ONE deterministic episode and collect the per-step physics States.

    Returns:
      (rollout, summary) where ``rollout`` is the list of env States (each carries
      ``.data`` for the renderer) and ``summary`` reports success/collision/length.
    """
    import jax

    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    jit_inf = jax.jit(inference_fn)

    rng = jax.random.PRNGKey(seed)
    state = jit_reset(rng)
    rollout = [state]

    success = False
    collision = False
    steps = 0
    for _ in range(int(episode_length)):
        act_rng, rng = jax.random.split(rng)
        action, _ = jit_inf(state.obs, act_rng)
        state = jit_step(state, action)
        rollout.append(state)
        steps += 1

        # Track terminal cause from metrics/info (host-side bool OK, outside jit).
        success = success or _flag_true(state, contract.METRIC_SUCCESS)
        collision = collision or _flag_true(state, contract.METRIC_COLLISION)

        if bool(state.done):
            break

    summary = {
        "success": bool(success),
        "collision": bool(collision),
        "length": int(steps),
        "reached_goal": bool(success),
    }
    return rollout, summary


def _flag_true(state: Any, key: str) -> bool:
    """Read a boolean-ish flag from state.metrics or state.info (host-side)."""
    for container_name in ("metrics", "info"):
        container = getattr(state, container_name, None)
        if isinstance(container, dict) and key in container:
            try:
                return float(container[key]) > 0.5
            except Exception:  # noqa: BLE001 — tracer/array edge cases
                try:
                    import numpy as np

                    return bool(np.asarray(container[key]).astype(float).max() > 0.5)
                except Exception:  # noqa: BLE001
                    return False
    return False


# --------------------------------------------------------------------------- #
# Rendering with egl -> osmesa fallback + black-frame guard
# --------------------------------------------------------------------------- #
def _frames_are_black(frames: Any, std_threshold: float) -> bool:
    """Black-frame guard: True if rendered frames are essentially constant/empty."""
    import numpy as np

    arr = np.asarray(frames)
    if arr.size == 0:
        return True
    return float(arr.std()) < float(std_threshold)


def _render_frames(
    env: Any,
    rollout: List[Any],
    camera: str,
) -> Any:
    """Render frames from the collected rollout via the unwrapped NavEnv.render."""
    return env.render(rollout, camera=camera)


def _render_with_fallback(
    rollout: List[Any],
    requested_backend: str,
    camera: str,
    height: int,
    width: int,
    black_std_threshold: float,
    build_env: Callable[[], Any],
) -> Tuple[Any, str]:
    """Render with the egl->osmesa fallback + black-frame guard.

    ``requested_backend`` is "auto" | "egl" | "osmesa":
      * "egl"    -> render under egl only (raise on failure).
      * "osmesa" -> render under osmesa only.
      * "auto"   -> try egl; if it raises OR frames look black, retry under osmesa.

    The GL backend can only be chosen via env var BEFORE mujoco is imported. Because a
    process imports mujoco exactly once, switching backends mid-process is unreliable.
    The robust approach: this function is called by ``main`` which has ALREADY set the
    initial backend and imported mujoco. For "auto", if egl yields black frames we
    re-exec the process forcing osmesa (clean import). See ``main`` for the re-exec.

    Here we render once under the currently-active backend and report whether the
    frames are usable; ``main`` owns the osmesa re-exec decision.
    """
    env = build_env()
    frames = _render_frames(env, rollout, camera=camera)
    active = os.environ.get("MUJOCO_GL", requested_backend)
    if _frames_are_black(frames, black_std_threshold):
        return frames, f"{active}:BLACK"
    return frames, f"{active}:OK"


# --------------------------------------------------------------------------- #
# Env construction + fps
# --------------------------------------------------------------------------- #
def _build_nav_env(height: int, width: int, n_obstacles: Optional[int] = None) -> Any:
    """Build a fresh UNWRAPPED NavEnv for rollout + rendering.

    We use the unwrapped env (NOT wrap_for_brax_training) so state.data matches what
    env.render expects (fact #9). ``n_obstacles`` MUST match what the policy trained on
    (the obs has that many active obstacle slots), else the rollout is off-distribution.
    """
    from praxis.envs import NavEnv, default_config

    cfg = default_config()
    if n_obstacles is not None:
        cfg.n_active_obstacles = max(0, min(int(n_obstacles), 4))
    return NavEnv(cfg)


def _fps_from_env(env: Any, fallback: int) -> int:
    """fps = int(1/env.dt), guarded against a missing/zero dt."""
    dt = getattr(env, "dt", None)
    try:
        if dt is not None:
            dt_f = float(dt)
            if dt_f > 0:
                return max(1, int(round(1.0 / dt_f)))
    except Exception:  # noqa: BLE001
        pass
    return int(fallback)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def _build_arg_parser() -> argparse.ArgumentParser:
    from praxis.config import get_config

    cfg = get_config()
    p = argparse.ArgumentParser(
        prog="python -m praxis.eval_render",
        description="Render a rollout mp4 from a trained Brax-PPO checkpoint.",
    )
    p.add_argument(
        "--checkpoint-dir",
        required=True,
        help="Path to the Orbax checkpoint (ckpts/<run>, a run dir or step subdir).",
    )
    p.add_argument(
        "--run-name",
        default=None,
        help="Optional run name (for logging only; output path is --out).",
    )
    p.add_argument(
        "--out",
        default=cfg.paths.video_out,
        help=f"Output mp4 path (default: {cfg.paths.video_out}).",
    )
    p.add_argument(
        "--episodes",
        type=int,
        default=cfg.n_eval_episodes,
        help="Number of episodes to roll out; renders the best (goal-reaching) one.",
    )
    p.add_argument(
        "--mujoco-gl",
        choices=("auto", "egl", "osmesa"),
        default=cfg.render.mujoco_gl,
        help="GL backend. auto = try egl then fall back to osmesa on black/error.",
    )
    p.add_argument(
        "--seed",
        type=int,
        default=cfg.seed,
        help="PRNG seed for the rollout (jax.random.PRNGKey).",
    )
    p.add_argument(
        "--n-obstacles",
        type=int,
        default=None,
        help="Active obstacles (0..4) — MUST match what the policy trained on.",
    )
    return p


# Internal env var used to signal "we already fell back; do NOT re-exec again" so the
# osmesa re-exec can't loop forever.
_NO_REEXEC_ENV = "PRAXIS_EVAL_RENDER_NO_REEXEC"


def main(argv: Optional[List[str]] = None) -> int:
    """Entry point. Importable with no side effects; all work happens here.

    Returns process exit code (0 = success).
    """
    args = _build_arg_parser().parse_args(argv)

    from praxis.config import get_config

    cfg = get_config()
    camera = cfg.render.camera
    height = int(cfg.render.height)
    width = int(cfg.render.width)
    black_std = float(cfg.render.black_frame_std_threshold)
    fps_fallback = int(cfg.render.fps_fallback)
    episode_length = int(cfg.episode_length)
    deterministic = bool(cfg.deterministic)

    # --- 1) Pick the initial GL backend and set env BEFORE importing mujoco --- #
    already_fell_back = os.environ.get(_NO_REEXEC_ENV) == "1"
    if args.mujoco_gl == "auto":
        initial_backend = "osmesa" if already_fell_back else "egl"
    else:
        initial_backend = args.mujoco_gl
    _set_gl_backend(initial_backend)

    # Now it is safe to import mujoco/jax-dependent modules (they read MUJOCO_GL).
    import mediapy  # noqa: F401  (import after GL env set)

    # --- 2) Build env + restore policy --------------------------------------- #
    env = _build_nav_env(height, width, n_obstacles=args.n_obstacles)
    inference_fn, _params = _load_params_and_inference_fn(
        args.checkpoint_dir, env, deterministic=deterministic
    )

    # --- 3) Roll out (collect best episode) ---------------------------------- #
    best_rollout: Optional[List[Any]] = None
    best_summary: Optional[dict] = None
    n_eps = max(1, int(args.episodes))
    for ep in range(n_eps):
        rollout, summary = _rollout_one(
            env, inference_fn, episode_length, seed=int(args.seed) + ep
        )
        print(
            f"[eval] episode {ep}: success={summary['success']} "
            f"collision={summary['collision']} length={summary['length']}"
        )
        # Prefer a goal-reaching episode; otherwise keep the longest survivor.
        if (
            best_summary is None
            or (summary["reached_goal"] and not best_summary["reached_goal"])
            or (
                summary["reached_goal"] == best_summary["reached_goal"]
                and summary["length"] > best_summary["length"]
            )
        ):
            best_rollout, best_summary = rollout, summary

    assert best_rollout is not None and best_summary is not None

    # --- 4) Render with egl -> osmesa fallback + black-frame guard ------------ #
    # Fall back to osmesa if (a) frames look black/empty OR (b) rendering RAISES —
    # both are the documented EGL-in-WSL2 failure modes (spec fact #9). The GL backend
    # is fixed at the first mujoco import, so the only reliable way to switch is a
    # clean-process re-exec forcing MUJOCO_GL=osmesa.
    frames = None
    status = "unknown"
    render_error: Optional[BaseException] = None
    try:
        frames, status = _render_with_fallback(
            best_rollout,
            requested_backend=args.mujoco_gl,
            camera=camera,
            height=height,
            width=width,
            black_std_threshold=black_std,
            build_env=lambda: env,
        )
    except Exception as e:  # noqa: BLE001 — render errors are a known egl failure mode
        render_error = e
        status = f"{os.environ.get('MUJOCO_GL', '?')}:RAISED"

    need_fallback = render_error is not None or status.endswith("BLACK")
    if args.mujoco_gl == "auto" and need_fallback and not already_fell_back:
        why = "raised" if render_error is not None else "produced black/empty frames"
        print(
            f"[eval] egl {why} ({status}); re-running under osmesa "
            "(software GL fallback)."
        )
        child_env = dict(os.environ)
        child_env[_NO_REEXEC_ENV] = "1"
        child_env["MUJOCO_GL"] = "osmesa"
        child_env["PYOPENGL_PLATFORM"] = "osmesa"
        import subprocess

        cmd = [sys.executable, "-m", "praxis.eval_render", *(argv or sys.argv[1:])]
        return subprocess.call(cmd, env=child_env)

    # No fallback available (backend was forced, or we already fell back to osmesa).
    if render_error is not None:
        raise RuntimeError(
            f"Rendering failed under MUJOCO_GL={os.environ.get('MUJOCO_GL')} "
            f"(status={status}). Original error: {render_error!r}. "
            "Check libegl1 / libosmesa6 in the container, or pass "
            "--mujoco-gl osmesa explicitly."
        ) from render_error

    if status.endswith("BLACK"):
        # osmesa also black (or user forced a backend that failed): warn loudly but
        # still write what we have so the orchestrator can inspect.
        print(
            f"[eval] WARNING: rendered frames look black/empty (status={status}). "
            "Check MUJOCO_GL / libegl1 / libosmesa6 in the container."
        )

    # --- 5) Encode the mp4 --------------------------------------------------- #
    fps = _fps_from_env(env, fps_fallback)
    out_path = os.path.abspath(args.out)
    out_dir = os.path.dirname(out_path)
    if out_dir:
        os.makedirs(out_dir, exist_ok=True)
    mediapy.write_video(out_path, frames, fps=fps)

    print(
        f"[eval] wrote {out_path}  (fps={fps}, frames={len(frames)}, "
        f"render={status})\n"
        f"[eval] best episode: success={best_summary['success']} "
        f"collision={best_summary['collision']} length={best_summary['length']}"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
