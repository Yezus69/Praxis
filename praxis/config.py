"""praxis/config.py — central ``ml_collections`` config for eval/render + plotting.

This holds the **eval / render / plotting** knobs that ``praxis/eval_render.py`` and
``praxis/plot_curves.py`` consume, plus a clearly-marked *convenience mirror* of the
training hyperparameters so everything is discoverable in one place.

IMPORTANT — ownership:
  * The AUTHORITATIVE training hyperparameters live in ``praxis/train.py`` (Agent-B).
    The ``train`` sub-config below is a NON-authoritative mirror for documentation /
    discoverability only. Do NOT read it back into the trainer; it can drift.
  * The AUTHORITATIVE obs/action/episode/reward shapes live in ``praxis/contract.py``.
    We import episode length etc. from there — never hardcode a second copy.

This module is intentionally light: it imports ``ml_collections`` and ``praxis.contract``
(pure Python) only. We deliberately do NOT import jax/mujoco at module top so this file
stays importable on ANY host (including native Windows with no JAX/GPU).
"""

from __future__ import annotations

import ml_collections

from praxis import contract


def get_config() -> ml_collections.ConfigDict:
    """Return the Praxis eval/render/plot config as an ``ml_collections.ConfigDict``.

    Returns:
      A fully-populated, mutable ``ConfigDict``. Callers may ``.unlock()`` and override
      fields; CLIs in ``eval_render.py`` / ``plot_curves.py`` override from argparse.
    """
    cfg = ml_collections.ConfigDict()

    # ----------------------------------------------------------------- #
    # Rollout / episode
    # ----------------------------------------------------------------- #
    # Episode length is shared with env + trainer — pull from the frozen contract.
    cfg.episode_length = int(contract.EPISODE_LENGTH)

    # Deterministic policy => clean, repeatable video (fact #9). Stochastic only for
    # debugging exploration.
    cfg.deterministic = True

    # PRNG seed for the host rollout loop (jax.random.PRNGKey(seed)).
    cfg.seed = 0

    # How many independent episodes to roll out. When > 1, eval_render renders the
    # "best" (goal-reaching) episode, else concatenates — see eval_render.py.
    cfg.n_eval_episodes = 1

    # ----------------------------------------------------------------- #
    # Rendering / video
    # ----------------------------------------------------------------- #
    render = ml_collections.ConfigDict()
    # Camera baked into nav_scene.xml — a trackcom chase cam named "track".
    render.camera = "track"
    # Offscreen frame size (height, width). 480x640 is a good watchable default and
    # cheap to encode. NavEnv.render forwards these to mujoco.Renderer.
    render.height = 480
    render.width = 640
    # GL backend selection: "auto" tries egl first then falls back to osmesa if the
    # frames come back black/empty or rendering raises (EGL in WSL2 is flaky).
    # eval_render.py sets os.environ['MUJOCO_GL'] / ['PYOPENGL_PLATFORM'] from this
    # BEFORE importing mujoco / mujoco_playground.
    render.mujoco_gl = "auto"  # {"auto", "egl", "osmesa"}
    # Below this stddev across all pixels we treat the rendered clip as black/empty
    # and trigger the osmesa fallback (the "black-frame guard").
    render.black_frame_std_threshold = 1e-3
    # FPS fallback when env.dt is unavailable. Real fps is int(1 / env.dt).
    render.fps_fallback = 20
    cfg.render = render

    # ----------------------------------------------------------------- #
    # Output paths
    # ----------------------------------------------------------------- #
    paths = ml_collections.ConfigDict()
    paths.video_out = "rollout.mp4"       # default --out for eval_render
    paths.curves_out = "curves.png"       # default basename for plot_curves
    paths.runs_dir = "runs"               # runs/<run>/metrics.csv, runs/<run>/curves.png
    paths.ckpt_dir = "ckpts"              # ckpts/<run> (Orbax) — eval_render --checkpoint-dir
    cfg.paths = paths

    # ----------------------------------------------------------------- #
    # Plotting (plot_curves.py)
    # ----------------------------------------------------------------- #
    plot = ml_collections.ConfigDict()
    plot.dpi = 120
    plot.fig_width = 12.0       # inches (3 panels side by side)
    plot.fig_height = 4.0
    # The x-axis column and the three y-series we expect in runs/<run>/metrics.csv.
    # These are the column NAMES the trainer (Agent-B) writes. plot_curves degrades
    # gracefully (skips the panel) if a column is missing.
    plot.x_column = "step"
    plot.reward_column = "eval/episode_reward"
    plot.success_column = "eval/success_rate"
    plot.collision_column = "eval/collision_rate"
    cfg.plot = plot

    # ----------------------------------------------------------------- #
    # Convenience MIRROR of training hyperparameters (NON-AUTHORITATIVE).
    # The real values live in praxis/train.py (Agent-B). Mirrored here only so the
    # full picture is discoverable from one config. Sourced from agent/README.md.
    # DO NOT feed these back into the trainer.
    # ----------------------------------------------------------------- #
    train = ml_collections.ConfigDict()
    train.num_timesteps = int(2e7)
    train.num_envs = 2048
    train.episode_length = int(contract.EPISODE_LENGTH)
    train.unroll_length = 20
    train.batch_size = 256
    train.num_minibatches = 32
    train.num_updates_per_batch = 4
    train.learning_rate = 3e-4
    train.entropy_cost = 1e-2
    train.discounting = 0.97
    train.reward_scaling = 1.0
    train.normalize_observations = True
    train.num_evals = 10
    train.seed = 0
    # Network sizes mirror praxis/agent/networks.py defaults.
    train.policy_hidden_layer_sizes = (256, 256, 256)
    train.value_hidden_layer_sizes = (256, 256, 256)
    cfg.train = train

    return cfg


__all__ = ["get_config"]
