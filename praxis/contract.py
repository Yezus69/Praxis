"""praxis/contract.py — the FROZEN integration contract.

The single source of truth for the obs/action/reward shapes that the simulator
(``praxis/envs``) and the agent/trainer (``praxis/agent``, ``praxis/train.py``)
must agree on EXACTLY. Everything imports these constants; nobody redefines them.

Design spec: ../sim/README.md (observation space) and ../agent/README.md (reward).

This module is intentionally **pure Python** — no jax/numpy import — so it is safe
to import from tests, the trainer, the env, and tooling without pulling in heavy
or platform-specific deps. Keep it that way.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Action contract
# --------------------------------------------------------------------------- #
# Holonomic planar velocity command [vx, vy], each in [-ACTION_LIMIT, ACTION_LIMIT].
# Applied through VELOCITY actuators inside env.step (scaled to AGENT_MAX_SPEED).
# NEVER set qpos/position directly (tunnels through obstacles; collision never fires).
ACT_DIM: int = 2
ACTION_LIMIT: float = 1.0          # raw policy action range is [-1, 1]
AGENT_MAX_SPEED: float = 1.5       # m/s that |action|==1 maps to inside step()

# --------------------------------------------------------------------------- #
# Obstacle / observation sizing
# --------------------------------------------------------------------------- #
# Fixed model topology under jit/vmap: the MJCF declares MAX_OBSTACLES movers.
# Per-env, extras are disabled via a boolean `active` mask (NOT by adding/removing
# geoms). We observe the K nearest *active* obstacles, sorted by distance every step.
MAX_OBSTACLES: int = 4             # mocap movers declared in the scene XML
K: int = 4                         # K nearest obstacles exposed to the policy

# --------------------------------------------------------------------------- #
# Observation layout (fixed-shape, MLP-friendly — sim/README.md)
# --------------------------------------------------------------------------- #
#   [ goal:       dx, dy, dist, heading_err          ] -> 4
#   [ agent vel:  vx, vy, omega                       ] -> 3
#   [ K obstacles: (px, py, vx, vy) each, goal-relative-agnostic (agent frame) ] -> 4*K
#   [ active mask: one bit per obstacle slot          ] -> K
# Slots are sorted by distance EVERY step; an MLP is not permutation-invariant, so a
# slot must always mean "the n-th nearest active obstacle". Zero-pad empty slots.
GOAL_DIM: int = 4                  # dx, dy, dist, heading_err
VEL_DIM: int = 3                   # vx, vy, omega
PER_OBSTACLE_DIM: int = 4          # px, py, vx, vy (relative to agent)

OBS_DIM: int = GOAL_DIM + VEL_DIM + PER_OBSTACLE_DIM * K + K   # 4 + 3 + 16 + 4 = 27

# Half-open slice bounds into the flat obs vector. Index with these, never literals.
GOAL_SLICE = (0, GOAL_DIM)                                     # [0, 4)
VEL_SLICE = (GOAL_SLICE[1], GOAL_SLICE[1] + VEL_DIM)           # [4, 7)
OBST_SLICE = (VEL_SLICE[1], VEL_SLICE[1] + PER_OBSTACLE_DIM * K)  # [7, 23)
MASK_SLICE = (OBST_SLICE[1], OBST_SLICE[1] + K)                # [23, 27)

assert MASK_SLICE[1] == OBS_DIM, "obs layout slices must tile [0, OBS_DIM) exactly"

# --------------------------------------------------------------------------- #
# Reward contract (agent/README.md)
# --------------------------------------------------------------------------- #
#   reward = + K1 * (prev_dist - dist)     # dense progress toward goal
#            - K2 * collision              # collision penalty (and TERMINATE)
#            - K3                           # small per-step time penalty
#            + K4 * success                # success bonus    (and TERMINATE)
#
# Notes that decide whether it learns at all (see agent/README.md):
#  * collision & success are TRUE terminations (done=1, NOT bootstrapped).
#  * timeout is a TRUNCATION (done=1 but info['truncation']=1 -> value IS bootstrapped).
#    A terminal collision must NOT be bootstrapped like a timeout or "suicide" to
#    escape the time penalty becomes optimal.
#  * keep K3 small vs per-step K1*progress so a successful path nets clearly positive.
REWARD_KEYS = ("k1", "k2", "k3", "k4")
DEFAULT_REWARD_WEIGHTS = {
    "k1": 1.0,    # progress-to-goal (per metre closed)
    "k2": 1.0,    # collision penalty (terminal)
    "k3": 0.005,  # per-step time penalty (small vs per-step progress)
    "k4": 10.0,   # success bonus (terminal)
}

# --------------------------------------------------------------------------- #
# Task geometry / episode
# --------------------------------------------------------------------------- #
GOAL_RADIUS: float = 0.5           # within this distance of goal site => success
COLLISION_DIST: float = 0.0        # contact-based; <=0 means "use actual MJX contacts"
EPISODE_LENGTH: int = 1000         # steps before timeout/truncation (env + trainer share)
ARENA_HALF: float = 3.0            # arena spans [-ARENA_HALF, ARENA_HALF] in x and y

# Metric names emitted into state.metrics every step. Brax surfaces these as
# eval/episode_<name>; the trainer maps them to success_rate / collision_rate curves.
METRIC_SUCCESS = "success"
METRIC_COLLISION = "collision"
METRIC_REWARD_COMPONENTS = ("reward_progress", "reward_collision", "reward_time", "reward_success")

__all__ = [
    "ACT_DIM", "ACTION_LIMIT", "AGENT_MAX_SPEED",
    "MAX_OBSTACLES", "K",
    "GOAL_DIM", "VEL_DIM", "PER_OBSTACLE_DIM", "OBS_DIM",
    "GOAL_SLICE", "VEL_SLICE", "OBST_SLICE", "MASK_SLICE",
    "REWARD_KEYS", "DEFAULT_REWARD_WEIGHTS",
    "GOAL_RADIUS", "COLLISION_DIST", "EPISODE_LENGTH", "ARENA_HALF",
    "METRIC_SUCCESS", "METRIC_COLLISION", "METRIC_REWARD_COMPONENTS",
]
