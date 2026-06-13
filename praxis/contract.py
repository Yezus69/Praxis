"""praxis/contract.py — the FROZEN integration contract (COVERAGE task).

Single source of truth for the obs/action/reward shapes shared by the simulator
(``praxis/envs``) and the trainer/eval (``praxis/train.py``, ``praxis/eval_render.py``).

TASK: the agent EXPLORES the arena and tries to COVER the whole area, with REAL
physics — it physically collides with (bounces off) walls and moving obstacles.

Pure Python (no jax/numpy) so it is safe to import anywhere.
"""

from __future__ import annotations

# --------------------------------------------------------------------------- #
# Action — holonomic planar velocity command [vx, vy] in [-1, 1], scaled to
# AGENT_MAX_SPEED inside step() and applied via velocity actuators.
# --------------------------------------------------------------------------- #
ACT_DIM: int = 2
ACTION_LIMIT: float = 1.0
AGENT_MAX_SPEED: float = 2.0          # ctrl scale; effective top speed ~1.85 m/s

# --------------------------------------------------------------------------- #
# Arena + coverage grid
# --------------------------------------------------------------------------- #
ARENA_HALF: float = 3.0               # arena spans [-3, 3] in x and y
GRID_SIZE: int = 6                    # GxG coverage cells (6x6 = 36; cell ~1.0 m)
N_CELLS: int = GRID_SIZE * GRID_SIZE  # 36
AGENT_RADIUS: float = 0.15
OBSTACLE_RADIUS: float = 0.25

# --------------------------------------------------------------------------- #
# Obstacles (REAL collisions). All MAX_OBSTACLES are active/colliding.
# --------------------------------------------------------------------------- #
MAX_OBSTACLES: int = 4
K: int = 4                            # K nearest obstacles exposed to the policy

# --------------------------------------------------------------------------- #
# Observation layout (fixed-shape, MLP-friendly)
#   [ agent: x/arena, y/arena, vx, vy                 ] -> 4
#   [ K obstacles: (px, py, vx, vy) relative          ] -> 4*K
#   [ per-obstacle active mask                         ] -> K
#   [ frontier: dx, dy, dist to NEAREST UNVISITED cell ] -> 3
#   [ covered fraction so far                          ] -> 1
# The "frontier" vector (direction to the nearest unvisited cell) is the key
# exploration signal — an MLP can follow it directly, which is far easier to learn
# than reasoning over a flattened occupancy grid. The visited grid itself lives in
# state.info (it is the env's memory), not in the obs.
# --------------------------------------------------------------------------- #
AGENT_DIM: int = 4
PER_OBSTACLE_DIM: int = 4
FRONTIER_DIM: int = 3        # dx, dy (unit) and distance to nearest unvisited cell
COVERED_DIM: int = 1         # fraction of cells covered so far

OBS_DIM: int = AGENT_DIM + PER_OBSTACLE_DIM * K + K + FRONTIER_DIM + COVERED_DIM  # 28

AGENT_SLICE = (0, AGENT_DIM)                                   # [0, 4)
OBST_SLICE = (AGENT_SLICE[1], AGENT_SLICE[1] + PER_OBSTACLE_DIM * K)  # [4, 20)
MASK_SLICE = (OBST_SLICE[1], OBST_SLICE[1] + K)               # [20, 24)
FRONTIER_SLICE = (MASK_SLICE[1], MASK_SLICE[1] + FRONTIER_DIM)  # [24, 27)
COVERED_SLICE = (FRONTIER_SLICE[1], FRONTIER_SLICE[1] + COVERED_DIM)  # [27, 28)

assert COVERED_SLICE[1] == OBS_DIM, "obs layout slices must tile [0, OBS_DIM)"

# --------------------------------------------------------------------------- #
# Reward
#   reward = + K_COV  * (new cells covered this step)
#            - K_COLL * (obstacle collision this step)   # NON-terminal
#            - K_TIME                                     # tiny per-step nudge
# No goal, no success bonus. Episodes run a fixed length; collisions do NOT end
# the episode (the agent bounces off and keeps exploring).
# --------------------------------------------------------------------------- #
REWARD_KEYS = ("k_cov", "k_coll", "k_time")
DEFAULT_REWARD_WEIGHTS = {
    "k_cov": 1.0,     # per newly-covered cell
    "k_coll": 0.1,    # per step in contact with an obstacle (light; don't drown coverage)
    "k_time": 0.01,   # tiny per-step penalty (discourages dithering)
}

COLLISION_MARGIN: float = 0.05        # contact band for the collision penalty
EPISODE_LENGTH: int = 600             # steps per episode (env + trainer share)

# Metrics emitted into state.metrics every step (Brax surfaces eval/episode_<name>).
METRIC_COVERAGE = "coverage"          # fraction of cells visited so far (0..1)
METRIC_COLLISION = "collision"
METRIC_REWARD_COMPONENTS = ("reward_cover", "reward_collision", "reward_time")

__all__ = [
    "ACT_DIM", "ACTION_LIMIT", "AGENT_MAX_SPEED",
    "ARENA_HALF", "GRID_SIZE", "N_CELLS", "AGENT_RADIUS", "OBSTACLE_RADIUS",
    "MAX_OBSTACLES", "K",
    "AGENT_DIM", "PER_OBSTACLE_DIM", "FRONTIER_DIM", "COVERED_DIM", "OBS_DIM",
    "AGENT_SLICE", "OBST_SLICE", "MASK_SLICE", "FRONTIER_SLICE", "COVERED_SLICE",
    "REWARD_KEYS", "DEFAULT_REWARD_WEIGHTS", "COLLISION_MARGIN", "EPISODE_LENGTH",
    "METRIC_COVERAGE", "METRIC_COLLISION", "METRIC_REWARD_COMPONENTS",
]
