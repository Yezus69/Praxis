"""praxis/envs/cover_env.py — Phase-0 COVERAGE / exploration env (REAL physics).

``CoverEnv`` is a MuJoCo-Playground / MJX env (functional Brax-style State API). The
agent explores the arena to COVER as many grid cells as possible while physically
colliding with (bouncing off) walls and moving obstacles.

Everything in reset/step is JAX-traceable (fixed shapes, no python `if` on tracers).
Real collisions come from MuJoCo contacts (cover_scene.xml: agent/walls/obstacles
share collision bitmasks); the agent floats (no floor contact) and has no yaw dof, so
there are no phantom contacts. The COLLISION PENALTY is computed geometrically
(distance to obstacles) — robust and traceable — and is NON-terminal.
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx
import ml_collections

try:  # pragma: no cover
    from mujoco_playground import mjx_env
except Exception:  # pragma: no cover
    from mujoco_playground._src import mjx_env  # type: ignore

from praxis import contract

_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_XML_PATH = os.path.join(_THIS_DIR, "cover_scene.xml")


def default_config() -> ml_collections.ConfigDict:
    cfg = ml_collections.ConfigDict()
    cfg.sim_dt = 0.01
    cfg.n_substeps = 5            # ctrl_dt = 0.05
    cfg.episode_length = contract.EPISODE_LENGTH

    cfg.reward = ml_collections.ConfigDict()
    cfg.reward.k_cov = contract.DEFAULT_REWARD_WEIGHTS["k_cov"]
    cfg.reward.k_coll = contract.DEFAULT_REWARD_WEIGHTS["k_coll"]
    cfg.reward.k_time = contract.DEFAULT_REWARD_WEIGHTS["k_time"]
    # --- WALL1 fix: opt-in reward shaping; defaults reproduce current behavior ---
    cfg.reward.terminate_on_full_coverage = False  # Variant A: true terminal when all cells covered
    cfg.reward.k_complete = 0.0                     # Variant A: completion bonus weight (0 => off)
    cfg.reward.collision_penalty_cap = 0.0          # Variant C.1: 0 => uncapped (current behavior)
    cfg.reward.patrol = False                       # Variant B: renewable freshness reward
    cfg.reward.k_fresh = 0.0                        # Variant B: weight on freshness restored/step
    cfg.reward.freshness_decay = 0.99               # Variant B: per-step freshness decay

    cfg.arena_half = contract.ARENA_HALF
    cfg.grid_size = contract.GRID_SIZE
    cfg.agent_radius = contract.AGENT_RADIUS
    cfg.obstacle_radius = contract.OBSTACLE_RADIUS
    cfg.agent_max_speed = contract.AGENT_MAX_SPEED
    cfg.collision_margin = contract.COLLISION_MARGIN
    cfg.spawn_margin = 0.5         # keep agent/obstacle spawns this far inside walls

    # Obstacle patrol. Some obstacles static (amp 0), some moving (sampled in reset).
    cfg.obstacle = ml_collections.ConfigDict()
    cfg.obstacle.max_amplitude = 1.0
    cfg.obstacle.min_frequency = 0.1
    cfg.obstacle.max_frequency = 0.25
    cfg.obstacle.height = 0.25
    cfg.obstacle.frac_moving = 0.5   # ~half the obstacles patrol, rest are static
    return cfg


class CoverEnv(mjx_env.MjxEnv):
    """Area-coverage exploration with real physics collisions (MJX)."""

    def __init__(self, config: Optional[ml_collections.ConfigDict] = None,
                 config_overrides: Optional[ml_collections.ConfigDict] = None) -> None:
        cfg = config if config is not None else default_config()
        try:
            super().__init__(cfg, config_overrides)  # type: ignore[arg-type]
        except Exception:
            try:
                super().__init__(cfg)  # type: ignore[misc]
            except Exception:
                pass
        self._config = cfg
        if config_overrides is not None:
            self._config.update(config_overrides)

        self._mj_model = mujoco.MjModel.from_xml_path(_XML_PATH)
        self._mj_model.opt.timestep = float(self._config.sim_dt)
        self._mjx_model = mjx.put_model(self._mj_model)
        self._n_substeps = int(self._config.n_substeps)

        # Agent dof addresses (slide x, slide y).
        self._qx = int(self._mj_model.jnt_qposadr[
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, "agent_x")])
        self._qy = int(self._mj_model.jnt_qposadr[
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, "agent_y")])
        self._vx = int(self._mj_model.jnt_dofadr[
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, "agent_x")])
        self._vy = int(self._mj_model.jnt_dofadr[
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, "agent_y")])

        # Grid cell centres (N_CELLS, 2) for the frontier obs. cell idx = gy*G + gx.
        g = int(self._config.grid_size)
        arena = float(self._config.arena_half)
        cell = 2.0 * arena / g
        cx = -arena + (jp.arange(g) + 0.5) * cell
        self._cell_centers = jp.stack([jp.tile(cx, g), jp.repeat(cx, g)], axis=-1)

    # ---- required MjxEnv interface ---- #
    @property
    def xml_path(self) -> str: return _XML_PATH
    @property
    def action_size(self) -> int: return contract.ACT_DIM
    @property
    def mj_model(self) -> mujoco.MjModel: return self._mj_model
    @property
    def mjx_model(self) -> Any: return self._mjx_model
    @property
    def dt(self) -> float: return float(self._config.sim_dt) * float(self._n_substeps)

    # ---- helpers (traceable) ---- #
    def _obstacle_state(self, info: Dict[str, Any], t: jax.Array):
        """Analytic patrol position (M,3) + planar velocity (M,2)."""
        c = info["obst_centre"]          # (M,2)
        axis = info["obst_axis"]         # (M,2)
        amp = info["obst_amp"]           # (M,)
        freq = info["obst_freq"]         # (M,)
        phase = info["obst_phase"]       # (M,)
        height = info["obst_height"]     # (M,)
        omega = 2.0 * jp.pi * freq
        ang = omega * t + phase
        offset = (amp * jp.sin(ang))[:, None] * axis
        vel = (amp * omega * jp.cos(ang))[:, None] * axis
        pos = jp.concatenate([c + offset, height[:, None]], axis=-1)
        return pos, vel

    def _agent_pose(self, data: Any):
        xy = jp.stack([data.qpos[self._qx], data.qpos[self._qy]])
        vel = jp.stack([data.qvel[self._vx], data.qvel[self._vy]])
        return xy, vel

    def _cell_index(self, xy: jax.Array) -> jax.Array:
        arena = float(self._config.arena_half)
        g = int(self._config.grid_size)
        gx = jp.clip(jp.floor((xy[0] + arena) / (2.0 * arena) * g).astype(jp.int32), 0, g - 1)
        gy = jp.clip(jp.floor((xy[1] + arena) / (2.0 * arena) * g).astype(jp.int32), 0, g - 1)
        return gy * g + gx

    def _build_obs(self, agent_xy, agent_vel, obst_pos, obst_vel, visited):
        arena = float(self._config.arena_half)
        vmax = float(self._config.agent_max_speed)
        agent_feat = jp.array([
            agent_xy[0] / arena, agent_xy[1] / arena,
            agent_vel[0] / vmax, agent_vel[1] / vmax,
        ])
        # K nearest obstacles (relative pos + vel)
        rel = obst_pos[:, :2] - agent_xy[None, :]          # (M,2)
        dist = jp.linalg.norm(rel, axis=-1)                # (M,)
        _, idx = jax.lax.top_k(-dist, contract.K)
        per = jp.concatenate([rel[idx] / arena, obst_vel[idx] / vmax], axis=-1).reshape(-1)
        mask = jp.ones((contract.K,))

        # Frontier: unit direction + distance to the NEAREST UNVISITED cell.
        unvisited = visited < 0.5                          # (N_CELLS,)
        cdist = jp.linalg.norm(self._cell_centers - agent_xy[None, :], axis=-1)
        cdist = jp.where(unvisited, cdist, jp.inf)
        fidx = jp.argmin(cdist)
        fvec = self._cell_centers[fidx] - agent_xy        # (2,)
        fnorm = jp.linalg.norm(fvec) + 1e-6
        all_done = jp.all(~unvisited)
        fdir = jp.where(all_done, jp.zeros(2), fvec / fnorm)
        fdist = jp.where(all_done, 0.0, fnorm / (2.0 * arena))
        frontier = jp.concatenate([fdir, fdist[None]])     # (3,)
        covered = (visited.sum() / float(contract.N_CELLS))[None]  # (1,)

        obs = jp.concatenate([agent_feat, per, mask, frontier, covered])
        return jp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)

    def _metrics_zero(self):
        z = jp.zeros(())
        return {
            contract.METRIC_COVERAGE: z, contract.METRIC_COLLISION: z,
            contract.METRIC_REWARD_COMPONENTS[0]: z,
            contract.METRIC_REWARD_COMPONENTS[1]: z,
            contract.METRIC_REWARD_COMPONENTS[2]: z,
            "completed": z,
            "mean_freshness": z,
        }

    # ---- reset ---- #
    def reset(self, rng: jax.Array) -> Any:
        cfg = self._config
        arena = float(cfg.arena_half)
        spawn = arena - float(cfg.spawn_margin)
        M = contract.MAX_OBSTACLES

        rng, ka, ko = jax.random.split(rng, 3)
        agent_xy0 = jax.random.uniform(ka, (2,), minval=-spawn, maxval=spawn)

        ks = jax.random.split(ko, 6)
        obst_centre = jax.random.uniform(ks[0], (M, 2), minval=-1.6, maxval=1.6)
        theta = jax.random.uniform(ks[1], (M,), minval=0.0, maxval=2.0 * jp.pi)
        obst_axis = jp.stack([jp.cos(theta), jp.sin(theta)], axis=-1)
        # ~frac_moving obstacles patrol; the rest are static (amp 0).
        moving = (jax.random.uniform(ks[2], (M,)) < float(cfg.obstacle.frac_moving))
        amp = jp.where(moving,
                       jax.random.uniform(ks[3], (M,), minval=0.5,
                                          maxval=float(cfg.obstacle.max_amplitude)),
                       0.0)
        freq = jax.random.uniform(ks[4], (M,), minval=float(cfg.obstacle.min_frequency),
                                  maxval=float(cfg.obstacle.max_frequency))
        phase = jax.random.uniform(ks[5], (M,), minval=0.0, maxval=2.0 * jp.pi)
        height = jp.full((M,), float(cfg.obstacle.height))

        info: Dict[str, Any] = {
            "rng": rng, "step": jp.zeros((), jp.int32), "time": jp.zeros(()),
            "truncation": jp.zeros(()), "time_out": jp.zeros(()),
            "obst_centre": obst_centre, "obst_axis": obst_axis, "obst_amp": amp,
            "obst_freq": freq, "obst_phase": phase, "obst_height": height,
        }

        data = mjx.make_data(self._mjx_model)
        qpos = data.qpos.at[self._qx].set(agent_xy0[0]).at[self._qy].set(agent_xy0[1])
        obst_pos0, obst_vel0 = self._obstacle_state(info, info["time"])
        mocap_pos = data.mocap_pos.at[:M].set(obst_pos0)
        data = data.replace(qpos=qpos, qvel=jp.zeros_like(data.qvel),
                            ctrl=jp.zeros_like(data.ctrl), mocap_pos=mocap_pos)
        data = mjx.forward(self._mjx_model, data)

        # Visited grid starts EMPTY; cells are marked as the agent enters them in
        # step(). The per-step coverage DELTA metric then sums (Brax aggregates
        # episode metrics by summation) to the final coverage fraction.
        visited = jp.zeros((contract.N_CELLS,))
        info["visited"] = visited
        info["freshness"] = jp.zeros((contract.N_CELLS,))  # Variant B; inert unless patrol/k_fresh
        info["covered"] = jp.zeros(())

        agent_xy, agent_vel = self._agent_pose(data)
        obs = self._build_obs(agent_xy, agent_vel, obst_pos0, obst_vel0, visited)
        metrics = self._metrics_zero()  # coverage delta + collision start at 0
        return mjx_env.State(data, obs, jp.zeros(()), jp.zeros(()), metrics, info)

    # ---- step ---- #
    def step(self, state: Any, action: jax.Array) -> Any:
        cfg = self._config
        info = dict(state.info)
        data = state.data
        step_idx = info["step"] + 1
        time = info["time"] + self.dt

        action = jp.clip(action, -contract.ACTION_LIMIT, contract.ACTION_LIMIT)
        ctrl = (action * float(cfg.agent_max_speed))
        ctrl = jp.zeros_like(data.ctrl).at[0].set(ctrl[0]).at[1].set(ctrl[1])

        # scripted obstacles at the NEW time; write mocap BEFORE stepping
        obst_pos, obst_vel = self._obstacle_state(info, time)
        M = contract.MAX_OBSTACLES
        data = data.replace(ctrl=ctrl, mocap_pos=data.mocap_pos.at[:M].set(obst_pos))
        data = jax.lax.fori_loop(0, self._n_substeps,
                                 lambda _, d: mjx.step(self._mjx_model, d), data)

        agent_xy, agent_vel = self._agent_pose(data)

        # coverage update
        prev_visited = info["visited"]
        cell = self._cell_index(agent_xy)
        visited = prev_visited.at[cell].set(1.0)
        new_cells = visited.sum() - prev_visited.sum()

        # Variant B: renewable freshness (inert when patrol=False / k_fresh=0).
        decay = float(cfg.reward.freshness_decay) if bool(cfg.reward.patrol) else 1.0
        prev_fresh = info["freshness"] * decay
        gain = 1.0 - prev_fresh[cell]
        freshness = prev_fresh.at[cell].set(1.0)
        info["freshness"] = freshness
        r_fresh = float(cfg.reward.k_fresh) * gain

        # collision penalty (geometric, vs obstacles; NON-terminal)
        rel = obst_pos[:, :2] - agent_xy[None, :]
        d = jp.linalg.norm(rel, axis=-1)
        thresh = float(cfg.agent_radius) + float(cfg.obstacle_radius) + float(cfg.collision_margin)
        collision = jp.any(d < thresh).astype(jp.float32)

        k_cov = float(cfg.reward.k_cov)
        k_coll = float(cfg.reward.k_coll)
        k_time = float(cfg.reward.k_time)
        r_cover = k_cov * new_cells
        r_coll = -k_coll * collision
        _cap = float(cfg.reward.collision_penalty_cap)
        r_coll = jp.where(_cap > 0.0, jp.maximum(r_coll, -_cap), r_coll)
        r_time = -k_time

        # Variant A: terminate on full coverage + completion bonus.
        timeout = (step_idx >= int(cfg.episode_length)).astype(jp.float32)
        fully_covered = (visited.sum() >= float(contract.N_CELLS)).astype(jp.float32)
        success = fully_covered * (1.0 if bool(cfg.reward.terminate_on_full_coverage) else 0.0)
        remaining_frac = jp.clip(
            (float(cfg.episode_length) - step_idx.astype(jp.float32)) / float(cfg.episode_length),
            0.0, 1.0)
        r_complete = float(cfg.reward.k_complete) * success * remaining_frac

        reward = r_cover + r_coll + r_time + r_complete + r_fresh

        done = jp.maximum(timeout, success)
        time_out = timeout * (1.0 - success)
        info["truncation"] = time_out
        info["time_out"] = time_out
        info["step"] = step_idx
        info["time"] = time
        info["visited"] = visited
        info["covered"] = visited.sum()
        rng, _ = jax.random.split(info["rng"])
        info["rng"] = rng

        obs = self._build_obs(agent_xy, agent_vel, obst_pos, obst_vel, visited)

        # Emit per-step DELTAS so Brax's episode-sum aggregation yields fractions:
        #   coverage  -> sum of (new_cells / N_CELLS) = final coverage fraction (0..1)
        #   collision -> sum of (collision / episode_length) = fraction of steps in contact
        metrics = dict(state.metrics)
        metrics[contract.METRIC_COVERAGE] = new_cells / float(contract.N_CELLS)
        metrics[contract.METRIC_COLLISION] = collision / float(cfg.episode_length)
        metrics[contract.METRIC_REWARD_COMPONENTS[0]] = r_cover
        metrics[contract.METRIC_REWARD_COMPONENTS[1]] = r_coll
        metrics[contract.METRIC_REWARD_COMPONENTS[2]] = jp.asarray(r_time)
        metrics["completed"] = success
        metrics["mean_freshness"] = freshness.mean() / float(cfg.episode_length)

        return mjx_env.State(data, obs, reward, done, metrics, info)
