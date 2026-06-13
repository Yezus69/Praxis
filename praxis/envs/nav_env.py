"""praxis/envs/nav_env.py — the Phase-0 navigation environment.

``NavEnv`` is a MuJoCo-Playground / MJX environment (functional Brax-style API,
NOT Gymnasium). It subclasses ``mujoco_playground._src.mjx_env.MjxEnv`` and
implements the FROZEN task contract in :mod:`praxis.contract`.

Everything inside :meth:`reset` / :meth:`step` is JAX-traceable: fixed shapes,
no Python ``if`` on tracers (``jp.where`` / ``jax.lax.select`` only), no
``.item()`` / numpy-on-tracers, no dynamic-length loops. The physics ``data`` is
a pytree — updated with ``.replace(...)``, never mutated.

Key design points (see CORRECTED TECHNICAL FACTS in the build brief):
  * Action = planar velocity command ``[vx, vy] in [-1, 1]`` scaled by
    ``AGENT_MAX_SPEED`` and written to ``data.ctrl`` (velocity actuators).
  * 4 scripted mocap obstacles on sinusoidal patrols; positions written to
    ``data.mocap_pos`` BEFORE stepping; obstacle velocity is the analytic
    time-derivative of the patrol.
  * Observation = fixed 27-vec, K=4 nearest ACTIVE obstacles sorted by distance.
  * Collision detection is GEOMETRIC (traceable, reliable) — not contact-list
    parsing — plus an arena-bounds check.
  * Collision / success = TRUE termination (info['truncation']=0); timeout =
    TRUNCATION (info['truncation']=1, Brax bootstraps value).
"""

from __future__ import annotations

import os
from typing import Any, Dict, Optional

import jax
import jax.numpy as jp
import mujoco
from mujoco import mjx

import ml_collections

# Playground base env + helpers. Import path per Playground convention.
# NOTE(orchestrator): both `from mujoco_playground import mjx_env` and
# `from mujoco_playground._src import mjx_env` are documented; we try the public
# one first and fall back. Verify against the installed package version.
try:  # pragma: no cover - import shim, exercised at runtime in the container
    from mujoco_playground import mjx_env
except Exception:  # pragma: no cover
    from mujoco_playground._src import mjx_env  # type: ignore

from praxis import contract


# Path to the MJCF scene authored alongside this module.
_THIS_DIR = os.path.dirname(os.path.abspath(__file__))
_XML_PATH = os.path.join(_THIS_DIR, "nav_scene.xml")

# Geometry radii used by the GEOMETRIC collision check. These mirror the geom
# sizes in nav_scene.xml. Obstacle radius can be randomized per-env (see
# randomize.py); the obs/collision use the per-env value carried in info.
_AGENT_RADIUS = 0.15
_OBSTACLE_RADIUS = 0.25

# Index of the cosmetic goal marker in data.mocap_pos (obstacles are 0..3).
_GOAL_MARKER_MOCAP_IDX = contract.MAX_OBSTACLES  # == 4

# A "far away" parking spot for inactive obstacles (well outside the arena so
# they can never collide and never become a nearest neighbour). Kept finite.
_FAR_AWAY = 1.0e3


def default_config() -> ml_collections.ConfigDict:
    """Default env config. Continuous fields here are overridable / randomizable.

    ``sim_dt * n_substeps == ctrl_dt`` is the control timestep the rollout uses
    for ``fps = int(1 / env.dt)``.
    """
    cfg = ml_collections.ConfigDict()
    cfg.sim_dt = 0.01            # MJX integration timestep
    cfg.n_substeps = 5          # physics substeps per control step -> ctrl_dt 0.05
    cfg.episode_length = contract.EPISODE_LENGTH

    # Reward weights (from the frozen contract; exposed so the trainer/eval can
    # read them, but defaults MUST equal the contract).
    cfg.reward = ml_collections.ConfigDict()
    cfg.reward.k1 = contract.DEFAULT_REWARD_WEIGHTS["k1"]
    cfg.reward.k2 = contract.DEFAULT_REWARD_WEIGHTS["k2"]
    cfg.reward.k3 = contract.DEFAULT_REWARD_WEIGHTS["k3"]
    cfg.reward.k4 = contract.DEFAULT_REWARD_WEIGHTS["k4"]

    # Task geometry.
    cfg.goal_radius = contract.GOAL_RADIUS
    cfg.arena_half = contract.ARENA_HALF
    cfg.agent_radius = _AGENT_RADIUS
    cfg.obstacle_radius = _OBSTACLE_RADIUS
    cfg.agent_max_speed = contract.AGENT_MAX_SPEED

    # Start / goal randomization. Start near one side, goal near the other,
    # with a minimum separation so the task is non-trivial but reachable.
    cfg.spawn_margin = 0.4          # keep spawns this far inside the walls
    cfg.min_start_goal_sep = 2.0    # metres

    # Obstacle patrol params (sinusoidal). These are per-obstacle defaults; the
    # randomizer perturbs centre / amplitude / freq / phase within ranges.
    cfg.obstacle = ml_collections.ConfigDict()
    cfg.obstacle.amplitude = 1.0    # metres of travel along the patrol axis
    cfg.obstacle.frequency = 0.3    # Hz (low — collisions avoidable)
    cfg.obstacle.height = 0.25      # z of the mocap obstacle centre
    # All 4 obstacles ACTIVE by default. randomize.py may set some inactive.
    cfg.n_active_obstacles = contract.MAX_OBSTACLES

    return cfg


class NavEnv(mjx_env.MjxEnv):
    """Goal-reaching navigation among scripted moving obstacles (MJX)."""

    # ------------------------------------------------------------------ #
    # construction
    # ------------------------------------------------------------------ #
    def __init__(
        self,
        config: Optional[ml_collections.ConfigDict] = None,
        config_overrides: Optional[ml_collections.ConfigDict] = None,
    ) -> None:
        cfg = config if config is not None else default_config()

        # NOTE(orchestrator): MjxEnv.__init__ signature varies slightly across
        # Playground versions (some take (config, config_overrides)). We call it
        # defensively; if the base stores config/sim_dt itself this is a no-op
        # on our side. Our own attributes below are authoritative.
        try:
            super().__init__(cfg, config_overrides)  # type: ignore[arg-type]
        except Exception:
            try:
                super().__init__(cfg)  # type: ignore[misc]
            except Exception:
                # Last resort: skip base init; we manage all needed state here.
                pass

        self._config = cfg
        if config_overrides is not None:
            self._config.update(config_overrides)

        # Build the MuJoCo model from the MJCF, then the MJX model.
        self._mj_model = mujoco.MjModel.from_xml_path(_XML_PATH)
        # Apply the configured sim timestep onto the model so dt is consistent.
        self._mj_model.opt.timestep = float(self._config.sim_dt)
        self._mjx_model = mjx.put_model(self._mj_model)

        self._n_substeps = int(self._config.n_substeps)

        # Cache static index lookups (python ints, fine outside trace).
        self._agent_body_id = mujoco.mj_name2id(
            self._mj_model, mujoco.mjtObj.mjOBJ_BODY, "agent"
        )
        # qpos/qvel addresses for the agent's three dofs (x, y, yaw). With only
        # the agent jointed, these are simply [0, 1, 2], but resolve explicitly.
        self._qadr_x = int(self._mj_model.jnt_qposadr[
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, "agent_x")
        ])
        self._qadr_y = int(self._mj_model.jnt_qposadr[
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, "agent_y")
        ])
        self._qadr_yaw = int(self._mj_model.jnt_qposadr[
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, "agent_yaw")
        ])
        self._vadr_x = int(self._mj_model.jnt_dofadr[
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, "agent_x")
        ])
        self._vadr_y = int(self._mj_model.jnt_dofadr[
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, "agent_y")
        ])
        self._vadr_yaw = int(self._mj_model.jnt_dofadr[
            mujoco.mj_name2id(self._mj_model, mujoco.mjtObj.mjOBJ_JOINT, "agent_yaw")
        ])

    # ------------------------------------------------------------------ #
    # required MjxEnv properties
    # ------------------------------------------------------------------ #
    @property
    def xml_path(self) -> str:
        return _XML_PATH

    @property
    def action_size(self) -> int:
        return contract.ACT_DIM

    @property
    def mj_model(self) -> mujoco.MjModel:
        return self._mj_model

    @property
    def mjx_model(self) -> Any:
        return self._mjx_model

    @property
    def dt(self) -> float:
        # Effective control timestep (sim_dt * n_substeps). The base may also
        # provide this; we override to be explicit for the rollout fps.
        return float(self._config.sim_dt) * float(self._n_substeps)

    # ------------------------------------------------------------------ #
    # helpers (all traceable when given tracers)
    # ------------------------------------------------------------------ #
    @staticmethod
    def _wrap_to_pi(angle: jax.Array) -> jax.Array:
        """Wrap an angle to (-pi, pi]. Traceable."""
        return (angle + jp.pi) % (2.0 * jp.pi) - jp.pi

    def _obstacle_state(
        self, info: Dict[str, Any], t: jax.Array
    ) -> tuple[jax.Array, jax.Array]:
        """Analytic patrol position + velocity for all MAX_OBSTACLES obstacles.

        Patrol (per obstacle i), a sinusoid along a per-obstacle unit axis
        ``d_i`` about a per-obstacle centre ``c_i``::

            p_i(t) = c_i + A_i * sin(2*pi*f_i*t + phi_i) * d_i
            v_i(t) = A_i * (2*pi*f_i) * cos(2*pi*f_i*t + phi_i) * d_i

        z is held constant at ``height`` (zero z-velocity). Inactive obstacles
        are parked at ``_FAR_AWAY`` with zero velocity so they never collide and
        never appear as a nearest neighbour.

        Returns:
            pos: (MAX_OBSTACLES, 3) world mocap positions.
            vel: (MAX_OBSTACLES, 2) world planar velocities (z-vel is 0).
        """
        centre = info["obst_centre"]        # (M, 2)
        axis = info["obst_axis"]            # (M, 2) unit vectors
        amp = info["obst_amp"]              # (M,)
        freq = info["obst_freq"]            # (M,)
        phase = info["obst_phase"]          # (M,)
        height = info["obst_height"]        # (M,)
        active = info["obst_active"]        # (M,) float 0/1

        omega = 2.0 * jp.pi * freq          # (M,)
        ang = omega * t + phase             # (M,)
        s = jp.sin(ang)                     # (M,)
        c = jp.cos(ang)                     # (M,)

        offset = (amp * s)[:, None] * axis            # (M, 2)
        planar_pos = centre + offset                  # (M, 2)
        planar_vel = (amp * omega * c)[:, None] * axis  # (M, 2)

        # Park inactive obstacles far away with zero velocity.
        act = active[:, None]                          # (M, 1)
        planar_pos = jp.where(act > 0.5, planar_pos, _FAR_AWAY)
        planar_vel = jp.where(act > 0.5, planar_vel, 0.0)

        pos = jp.concatenate([planar_pos, height[:, None]], axis=-1)  # (M, 3)
        return pos, planar_vel

    def _agent_pose(self, data: Any) -> tuple[jax.Array, jax.Array, jax.Array]:
        """Return agent (xy position (2,), yaw scalar, planar+omega vel (3,))."""
        ax = data.qpos[self._qadr_x]
        ay = data.qpos[self._qadr_y]
        yaw = data.qpos[self._qadr_yaw]
        vx = data.qvel[self._vadr_x]
        vy = data.qvel[self._vadr_y]
        omega = data.qvel[self._vadr_yaw]
        pos = jp.stack([ax, ay])
        vel = jp.stack([vx, vy, omega])
        return pos, yaw, vel

    def _build_obs(
        self,
        agent_xy: jax.Array,
        agent_yaw: jax.Array,
        agent_vel: jax.Array,
        goal_xy: jax.Array,
        obst_pos: jax.Array,
        obst_vel: jax.Array,
        obst_active: jax.Array,
    ) -> jax.Array:
        """Assemble the fixed 27-d observation per the contract slices."""
        # --- GOAL_SLICE [0:4] ---
        dxy = goal_xy - agent_xy                       # (2,)
        dist = jp.linalg.norm(dxy)                     # scalar
        heading_err = self._wrap_to_pi(
            jp.arctan2(dxy[1], dxy[0]) - agent_yaw
        )
        goal_feat = jp.stack([dxy[0], dxy[1], dist, heading_err])  # (4,)

        # --- VEL_SLICE [4:7] --- agent vx, vy, omega (world frame)
        vel_feat = agent_vel                           # (3,)

        # --- OBST_SLICE [7:23] --- K nearest ACTIVE obstacles, sorted by dist.
        rel_pos = obst_pos[:, :2] - agent_xy[None, :]  # (M, 2) relative position
        obst_dist = jp.linalg.norm(rel_pos, axis=-1)   # (M,)

        # Inactive slots pushed to +inf distance so top_k never selects them.
        is_active = obst_active > 0.5                  # (M,) bool
        sort_key = jp.where(is_active, obst_dist, jp.inf)  # (M,)

        # K nearest = K smallest sort_key. top_k of the negated key gives the
        # indices of the K smallest, already ordered nearest->farthest.
        _, idx = jax.lax.top_k(-sort_key, contract.K)  # idx: (K,)

        sel_rel = rel_pos[idx]                          # (K, 2)
        sel_vel = obst_vel[idx]                         # (K, 2)
        sel_active = is_active[idx].astype(jp.float32)  # (K,)

        # Zero out any selected slot that is actually inactive (when fewer than
        # K active obstacles exist, top_k still returns K indices).
        sel_mask = sel_active[:, None]                  # (K, 1)
        sel_rel = sel_rel * sel_mask
        sel_vel = sel_vel * sel_mask

        # Per slot: (px, py, vx, vy)  -> flatten to (4K,)
        per_obst = jp.concatenate([sel_rel, sel_vel], axis=-1)  # (K, 4)
        obst_feat = per_obst.reshape(-1)                # (4K,)

        # --- MASK_SLICE [23:27] --- one active bit per sorted slot.
        mask_feat = sel_active                          # (K,)

        obs = jp.concatenate([goal_feat, vel_feat, obst_feat, mask_feat])
        # Guarantee finiteness (defensive — inf only existed in sort_key).
        obs = jp.nan_to_num(obs, nan=0.0, posinf=0.0, neginf=0.0)
        return obs

    @staticmethod
    def _empty_metrics() -> Dict[str, jax.Array]:
        """Metric dict with the SAME keys used in reset and step (Brax req.)."""
        z = jp.zeros(())
        return {
            contract.METRIC_SUCCESS: z,
            contract.METRIC_COLLISION: z,
            contract.METRIC_REWARD_COMPONENTS[0]: z,  # reward_progress
            contract.METRIC_REWARD_COMPONENTS[1]: z,  # reward_collision
            contract.METRIC_REWARD_COMPONENTS[2]: z,  # reward_time
            contract.METRIC_REWARD_COMPONENTS[3]: z,  # reward_success
        }

    # ------------------------------------------------------------------ #
    # reset
    # ------------------------------------------------------------------ #
    def reset(self, rng: jax.Array) -> Any:
        cfg = self._config
        arena = float(cfg.arena_half)
        margin = float(cfg.spawn_margin)
        spawn_half = arena - margin

        # Split the rng for each sampled quantity + a stored stream for per-step
        # randomness. Threading rng deterministically is gate DoD-4.
        rng, k_start, k_goal, k_obst, k_store = jax.random.split(rng, 5)

        # --- agent start: near the -x side; y anywhere within spawn band ---
        start_x = jax.random.uniform(
            k_start, (), minval=-spawn_half, maxval=-0.5 * spawn_half
        )
        start_y = jax.random.uniform(
            jax.random.fold_in(k_start, 1), (), minval=-spawn_half, maxval=spawn_half
        )
        agent_xy0 = jp.stack([start_x, start_y])

        # --- goal: near the +x side, enforce min separation by construction ---
        goal_x = jax.random.uniform(
            k_goal, (), minval=0.5 * spawn_half, maxval=spawn_half
        )
        goal_y = jax.random.uniform(
            jax.random.fold_in(k_goal, 1), (), minval=-spawn_half, maxval=spawn_half
        )
        goal_xy = jp.stack([goal_x, goal_y])
        # If (rarely) too close, push the goal out along the separation axis.
        sep_vec = goal_xy - agent_xy0
        sep = jp.linalg.norm(sep_vec) + 1e-6
        min_sep = float(cfg.min_start_goal_sep)
        unit = sep_vec / sep
        goal_xy = jp.where(
            sep < min_sep, agent_xy0 + unit * min_sep, goal_xy
        )
        # Keep goal inside the spawn band after the push.
        goal_xy = jp.clip(goal_xy, -spawn_half, spawn_half)

        # --- obstacle patrol params ---
        M = contract.MAX_OBSTACLES
        ko = jax.random.split(k_obst, 6)
        # Centres spread across the middle band of the arena.
        obst_centre = jax.random.uniform(
            ko[0], (M, 2), minval=-1.5, maxval=1.5
        )
        # Patrol axis: random unit direction per obstacle.
        theta = jax.random.uniform(ko[1], (M,), minval=0.0, maxval=2.0 * jp.pi)
        obst_axis = jp.stack([jp.cos(theta), jp.sin(theta)], axis=-1)  # (M,2)
        obst_amp = jax.random.uniform(
            ko[2], (M,),
            minval=0.5 * float(cfg.obstacle.amplitude),
            maxval=1.0 * float(cfg.obstacle.amplitude),
        )
        obst_freq = jax.random.uniform(
            ko[3], (M,),
            minval=0.5 * float(cfg.obstacle.frequency),
            maxval=1.0 * float(cfg.obstacle.frequency),
        )
        obst_phase = jax.random.uniform(
            ko[4], (M,), minval=0.0, maxval=2.0 * jp.pi
        )
        obst_height = jp.full((M,), float(cfg.obstacle.height))
        # Active mask: first n_active_obstacles active. n_active is a python int
        # from config (static) so this comparison is fine outside the trace.
        n_active = int(cfg.n_active_obstacles)
        obst_active = (jp.arange(M) < n_active).astype(jp.float32)
        obst_radius = jp.full((M,), float(cfg.obstacle_radius))

        # --- build the physics data at the sampled start pose ---
        data = mjx_env.make_data(self._mjx_model) if hasattr(mjx_env, "make_data") \
            else mjx.make_data(self._mjx_model)
        # NOTE(orchestrator): some Playground versions expose make_data; mjx
        # always does. Fall back used above.

        qpos = data.qpos
        qpos = qpos.at[self._qadr_x].set(agent_xy0[0])
        qpos = qpos.at[self._qadr_y].set(agent_xy0[1])
        qpos = qpos.at[self._qadr_yaw].set(0.0)
        qvel = jp.zeros_like(data.qvel)

        info: Dict[str, Any] = {
            "rng": k_store,
            "step": jp.zeros((), dtype=jp.int32),
            "time": jp.zeros(()),
            "goal": goal_xy,
            "prev_dist": jp.linalg.norm(goal_xy - agent_xy0),
            "truncation": jp.zeros(()),
            # Brax PPO (bootstrap_on_timeout=True) reads info['time_out']: 1.0 when the
            # episode ends due to the time limit (so value is bootstrapped, not
            # terminated). Same semantics as our truncation. Required as an extra_field.
            "time_out": jp.zeros(()),
            # obstacle patrol params (continuous; randomizable via model too)
            "obst_centre": obst_centre,
            "obst_axis": obst_axis,
            "obst_amp": obst_amp,
            "obst_freq": obst_freq,
            "obst_phase": obst_phase,
            "obst_height": obst_height,
            "obst_active": obst_active,
            "obst_radius": obst_radius,
        }

        # Place obstacles + goal marker at t=0 via mocap_pos, then forward.
        obst_pos0, obst_vel0 = self._obstacle_state(info, info["time"])
        mocap_pos = data.mocap_pos
        mocap_pos = mocap_pos.at[:M].set(obst_pos0)
        goal_marker_pos = jp.array(
            [goal_xy[0], goal_xy[1], 0.05]
        )
        mocap_pos = mocap_pos.at[_GOAL_MARKER_MOCAP_IDX].set(goal_marker_pos)

        data = data.replace(qpos=qpos, qvel=qvel, ctrl=jp.zeros_like(data.ctrl),
                            mocap_pos=mocap_pos)
        # Forward to populate derived quantities (xpos, contacts, etc.).
        data = mjx.forward(self._mjx_model, data)

        agent_xy, agent_yaw, agent_vel = self._agent_pose(data)
        obs = self._build_obs(
            agent_xy, agent_yaw, agent_vel, goal_xy,
            obst_pos0, obst_vel0, obst_active,
        )

        metrics = self._empty_metrics()
        reward = jp.zeros(())
        done = jp.zeros(())

        return mjx_env.State(data, obs, reward, done, metrics, info)

    # ------------------------------------------------------------------ #
    # step
    # ------------------------------------------------------------------ #
    def step(self, state: Any, action: jax.Array) -> Any:
        cfg = self._config
        info = dict(state.info)  # shallow copy; we return a fresh dict
        data = state.data

        # --- advance time (before computing this step's obstacle state) ---
        step_idx = info["step"] + 1
        time = info["time"] + self.dt

        # --- map policy action [-1,1] -> velocity ctrl, write to data.ctrl ---
        action = jp.clip(action, -contract.ACTION_LIMIT, contract.ACTION_LIMIT)
        ctrl_target = action * float(cfg.agent_max_speed)  # (2,)
        ctrl = jp.zeros_like(data.ctrl)
        ctrl = ctrl.at[0].set(ctrl_target[0])  # vel_x actuator
        ctrl = ctrl.at[1].set(ctrl_target[1])  # vel_y actuator

        # --- scripted obstacle positions at the NEW time; write mocap BEFORE
        #     stepping (mujoco#2606: set mocap pose on the data you step) ---
        obst_pos, obst_vel = self._obstacle_state(info, time)
        M = contract.MAX_OBSTACLES
        mocap_pos = data.mocap_pos
        mocap_pos = mocap_pos.at[:M].set(obst_pos)
        # Goal marker stays put (info['goal'] is fixed within an episode).
        data = data.replace(ctrl=ctrl, mocap_pos=mocap_pos)

        # --- physics step (n_substeps fixed python int -> python loop OK) ---
        data = self._physics_step(data)

        # --- read new agent pose ---
        agent_xy, agent_yaw, agent_vel = self._agent_pose(data)
        goal_xy = info["goal"]
        obst_active = info["obst_active"]
        obst_radius = info["obst_radius"]

        # --- distances ---
        dxy = goal_xy - agent_xy
        dist = jp.linalg.norm(dxy)
        prev_dist = info["prev_dist"]

        # --- GEOMETRIC collision detection (traceable) ---
        rel = obst_pos[:, :2] - agent_xy[None, :]      # (M, 2)
        obst_d = jp.linalg.norm(rel, axis=-1)          # (M,)
        contact_thresh = float(cfg.agent_radius) + obst_radius  # (M,)
        hit_obstacle = jp.any(
            (obst_active > 0.5) & (obst_d < contact_thresh)
        )
        arena = float(cfg.arena_half)
        out_of_bounds = jp.any(jp.abs(agent_xy) > arena)
        collision = (hit_obstacle | out_of_bounds).astype(jp.float32)

        # --- success / timeout / done ---
        success = (dist < float(cfg.goal_radius)).astype(jp.float32)
        timeout = (step_idx >= int(cfg.episode_length)).astype(jp.float32)

        terminal = jp.maximum(collision, success)            # true termination
        # done on terminal OR timeout.
        done = jp.clip(terminal + timeout, 0.0, 1.0)
        # truncation = 1 ONLY when timeout fires WITHOUT a true termination.
        info["truncation"] = ((timeout > 0.5) & (terminal < 0.5)).astype(jp.float32)
        # Brax PPO's bootstrap_on_timeout reads info['time_out'] (same semantics).
        info["time_out"] = info["truncation"]

        # --- reward ---
        k1 = float(cfg.reward.k1)
        k2 = float(cfg.reward.k2)
        k3 = float(cfg.reward.k3)
        k4 = float(cfg.reward.k4)
        r_progress = k1 * (prev_dist - dist)
        r_collision = -k2 * collision
        r_time = -k3
        r_success = k4 * success
        reward = r_progress + r_collision + r_time + r_success

        # --- observation (uses obstacle state at this step's time) ---
        obs = self._build_obs(
            agent_xy, agent_yaw, agent_vel, goal_xy,
            obst_pos, obst_vel, obst_active,
        )

        # --- update info carried across steps ---
        info["step"] = step_idx
        info["time"] = time
        info["prev_dist"] = dist
        # Re-split the stored rng so any future per-step randomness is fresh and
        # deterministic from the reset seed (gate DoD-4).
        rng, _sub = jax.random.split(info["rng"])
        info["rng"] = rng

        # --- metrics EVERY step (Brax surfaces eval/episode_<name>) ---
        # PRESERVE keys already in state.metrics, then update ours. Brax's EvalWrapper
        # injects an extra metrics['reward'] key during evaluation; if step returned a
        # fresh dict with only our keys, the EpisodeWrapper's lax.scan carry would
        # mismatch (7 keys in vs 6 out). Copying keeps the pytree structure stable.
        metrics = dict(state.metrics)
        metrics[contract.METRIC_SUCCESS] = success
        metrics[contract.METRIC_COLLISION] = collision
        metrics[contract.METRIC_REWARD_COMPONENTS[0]] = r_progress
        metrics[contract.METRIC_REWARD_COMPONENTS[1]] = r_collision
        metrics[contract.METRIC_REWARD_COMPONENTS[2]] = jp.asarray(r_time)
        metrics[contract.METRIC_REWARD_COMPONENTS[3]] = r_success

        return mjx_env.State(data, obs, reward, done, metrics, info)

    # ------------------------------------------------------------------ #
    # physics stepping
    # ------------------------------------------------------------------ #
    def _physics_step(self, data: Any) -> Any:
        """Advance physics by ``n_substeps`` integration steps.

        Prefer the Playground helper if available; otherwise a fixed python
        for-loop over ``mjx.step`` (n_substeps is a static python int, so the
        loop unrolls cleanly under jit).
        """
        # NOTE(orchestrator): mjx_env.step(model, data, action_or_ctrl, n_substeps)
        # exists in recent Playground. Its action handling differs (some apply
        # ctrl internally). We already wrote data.ctrl ourselves, so we use the
        # plain mjx.step loop to avoid double-applying control. Verify the
        # helper's semantics if you switch to it.
        def body(_, d):
            return mjx.step(self._mjx_model, d)

        return jax.lax.fori_loop(0, self._n_substeps, body, data)
