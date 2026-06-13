# Praxis — Simulation Environment

A **real-physics** (not toy) 3D world where an agent body must navigate to a goal among static and dynamic obstacles, exposed through the **Brax/MJX functional environment API**.

---

## Recommended stack

### Phase 0 — today: MuJoCo Playground / MJX
- **Real physics:** MuJoCo contact dynamics (gold standard for rigid bodies), GPU-parallel via **MJX** (JAX/XLA) — thousands of envs on one GPU through `jax.vmap`.
- **Why it wins for the MVP:** `pip install playground` (no Omniverse install to burn the day), Apache-2.0, pairs natively with Brax PPO.
- **Honest caveats:**
  - Playground ships **dm_control_suite / locomotion / manipulation** tasks — **no navigation task.** We author the nav scene ourselves (MJCF + a custom env class). That is the real build work.
  - MJX has **no built-in pedestrians.** For the MVP, moving obstacles are **scripted mocap bodies** (see below). Learned/animated NPCs are Phase 1.

### Install (Linux or WSL2 — JAX GPU is not available on native Windows)
```bash
# Linux/WSL2, CUDA-12-capable NVIDIA driver (>=525), Python >=3.11
pip install playground                 # PyPI name 'playground'; import as `mujoco_playground`
pip install -U "jax[cuda12]"           # CUDA 12 wheels — do NOT install plain `jax` (CPU-only)
pip install mediapy "imageio[ffmpeg]" ml_collections pytest
python -c "import jax; print(jax.default_backend())"   # MUST print: gpu
export MUJOCO_GL=egl                    # headless offscreen render (also need system ffmpeg)
export JAX_DEFAULT_MATMUL_PRECISION=highest   # Ampere/Ada (RTX 30/40/50) reproducibility
```

### Phase 1 — fidelity: NVIDIA Isaac Sim + Isaac Lab
NPC crowds via **Isaac Sim Replicator Agent (IRA)** (NavMesh walking, reactive triggers), PhysX dynamic clutter, USD procedural scenes. PyTorch tensorized VecEnv (per-library wrappers `isaaclab_rl.{rsl_rl,rl_games,skrl}`). Isaac Lab is BSD-3; Isaac Sim under the Omniverse EULA (review before distribution).

### Avoid (verified deprecated)
Habitat (maintenance-only past v0.3.4) · Isaac Gym Preview / IsaacGymEnvs (deprecated; Isaac Lab is the successor).

---

## The task contract (semantic — backend APIs differ)

The agent depends on the *semantics* of obs/action/reward, not on one literal API. **Phase 0 is the Brax/MJX functional `State` API — NOT a Gymnasium 5-tuple:**

```python
# Custom env subclasses mujoco_playground._src.mjx_env.MjxEnv  (from mujoco_playground import mjx_env)
state = env.reset(rng)            # rng = jax.random.PRNGKey(...), NOT an int seed
state = env.step(state, action)   # returns ONE State dataclass
# state.data    : mjx.Data (the physics state — a JAX pytree; update with .replace(...), never mutate)
# state.obs     : jax.Array (or dict with 'state' / 'privileged_state')
# state.reward  : jax.Array
# state.done    : jax.Array, SINGLE float 0/1 — there is NO terminated/truncated split
# state.metrics : dict (put success / collision / reward components here — see agent/README.md)
# state.info    : dict (carry prev_dist, goal, step count, rng, info['truncation'], ...)
```

obs/reward/done are computed **inside** `reset`/`step` and packed into the returned `State`. Batching is `jax.vmap`, not an env method. Everything inside `reset`/`step` must be **JAX-traceable**: no Python `if` on traced arrays (use `jp.where` / `jax.lax.select`), no `.item()`/numpy on tracers, no dynamic-length loops. A Gymnasium 5-tuple adapter is a Phase-1 nicety (and breaks vmap-on-GPU) — do not build it now.

## Scene spec (MVP)

- **Agent body:** a planar/free joint; `[vx, vy] ∈ [-1, 1]` applied via **velocity actuators** (`<velocity>`/`<motor>` `ctrl`) or by setting the agent's planar `qvel` via `data.replace(qvel=...)`. **Do NOT set `qpos`/position directly** — that tunnels the agent through obstacles and the collision penalty never fires.
- **Goal:** a randomized target position (a `site`).
- **Static obstacles:** walls + boxes + thin "wire/clutter" geoms.
- **Dynamic obstacles:** 2–4 **mocap bodies** (`<body mocap="true">`). Each functional step, compute scripted positions (e.g. sinusoidal patrol) and write them via `data = data.replace(mocap_pos=new_pos)` (and `mocap_quat` if rotating) **before** `mjx.step`. Mocap bodies aren't physics-integrated (no qpos) but still **collide** with the agent — the "real dynamics seed." Footgun (mujoco#2606): set mocap pose explicitly on the data you step; `make_data` doesn't reliably copy it.

## Observation space (MVP — fixed-shape, MLP-friendly)

Frozen layout (also lives in `praxis/contract.py`):
```
[ goal: dx, dy, dist, heading_err            (4) ]
[ agent vel: vx, vy, omega                    (3) ]
[ K nearest obstacles: (px, py, vx, vy) each (4K) ]   # K = 4
[ per-slot active mask                        (K) ]
```
**Use a FIXED K (=4):** sort the K nearest obstacles by distance **every step** (an MLP is not permutation-invariant — a slot must always mean the same thing), zero-pad when fewer exist, and append the active-mask bits. **Never emit a ragged/variable-length obstacle vector.** The occupancy-grid alternative is fixed-shape but implies a CNN — it is **not** interchangeable with the stated MLP; ship the fixed-K sorted+masked vector for the MVP.

## Domain randomization (MJX requires FIXED model topology)

**You cannot randomize obstacle COUNT by adding/removing geoms or loading different XMLs per env** — jit/vmap require fixed topology. Declare **max-N obstacles** in the MJCF and disable extras per-env via a boolean `active` mask (zero size / `contype=0` / move far away). Randomize only **continuous fields** (position / size / speed / friction / goal / start) via a Brax-style `randomization_fn(model, rng) -> (batched_model, in_axes)`, applied by Playground's `BraxDomainRandomizationVmapWrapper` (`get_domain_randomizer`).

## Parallelism, determinism, rendering

- **Parallel envs:** Playground's `wrap_for_brax_training` auto-resets and vmaps the env; Brax PPO's `num_envs` sets the batch.
- **Determinism:** split a `PRNGKey` per env (`jax.random.split`); same key → reproducible. Set `JAX_DEFAULT_MATMUL_PRECISION=highest` on Ampere/Ada.
- **Rollout video (the deliverable) — runs OUTSIDE the jit/training loop, from a saved checkpoint:**
  ```python
  import jax, mediapy
  jit_reset, jit_step, jit_inf = jax.jit(env.reset), jax.jit(env.step), jax.jit(inference_fn)
  rng = jax.random.PRNGKey(0)
  state = jit_reset(rng); rollout = [state]
  for _ in range(1000):
      act_rng, rng = jax.random.split(rng)
      action, _ = jit_inf(state.obs, act_rng)      # deterministic inference for a clean video
      state = jit_step(state, action); rollout.append(state)
      if state.done: break
  frames = env.render(rollout, camera="track")      # CPU/offscreen via mujoco.Renderer
  mediapy.write_video("rollout.mp4", frames, fps=int(1 / env.dt))
  ```
  `env.render` takes the list of physics **States/`.data`** (not raw obs, not the wrapped Brax state). Headless WSL/Linux needs `MUJOCO_GL=egl` + ffmpeg, or it silently returns black frames. (In-loop pixel rendering = Madrona batch renderer = Phase 1 only.)

## Deferred (Phase 1+)

Cameras / pixels (the perception gap), photorealism, learned & animated NPC behavior, real-world assets.
