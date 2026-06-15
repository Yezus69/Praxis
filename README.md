# Praxis

> An MVP to train — and watch — a reinforcement-learning agent learn to navigate a 3D physics simulation with dynamic obstacles.
>
> *(Codename. Rename freely.)*

Two components, joined by a shared task definition:

- **[`sim/`](sim/README.md)** — *the world.* A real-physics 3D environment with a controllable agent body, a goal, static clutter, and scripted moving obstacles.
- **[`agent/`](agent/README.md)** — *the brain.* A neural-network policy trained with reinforcement learning to reach the goal while avoiding obstacles.

## Current CSN-PPO Target

Current CSN-PPO implementation target:
- 28-D coverage/exploration task
- no goal-reaching reward
- collisions are non-terminal by default
- metric is coverage retention, not success-rate retention

This is not the original 27-D navigation contract. The active CSN-PPO contract is the coverage/exploration task in `praxis/contract.py` and `praxis/envs/cover_env.py`. Optimize coverage retention, collision discipline, and guard/sentinel stability, not goal success.

Operational rule: sentinel required for long runs.

P0-P8 hardening now exists: sentinel mandatory for long runs, per-cluster mosaic teachers, adaptive guard pressure, curriculum mixture, validation bank, stratified memory, `--long-run` preset, and guard-KL conditioning.

To use CSN-PPO for 27-D navigation:
1. replace coverage criticality with nav criticality,
2. replace coverage probes with nav probes,
3. use success/collision sentinels instead of coverage/collision sentinels,
4. restore goal-relative observation contract,
5. label synthetic probes using goal-directed analytic teacher.

---

## ⚠️ GPU prerequisite — read this first

Phase 0 runs **JAX on GPU**, and **JAX has no native-Windows CUDA wheel** (prebuilt for Linux x86_64/aarch64 only). On Windows 10, run everything inside **WSL2 (Ubuntu)** or a Linux/CUDA container. **Verify the GPU is visible before any other work:**

```bash
python -c "import jax; print(jax.default_backend())"   # MUST print: gpu
```

CPU JAX runs the smoke test fine but is ~100–1000× too slow to "watch it learn." This single check is the most likely day-one blocker — do it first.

## The MVP, precisely

**Success = the agent reliably reaches a goal in a physics-simulated scene while avoiding static and simple moving obstacles — with a visible learning curve (rising reward + success rate, falling collisions) and a watchable rollout `.mp4`.**

Realistic end-of-day result: **"the loop runs and the agent is clearly learning,"** not a polished navigator. Prove the loop learns; everything else is later.

## In / out — today

**In:** real rigid-body physics · goal-reaching · static obstacles (boxes, walls, wire-like clutter) · a few *scripted* moving obstacles · PPO training · reward/success/collision curves + a rollout video.

**Out (deferred on purpose):** learning from raw pixels — the agent gets **privileged low-dim state** for now · photorealistic rendering · realistic crowd/NPC *behavior* · anything beyond the simulator.

> Deferring perception and NPC realism is a deliberate scoping decision — it is what makes a working agent achievable quickly.

## Architecture

```
            ┌──────────────────────────┐    State (obs, reward,    ┌─────────────────────────┐
            │          sim/            │     done, metrics)        │         agent/          │
            │  physics + scene + task  │ ────────────────────────► │  policy/value network   │
            │  (MuJoCo Playground/MJX) │                           │      (PPO learner)      │
            │   functional JAX State   │ ◄──────────────────────── │      (Brax PPO)         │
            └──────────────────────────┘        action             └─────────────────────────┘
                         ▲                                                     ▲
                         └──── shared TASK contract: obs/action/reward ────────┘
                              semantics + scene definition
                              (the concrete env API differs per backend)
```

- **Observation (MVP):** privileged low-dim state — goal-relative vector, agent velocity, and the **K=4 nearest obstacles** (sorted by distance, fixed-K, zero-padded + active mask). **No camera yet.**
- **Action (MVP):** continuous `[vx, vy]` (holonomic), applied through **velocity actuators** — never by setting position directly.
- **Reward:** progress-to-goal − collision penalty − small time penalty + success bonus.

### What is (and isn't) shared between sim and agent

The two halves share a **conceptual task contract** — the *semantics* of observation, action, reward, and termination, plus the scene/task definition. **The concrete environment API does NOT port across backends:**

- **Phase 0 (Brax/MJX): a functional JAX `State` API** — `state = env.step(state, action)`, a single `State` dataclass in and out, batched with `jax.vmap`. **Not** the imperative Gymnasium `obs, reward, terminated, truncated, info = env.step(action)` 5-tuple, and **not** an int seed (`reset(rng)` takes a `jax.random.PRNGKey`).
- **Phase 1 (Isaac Lab): a PyTorch tensorized VecEnv** (batched tensors, built-in autoreset) — also *not* classic Gymnasium.

So "swapping simulators" means **re-implementing the same task spec against a new backend API**, not reusing one literal `reset`/`step`. Design for that honestly.

## Phased plan

| Phase | Simulator | Trainer | Purpose |
|---|---|---|---|
| **0 — now** | **MuJoCo Playground / MJX** (real physics, GPU-parallel, `pip install playground`, Apache-2.0) | **Brax PPO** (`brax.training.agents.ppo`; Brax ≥0.12, latest 0.14.x — physics deprecated, trainer maintained) | Prove the loop learns *today*. Scripted moving obstacles. |
| **1 — later** | **NVIDIA Isaac Lab + Isaac Sim Replicator Agent (IRA)** (PyTorch VecEnv; per-library wrappers) | RSL-RL / rl_games | Animated NPC crowds, PhysX clutter, procedural scenes. |

**Honest cost of the Phase-0→1 swap:** the env API *and* the trainer both change — functional JAX `State` vs PyTorch VecEnv tensors, single `done` vs terminated/truncated, MJCF vs USD scenes. **Only the task/obs/reward *design* carries over; a trained Brax/JAX policy does not transfer to Isaac.** Phase 1 is a full env re-implementation, not a drop-in.

**Time-to-learning:** minutes on a modern GPU (comparable Playground locomotion tasks train in ~6 min on an A100). Budget **30–60s for first-run JIT compile** — it is not a hang.

## Layout

```
Praxis/
├── README.md                 ← you are here — overview + what the two halves share
├── sim/README.md             ← the environment design
├── agent/README.md           ← the NN + RL design
└── NEXT_SESSION_PROMPT.md     ← paste into the next (Opus + Codex) build session
```

## Beyond the MVP (noted, not built now)

1. **Perception swap** (privileged state → `pixels/depth → occupancy`): this **deliberately changes the observation contract and the network** (MLP → encoder/CNN) and **requires retraining**. It preserves the *action/reward/task* contract and the train-rollout harness, but **not** the obs contract — that change *is* the geometric bottleneck. The trained MLP does not transfer.
2. **World-model swap** (DreamerV3 / MuZero-style): keeps the env contract fixed and changes **only the learner** — for sample-efficiency and planning. This sub-claim is accurate.

## Tooling rationale

Verified, current (2026), license-checked — see [`sim/README.md`](sim/README.md) and [`agent/README.md`](agent/README.md), including the honest note on why **PufferLib** (an excellent MIT RL library) is **not** the trainer here.
