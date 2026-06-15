# Praxis — Neural Network & RL Agent

The policy that maps the environment observation to an action and **learns to navigate** via reinforcement learning.

## Current CSN-PPO Target

Current CSN-PPO implementation target:
- 28-D coverage/exploration task
- no goal-reaching reward
- collisions are non-terminal by default
- metric is coverage retention, not success-rate retention

This is not the original 27-D navigation contract. The active CSN-PPO path uses the 28-D coverage observation contract, coverage rewards, coverage probes, coverage criticality, and coverage/collision sentinels. The older goal-reaching notes below are MVP navigation context, not the CSN-PPO optimization target.

Operational rule: sentinel required for long runs.

P0-P8 hardening now exists: sentinel mandatory for long runs, per-cluster mosaic teachers, adaptive guard pressure, curriculum mixture, validation bank, stratified memory, `--long-run` preset, and guard-KL conditioning.

To use CSN-PPO for 27-D navigation:
1. replace coverage criticality with nav criticality,
2. replace coverage probes with nav probes,
3. use success/collision sentinels instead of coverage/collision sentinels,
4. restore goal-relative observation contract,
5. label synthetic probes using goal-directed analytic teacher.

---

## Recommended approach

- **Algorithm: PPO** — on-policy, robust, the standard for navigation.
- **Trainer: Brax PPO** (`brax.training.agents.ppo`), JAX-native, the standard MuJoCo Playground trainer. Brax *physics* is deprecated; Brax *training* is maintained (Brax 0.14.x). Phase 1 (Isaac Lab) switches to **RSL-RL / rl_games** (PyTorch) — a full re-wire, not a drop-in.
- **Network: a small MLP actor-critic** — the observation is low-dim privileged state, so **no CNN is needed yet.** Brax uses **separate** policy/value MLPs (not a shared trunk).

### Wiring (current API — give this to the coding agent verbatim)
```python
import functools, jax
from brax.training.agents.ppo import train as ppo, networks as ppo_networks
from mujoco_playground import wrapper   # wrap_for_brax_training: MjxEnv -> Brax envs.Env

# Network sizing MUST be explicit. Brax DEFAULTS are policy=(32,)*4, value=(256,)*5 — NOT what we want.
network_factory = functools.partial(
    ppo_networks.make_ppo_networks,
    policy_hidden_layer_sizes=(256, 256, 256),
    value_hidden_layer_sizes=(256, 256, 256),
)

make_inference_fn, params, metrics = ppo.train(   # NOTE: returns a 3-TUPLE
    environment=wrapper.wrap_for_brax_training(env),
    network_factory=network_factory,
    num_timesteps=int(2e7),      # first learnable run: 2e7–5e7
    num_envs=2048,               # start here; scale up later (watch for OOM)
    episode_length=1000,
    unroll_length=20,
    batch_size=256,
    num_minibatches=32,          # (num_envs*unroll_length) MUST be divisible by (batch_size*num_minibatches)
    num_updates_per_batch=4,
    learning_rate=3e-4,
    entropy_cost=1e-2,
    discounting=0.97,
    reward_scaling=1.0,
    normalize_observations=True, # Brax default is False — MUST enable for low-dim mixed-scale nav obs
    num_evals=10,
    log_training_metrics=True,
    save_checkpoint_path="ckpts/",  # Orbax
    seed=0,
)
```

## On PufferLib — honest, verified

PufferLib is an excellent, **MIT-licensed, actively maintained** RL trainer (PPO + Muon + V-trace + prioritized replay). **But it is not a physics simulator, and its speed advantage is built for fast CPU environments written in C.** For a GPU-physics sim (MJX), envs are already GPU-vectorized; routing them through PufferLib's CPU-oriented vectorizer adds host↔device round-trips and buys little. **→ PufferLib is not the MVP trainer here.** Use Brax PPO. Keep PufferLib in mind only if we later build a fast *custom CPU* env where its throughput (1M+ steps/s) is the point.

## Observation / action contract

Must match [`../sim/README.md`](../sim/README.md) and `praxis/contract.py` **exactly** (fixed-K=4, sorted, masked obs; `[vx,vy]∈[-1,1]` action). The agent code depends only on this contract, never on simulator internals.

## Reward shaping + termination semantics (read carefully — top silent-failure source)

```
reward =  + k1 * (prev_dist_to_goal - dist_to_goal)   # progress toward goal
          - k2 * collision                            # and TERMINATE the episode
          - k3                                         # small per-step time penalty
          + k4 * reached_goal                          # success bonus (and terminate)
```

- **Collision and success are TRUE terminations** (`done=1`); **timeout is a TRUNCATION.** Brax bootstraps value at the unroll boundary — a terminal collision must **not** be bootstrapped like a timeout, or the time penalty makes deliberate early collision (suicide) optimal. Bootstrap on truncation, terminate on collision/success (`bootstrap_on_timeout`).
- Keep `k3` **small** relative to per-step `k1*progress`, so cumulative reward along a successful path clearly exceeds total time penalty.
- `normalize_observations=True` (Brax default is False) — required for low-dim mixed-scale features.
- Scale actions to the control range inside `step`.
- **Emit `success`, `collision`, and reward components into `state.metrics` every step** — Brax logs `eval/episode_reward` by default but **not** success/collision unless they're in `metrics` (they surface as `eval/episode_<name>`). Without this you can't evaluate "is it learning."

## Training & evaluation

- Train across thousands of parallel envs; log **episode reward, success rate, collision rate** via a `progress_fn`.
- Checkpoint policies (Orbax, `save_checkpoint_path`).
- Run an **eval/rollout that records a video** (recipe in `sim/README.md`) — from the checkpoint, deterministic inference, `MUJOCO_GL=egl`.
- **MVP success metric:** success rate climbing well above chance with collision rate trending down, and a rollout where the agent visibly reaches goals *around* moving obstacles.

## Path beyond the MVP (noted, not built now)

1. **Perception swap:** privileged state → a learned `pixels/depth → occupancy` front-end. This **changes the observation contract and the network (MLP → encoder/CNN) and requires retraining** — it preserves the action/reward/task contract and the harness, but not the obs contract (that's the geometric bottleneck). The trained MLP does **not** transfer.
2. **World-model swap:** model-free PPO → a latent world model (**DreamerV3 / MuZero-style**) that imagines rollouts. The env contract is unchanged; only the learner changes — this one *is* a clean swap.

> Implementation note: this design doc is the spec for the coding session. Hand the implementation to a coding agent (e.g. Codex) against these contracts, and review its output against the MVP success metric and the acceptance gates in `NEXT_SESSION_PROMPT.md`.
