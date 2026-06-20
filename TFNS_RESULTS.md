# TFNS Results — Task-Free Synaptic Null-Space Continual PPO on Atari

Empirical validation of the architecture in `README_TASK_FREE_CONTINUAL_ATARI.md`.
Branch: `task-free-ns`. Hardware: 2× RTX 4090 (WSL `praxis`, JAX 0.9.2 + envpool).

## Headline result: the protection prevents catastrophic forgetting

Sequential curriculum **SpaceInvaders → BeamRider** (seed 0, 5M env-steps/game, 128 envs,
4 epochs × 4 minibatches, entropy-annealed). Same architecture, optimizer, and budget for both
the protected (TFNS) run and the plain-PPO baseline. Evaluation: 30 completed episodes, true
(unclipped) returns, zero recurrent state at reset, **no game id, no episodic-memory action read**.

**SpaceInvaders score after learning each game:**

| | after SpaceInvaders | after BeamRider | retention |
|---|---|---|---|
| **TFNS (protected)** — stochastic | 449.4 | **379.1** | **0.78** |
| **TFNS (protected)** — greedy | 325.0 | **329.0** | **~1.00** |
| Plain PPO — stochastic | 386.4 | 146.1 | **0.07** |
| Plain PPO — greedy | 450.0 | 237.6 | 0.34 |
| random | 126.7 | | |

- **TFNS retains 78% (stochastic) / ~100% (greedy) of SpaceInvaders** after learning a second game.
- **Plain PPO catastrophically forgets**: down to 7% (stochastic) / 34% (greedy).
- Normalized forgetting: TFNS 0.22 vs plain ~0.93. Retention AUC 0.89.
- Consolidation of SpaceInvaders was **accepted** (closed-loop gate retention 1.07), building **6
  protected sentinel clusters**; BeamRider then learned **under** that protection (it learns in the
  complementary null-space subspace while SpaceInvaders activations are preserved).

This is the spec's core claim demonstrated on real Atari: the live network alone (no id, no memory
read) still plays the old game, because committed optimizer deltas were projected out of SpaceInvaders'
protected activation subspaces.

## Single-task references (matched budget, no protection)

| game | random | single-task (S_single) |
|---|---|---|
| SpaceInvaders | 126.7 | 359.5 |
| BeamRider | 259.6 | 343.9 |

Both games learn well above random under the recurrent agent.

## Architecture & mechanism validation (all built, codex-implemented, Claude-verified)

- **75+ unit/property tests pass** covering every §23 invariant: linear/bias/conv/GRU null-space
  invariance, **applied-Adam-delta projection** (raw-grad projection alone is insufficient),
  first-moment projection, multiple non-cancelling constraints QP, sentinel backtracking, atomic
  rollback, gate orientation, identity-leakage, no-inference-memory-path, delayed-credit indexing,
  burn-in replay reconstruction, byte-budget memory + diversity.
- **8-dimension adversarial spec-audit** (multi-agent) found 6 real divergences (all fixed); the core
  projection/optimizer/identity/gate logic audited clean.
- **2-game Atari smoke (§24.4)**: every protection mechanism verified end-to-end on real games
  (bases grow, consolidation accepted, applied-delta projection active, memory-disabled eval identical).
- **No task identity** enters the policy/memory/optimizer; **memory never selects actions**
  (memory-disabled eval is byte-identical).

## Diagnostics / honesty notes

- **Breakout** was dropped from the headline: our recurrent agent learns the *shooters*
  (SpaceInvaders, BeamRider) but is slow on Breakout's *paddle-control* task. A baseline feed-forward
  PPO (`baseline_ppo.py`) with the same envpool learns Breakout (1.3 → 33 at 2M) once a **FireReset**
  is added — confirming the env is fine; we added FireReset to the adapter, but the recurrent agent
  still needs far more steps on Breakout than was budgeted here. Reported, not hidden.
- This is a **2-game, seed-0** core proof (the user-chosen scope). The same driver extends to more
  games/seeds (`tfns/train/curriculum.py --games ... --seed ...`).
- The recurrent + protection machinery is heavier than feed-forward PPO; throughput is ~500–750
  env-SPS at N=128 after fixing several eager/O(n²) hot paths (jitted update + predictor, vectorized
  memory eviction, capped bank, protected-only replay conservation).

## Reproduce

```
# single-task refs (per game): python -m tfns.train.curriculum --mode refs --games SpaceInvaders-v5 ...
# plain baseline:              python -m tfns.train.curriculum --mode plain  --games SpaceInvaders-v5 BeamRider-v5 --steps-per-game 5000000 ...
# TFNS curriculum:             python -m tfns.train.curriculum --mode curriculum --games SpaceInvaders-v5 BeamRider-v5 \
#   --steps-per-game 5000000 --num-envs 128 --update-epochs 4 --num-minibatches 4 --ent-coef 0.01 \
#   --learned-threshold 0.7 --retention-accept 0.7 --refs-json <refs.json>
```
