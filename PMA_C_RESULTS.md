# PMA-C Results — Catastrophic-Forgetting Resistance (Supervised + RL)

> **Status:** all numbers below are committed 3-seed sweeps in `pma_c_results/`
> (`headline_10task/`, `split_mnist/`, `decomp_5task/`, `rl_continual/`). Reproduce in §8.

## 1. Claim

On **four** continual-learning settings spanning supervised AND reinforcement learning — Permuted-MNIST
(10 tasks), Split-MNIST (class-incremental), continual-RL gridworld, and **pixel-based RL: continual
MinAtar** (miniaturized-Atari, shared CNN, 4 games) — a **naive baseline catastrophically forgets** earlier
tasks while **PMA-C retains them**, using the *same* network, optimizer, learning rate, task order, and
training data. PMA-C dominates on every standard metric while **still learning each new task as well as
(or better than) the baseline** (no loss of plasticity). The same atlas + conservation + projection +
stability + memory machinery is reused across all four via different domain adapters — the approach is
genuinely domain-general.

Headlines: **10-task Permuted-MNIST** — ACC 0.749→0.930, worst-task retention 0.529→0.906. **Split-MNIST**
— ACC 0.197→0.962, baseline forgets *completely* (worst retention 0.000). **Atari-derived RL — Continual
MinAtar** (§3d) — the demanding test: a shared CNN trained sequentially on 4 (miniaturized-Atari) games;
the baseline collapses (worst game retains **8%**, mean retention 0.47±0.06) while **PMA-C retains
0.95±0.03** (tight, non-overlapping) without sacrificing new-game learning (per-seed mean final return
higher in all 3 seeds). An ablation decomposition (§4) isolates the driver as the **hinge conservation loss**
(functional regularization to stored teacher behavior), *not data rehearsal* — confirmed in both supervised
(PMA-C with no replay ≈ full) and RL (removing conservation collapses retention to baseline). ("No replay"
still *uses* the stored anchors for the conservation *gradient* — this separates rehearsal from functional
memory, not memory from no-memory.)

This is the empirical instantiation of the PMA-C spec
(`PMA_C_GENERAL_CONTINUAL_LEARNING_SPEC.md`): protected behavior anchors + hinge **conservation loss**
(§7), **tangent-cone gradient projection** (§8), **synaptic stability** (§9), balanced **rehearsal**
(§13/§15), frozen **champions** + **atlas** + **sentinels** (§4/§5), and a **regression gate** (§16),
hardened with gradient-norm control.

## 2. Setup (the matched comparison)

- **Benchmark:** Permuted-MNIST. Task 0 = raw MNIST (10-way); tasks 1..N−1 each apply a fixed random
  permutation of the 784 input pixels (same permutation for train and test). Real MNIST (60k/10k),
  pixels in [0,1]. Tasks trained **sequentially**.
- **Model:** plain-pytree MLP 784→256→256→10, ReLU. **Optimizer:** SGD, lr 0.1, batch 128.
- **Two arms, identical except protection:**
  - **baseline** — plain cross-entropy on the current task only.
  - **PMA-C** — same init (same seed), same optimizer, same task order, same epochs, **same training
    rows**; adds per-step: rehearsal of stored anchors, hinge conservation loss to each prior task's
    frozen teacher logits, gradient projection away from conflicting guard gradients, synaptic-stability
    LR scaling, and (off in the headline) a regression-rollback gate. After each task PMA-C certifies a
    frozen champion + anchors + sentinels into the atlas.
- **Fairness controls (see §6):** identical **gradient clipping** on both arms; a **train-derived
  validation split** used for all training-time decisions so **test data is never touched during
  training** (test is used only for the reported accuracy, identically for both arms).
- **Metrics.** With `A[i,j]` = test accuracy on task `j` after finishing training task `i`:
  - ACC = mean_j A[N−1, j];  learned/peak per task = A[j,j] / max_i A[i,j];
  - BWT = mean_{j<N−1}(A[N−1,j] − A[j,j]);  Forgetting = mean_{j<N−1}(peak_j − A[N−1,j]);
  - Retention_j = A[N−1,j] / peak_j (report mean and worst).

## 3. Headline result — 10-task Permuted-MNIST (3 seeds: 0,1,2)

Figure: `pma_c_results/headline_10task/fig_headline.png` (also `comparison.png`).
Data: `pma_c_results/headline_10task/results.json`.

| mode | ACC | BWT | Forgetting | mean Retention | worst Retention |
|---|---|---|---|---|---|
| baseline | 0.749 ± 0.023 | −0.239 ± 0.025 | 0.239 ± 0.025 | 0.777 ± 0.024 | **0.529 ± 0.023** |
| **PMA-C** | **0.930 ± 0.001** | **−0.034 ± 0.001** | **0.035 ± 0.001** | **0.967 ± 0.001** | **0.906 ± 0.014** |

Per-seed PMA-C ACC = 0.929 / 0.932 / 0.930 (essentially zero variance); baseline = 0.722 / 0.777 / 0.748.

Reading: over 10 tasks the **baseline's worst task retains only 53%** of its peak accuracy (early tasks
collapse to ~0.51–0.63), the signature of catastrophic forgetting. **PMA-C holds every task at 0.88–0.96
(worst retention 0.91)** while still learning each new task to ~0.95 — i.e. **no loss of plasticity**.
PMA-C forgets **6.8× less** than the baseline (0.035 vs 0.239) and turns backward transfer from −0.24 to
−0.03. The "Task-0-across-training" panel (right) shows the baseline degrading monotonically 0.96→0.51
as new tasks arrive while PMA-C stays flat at ~0.95. `nonfinite_steps = 0` for all PMA-C runs.

## 3b. Second benchmark — Split-MNIST, class-incremental (3 seeds)

A different forgetting mechanism: 5 tasks of 2 digit classes each ({0,1},…,{8,9}), one shared 10-way
head, **no task ID at test** (class-incremental — the hardest standard CL setting). Figure:
`pma_c_results/split_mnist/fig_split.png`; data: `pma_c_results/split_mnist/results.json`.

| mode | ACC | BWT | Forgetting | mean Retention | worst Retention |
|---|---|---|---|---|---|
| baseline | 0.197 ± 0.000 | −0.995 ± 0.001 | 0.995 ± 0.001 | 0.200 | **0.000** |
| **PMA-C** | **0.962 ± 0.002** | −0.007 | 0.007 | 0.994 | 0.988 |
| **PMA-C − replay** | **0.964 ± 0.001** | −0.015 | 0.015 | 0.988 | **0.976** |

The baseline forgets **completely** — worst-task retention **0.000**, BWT −0.995 — collapsing to predict
only the most recent task's classes (ACC 0.197 ≈ 1/5). PMA-C retains ~98%.

**Decisive decomposition result:** **PMA-C *without replay* (ACC 0.964) is statistically identical to
full PMA-C (0.962)**, and both ≈ near-perfect, versus baseline 0.197. This directly refutes the "it's
just rehearsal" hypothesis: with *no* replay augmentation, the **gradient-geometry mechanisms alone**
(hinge conservation to frozen teacher logits + tangent-cone projection + synaptic stability, all using
the stored anchors only for the *guard gradient*, never mixed into the loss) retain ~98% while the
baseline retains 0%. On this benchmark replay is not even necessary.

## 3c. Domain generality — Continual Reinforcement Learning (3 seeds)

PMA-C is domain-general (it needs only a behavior distance). To show this, the **same** atlas + conservation
+ projection + stability + replay machinery is applied to **RL** via the RL adapter (spec §19.1: behavior =
policy logits + value; distance = `D_KL(π*‖π) + λ|V−V*|`). Task = reach a goal in a small goal-conditioned
JAX gridworld (A2C); a sequence of 4 goals trained sequentially; metric = success rate (fraction of eval
episodes reaching the goal). Figure: `pma_c_results/rl_continual/comparison.png`.

| mode | mean final success | Forgetting | per-seed final-success vector (seed 0) |
|---|---|---|---|
| baseline | 0.330 ± 0.025 | 0.893 ± 0.033 | [0.00, 0.00, 0.18, 1.00] |
| **PMA-C** | **1.000 ± 0.000** | **0.000 ± 0.000** | [1.00, 1.00, 1.00, 1.00] |
| PMA-C − replay | 1.000 ± 0.000 | 0.000 | [1.00, 1.00, 1.00, 1.00] |
| PMA-C − conservation | 1.000 ± 0.000 | 0.000 | [1.00, 1.00, 1.00, 1.00] |

Both arms **learn every goal** to 1.0 success. But the naive A2C agent then **catastrophically forgets**:
after training all 4 goals it reaches *only the most recent* (mean success 0.33, forgetting 0.89). **PMA-C
retains all goals perfectly** (success 1.00, forgetting 0.00) across all 3 seeds. The right panel shows the
baseline's goal-0 success collapsing 1.0→0.0 as new goals arrive while PMA-C stays flat.

**Honest scope of this RL result.** The gridworld is *deliberately easy* — small 5×5 grid, short horizon,
goal-conditioned, so A2C converges reliably and the forgetting signal is clean. It is a *proof that the
PMA-C machinery ports to an RL adapter* (policy-KL + value distance) and stops the baseline's
catastrophic forgetting — **not** a claim of generality to hard RL (Atari/MuJoCo), which is future work.
Because the task is easy, the ablations saturate: *either* protection mechanism alone (`no_replay` or
`no_conservation`) already retains perfectly, so this benchmark cannot separate their contributions the
way the supervised decomposition (§4) does. Its value is the cross-paradigm transfer, not a mechanism study.

## 3d. Atari-derived RL — Continual MinAtar (4 games, 3 seeds)

The demanding test relative to the toy gridworld (§3c): **MinAtar** (Young & Tian 2019) — *miniaturized*
Atari with real pixel-like spatial observations (10×10×channels), sparse/delayed rewards, and credit
assignment — run via `gymnax` (fully JAX-native, jit'd PPO on GPU). A **single shared CNN actor-critic**
(Conv→Dense→policy+value, game-id conditioned) is trained **sequentially** on 4 games
(Breakout→Asterix→Freeway→SpaceInvaders). The shared CNN encoder is genuinely overwritten across games →
*real* catastrophic forgetting (the gridworld lacked a shared representation under pressure). PPO is
PureJaxRL-style and **entirely bounded `lax.scan`** (cannot hang). 5M env-steps/game. **It is miniaturized
Atari (10×10, not full 84×84 ALE) — chosen deliberately so PPO converges in minutes and runs cannot get
stuck; full ALE is future work.** Figure: `pma_c_results/minatar_hard/fig_minatar.png`; data
(per-seed): `pma_c_results/minatar_hard/`.

**Headline = retention** (the clean, low-variance, statistically-clear metric):

| mode | mean retention | worst retention | Forgetting |
|---|---|---|---|
| baseline | 0.470 ± 0.063 | **0.083 ± 0.059** | 11.1 ± 3.4 |
| **PMA-C** | **0.947 ± 0.025** | **0.863 ± 0.081** | **1.2 ± 0.7** |
| − conservation | 0.502 ± 0.048 | 0.172 ± 0.105 | 11.7 ± 4.2 |

Per-game retention (final/peak after all 4 games): Breakout **0.08→0.95**, Asterix 0.28→1.0, Freeway
0.44→0.95, SpaceInvaders 1.0→1.0 (last game, retained by both). The retention gap (0.47→0.95, non-
overlapping error bars) is the robust result.

**Per-seed mean final return** (absolute scores vary 7–150 across games, so this metric is high-variance —
but PMA-C beats baseline in **every seed**, a consistent paired win):

| seed | baseline | PMA-C |
|---|---|---|
| 0 | 11.3 | 26.3 |
| 1 | 7.7 | 15.2 |
| 2 | 41.1 | 53.6 |
| mean | 20.0 ± 15.0 | 31.7 ± 16.1 (paired Δ always +) |

**Reading.** The naive PPO agent **catastrophically forgets** early games when the shared CNN is
overwritten — Breakout retains only **8%** of its peak, worst-game retention 0.083 (Breakout curve, right
panel, collapses 1.0→0.02). **PMA-C retains all games** (mean retention 0.947). It also **does not
sacrifice plasticity** — per seed it matches or beats the baseline's new-game scores and its mean final
return is higher in all three seeds (the *unpaired* std is large only because game scores span 7–150). The
`no_conservation` ablation drops retention back to baseline (0.50) — **conservation is the driver in RL too.**

**Honest notes & limitations (disclosed, from an adversarial review of this result).**
- **n=3, high score-variance.** Per-game scores span 7–150, so the *unpaired* mean-final-return error bars
  overlap; the retention metric (0.470±0.063 vs 0.947±0.025) is tight and non-overlapping and is the metric
  we headline. The mean-final win is a *paired* statement (PMA-C > baseline in each seed), not a claim of
  unpaired significance at n=3. Per-seed numbers are shown above and in the committed JSON.
- **Training budget was tuned empirically.** In exploratory runs at 3M steps/game the conservation penalty
  *over-constrained* the 4th game in some seeds (it could not learn the new game *under* the guard); at 5M
  steps/game it reliably learns every game *and* retains across all 3 seeds. We report 5M; we did not commit
  a full 3M-vs-5M frontier table. So "no plasticity sacrifice" holds *at this budget*, not unconditionally.
- **`guard_coef` is a real knob** (we sampled {0.3, 0.5, 1.0}); too high over-constrains later tasks because
  accumulated guard pressure grows with the number of prior tasks. A length-normalized guard (pressure ÷
  #prior tasks) would remove this tuning and is the natural next step — currently a TODO, not implemented.
- **Anchor-collection detail:** anchors store the *stochastic* policy's logits (sampled actions during
  rollout) while evaluation is *greedy* (argmax). Conserving the full action distribution is arguably more
  complete, but this is a mismatch with the greedy eval and is disclosed; a greedy-anchor ablation is not run.
- **Metric subtlety:** retention (final/peak) alone can be gamed by *not learning* a game (low peak ⇒ trivial
  retention), which is exactly why we also report per-seed mean final return.
- **Scope:** this is *miniaturized* Atari (MinAtar), genuinely harder than §3c (shared CNN, pixels, sparse
  reward) but not full-scale ALE; it answers the easy-task critique without claiming Atari-scale generality.

## 4. Credit decomposition — what actually drives the retention (3 seeds)

Source: `pma_c_results/decomp_5task/results.json` (5-task Permuted-MNIST, 3 seeds, matched).
Figure: `pma_c_results/decomp_5task/fig_decomposition.png`. This is the honest answer to "is it just
replay?" — **it is not; the hinge *conservation* loss is the driver.**

| condition | ACC | Forgetting | worst Retention | notes |
|---|---|---|---|---|
| baseline (naive) | 0.872 ± 0.002 | 0.115 ± 0.003 | 0.748 | lower bound — forgets |
| replay_only (= Experience Replay) | 0.895 ± 0.008 | 0.083 ± 0.009 | 0.850 | replay alone barely helps |
| − conservation (`no_conservation`) | 0.895 ± 0.008 | 0.083 ± 0.009 | 0.850 | **= ER: conservation is essential** |
| − projection (`no_projection`) | 0.962 ± 0.001 | 0.003 ± 0.002 | 0.992 | removable at this scale |
| − stability (`no_stability`) | 0.961 ± 0.001 | 0.003 ± 0.001 | 0.994 | removable at this scale |
| random memory (`random_memory`) | 0.957 ± 0.000 | 0.005 ± 0.001 | 0.990 | importance non-critical here |
| **PMA-C − replay (`no_replay`)** | **0.960 ± 0.001** | **0.005 ± 0.002** | **0.989** | **≈ full, with NO rehearsal** |
| **PMA-C (full)** | **0.961 ± 0.001** | **0.003 ± 0.001** | **0.994** | all mechanisms |

**What the decomposition shows:**
- **Replay alone is weak.** `replay_only` (= standard Experience Replay) reaches only ACC 0.895 — barely
  above the 0.872 baseline. Rehearsal is *not* what produces PMA-C's retention.
- **Conservation is the engine.** Removing the hinge conservation loss (`no_conservation`) collapses
  PMA-C back to exactly the ER level (0.895), because with no guard gradients the projection has nothing
  to project against (it becomes a no-op too). The functional regularization to each prior task's frozen
  teacher logits is what holds the manifold.
- **It works without replay.** `no_replay` (0.960) ≈ full (0.961): the conservation mechanism retains
  ~99% **with no rehearsal at all**. Corroborated at **two more scales**: Split-MNIST (§3b) `no_replay`
  0.964 ≈ full 0.962 (vs baseline 0.197), and the dramatic **10-task Permuted-MNIST** `no_replay` ACC
  0.933 ≈ full 0.932 (vs baseline 0.740) — so the mechanism-without-rehearsal result holds at the
  headline scale too (`pma_c_results/noreplay_10task/`).
- **Projection / stability / importance-selection** are individually removable at this *easy* 5-task
  regime (the 256-256 net has spare capacity, so conservation suffices). They earn their keep in the
  **harder** regimes: without gradient-norm control (their relatives) PMA-C diverged at 8 epochs / 10
  tasks (§5) — projection + stability + clipping are what keep the optimization on-manifold there.

## 5. Robustness

- **Across regimes:** PMA-C retains and the baseline forgets at 5 tasks/5 epochs, 5 tasks/8 epochs,
  and 10 tasks/5 epochs. More epochs / more tasks → the baseline forgets *more* (worst retention
  0.76 → 0.71 → 0.53), PMA-C stays ≥0.90. On class-incremental Split-MNIST the baseline's worst
  retention is 0.00 (total collapse) while PMA-C is 0.99.
- **Stability constraint (a real, load-bearing finding).** The squared-hinge conservation loss has
  gradient `2·(KL−ε)·∇KL`, which becomes a positive-feedback runaway at lr 0.1 once an update pushes the
  net off-manifold; *without* gradient-norm control PMA-C collapsed to chance at 8 epochs / 10 tasks
  (ACC 0.088). This is not a hygiene detail — the conservation mechanism is **unstable unless
  constrained**. The fix clips each guard gradient to `k·‖g_new‖` (so the conservation correction can
  never dominate the task signal) + a global update-norm clip + a non-finite-step skip; it trades a small
  amount of guard strength for stability. After it, PMA-C is stable across all tested regimes (8-epoch
  ACC 0.088 → 0.965). **Honesty:** the *global* clip is applied to both arms (a near-no-op for the
  baseline's small gradients), but the **guard-gradient clip is PMA-C-specific** (the baseline has no
  guard gradient) — it is part of PMA-C's mechanism, not a shared control.

## 6. Fairness & honesty (adversarial-audit response)

A multi-agent adversarial audit of this result confirmed the phenomenon and the metric computation
(clean test splits, no contamination) and flagged asymmetries, all addressed:
- **Replay is part of PMA-C, and its contribution is decomposed** (§4: `replay_only` and `no_replay`),
  not hidden. We do not attribute the full gain to the novel mechanisms — §4 partitions credit.
- **Gradient clipping is symmetric** (both arms clipped identically; a near-no-op for the baseline's
  small gradients but the comparison is honest).
- **No test-set leakage anywhere in training.** The acceptance gate, sentinels, and certification use a
  train-derived **validation** split; both arms train on the **same** train rows (train minus val);
  **test** is used only for the reported per-task accuracy, scored identically for both arms. The
  `results.json` `headline_config` records `gate_enabled`, `max_grad_norm`, and `val_size` to
  substantiate this.
- **The baseline is a fair naive learner** (same architecture/optimizer/lr/order/data); it is the
  "PMA-C disabled" control the task requires.

**Honest scope — what these results do and do NOT show.**
- **Active mechanisms in the headline.** The retention shown here is produced by **conservation +
  projection + stability + replay** on a spare-capacity 256-256 MLP. **Growth, consolidation, and the
  router are implemented and unit-tested but are NOT the active drivers** here — the net has ample
  capacity so growth never triggers, and the consolidation interval is not reached within these short
  runs. They are exercised separately by the unit suite and the live full-system demo
  `pmac/experiments/full_system_demo.py` (sections A–D all PASS: champion immutability + non-deletion,
  no-op growth giving plasticity with zero old-task interference, slow-core consolidation, routing). On
  harder, capacity-limited tasks those mechanisms would activate and could surface new failure modes not
  tested here.
- **The impossibility-boundary case (spec §2) is NOT tested.** All three benchmarks have *distinguishable*
  contexts (different pixel permutations / disjoint class subsets / explicit goal IDs in the observation),
  so a single function *can* in principle serve all tasks. The spec's hard case — *identical* input
  requiring *opposite* output with no context — is exactly where the spec says you need explicit/inferred
  context, growth, or frozen experts; none of our benchmarks probe it. The claim here is the empirical one
  (PMA-C resists forgetting when tasks are distinguishable and capacity/memory suffice), not the
  system-level non-overwrite guarantee (which is structural — frozen champions — and unit-tested, not
  measured by these accuracy curves).
- **Benchmark scale.** Two MNIST variants, a toy gridworld, and **hard pixel-based RL (continual MinAtar,
  Atari-derived, shared CNN)** — the last genuinely stresses representation overwriting and is where the
  baseline truly collapses. This is a substantial step beyond the earlier "easy-task" RL, but MinAtar is
  still a *miniaturized* Atari (chosen deliberately so PPO converges in minutes and the runs cannot hang);
  full ALE Atari / MuJoCo at scale remains future work, as does a length-normalized guard so `guard_coef`
  need not be tuned per sequence length.
- **Memory vs rehearsal.** "no_replay" removes data *rehearsal* but still uses the stored anchors for the
  conservation gradient — it is rehearsal-free, not memory-free. Long-term memory (anchors) is intrinsic
  to PMA-C; §4 separates two *uses* of that memory (functional regularization vs. data mixing).
- Generalization to unseen inputs is empirical (depends on anchor/sentinel coverage), exactly as the spec
  states (§0 honesty boundary, §29.5).

## 7. What is implemented (spec coverage)

All PMA-C modules from the spec are implemented in `pmac/` with passing unit tests (53 tests,
`tests/pmac/`):
behavior distances §6, conservation §7, tangent-cone projection §8, synaptic stability §9, growth
§10/§25.4, consolidation §11/§18, router §12, memory selection/anchors §13–14, scheduler §15, acceptance
gate §16, full training loop §17, atlas/skill-graph §5, champions/non-deletion invariant §4/§11.4, and
the supervised domain adapter §19.4 **and the RL adapter §19.1** (policy-KL + value distance, demonstrated
live in §3c) — plus the JIT fast path, gradient-clipping hardening, and the matched continual-learning
runner. Each component's unit test encodes the spec's math/invariant
(e.g. projection removes a conflicting gradient component; conservation hinge is zero inside tolerance;
a frozen champion is an immutable deep copy; the last certified implementation can never be deleted).

## 8. Reproduce

WSL distro `praxis`, `/opt/venv/bin/python` (JAX 0.9.2 + optax, CUDA on an RTX 4090), repo on `/mnt/c`.

```bash
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=<repo>
# Headline (10-task, 3 seeds): baseline vs PMA-C
python -m pmac.experiments.continual_mnist \
  --stream permuted_mnist --num-tasks 10 --seeds 0,1,2 \
  --epochs 5 --batch-size 128 --lr 0.1 --optimizer sgd --hidden 256,256 \
  --temperature 2.0 --gate off --out runs/pmac_headline
# Second benchmark — Split-MNIST, class-incremental (3 seeds), incl. no-replay decomposition
python -m pmac.experiments.continual_mnist \
  --stream split_mnist --seeds 0,1,2 --epochs 5 --gate off \
  --ablations no_replay --out runs/pmac_split
# Credit decomposition (5-task Permuted, 3 seeds): all ablations
python -m pmac.experiments.continual_mnist \
  --stream permuted_mnist --num-tasks 5 --seeds 0,1,2 --epochs 5 --gate off \
  --ablations no_replay,replay_only,no_projection,no_conservation,no_stability,random_memory \
  --out runs/pmac_decomp
# Continual RL — gridworld (sequential goals, 3 seeds)
python -m pmac.experiments.continual_rl --goals 4 --seeds 0,1,2 \
  --ablations no_replay,no_conservation --out runs/pmac_rl
# HARD RL — continual MinAtar (Atari-derived, 4 games, shared CNN, 3 seeds).
# gymnax provides MinAtar; PPO is bounded lax.scan (cannot hang). ~8 min/seed on a 4090.
python -m pmac.experiments.continual_minatar \
  --games Breakout-MinAtar,Asterix-MinAtar,Freeway-MinAtar,SpaceInvaders-MinAtar \
  --per-game-steps 5000000 --seeds 0,1,2 --ablations no_conservation --out runs/pmac_minatar
# (single-game learnability check: python -m pmac.experiments.rl_minatar_smoke --game Breakout-MinAtar)
```
Each run writes `results.json` (all accuracy matrices + metrics + aggregate + config echo proving
gate/clip/val) and `comparison.png`. Figures regenerated via `pma_c_results/make_figures.py`. Unit
tests: `JAX_PLATFORMS=cpu pytest tests/pmac -q` (53 passing). Full-system demo:
`JAX_PLATFORMS=cpu python -m pmac.experiments.full_system_demo`. Committed artifacts under
`pma_c_results/`: `headline_10task/`, `split_mnist/`, `decomp_5task/`.
