# PMA-C Results — Catastrophic-Forgetting Resistance on Continual MNIST

> **Status:** all numbers below are committed 3-seed sweeps in `pma_c_results/`
> (`headline_10task/`, `split_mnist/`, `decomp_5task/`). Reproduce commands in §8.

## 1. Claim

On two standard continual-learning benchmarks — **Permuted-MNIST** (10 tasks) and **Split-MNIST**
(class-incremental) — a **naive baseline catastrophically forgets** earlier tasks while **PMA-C retains
them**, using the *same* network, optimizer, learning rate, task order, and training data. PMA-C
dominates the baseline on every standard metric (Average Accuracy, Forgetting / Backward Transfer,
Retention) while **still learning each new task as well as the baseline** (no loss of plasticity).

Headlines: **10-task Permuted-MNIST** — ACC 0.749→0.930, worst-task retention 0.529→0.906, forgetting
6.8× lower. **Split-MNIST** — ACC 0.197→0.962, the baseline forgets *completely* (worst retention
0.000). An ablation decomposition (§4) shows the driver is the **hinge conservation loss**, not replay:
PMA-C *without any rehearsal* retains ~99% (≈ full system), while plain Experience Replay barely helps.

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
  ~99% **with no rehearsal at all**. (Corroborated on Split-MNIST §3b: `no_replay` 0.964 ≈ full 0.962,
  vs baseline 0.197.)
- **Projection / stability / importance-selection** are individually removable at this *easy* 5-task
  regime (the 256-256 net has spare capacity, so conservation suffices). They earn their keep in the
  **harder** regimes: without gradient-norm control (their relatives) PMA-C diverged at 8 epochs / 10
  tasks (§5) — projection + stability + clipping are what keep the optimization on-manifold there.

## 5. Robustness

- **Across regimes:** PMA-C retains and the baseline forgets at 5 tasks/5 epochs, 5 tasks/8 epochs,
  and 10 tasks/5 epochs. More epochs / more tasks → the baseline forgets *more* (worst retention
  0.76 → 0.71 → 0.53), PMA-C stays ≥0.90. On class-incremental Split-MNIST the baseline's worst
  retention is 0.00 (total collapse) while PMA-C is 0.99.
- **Hardening (a real finding, fixed).** Without gradient-norm control, the squared-hinge conservation
  loss `∇ = 2·(KL−ε)·∇KL` becomes a positive-feedback runaway at lr 0.1 once an update pushes the net
  off-manifold; at 8 epochs / 10 tasks PMA-C collapsed to chance. Fix: clip each guard gradient to
  `k·‖g_new‖` (the conservation correction can never dominate the task signal) + a global update-norm
  clip + a non-finite-step skip. After hardening PMA-C is stable across all tested regimes (e.g. 8-epoch
  PMA-C went from ACC 0.088 → 0.965). The clip is applied to **both** arms.

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

**Honest scope.** Generalization to unseen inputs is empirical (depends on anchor/sentinel coverage),
exactly as the spec states. Consolidation/growth/router are implemented and unit-tested but are not the
*active* drivers in this headline (the 256-256 MLP has ample capacity, so growth never triggers and the
consolidation interval is not reached within these short runs); they are exercised by the unit suite and
the live full-system demo `pmac/experiments/full_system_demo.py` (sections A–D all PASS: champion
immutability + non-deletion, no-op growth giving plasticity with zero old-task interference, slow-core
consolidation, and context routing).

## 7. What is implemented (spec coverage)

All PMA-C modules from the spec are implemented in `pmac/` with passing unit tests (53 tests,
`tests/pmac/`):
behavior distances §6, conservation §7, tangent-cone projection §8, synaptic stability §9, growth
§10/§25.4, consolidation §11/§18, router §12, memory selection/anchors §13–14, scheduler §15, acceptance
gate §16, full training loop §17, atlas/skill-graph §5, champions/non-deletion invariant §4/§11.4, and
the supervised domain adapter §19.4 — plus the JIT fast path, gradient-clipping hardening, and the
matched continual-learning runner. Each component's unit test encodes the spec's math/invariant
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
```
Each run writes `results.json` (all accuracy matrices + metrics + aggregate + config echo proving
gate/clip/val) and `comparison.png`. Figures regenerated via `pma_c_results/make_figures.py`. Unit
tests: `JAX_PLATFORMS=cpu pytest tests/pmac -q` (53 passing). Full-system demo:
`JAX_PLATFORMS=cpu python -m pmac.experiments.full_system_demo`. Committed artifacts under
`pma_c_results/`: `headline_10task/`, `split_mnist/`, `decomp_5task/`.
