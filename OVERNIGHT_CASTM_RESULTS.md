# Overnight CASTM Results — genuinely task-free, fully-plastic continual learning

**Status: IN PROGRESS** (this file is updated as runs complete). Branch `task-free-ns`.
Hardware: 2× RTX 4090 (CUDA 1,2). All claims below are evidence-linked; oracle
(diagnostic) and inferred (primary, task-free) results are kept separate.

> Honesty contract: nothing here is called "solved"/"proven"/"foolproof" unless the
> corresponding strict gate actually passed on completed-episode evaluations.

## 1. Commit hashes
- `11b1f17` — task-free machinery (context_manager, online_resolve, sparse_exec,
  train_taskfree, analyze_taskfree, _ppo_terms forced-FIRE mask) + 21 tests.
- (subsequent routing-robustness + review-fix commits listed at the end)

## 2. What was implemented (removing the research-invalid shortcuts)
| Shortcut (was) | Now |
|---|---|
| Addresses preallocated from the game list | Allocated **online** by the context manager from content novelty |
| Active context = curriculum index | **Inferred online** from content (no id/label reaches agent/router/memory/optimizer) |
| Policy/value heads re-initialised at game boundaries | **No re-init**; novelty-triggered entropy boost + Adam-moment reset; weights continuous |
| Prototypes built post-hoc per labelled game | Built **online from raw anchors**, refreshed under encoder drift |
| Retention oracle-addressed | **Inferred routing is the primary** retention eval; oracle is a diagnostic only |
| `forward_sparse` masks but computes all slots | **Blocked gather-before-matmul** path; provably O(1) in #contexts |
| Forced FIRE in PPO ratio | **free_mask** excludes forced transitions from policy-grad/entropy/KL (value keeps all) |
| Value head excluded from resolve | **Value protected by default** (ablatable) |

## 3. Tests and profiling
- **Unit/synthetic (CPU):** full `tests/castm/` suite — **PASS** (51 prior + new
  modules). New coverage: online resolve invariants (1,2,3,12,16), online context
  discovery/recall/false-split/per-stream/refresh/identity (4,5,6,8,9,10), true-sparse
  bit-identity + constant-work (11), forced-FIRE mask (14), completed-episode gate (15),
  serialization/resume (13).
- **Mathematical memory gate §21.1** (prior): exact noninterference, 32 contexts,
  worst-case 5.3e-7 < 1e-6 (float32) — retained.
- **True-sparse execution benchmark (RTX 4090):** blocked path 1.41→1.57 ms over
  1→57 contexts (**6.7% overhead 5→57**, < 20% requirement); functional path
  1.66→4.97 ms (**200%**). Artifact: `castm_runs/taskfree/sparse_benchmark.json`.

## 4. GPU runs (table)
_TBD — filled as runs complete. Each row: config, seed, steps, contexts discovered,
raw scores, normalized progress (vs matched 500k ref), retention, router top-1, throughput._

Matched single-task references (500k, this budget):
| game | random | 500k ref (final/best) |
|---|---|---|
| SpaceInvaders-v5 | ~138 | 147.9 / 170.0  (weak gap → weak normalization at 500k) |
| Seaquest-v5 | ~45 | 591.7 / 595.0  (strong gap → clean normalization) |

## 5. Strict-gate status
_TBD._ Two-context gate: R_old≥0.90, P_new≥0.90, A_router≥0.99. Three-context and
five-context gates per the ladder.

## 6. Raw and normalized scores
_TBD._

## 7. Retention matrices
_TBD._

## 8. Routing metrics
_TBD._ (router top-1, confidence margin, false splits/merges, switch latency,
contexts discovered vs regimes.)

## 9. Compactness and throughput
Per-context memory ≈ 0.6 MB across all contextualised layers (prior measurement,
retained). Throughput: ~700–800 sps steady-state for the task-free trainer (vs
~2000 sps single-task baseline; the delta is the per-rollout routing + periodic
resolve + prototype refresh).

## 10. Negative findings / honest caveats
_TBD._ Known: online routing under a plastic encoder is the hard part — the
content representation's within-regime coherence is low early in training (weak
game separation), which stressed the discovery logic (documented iteration in the
research log). SpaceInvaders normalizes weakly at the 500k budget (high random
baseline vs achievable score) — Seaquest is the clean target.

## 11. Next three highest-value experiments
_TBD._
