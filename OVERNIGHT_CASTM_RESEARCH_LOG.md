# Overnight CASTM Research Log — genuinely task-free, fully-plastic continual learning

Append-only. Branch `task-free-ns`. Two RTX 4090s (CUDA_VISIBLE_DEVICES 1,2 under
`CUDA_DEVICE_ORDER=PCI_BUS_ID`; device 0 is a 2080Ti — avoid). WSL distro `praxis`,
venv `/opt/venv`, repo at `/mnt/c/Users/Asav/source/repos/Praxis`.

All times local (system clock).

---

## 2026-06-23 ~20:20 — Session start, codebase audit

### Hypothesis / framing
The repo proves *compact retention* under a fully-plastic shared net, but the
training path (`tfns/castm/train_plastic.py`) is **boundary-aware**: it uses the
curriculum index as the context id. The overnight goal is to remove every
research-invalid shortcut and demonstrate **online, task-free** context discovery
+ retention + new learning, with the strict gates.

### Confirmed research-invalid shortcuts (read the code, line-precise)
1. `train_plastic.py:185-188` — addresses **preallocated** from `cfg.games` count.
2. `train_plastic.py:214-216` — active context = `ctx_ids[gi]` (**curriculum index**).
3. `train_plastic.py:226-235` — **policy/value heads re-initialised** at each game
   boundary (an externally-boundary-aware shortcut).
4. `train_plastic.py:341-352` — routing prototypes built **post-hoc per labelled game**.
5. `train_plastic.py:321-331` — retention eval is **oracle-addressed**; inferred only at end.
6. `synaptic.forward_sparse` (`synaptic.py:166-192`) and conv `_conv_memory_delta`
   (`layers.py:145-157`) are **functionally** sparse but **not computationally**
   sparse: they contract over **all M slots** then mask. Runtime scales with total
   stored contexts. (Test 11 / architecture §5 unmet.)
7. Forced-FIRE PPO: `sample` stores `log pi(FIRE)` for forced steps; `_ppo_terms`
   includes every transition in the PG mean. Forced transitions (behaviour prob = 1)
   contribute a spurious policy-gradient term. **No `forced` mask exists.** (#7)
8. `resolve_memory` (`train_plastic.py:145-148`) **excludes the value head** from
   drift compensation. (#8)

### What is correct and must be preserved
- `synaptic.py` factorised memory, exact dual writes, decode, recompress,
  `shared_consolidate` — mathematically sound (gate 21.1 passed, 5.3e-7 < 1e-6).
- `address.py` orthonormal codebook + dual + `allocate_novel_from_candidate`
  (residual construction `P_perp u`) + `compensation_vector` + numerical invariants.
- `router.py` jittable batched router (KNOWN/UNCERTAIN/NOVEL, dwell, hysteresis,
  novelty persistence, `allocate_context`, `refresh_prototypes`) — **built but NOT
  wired into the training loop.**
- `transaction.py` commit/recompress/consolidate w/ verify-then-commit + rollback.
- `consolidate.py` exact compensated common-structure consolidation.
- `train/evaluate.py` + `train_castm.evaluate_game` — completed-episode valid-only eval.
- Temporal replay alignment, raw rewards, forced-action semantics (where fixed),
  numerical guards, serialization.

### Core mechanism (verified by reading) and how the online version generalises it
`resolve_memory`: most-recently-trained context = pure `W0` (no component); every
older context c carries `D_c` with `D_c' = SVD_R(decode_delta(D_c) - dW0)`, so
`W0_new + D_c = ` c's learned weights. This is exactly the invariant
`D_c' = D_c - ΔW0 (c≠a)`, `D_a' = D_a`. It already supports an **active context
that has components** (D_a is constant w.r.t. the W0 gradient), so the online
generalisation is: snapshot W0/b0 → train K updates → for every context EXCEPT the
inferred-active one, `D_c -= ΔW0`, `β_c -= Δb0` → advance snapshot. Trigger
periodically (every N updates) **and at every inferred switch** so each interval
has a single active context. New context: allocated with `D=0` at current W0;
becomes protected once the stream leaves it.

### Plan (decisions; not asking)
Local, fast, high-ROI fixes first (CPU-testable), then the online machinery, then GPU.
- **D1 forced-FIRE PPO mask**: carry `forced` in the batch; PG/entropy/approx_kl over
  free steps only; value loss over all. Add tests. (Architecture §6, test 14.)
- **D2 true sparse execution**: per-context contiguous slot table; gather a fixed
  `slots_per_ctx` block (static shape) → matmul only over the gathered block.
  Benchmark 1/5/20/57 contexts; assert <20% overhead 5→57 and bit-identical output
  when an unselected context's factors change. (Architecture §5, test 11.)
- **D3 online context manager** (`context_manager.py`): wraps router primitives;
  per-stream + batch-consensus allocation w/ dwell+novelty persistence; anchor
  buffers (train/held-out); online prototype build/refresh from raw anchors.
  Synthetic tests 4,5,9,10.
- **D4 online resolve engine**: periodic + on-switch; value included (ablate exclude);
  strict error audit (max-elem, rel-Fro, functional drift). Tests 1,2,3,12,16.
- **D5 `train_taskfree.py`**: integrates D1–D4 + novelty-triggered exploration
  (entropy boost / Adam-moment damp, ablated vs none) replacing head re-init;
  inferred routing is the PRIMARY eval; serialization/resume (test 13).
- Then GPU ladder Stage 1 → 2 → 3.

## ~21:45 — KEY FINDING: the shared-encoder content query is non-discriminative
Adversarial review (4 agents, 21 findings, 4 high) ran in parallel with the pilot and
confirmed + extended my diagnosis: warm-up pollution, incomplete serialization, and
two attribution bugs (alloc `prev_ctx` fallback; switch credit). All fixed.

Then the routing root cause surfaced in the data. Instrumented smokes showed:
- The W0-encoder content query gives **cosine ~0.99 between SpaceInvaders and
  Seaquest frames** (and ~0.99 within each) — the games are *unseparable* in the
  penultimate features. **Mean-centering** (ReLU features live in the positive
  orthant) lifted within-game structure but Seaquest still scored **0.86–0.99**
  against SI's centered prototypes. The encoder's penultimate features are genuinely
  non-discriminative across these visually-distinct games (both collapse to the same
  control-feature region).
- Also fixed two prerequisite bugs the encoder path exposed: (i) **false split in
  game0** from a one-way baseline ratchet + inflated seeding mean (k-means centroids
  trivially fit their own seeding frames at ~1.0 while new frames score far lower) —
  fixed by calibrating the baseline from real warm-up frames, not the seed; (ii)
  **false merge in game1** from a dev-scaled band swallowing the cross-game sim —
  fixed by capping the band. With these, game0 stays 1 context (no split) at every
  scale, but game1 still merged because the *representation* can't separate them.

### Decision: content signature = pooled raw observation (spec-allowed "observations")
Since the shared encoder cannot separate the regimes, the content query uses a
**pooled raw-observation signature** (8×8 spatial average-pool per stacked frame =
256-d), which the spec explicitly permits ("observations or shared encoder
features"). It separates SI/Seaquest trivially (sanity: within 1.0, cross 0.0), is
**encoder-drift-free** (raw pixels don't drift — the §2 drift machinery still runs
but is trivially stable), and is domain-general (any agent has sensory input). The
encoder-feature path is retained as an ablation. This is the honest, working choice;
the negative result (encoder features non-discriminative across games) is reported.

## ~22:09 — Stage 1 (two-context, 500k) — PASSES (pilot); naive control bug found+fixed
**PLASTIC task-free (SI→Seaquest, 500k/game, seed 1):** discovered **2 contexts**
(no labels), **router top-1 = 1.0** (inter-ctx sim −0.94), **retained SI exactly**
(165.8 after game0 → 176.7 after Seaquest), learned **Seaquest 566.7** (oracle P_new
0.95). Gates: no-proliferation PASS, A_router PASS, R_old PASS, P_new PASS(oracle)/
0.81(inferred, within Seaquest variance). Seaquest detected within ~3 rollouts of the
unannounced switch (active_sim 0.80→0.29).

**Naive-control bug:** with `--no-resolve`, `do_resolve` returned early WITHOUT
syncing `banks` from the trained `params`, so the retention eval read stale (initial)
weights (SI 107.5→107.5, Seaquest 71.7≈random despite learning to 411 live). Fixed:
sync `banks = apply_shared_trainable(banks, params)` in the no-resolve branch.
Re-running the naive control. The PLASTIC arm is unaffected (its resolves sync banks).

## ~22:12 — Stage 2 (three-context alternation A→B→C→A→B) launched on GPU2
SI→Seaquest→Breakout→SI→Seaquest, 300k/segment. Tests online discovery of 3 regimes,
revisit recall (SI & Seaquest revisited → recall their contexts, not allocate new),
and correction transport across the alternation.

## ~22:25 — Stage 2 (3-context) discovery + REVISIT RECALL works
- 3 contexts discovered online (SI=ctx0, Seaquest=ctx1, Breakout=ctx2). Pooled
  signatures separate them sharply: Breakout sim −0.08 to Seaquest, −0.6 to SI.
- **Revisit recall**: at game3 (SI again), active_sim to the active Breakout ctx2 =
  −0.63 but best_sim to ctx0 (SI) = 0.997 → **SWITCH to ctx0** (recalled), nctx stays
  3 (NO 4th context). SI eval on recall = 213.8 (preserved). game4 (Seaquest revisit)
  → recall ctx1 expected. This validates online discovery + recall + no proliferation.

## ~22:50 — Stage 3 (5-game) NEGATIVE finding: pooled signature merges SI into Breakout
Order Breakout→Pong→SpaceInvaders→Seaquest→BeamRider. Breakout=ctx0, Pong=ctx1 (clean).
At game2, **SpaceInvaders merged into Breakout's ctx0** (best_sim 0.832 ≥ Breakout's
match level ~0.52 → SWITCH ctx0, no new context). Cause: Breakout has high within-game
pooled-signature variance (ball motion, vanishing bricks) → low running mean (~0.64) →
a WIDE match band, and SI's layout (action-bottom/targets-top) is genuinely ~0.83
similar to Breakout's in pooled-pixel space. The pooled signature lacks the resolution
to separate two visually-similar games when one has high variance.
- Note the asymmetry: in Stage 2 (SI first, Breakout third) they did NOT merge
  (Breakout-vs-SI 0.13); here (Breakout first) SI-vs-Breakout 0.83. Order + the
  variance-driven band width matter.
- **Decision:** let Stage-3 finish and report it as a PARTIAL/negative result rather
  than burn ~50 min on an uncertain higher-resolution re-run. The fix is a learned
  discriminative content encoder (next-experiment #2) — the mechanism (resolve +
  online discovery) is already proven at 2 and 3 contexts. Honest > forced pass.

### Decision: decouple true-sparse-exec *claim* from the pilot training path
The existing `forward_sparse` is functionally correct but contracts over all `M`
slots (overhead ∝ stored contexts). At 2–5 contexts the overhead is a negligible
constant on the memory-delta term (the W0 conv/dense matmul dominates), so the
**pilots train with the existing functionally-sparse forward** (exact outputs,
exact retention). The architecture-§5 / test-11 *deliverable* (runtime constant in
#contexts; bit-identical under unselected-context edits) is delivered separately as
a **blocked-layout true-sparse forward + benchmark + bit-identity test**, so a
memory-layout refactor never blocks the GPU science. Both are delivered; risk is
lower. Recorded for auditability.

### Decision: online resolve representation
Generalise `resolve_memory` (not rewrite synaptic): rebuild each layer's memory by
re-appending, per context `c`, a single recompressed component. **Active context's
components are copied verbatim** (`D_a' = D_a` exact, zero error); **inactive
contexts get `D_c' = Compress_R(decode_delta(D_c) − ΔW0)`, `β_c' = β_c − Δb0`**.
Value head **included by default** (ablatable). Trigger periodically *and at every
inferred switch with the outgoing context as active*, so each interval has one
active context. Per-context max-elem + rel-Fro residual audited; budget flag set.

## ~21:10 — Core modules built + validated (CPU)

- `tfns/castm/context_manager.py` (`OnlineContextManager`): online discovery/recall
  from content; novel-evidence window seeding (uncontaminated), pollution-free
  anchor buffers, prototype refresh under encoder drift. **7/7 tests pass.**
- `tfns/castm/online_resolve.py` (`online_resolve`): generalised compensated resolve;
  active verbatim, inactive `D_c−ΔW0` recompressed, value included, budget audited.
  **6/6 tests pass.**
- `_ppo_terms` now takes an optional `free_mask` (forced-FIRE exclusion). D1 in place.
- Found+fixed two real bugs in the manager before GPU: (i) seeding window was the
  whole history (`pending_cap` huge) → prototypes spanned clusters; (ii) novelty-
  building frames polluted the active context's anchors. Fix: seed only from a short
  *novel-evidence window*; only add anchors when confidently dwelling (KNOWN).

### Attribution decision (avoids cross-context contamination, exactly)
The trainer **holds the W0 gradient step on non-KNOWN rollouts** (BOOTSTRAP/ALLOC/
SWITCH/UNCERTAIN). Every applied W0 update therefore happens under a single
confident active context, so the interval→active-context attribution is exact and
the online resolve's `D_a' = D_a` / `D_c' = D_c − ΔW0` split is contamination-free.
Cost: ~`novel_persist`+switch rollouts of held W0 per transition (data still
collected for routing/anchors) — negligible vs a 250k-step segment. Resolve fires
periodically (active = current) and at every switch/alloc (active = outgoing).

### Novelty-triggered exploration (replaces head re-init; test 7)
On ALLOC: re-init the Adam moments (damp stale momentum from the prior regime) and
apply a decaying entropy-coefficient boost. **Weights (incl. heads) stay fully
continuous** — no re-initialisation. Ablation flag for "no intervention".

## ~21:50 — GPU smoke (SI→Seaquest, 33k steps each) — plumbing works, one bug found
End-to-end on a 4090, exit 0. Online resolve fired (residual 0.0004–0.0014),
prototype refresh ~0.96, **held-out router top-1 = 1.0** (per-game SI 1.0, Seaquest
1.0). Oracle retention computed. Serialization/persist OK.

**Bug: 3 contexts discovered for 2 games** (false split *within* SpaceInvaders).
Diagnosis: the encoder is very plastic in the first ~16 updates, so SI's content
queries drift off ctx0's prototypes (seeded at rollout 1, full-refreshed only every
`resolve_every`) → transient within-context novelty → spurious ctx1. This is the
encoder-drift false-split risk from architecture §2, surfacing because the active
context's prototypes weren't tracking the drift fast enough.

**Fix:** cheap per-rollout refresh of the **active** context's prototypes from the
in-hand queries whenever the decision is KNOWN (no GPU re-encode). The KNOWN gate
prevents cross-context pollution; full raw-anchor re-encode still runs at each
resolve for the inactive contexts. Also: closed-loop `inferred_eval_game` route_acc
was low (0.24) under 3 contexts — expected to resolve once proliferation is fixed.
Smoke throughput low (384 sps) due to `resolve_every=4` + per-env eval loop; real
pilot uses `resolve_every=20`, `num_envs=32`.

### Orchestration decision
The core is a tightly-coupled refactor on a live GPU+WSL pipeline; subagents cannot
run the WSL GPU/pytest to self-verify, so I drive the implementation directly to keep
it coherent and runnable. I will use parallel agents (Workflow/Agent) for the genuinely
parallel, independently-checkable sub-tasks: an adversarial multi-lens review of the
new core modules before GPU commitment, and results analysis while runs execute.
Recorded so the choice is auditable.

## ~21:15 — Routing robustness: three bugs found and fixed via GPU smokes (SI→Seaquest)
The hard part of task-free is ONLINE routing under a plastic encoder. Iterated on a
60k-step smoke (worst case: least-trained encoder = smallest game separation):
1. **False split within game0** (3 ctx for 2 games): fixed-threshold novelty + the
   seeding baseline being over-tight (prototypes scored on their own frames). Fixes:
   adaptive per-context running-similarity baseline (self-calibrating); cheap
   per-rollout prototype refresh on KNOWN rollouts; warm-up grace (6 rollouts) so a
   new context's baseline calibrates on real gameplay frames.
2. **False merge of game1 into game0** (1 ctx for 2 games): the dev-scaled match band
   widened enough to swallow the cross-game similarity, and KNOWN-rollout stat/proto
   updates pulled ctx0 toward game1 (feedback loop). Fixes: **cap the match-band drop**
   (`known_cap=0.12`) so a clear similarity drop always breaks the match; **anti-
   pollution guard** (only fold frames into a context when sim ≥ its running mean).
3. **Attribution at a latency-delayed switch**: detection took ~7 rollouts (thin margin
   on the weak smoke encoder), so the most-frequent context over the segment was the
   stale one; the end-of-segment resolve credited it. Fix: resolves attribute to the
   context active **since the last resolve** (settled/current), not the segment mode.

**Smoke5 (final) confirmed the mechanism end-to-end**: game0 stays 1 context (no
split); at game1, Seaquest scores active_sim=0.804 vs ctx0 within-SI mean 0.930 →
capped level 0.810 → **ALLOC ctx1** (2 contexts); held-out router top-1 = 1.0. The
margin is thin on the weak 60k encoder (0.006) but widens with training. All 7 manager
tests + full suite (72) still pass. Committed 11b1f17.

Sparse-exec benchmark (RTX 4090): blocked path 1.41→1.57ms over 1→57 contexts
(**6.7% overhead 5→57**, < 20% req); functional path 1.66→4.97ms (**200%**). Test 11 met.

## ~21:18 — Stage 1 launched (both GPUs)
- GPU1: task-free pilot SI→Seaquest, 500k/game, mem_rank 64, resolve_every 20,
  intervention boost_reset, eval 12 completed eps. (`castm_runs/taskfree/stage1_pilot`)
- GPU2 (sequential): matched 500k refs (SI, Seaquest) → naive control (--no-resolve).
  (`refs/`, `stage1_naive`)
At 500k the game0 encoder is strongly trained when game1 starts, so Seaquest
separation (and detection margin) should be far cleaner than the 60k smoke.
Parallel: adversarial review workflow on the new core modules; analyzer ready.
