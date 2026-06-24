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
