# CSN-PPO build tracker (conductor: Claude; coder: Codex gpt-5.5 xhigh)

Source of truth: `CSN_PPO_README.md`. Build rules: `AGENTS.md`. Each phase: Codex implements →
Claude runs tests/verifies math in WSL → math-fidelity review → next phase.

DECISION (env target): CSN-PPO targets the **28-D COVERAGE env** (user: solve forgetting+overfitting
on the current task, where the collapse was measured). The verified env-agnostic core (memory,
guarded_loss, gradient_projection, config) is REUSED UNCHANGED. Only obs-coupled parts adapt to the
28-D coverage obs [agent_feat 0:4, obstacles 4:20 (4x4), mask 20:24, frontier 24:27, covered 27]:
criticality features (§19), synthetic probes (§16) → coverage probes, teacher = policy-at-mining-time
(mosaic best-policy teacher arrives Phase 3; no goal-reaching analytic teacher). All README MATH/
STRUCTURE kept exact; only obs plumbing changes. Killer test: does CSN-PPO prevent the 0.82→0.27
coverage collapse over 10M steps vs the baseline.

| Phase | Scope (README §) | Files | Status |
|-------|------------------|-------|--------|
| 1a | pure-functional core + unit tests (§4–11,16–20,24,33) | config, memory, guarded_loss, gradient_projection, synthetic_probes + 4 tests | **DONE — 12/12 tests pass; math audit GO (0 mismatches, all formulae exact)** |
| 1b | CSN loop on COVERAGE: guard+projection+minibatched holdout early-stop + **champion teacher** (pulled fwd from P3 — adversarial review found the §36 MVP FAILS without it: memory turnover 0.87M < collapse-onset 1.3M ⇒ guard follows policy down) | criticality_coverage, coverage_probes, train, metrics, rollout_mining, mosaic_teacher(min), praxis/train_csn.py | **BUILT — runs end-to-end (smoke clean, no NaN, ~13-15k steps/s); C1 champion + C2 minibatch reviewed PASS; 3 bug-classes fixed (jax.random API, pytree-register memory, jit boundaries). 10M killer exp running. WATCH: early cov ~0.39 (loop/params underperform baseline 0.82 — lr/entropy differ + holdout early-stop); analyzing full curves.** |
| 2 | sentinel bank: fixed-seed eval, regression detect, failed-state mining (§13) | sentinel.py + test_csn_sentinel.py | pending |
| 3 | mosaic teacher: per-cluster champions, labeling, update rule (§14) | mosaic_teacher.py | pending |
| 4 | curriculum mixture 70/20/10 + freeze-on-regression (§22) | curriculum.py | pending |
| 5 | JIT boundaries, metric cleanup, checkpoint meta, smoke + 100M launch config (§32.5) | (cross-cutting) | pending |

Acceptance gates (§31): A–I. Hard metric gates (§30). Math-fidelity is non-negotiable: every
formula in code must equal the README equation.
