# CSN-PPO build tracker (conductor: Claude; coder: Codex gpt-5.5 xhigh)

Source of truth: `CSN_PPO_README.md`. Build rules: `AGENTS.md`. Each phase: Codex implements →
Claude runs tests/verifies math in WSL → math-fidelity review → next phase.

NOTE (env contract): README targets the 27-D goal-reaching nav contract; repo's live env is the
28-D coverage task. Algorithm core (Phases 1–3 math) is env-agnostic and built to the 27-D contract
per spec. End-to-end training (Phase 1b train loop + acceptance §31) needs a 27-D nav env — resolve
restore-nav-env vs adapt at integration.

| Phase | Scope (README §) | Files | Status |
|-------|------------------|-------|--------|
| 1a | pure-functional core + unit tests (§4–11,16–20,24,33) | config, memory, guarded_loss, gradient_projection, synthetic_probes + 4 tests | **DONE — 12/12 tests pass; math audit GO (0 mismatches, all formulae exact)** |
| 1b | forked PPO train loop w/ guard + projection + holdout early-stop (§21,§25,§27,§28,§30) | train.py, train_csn_ppo.py, metrics.py, rollout_mining.py | pending |
| 2 | sentinel bank: fixed-seed eval, regression detect, failed-state mining (§13) | sentinel.py + test_csn_sentinel.py | pending |
| 3 | mosaic teacher: per-cluster champions, labeling, update rule (§14) | mosaic_teacher.py | pending |
| 4 | curriculum mixture 70/20/10 + freeze-on-regression (§22) | curriculum.py | pending |
| 5 | JIT boundaries, metric cleanup, checkpoint meta, smoke + 100M launch config (§32.5) | (cross-cutting) | pending |

Acceptance gates (§31): A–I. Hard metric gates (§30). Math-fidelity is non-negotiable: every
formula in code must equal the README equation.
