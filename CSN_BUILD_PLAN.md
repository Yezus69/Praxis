# CSN-PPO build tracker — FINAL STATUS (2026-06-14)

Conductor: Claude (orchestration/review). Coder: Codex CLI (gpt-5.5, xhigh).
Spec: `CSN_PPO_README.md`. Build rules: `AGENTS.md`. Results: `CSN_RESULTS.md`.
Target env: **28-D coverage**. This is not the original 27-D navigation contract. Env-agnostic core reused
verbatim; obs-coupled parts adapted to coverage; all README MATH kept exact (audited MATH_OK).

Current CSN-PPO implementation target:
- 28-D coverage/exploration task
- no goal-reaching reward
- collisions are non-terminal by default
- metric is coverage retention, not success-rate retention

Operational rule: sentinel required for long runs.

P0-P8 hardening now exists: sentinel mandatory for long runs, per-cluster mosaic teachers, adaptive guard
pressure, curriculum mixture, validation bank, stratified memory, `--long-run` preset, and guard-KL
conditioning.

To use CSN-PPO for 27-D navigation:
1. replace coverage criticality with nav criticality,
2. replace coverage probes with nav probes,
3. use success/collision sentinels instead of coverage/collision sentinels,
4. restore goal-relative observation contract,
5. label synthetic probes using goal-directed analytic teacher.

## Status: all 8 CSN mechanisms implemented + math-verified; goal demonstrated.

| Phase | README § | Module(s) | Status |
|-------|----------|-----------|--------|
| 1a core | 4–11,16–20,24 | config, memory, guarded_loss, gradient_projection, synthetic_probes | ✅ DONE, tested, MATH_OK |
| 1b loop | 21,27,28 | train.py (PPO+guard+projection+holdout+champion), metrics, rollout_mining, criticality_coverage, coverage_probes, mosaic_teacher(min) + praxis/train_csn.py | ✅ DONE, runs, **demonstrably reduces forgetting** |
| 2 sentinel | 13 | sentinel.py + test | DONE + loop-integrated; default enabled and required for long runs unless debug-bypassed |
| 3 mosaic | 14 | mosaic_teacher.py (per-cluster) + test | DONE; per-cluster teacher selection used for rollout/probe/sentinel-failure labeling |
| 4 curriculum | 22 | curriculum.py, env_wrappers.py, cover_env.py + test | DONE; live reset/autoreset path samples difficulty mixture into CoverEnv |
| 5 opt/tests | 32.5,33 | validation, stratified memory, long-run preset, guard-KL conditioning | DONE; smoke tests pass; MATH_OK audit; JIT boundaries in place. 100M cmd below |

## Headline result (CSN solves much of the forgetting)
5M, identical config: plain PPO collapses 0.83→**0.31** (retains 38%); CSN-PPO holds 0.83→**0.567**
(retains 68%), gap grows to +0.25 (+81%). Plot: `runs/csn_compare.png`. The memory hinge-KL guard +
champion anchor + nullspace projection roughly HALVE catastrophic forgetting. Math: MATH_OK vs README.

## Math fidelity: MATH_OK
10-auditor adversarial audit, 0 mismatches across all 9 modules (gaussian_kl §7, hinge guards §6/§8,
projection §9-11, memory §5, criticality+budgets §18/§19, sentinel §13, curriculum §22, champion §14,
PPO+holdout §21/§28). Coverage adaptations are intentional & form-preserving.

## Key knobs (praxis/train_csn.py)
`--long-run`, `--guard-warmup-steps` (delay guard past the peak so the champion captures the good policy),
`--guard-kl-budget` (active engagement lever), `--guard-lambda-mem`, `--[no-]guard`,
`--[no-]projection`, `--[no-]holdout-early-stop`, `--[no-]deterministic-eval`, `--enable-sentinel`,
`--allow-no-sentinel-for-debug`, `--learning-rate/--entropy-cost/--discounting`.

## 100M launch (§26/§32.5)
wsl -d praxis -u root -- bash -c 'cd /root/praxis; export PYTHONPATH=/root/praxis LD_LIBRARY_PATH=/usr/lib/wsl/lib CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1; /opt/venv/bin/python -m praxis.train_csn --long-run --seed 0 --run-name csn_100m'

## Remaining validation item
P0-P8 implementation hardening is complete in code. The remaining work is the P10 multi-seed 100M
acceptance protocol and plots, not another algorithmic hardening pass.
