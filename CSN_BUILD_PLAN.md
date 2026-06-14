# CSN-PPO build tracker — FINAL STATUS (2026-06-14)

Conductor: Claude (orchestration/review). Coder: Codex CLI (gpt-5.5, xhigh).
Spec: `CSN_PPO_README.md`. Build rules: `AGENTS.md`. Results: `CSN_RESULTS.md`.
Target env: **28-D coverage** (not the README's 27-D nav). Env-agnostic core reused verbatim; obs-coupled
parts adapted to coverage; all README MATH kept exact (audited MATH_OK).

## Status: all 8 CSN mechanisms implemented + math-verified; goal demonstrated.

| Phase | README § | Module(s) | Status |
|-------|----------|-----------|--------|
| 1a core | 4–11,16–20,24 | config, memory, guarded_loss, gradient_projection, synthetic_probes | ✅ DONE, tested, MATH_OK |
| 1b loop | 21,27,28 | train.py (PPO+guard+projection+holdout+champion), metrics, rollout_mining, criticality_coverage, coverage_probes, mosaic_teacher(min) + praxis/train_csn.py | ✅ DONE, runs, **demonstrably reduces forgetting** |
| 2 sentinel | 13 | sentinel.py + test | ✅ DONE + loop-integrated (flag `--enable-sentinel`, default off) |
| 3 mosaic | 14 | mosaic_teacher.py (per-cluster) + test | ✅ DONE (minimal champion in core loop; per-cluster in sentinel block) |
| 4 curriculum | 22 | curriculum.py + test | ✅ module DONE+tested; ⚠️ loop-integration pending (needs cover_env difficulty hook) |
| 5 opt/tests | 32.5,33 | — | ✅ 26 unit tests green; smoke tests pass; MATH_OK audit; JIT boundaries in place. 100M cmd below |

## Headline result (CSN solves much of the forgetting)
5M, identical config: plain PPO collapses 0.83→**0.31** (retains 38%); CSN-PPO holds 0.83→**0.567**
(retains 68%), gap grows to +0.25 (+81%). Plot: `runs/csn_compare.png`. The memory hinge-KL guard +
champion anchor + nullspace projection roughly HALVE catastrophic forgetting. Math: MATH_OK vs README.

## Math fidelity: MATH_OK
10-auditor adversarial audit, 0 mismatches across all 9 modules (gaussian_kl §7, hinge guards §6/§8,
projection §9-11, memory §5, criticality+budgets §18/§19, sentinel §13, curriculum §22, champion §14,
PPO+holdout §21/§28). Coverage adaptations are intentional & form-preserving.

## Key knobs (praxis/train_csn.py)
`--guard-warmup-steps` (delay guard past the peak so the champion captures the good policy — the fix for
the guard capping the peak), `--guard-kl-budget` (active engagement lever), `--guard-lambda-mem`,
`--[no-]guard`, `--[no-]projection`, `--[no-]holdout-early-stop`, `--[no-]deterministic-eval`,
`--enable-sentinel`, `--learning-rate/--entropy-cost/--discounting`. (Dead flag `--guard-policy-coef`:
unused; the §9 combine uses guard_lambda_mem.)

## 100M launch (§26/§32.5)
wsl -d praxis -u root -- bash -c 'cd /root/praxis; export PYTHONPATH=/root/praxis LD_LIBRARY_PATH=/usr/lib/wsl/lib CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1; /opt/venv/bin/python -m praxis.train_csn --num-timesteps 100000000 --num-envs 2048 --num-evals 50 --seed 0 --guard-warmup-steps 1500000 --guard-kl-budget 0.003 --enable-sentinel --run-name csn_100m'

## ONE remaining item for 100% spec coverage
Curriculum (§22) loop-integration: curriculum.py is built + unit-tested + MATH_OK, but the live loop does
not yet sample world difficulties into the env, because cover_env.py needs a difficulty hook (scale
obstacle speed/amplitude/frac_moving). That's a cover_env change + an --enable-curriculum gate in the loop
(mirror the sentinel integration pattern). Everything else is implemented, integrated, and verified.
