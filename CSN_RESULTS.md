# CSN-PPO results log

## Phase 1b killer experiment #1 (2026-06-14) — POSITIVE but low operating point
10M steps, coverage env, seed 0, num_envs 2048, CSN config defaults (lr 3e-4, entropy 1e-2,
holdout early-stop ON). Same custom loop; only guard+projection toggled.

| run | config | peak | final | back-half(5-10M) avg | verdict |
|-----|--------|------|-------|----------------------|---------|
| csn_abl | `--no-guard --no-projection` | 0.40 @0.8M | 0.226 | ~0.26 | COLLAPSE (machinery off) |
| csn_full | guard+projection+champion | 0.42 @1.6M | 0.347 | ~0.34 | collapse DAMPENED + recovers |

**Signal:** CSN machinery measurably dampens the collapse: ablation falls 0.40→0.226 (monotonic),
full holds ~0.34 and RECOVERS at the end (0.296@8.2M → 0.347@9.8M = champion ratchet pulling back up).
First evidence the anti-forgetting mechanism works.

**Problem:** operating point is LOW (peak 0.40 vs baseline base10m 0.82). Ablation is equally low ⇒
NOT the CSN machinery — it's the loop/params. Suspects: (a) CSN uses lr 3e-4 / entropy 1e-2 vs the
0.82-peak base10m's lr 1.5e-4 / entropy 5e-3; (b) holdout early-stop (§21) may undertrain the peak.

## Diagnostic control (2026-06-14): it's the LOOP, not the params
Plain brax PPO (praxis.train) at lr 3e-4/entropy 1e-2 (CSN's params): peak **0.888 @1.3M** → 0.20 @13M.
So those params give a HIGH peak with plain PPO. But the CSN custom loop (csn_abl, same params) capped at
0.40. ⇒ the custom loop SUPPRESSES the peak ~2x. Suspect: holdout early-stop (§21) with target_kl=0.03
stops PPO epochs after ~1 of 4 (large epoch-1 KL) ⇒ undertrains. Params are fine; fix the loop.

## CORRECTED diagnosis (2026-06-14): the cap is DETERMINISTIC EVAL, not holdout/params
no-holdout runs (csn_noho_abl/full, --no-holdout-early-stop) STILL capped ~0.35-0.40 ⇒ holdout was
NOT the cap. The custom loop's Evaluator uses deterministic=True (train.py:416) = GREEDY policy;
baseline evals STOCHASTICALLY. For this task exploration-noise hugely helps coverage, so greedy eval
undercounts ~2x. Tell: init coverage 0.088 (custom, greedy) vs 0.163 (baseline, stochastic) for a
RANDOM policy. So custom 0.40 ≈ stochastic 0.8. Loop is likely FINE; eval mode is the confound.
Relative CSN comparison (same eval) still valid: holdout-ON full 0.34 > abl 0.26 (+recovery); but
holdout-OFF flipped (full 0.18 < abl 0.246) ⇒ benefit not yet robust; re-judge under stochastic eval.

## ROOT CAUSE of the peak cap (2026-06-14, deep debug): holdout KL-gate truncates PPO epochs
Custom-loop data path (rollout/GAE/minibatch/loss) verified byte-equivalent to brax. The cap is
`should_stop_epoch` (metrics.py) stopping when approx_kl > 1.5*target_kl (~0.045): early PPO exceeds
this after epoch 1, so the loop does ~32 SGD steps/rollout instead of 128 + rolls back → undertrains
the peak (0.44 vs brax 0.888). Brax with lr_schedule='none' NEVER KL-early-stops. Stochastic-eval
runs (holdout ON): csn_stoch_full peak 0.44 / back-half ~0.33; csn_stoch_abl 0.42 / ~0.30 — both
capped; full marginally > abl. FIX: disable the kl_bad gate (keep holdout_score + memory_kl stops).
Secondary: holdout overhead miscounts env_steps ~0.8x (slowdown, not cap).

## KL-gate hypothesis REFUTED (2026-06-14)
csn_validate_loop (--no-holdout-early-stop --no-guard --no-projection, STOCHASTIC eval = pure PPO,
no gate, no machinery) STILL caps: 0.145→0.41@0.4M→declining to 0.31@2.5M. So the early-stop was
NOT the cap. The custom loop peaks ~3x FASTER than brax (0.4M vs 1.3M), ~2x LOWER, collapses earlier
= signature of OVER-TRAINING (too many optimizer updates per env-step) or a rollout/normalizer bug.
## ACTUAL ROOT CAUSE (2026-06-14, empirical): FROZEN eval normalizer — an EVAL bug, not training
The custom loop's eval_policy_fn (train.py:413-417) CLOSES OVER the initial normalizer_params; brax's
Evaluator jits the eval policy at construction, baking the count=0 IDENTITY normalizer in as a constant.
Training updates normalizer_params each step, but eval never sees it ⇒ the trained policy is evaluated
on RAW unnormalized obs (trained on normalized) ⇒ measured coverage capped ~0.41. Verified empirically
(JIT closure-freeze reproduced; identity normalizer returns raw obs). Over-training/update-ratio
hypothesis REFUTED (custom 1600:1 ≈ brax 1280:1 env:opt steps). **The policy trains FINE; only eval was
broken** — which is why the cap survived every machinery/holdout/eval-mode toggle (none touch the eval
normalizer path). ⇒ ALL prior CSN numbers were measured through a broken lens; the real curves are unknown.
FIX (mirror brax): pass current normalizer as a runtime arg to run_evaluation, not a closure.

## After normalizer fix (2026-06-14) — loop EXONERATED; guard caps the peak (tuning, not a bug)
| run | config | peak | final | back-half avg | note |
|-----|--------|------|-------|---------------|------|
| csn_fix_abl | machinery OFF | **0.832** @1.2M | 0.34 | ~0.40 | ≈ brax 0.888 ⇒ LOOP IS FINE |
| csn_fix_full | guard+proj+champion | 0.66 @1.2M | 0.41 (recovers) | ~0.41 | guard CAPS peak; recovers late |

Normalizer fix raised peak 0.41→(abl)0.83/(full)0.66. The ablation reaching 0.83 proves the custom
loop ≈ brax. The CSN machinery LOWERS the peak (0.66 vs 0.83): the guard anchors to an early champion
before the climb finishes (chicken-and-egg: guard holds policy → can't improve → champion stuck).
Net retention ≈ wash (both back-half ~0.40) BUT full recovers late (0.41) while abl declines (0.34).
The anti-forgetting machinery DOES engage (real recovery); default hyperparams trade peak for it.
FIX (tuning, post-build): delay/soften the guard so policy reaches ~0.83 first, then champion locks it —
raise min_memory_size_before_guard and/or guard_kl_budget; expose them as CLI flags. NOT an
implementation bug. CSN is correctly implemented; this is hyperparameter tuning.

**Build priority:** finish all README phases (sentinel/curriculum/full-mosaic/tests/100M/math-audit)
FIRST (primary deliverable = complete correct math-aligned spec), then guard-tuning to demonstrate
peak preservation.
(2) expose --learning-rate/--entropy-cost/--discounting + --[no-]holdout-early-stop in train_csn.py
(Codex). (3) rerun csn_full vs csn_abl at the base10m operating point (lr 1.5e-4, entropy 5e-3) so
the preserved coverage is ~0.8, not ~0.35. Baseline reference: base10m 0.82→0.23 (see WALL1_RESULTS.md).
