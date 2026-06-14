# Codex task — CSN-PPO Phase 1b (custom training loop on the 28-D coverage env)

Read `AGENTS.md`, then `CSN_PPO_README.md` (algorithm spec), then `CODEX_CSN_PHASE1B_SPEC.md`
(the detailed, brax-grounded implementation spec). Implement the training loop per
`CODEX_CSN_PHASE1B_SPEC.md` — BUT the **MANDATORY CORRECTIONS** below OVERRIDE that spec wherever
they conflict. An adversarial review found the un-corrected spec would FAIL the killer test (the
guard would follow the policy into collapse) and be uninterpretable (wrong optimizer dynamics).
Apply ALL corrections.

The Phase 1a core (`agent/csn_ppo/{config,memory,guarded_loss,gradient_projection,synthetic_probes}.py`)
is DONE and formula-verified — REUSE the env-agnostic parts UNCHANGED (gaussian_kl, memory_guard_loss,
project_conflicting_gradient, combine_safe_and_guard_grads, BehavioralMemory/insert/sample). Only obs-
coupled coverage pieces are new.

================================================================================
## MANDATORY CORRECTIONS (override the spec; ordered by impact)
================================================================================

### C1 — CHAMPION TEACHER (THE ACTUAL FIX; without it CSN follows the policy DOWN into collapse)
The base spec sets `teacher = current policy at mining time` + a plain ring buffer. Memory turnover
(~0.87M steps for slow memory) is FASTER than collapse onset (~1.3M steps), so the guard would anchor
to already-drifted behavior — a moving trust region, not a fixed anchor. Bring forward a **minimal
1-cluster mosaic/champion teacher** (README §14, §35-#3):
- Maintain a frozen `champion` = {normalizer_params, policy params, value params} snapshot, plus
  `champion_best_coverage` (float, init -inf) and a `champion_wins` counter.
- After each periodic deterministic eval (reuse the eval that already runs; eval metric
  `eval/episode_coverage`), if `eval_coverage > champion_best_coverage + champion_min_margin`
  (margin 0.02) increment `champion_wins`, else reset to 0. When `champion_wins >= champion_patience`
  (2), set `champion := current snapshot`, `champion_best_coverage := eval_coverage`, `champion_wins := 0`.
- **Teachers for mined atoms AND probe atoms are computed with `champion` params** (via
  `apply_policy_value(champion.normalizer, champion_policy_value_params, obs)`), NOT the live policy.
  Before any champion exists (early training), fall back to the current policy.
- This makes memory a RATCHET that preserves best-so-far (e.g. peak-0.82) behavior so the guard can
  actually anchor against the drift. Add config: `champion_min_margin=0.02`, `champion_patience=2`,
  `champion_eval_interval` (reuse the existing eval cadence; e.g. every N updates so there are ~20 evals
  over the run). Snapshot the champion params on the host (these are small MLPs).

### C2 — MINIBATCH THE PPO GRADIENT (required for a valid baseline comparison)
Do NOT take one full-batch PPO grad per epoch. Match the baseline brax dynamics: per epoch, shuffle
`train_data` into `num_minibatches` minibatches; for EACH minibatch: compute PPO grad + the (bucketed)
guard grads, `project_conflicting_gradient` (§9/§11), `combine_safe_and_guard_grads`, then
`optimizer.update` + `optax.apply_updates`. That is `max_updates_per_batch * num_minibatches` SGD
steps per rollout (≈128), not 4. Evaluate holdout/early-stop ONCE per epoch AFTER the minibatch scan.

### C3 — SMOKE DIVISIBILITY
The entry point / `--smoke` must choose `(batch_size, num_minibatches)` so
`(num_envs*unroll_length) % (batch_size*num_minibatches) == 0`. For a smoke with `num_envs=256, unroll=20`
(lhs=5120) use `batch_size=128, num_minibatches=40` (or `256, 20`). Auto-derive or hard-code a known-good
smoke config; do NOT leave it to a note. Keep the config-time assert.

### C4 — API CORRECTNESS
`PPONetworkParams` is a `flax.struct` dataclass: use `.replace()`, `.policy`, `.value` — NEVER
`._replace`/`._fields`. The PPO approx-KL metric key is **`'kl_mean'`** (NOT `'ppo/approx_kl'`)
everywhere (compute_ppo_loss returns: total_loss, policy_loss, v_loss, entropy_loss, kl_mean, ...).
Confirmed-correct (keep): pre-tanh mean = `dist.loc`, std = `dist.scale` (= softplus(split[1])+0.001),
so `logstd = log(dist.scale)`; `apply_policy_value` passes normalizer_params FIRST to network.apply;
guard loss == 0 when teacher==current; surrogate = `-policy_loss` (higher better).

### C5 — EVALUATOR CONSTRUCTION
Remove any `if False else`. Pass `params` (a `PPONetworkParams`) as `policy_params` to
`run_evaluation`; `eval_policy_fn = lambda p: make_inference_fn(net)((normalizer_params, p.policy, p.value))`
closing over `normalizer_params`. Use `deterministic=True` for sentinel/champion eval. Verify it emits
`eval/episode_coverage` (CoverEnv emits a per-step `coverage` delta brax SUMS).

### C6 — PERFORMANCE (required for the 10M run)
Do NOT run 7 eager Python-loop guard-grad backward passes per epoch (intractable). JIT the whole update
step; compute the bucketed guard gradient with a single vmapped/stacked-mask pass (or jit the per-bucket
body). Drop buckets that are empty pre-Phase-2 (the sentinel-regression `source` bucket is always empty
now). Keep the projection over the remaining (populated) buckets.

### C7 — PROTECTIVE SLOW MEMORY
Only insert an atom into SLOW memory when `criticality > slow_memory_threshold` (don't advance the slow
ring with zero-weight atoms — that evicts peak-era high-criticality atoms). Fast memory keeps the plain
ring. Probe teachers use `champion` params (per C1).

================================================================================
## ACCEPTANCE
================================================================================
- AST-check every new/edited file (`python -c "import ast; ast.parse(...)"`) — you cannot run jax here.
- The human runs in WSL: (a) a SHORT smoke (no NaN; logs `memory/kl_p95`, `ppo/holdout_surrogate`,
  `memory/guard_loss`, `eval/episode_coverage`; champion updates at least once), then (b) a 10M-step run
  vs the baseline collapse curve.
- Files: `agent/csn_ppo/criticality_coverage.py`, `agent/csn_ppo/train.py`, `agent/csn_ppo/metrics.py`,
  `agent/csn_ppo/rollout_mining.py`, `agent/csn_ppo/mosaic_teacher.py` (minimal champion), coverage probe
  additions, and an entry point `praxis/train_csn.py` (argparse mirroring praxis/train.py: --num-timesteps,
  --num-envs, --num-evals, --seed, --run-name, --smoke, plus CSN toggles --no-guard/--no-projection so we
  can ablate). Reuse Phase-1a core unchanged.

## Deliverable
Print a summary: files created/changed; how C1 (champion teacher) and C2 (minibatching) are implemented
(cite the functions); confirmation AST checks pass; and any deviations/assumptions.
