# WALL 1 (PPO catastrophic forgetting) — experiment log

Task: long PPO runs collapse (coverage peaks then regresses). Goal: coverage rises and
STAYS high through 10M steps, avg eval coverage > 0.9. All runs: 10M steps, num_envs=2048,
num_evals=21, lr=1.5e-4, entropy_cost=0.005, seed=0, ~6 min each on a 4090.
Code: praxis/train.py with opt-in stability flags (Codex-implemented, review GO).

## Results

| Run | Config | Peak cov | Final cov | Verdict |
|-----|--------|----------|-----------|---------|
| R0 (base10m) | baseline, stochastic eval | 0.82 @1.3M | 0.23 | COLLAPSE (control) |
| R0c | baseline, **deterministic eval** | 0.72 @1.3M | 0.21 | COLLAPSE — real, NOT eval-noise artifact |
| R1 | `--lr-schedule adaptive_kl` (desired_kl 0.01) | **0.99** @2.0M | 0.25 | higher peak, STILL collapses → KL control insufficient |
| R4 | `--no-normalize-advantage` | 0.82 @1.3M | 0.19 | COLLAPSE (faster) — hypothesis REFUTED |
| R4b | `--no-normalize-advantage` + adaptive_kl | 0.97 @1.3M | 0.16 | COLLAPSE (hard) |

## CONCLUSION: the collapse is STRUCTURAL (reward), not a PPO-knob problem
5/5 runs collapse regardless of LR, KL, entropy, grad-norm, advantage-norm. Every policy
regresses to ~0.16–0.23 = random-policy coverage level. Loss data shows zero advantage signal
at convergence. ROOT CAUSE: the coverage reward is a one-shot, non-renewable bonus; once the
arena is covered there is NO sustained advantage to MAINTAIN covering, while the −0.1 collision
penalty is a persistent gradient toward PASSIVITY. On-policy PPO therefore drifts the policy to
"stop moving" (which also minimizes collisions). adaptive_kl raises the peak (0.82→0.99) but
can't stop cumulative drift; no PPO knob can fix a degenerate reward optimum.

→ FIX must be on the reward/env side (Track A) and/or via a general anti-forgetting anchor
  (Track B, north-star). Pursuing Track A first to confirm the collapse is beatable.

## Track A (reward shaping) — ALSO collapses. Hypothesis refined again.
| Run | Config | Peak | Final | Verdict |
|-----|--------|------|-------|---------|
| A | `--terminate-on-full-coverage --k-complete 5 --k-coll 0.02` | 0.88 | 0.27 | COLLAPSE |
| B | `--patrol --k-fresh 0.2 --freshness-decay 0.98 --k-coll 0` | 0.94 | 0.12 | COLLAPSE |

Why A failed: full 36/36 coverage is rarely achieved (peak completion rate only 12.5%, avg_ep_len
stays ~600), so the termination almost never fires → A ≈ baseline.
Why B is inconclusive: B's frontier obs is computed from the MONOTONIC `visited` grid (=[0,0,0]
after first full pass), so the agent is BLIND to which cells are stale → the freshness reward is
partly unlearnable. B's reward fell WITH coverage (38→2) so it's genuine degradation, NOT hacking.

KEY NEW INSIGHT: the collapse is robust to BOTH optimizer AND reward changes, always peaks ~1.3M,
and the endpoint (~0.12–0.27) ≈ the random-policy coverage (step-0 = 0.16). The policy SHARPENS
(entropy ↓) onto a confident LOW-coverage behavior. Cause is something reward-independent +
step/competence-correlated. Loss telemetry: v_loss tiny+stable (critic tracks perfectly → zero
advantage for current policy → no restoring force), policy_loss≈0.

## Diagnostics running (single-variable, vs R0 baseline)
| Run | Tests | Status |
|-----|-------|--------|
| N | `--normalize-until-count 2000000` (freeze obs-normalizer) | 0.85→0.27 COLLAPSE — refuted |
| E | `--entropy-cost 0.0` (remove entropy pressure) | 0.84→0.28 COLLAPSE — refuted |
| render | collapsed policy CONFINES to a small patch, ~0 collisions, never explores | confirmed "confident passive" |

## Collapse is robust to EVERYTHING tried. Peak ALWAYS ~1.31M. Endpoint ~0.27 ≈ random.
Refuted causes: LR, KL-control, entropy (incl ZERO), grad-norm, advantage-norm, reward shaping
(terminate/patrol), obs-normalizer drift. The ONE major untested knob: the DISCOUNT FACTOR.

## Discounting hypothesis (γ=0.97 is myopic for a long-horizon task)
γ=0.97 → effective horizon ~33 steps, but covering 36 cells over a 600-step episode needs
long-horizon credit. Story: early high-entropy wandering covers cells incidentally; PPO then
converges to the MYOPICALLY-optimal policy (grab local reward, idle, avoid collisions) which has
low global coverage → peak-then-collapse. Reward/optimizer-independent ⇒ matches the invariance.
| Run | Config | Status |
|-----|--------|--------|
| D99 | `--discounting 0.99` (horizon ~100) | 0.87→0.27 COLLAPSE — refuted |
| D995 | `--discounting 0.995` (horizon ~200) | 0.75→0.26 COLLAPSE (peak hurt) — refuted |

## ROOT CHARACTERIZATION (from N train_metrics dynamics) — the real story
Sharp transition AT the 1.31M peak. BEFORE: kl≈0.033, policy_loss≈-0.02, v_loss≈0.02 (hot,
active optimization). AFTER (≥1.97M): kl≈0.007, policy_loss≈-0.001, v_loss≈0.0002 (everything
→0). The optimization CONVERGES exactly when coverage peaks.
⇒ The high-coverage peak is a TRANSIENT of the still-exploring policy. The CONVERGED fixed point
is a genuinely low-coverage policy (the confined wanderer in collapsed_rollout.mp4). Coverage is
produced by exploration "heat", not a learned deterministic sweep strategy. As PPO cools to its
fixed point, coverage falls to ~0.27. Once policy_loss≈0, the policy drifts there on noise.
This is why ALL 11 interventions failed: the converged fixed point itself is bad.

## 11 interventions refuted (do not re-run): LR, KL-adaptive-LR, entropy(incl 0), grad-norm,
## advantage-norm, terminate-on-coverage, patrol-reward, obs-normalizer-freeze, γ=0.99, γ=0.995.

## Strategic fork (awaiting user): (1) keep-best+early-stop [secures 0.87 now]; (2) curriculum
## to harder tasks [north-star: always a learning signal, path to 3D]; (3) anti-forgetting anchor
## [custom loop, reference-KL/replay]; (4) keep hunting (net size, updates_per_batch, lower LR).


## Diagnosis (from R1 train_metrics.csv loss trajectory)
At the 0.99 peak AND throughout the collapse, ALL losses are tiny & stable:
- `v_loss ≈ 0.0002`, flat → value function does NOT diverge (hypothesis #2 REJECTED).
- `policy_loss ≈ 0` → once solved, the true advantage signal vanishes.
- `kl_mean` held at ~0.011 by the controller → per-step trust region IS bounded.
- `entropy_loss` magnitude 0.0055 → 0.0013 → entropy quietly collapsing (~1.1 → ~0.27).

**Mechanism:** after the task is solved, real advantages ≈ 0, but brax default
`normalize_advantage=True` standardizes them to unit variance → PPO takes KL-bounded
**random-walk** steps on pure noise. Each step is small (KL~0.011) but millions of them
accumulate into a large drift away from the good policy; entropy regularization slowly
locks in the drifted (worse) behavior. The KL bound caps per-step size, not cumulative drift.
→ This is why the collapse survived LR, entropy, grad-norm, AND KL changes.

## Next decisions
- If R4/R4b hold coverage high: advantage-normalization-on-saturated-signal is THE cause;
  candidate production config = no-adv-norm (+ optionally KL control + value clip).
- If R4/R4b still collapse: the root is the reward structure (no sustained signal after
  coverage saturates) → reward shaping in cover_env.py (Codex task): sustained/denser
  coverage signal, or entropy_cost anneal-to-0, or KL-to-reference-policy regularization.
