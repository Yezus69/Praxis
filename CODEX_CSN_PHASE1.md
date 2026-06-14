# Codex task — CSN-PPO Phase 1 (pure-functional core + unit tests)

Read `AGENTS.md` and `CSN_PPO_README.md` IN FULL first. This task builds **Phase 1** = the
env-agnostic, pure-functional core of CSN-PPO (NO Brax/MJX/env/training-loop dependency — that is a
later phase). Everything must match the README formulae EXACTLY.

Create the package at the repo ROOT: **`agent/csn_ppo/`**. Implement these files:

## `agent/csn_ppo/__init__.py`
Package marker; may re-export the main public symbols.

## `agent/csn_ppo/config.py`
`CSNPPOConfig` dataclass EXACTLY per **README §24** — every field, every default value, unchanged.

## `agent/csn_ppo/gradient_projection.py`  (README §9, §10, §11)
- `tree_dot`, `tree_add_scaled`, `tree_scalar_mul`, `tree_add` — verbatim per §10.
- `project_conflicting_gradient(g_ppo, memory_grads, eps=1e-8)` — per §11. The math is §9:
  `g_safe = g_ppo - (min(0, g_ppo·g_mem) / (||g_mem||²+eps)) · g_mem`, applied per bucket in a loop.
- `combine_safe_and_guard_grads(g_safe, memory_grads, memory_coefs)` — per §11.

## `agent/csn_ppo/guarded_loss.py`  (README §6, §7, §8)
- `gaussian_kl(mean0, logstd0, mean1, logstd1)` — the diagonal-Gaussian KL EXACTLY per §7 formula
  (var0=exp(2·logstd0); kl_per_dim = 0.5·((var0+(mean0-mean1)²)/(var1+1e-8) - 1 + 2·(logstd1-logstd0));
  return sum over last axis). KL[N0 || N1].
- `memory_guard_loss(params, normalizer_params, memory_batch, apply_policy_value)` — per §8:
  hinge policy loss `mean(w·relu(kl-kl_budget)²)` where kl = gaussian_kl(teacher_mean, teacher_logstd,
  pred_mean, pred_logstd); hinge value loss `mean(w·relu(|pred_value-value|-value_budget)²)`; return
  `(policy_loss + 0.25·value_loss, metrics)` with the §8 metrics dict (memory/kl_mean, kl_p95,
  policy_violation_frac, value_violation_frac, policy_loss, value_loss). Use a `jnp.sort`-based
  percentile for kl_p95 (JIT-safe; the README §8 note).
- A bucketed-guard helper consistent with §11 (compute guard loss/grad per memory bucket). Buckets per
  §11 list. Keep the per-bucket structure; bucket assignment can use cluster_id/source_id from the batch.

## `agent/csn_ppo/memory.py`  (README §5, §18, §20)
- `BehavioralMemory`, `BehavioralMemoryBatch` dataclasses EXACTLY per §5 (all fields, shapes).
- `init_behavioral_memory(capacity, obs_dim=27, action_dim=2)` → zero-initialized BehavioralMemory.
- `insert_atoms(memory, atoms)` — ring-buffer insertion EXACTLY per §5 (write_idx/size update, modulo).
- `sample_memory(memory, rng, batch_size)` — EXACTLY per §5.
- `should_insert_slow_memory(criticality, threshold=3.0)` — per §20.
- An `age_memory(memory)` helper that increments `age` (per §4/§5 the atom has an age field).

## `agent/csn_ppo/synthetic_probes.py`  (README §16, §17, §18, §19)
- `pack_contract_obs(goal, agent_vel, obstacles, mask)` — §16 (returns 27-D float32).
- `sort_and_pad_obstacles(obstacles, max_k=4)` — §16 (sort by distance, pad to K=4). Provide the
  JIT-safe fixed-size approach the README §16 recommends.
- `make_probe_no_obstacle(rng)` — §16 (exact).
- `make_probe_blocked_path(rng)` — §16 (exact).
- `analytic_no_obstacle_teacher(obs, speed=1.0)` — §17 (a* = clip(k·g/||g||, [-1,1])).
- `analytic_obstacle_teacher(obs)` — §17 (evade perpendicular to blocking obstacles).
- `make_teacher_distribution(action_mean, logstd=-2.0)` — §17.
- Criticality features §19: `obstacle_distances(obs)`, `collision_proximity(obs, radius=0.75)`,
  `success_proximity(obs, goal_radius=0.5)`, `dynamic_obstacle_score(obs)`, and
  `criticality_score(obs, advantage_abs, novelty, sentinel_failure)` (the exact weighted sum + clip(0.1,10)).
- Memory budget formulas §18: helpers for `memory_weight = clip(c, w_min, w_max)`,
  `kl_budget = δ0/(1+c)`, `value_budget = ρ0/(1+β·c)`.

## Unit tests (README §33) — `tests/`
Implement the §33 tests EXACTLY, and FLESH OUT the two `pass` stubs with real assertions:
- `tests/test_csn_gradient_projection.py` — `test_projection_removes_conflict`,
  `test_projection_leaves_non_conflict_alone` (§33).
- `tests/test_csn_guarded_loss.py` — `test_gaussian_kl_zero_for_identical_distributions`,
  `test_guard_loss_zero_inside_budget` (current==teacher, KL & value error below budget ⇒ guard loss ≈ 0),
  `test_guard_loss_positive_outside_budget` (teacher far from current beyond budget ⇒ guard loss > 0).
  For these two, construct a tiny `apply_policy_value` stub (returns fixed mean/logstd/value arrays) and a
  small BehavioralMemoryBatch.
- `tests/test_csn_synthetic_probes.py` — `test_probe_obs_shape` (==(27,)), `test_probe_masks_valid`
  (mask ∈ {0,1}), plus a `gaussian_kl`-consistency check and a `criticality_score` monotonicity check
  (higher collision_proximity ⇒ higher criticality).
- `tests/test_csn_memory.py` — insert/sample roundtrip; ring-buffer wrap (insert > capacity wraps
  write_idx and caps size at capacity); sampled batch shapes/fields correct.

## Constraints
- 27-D obs contract per §1. action ∈ [-1,1]^2. K=4.
- JIT-safe: fixed shapes, no Python lists/dynamic shapes in jitted paths (§35.6).
- Do NOT touch the existing `praxis/` coverage code.
- You cannot run jax here — AST-check every new file (`python -c "import ast; ast.parse(...)"`). The
  human runs pytest in WSL.

## Deliverable
Implement all files above. Print a summary mapping each function → README section + formula, list the
test cases, and confirm AST checks passed for every file.
