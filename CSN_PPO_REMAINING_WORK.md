# CSN-PPO Remaining Build Work

**Purpose:** Convert the current CSN-PPO implementation from a promising short-run anti-forgetting prototype into a 100M+ step longitudinal optimizer that aggressively resists both catastrophic forgetting and overfitting.

**Target repo:** `Yezus69/Praxis`

**Current implementation target:** the live implementation currently targets the **28-D coverage/exploration task**, not the original 27-D goal-navigation contract from the first CSN-PPO README.

**Primary objective:** preserve peak learned behavior over long training runs while still allowing new behavior to improve.

---

## 0. Current state

The current repo already contains the important CSN-PPO core:

- fixed-size behavioral memory,
- hinge-KL policy/value guard loss,
- nullspace/conflict gradient projection,
- current-rollout holdout early stopping,
- global champion teacher,
- optional sentinel evaluation,
- synthetic coverage probes,
- standalone curriculum module,
- custom PPO loop.

This is a real start. The core math is mostly aligned with the CSN-PPO spec.

However, the implementation is **not yet 100M-step safe** because several critical pieces are incomplete or wired in a weaker form:

1. Sentinel failure states are currently at risk of being labeled by the **currently regressed policy**, not the best historical teacher.
2. Per-cluster mosaic teachers exist but are not used everywhere.
3. Sentinel evaluation is optional/default-off, even though long-run anti-forgetting requires it.
4. Curriculum mixture exists as a module but is not integrated into the live environment reset distribution.
5. Guard pressure is global/static, not adaptive by regressed cluster.
6. Synthetic probes are too sparse and lack an analytic safety/frontier teacher.
7. Holdout early stopping only protects against same-rollout overfitting, not closed-loop validation overfitting.
8. Memory size and replacement policy are probably too weak for 100M+ steps.
9. Long-run acceptance testing is not yet strong enough to claim that catastrophic forgetting and overfitting are solved.

The goal of this document is to give Codex/Claude Code an exact prioritized build plan.

---

# Priority 0 — Fix sentinel failure labeling

## Problem

When a sentinel cluster regresses, the system mines failed sentinel states into slow memory. That is correct.

The dangerous part is the teacher used to label those failed states.

If failed states are labeled by the **current policy**, and the current policy is the one that just regressed, the system can accidentally preserve the bad behavior that caused the sentinel failure.

Bad loop:

```text
policy regresses on sentinel cluster
→ failed states are mined
→ current regressed policy labels those states
→ memory preserves the failed behavior
→ guard protects the wrong action distribution
```

This is the highest-priority semantic bug.

## Desired behavior

When a sentinel cluster regresses, label mined failed states using the best available teacher:

1. best per-cluster champion policy,
2. else global champion policy,
3. else analytic safety teacher if available,
4. else skip insertion until a valid teacher exists.

Never label sentinel-failure atoms with the current regressed policy unless there is absolutely no alternative and the atom is explicitly marked as weak/untrusted.

## Files likely involved

```text
agent/csn_ppo/train.py
agent/csn_ppo/sentinel.py
agent/csn_ppo/mosaic_teacher.py
agent/csn_ppo/rollout_mining.py
agent/csn_ppo/memory.py
```

## Required implementation

Add a cluster-specific sentinel-failure mining path.

Pseudo-code:

```python
def label_failed_sentinel_atoms_with_best_teacher(
    sentinel_trajectories,
    regressions,
    champions,
    global_champion,
    current_params,
    current_normalizer,
    apply_policy_value,
    cfg,
):
    batches = []

    regressed = np.asarray(regressions["regressed"])

    for cluster_id in np.where(regressed)[0]:
        teacher_normalizer, teacher_params = mosaic_teacher.get_cluster_teacher(
            champions,
            cluster_id,
        )

        if teacher_params is None or teacher_normalizer is None:
            teacher_normalizer, teacher_params = mosaic_teacher.teacher_snapshot(
                global_champion,
                current_normalizer,
                current_params,
            )

        if teacher_params is None or teacher_normalizer is None:
            # No valid teacher yet. Do not preserve failed current behavior.
            continue

        cluster_atoms = sentinel.mine_failed_sentinel_states_for_cluster(
            failed_trajectories=sentinel_trajectories,
            cluster_id=cluster_id,
            regressions=regressions,
            teacher_params=teacher_params,
            teacher_normalizer=teacher_normalizer,
            apply_policy_value=apply_policy_value,
            cfg=cfg,
        )

        batches.append(cluster_atoms)

    return concat_memory_batches(*batches) if batches else None
```

Add or refactor this helper:

```python
def mine_failed_sentinel_states_for_cluster(
    failed_trajectories,
    cluster_id: int,
    regressions,
    teacher_params,
    teacher_normalizer,
    apply_policy_value,
    cfg,
) -> BehavioralMemoryBatch:
    ...
```

The teacher used for labels must be `teacher_params`, not `current params`.

## Acceptance tests

Add tests that force a regression and verify teacher provenance.

### Test 0.1 — failed sentinel labels do not use current policy

Create two dummy policies:

```text
current policy mean:  [0, 0]
champion policy mean: [1, 0]
```

Mine failed sentinel atoms.

Assert:

```python
assert allclose(failed_atoms.mean, champion_mean)
assert not allclose(failed_atoms.mean, current_mean)
```

### Test 0.2 — fallback to global champion

If no per-cluster champion exists but a global champion exists, labels must come from the global champion.

### Test 0.3 — no teacher means no insertion or zero-weight weak insertion

If neither per-cluster nor global champion exists, do not insert full-strength sentinel-failure atoms.

Expected behavior:

```python
failed_atoms is None
```

or:

```python
failed_atoms.weight == 0
failed_atoms.source_id == SOURCE_SENTINEL_FAILURE_UNTRUSTED
```

Preferred: skip insertion.

---

# Priority 1 — Use per-cluster mosaic teachers everywhere

## Problem

The repo has per-cluster mosaic champion structures, but the main rollout mining path still appears to rely on a global champion snapshot. This weakens the original mosaic teacher design.

The system should not preserve only one global best policy. It should preserve the best historical behavior per behavioral cluster.

Correct idea:

```text
cluster 0: collision-boundary behavior       → teacher = best collision-boundary champion
cluster 1: open/frontier exploration         → teacher = best exploration champion
cluster 2: dynamic-obstacle interaction      → teacher = best dynamic-obstacle champion
cluster 3: near-obstacle/non-collision path  → teacher = best near-obstacle champion
```

The live policy should distill the union of best historical behaviors.

## Desired behavior

Every atom should be labeled by the best teacher available for its cluster.

For each mined or synthetic observation:

```python
cluster_id = cluster_id_for(obs)
teacher = get_cluster_teacher(cluster_id)
label = teacher(obs)
```

Fallback order:

```text
per-cluster champion
→ global champion
→ current policy only before any champion exists
```

## Files likely involved

```text
agent/csn_ppo/train.py
agent/csn_ppo/rollout_mining.py
agent/csn_ppo/mosaic_teacher.py
agent/csn_ppo/coverage_probes.py
agent/csn_ppo/criticality_coverage.py
```

## Required implementation

Refactor label generation so that teacher selection is cluster-aware.

Add helper:

```python
def label_atoms_with_mosaic_teacher(
    obs: jnp.ndarray,                    # [B, obs_dim]
    cluster_id: jnp.ndarray,             # [B]
    champions,
    global_champion,
    current_params,
    current_normalizer,
    apply_policy_value,
    cfg,
) -> BehavioralMemoryBatch:
    ...
```

Implementation can be host-looped over a small fixed number of clusters. It does not need to be fully vectorized at first because teacher snapshots are host-side objects.

Pseudo-code:

```python
def label_atoms_with_mosaic_teacher(...):
    outputs = []

    for cid in range(cfg.num_clusters):
        mask = np.asarray(cluster_id) == cid
        if not mask.any():
            continue

        obs_c = obs[mask]

        teacher_norm, teacher_params = mosaic_teacher.get_cluster_teacher(
            champions,
            cid,
        )

        if teacher_params is None:
            teacher_norm, teacher_params = mosaic_teacher.teacher_snapshot(
                global_champion,
                current_normalizer,
                current_params,
            )

        if teacher_params is None:
            teacher_norm = current_normalizer
            teacher_params = current_params

        mean, logstd, value = apply_policy_value(
            teacher_params,
            teacher_norm,
            obs_c,
        )

        outputs.append(make_batch(obs_c, mean, logstd, value, cid, ...))

    return concat_memory_batches(*outputs)
```

Use this in:

```text
rollout_mining.mine_atoms
rollout_mining.label_probe_atoms
sentinel.mine_failed_sentinel_states
```

or call it from `train.py` after selecting candidate observations.

## Acceptance tests

### Test 1.1 — cluster-specific teacher selection

Create four dummy cluster teachers with different action means:

```text
cluster 0 → [1, 0]
cluster 1 → [0, 1]
cluster 2 → [-1, 0]
cluster 3 → [0, -1]
```

Create observations assigned to all four clusters.

Assert that atom labels match the correct cluster teacher.

### Test 1.2 — fallback order

For a missing cluster champion, assert fallback to global champion.

For missing global champion, assert fallback to current policy.

### Test 1.3 — rollout mining uses mosaic path

Add a regression test that fails if `mine_atoms` labels all clusters with the same global teacher when per-cluster teachers exist.

---

# Priority 2 — Make sentinels mandatory for long runs

## Problem

Sentinel evaluation is the closed-loop forgetting detector. Without it, the system only protects pointwise memory states and same-rollout holdout batches. That is not enough for 100M+ training.

Currently, sentinel can be disabled. That is acceptable for smoke tests but unsafe for serious long runs.

## Desired behavior

For long CSN-PPO training, sentinel should be mandatory.

Recommended rule:

```python
if config.num_timesteps >= 10_000_000 and not config.enable_sentinel:
    raise ValueError(
        "Long CSN-PPO runs require --enable-sentinel. "
        "Use --allow-no-sentinel-for-debug only for ablations."
    )
```

Alternative: set `enable_sentinel=True` by default.

## Files likely involved

```text
agent/csn_ppo/config.py
praxis/train_csn.py
agent/csn_ppo/train.py
```

## Required implementation

Add config:

```python
enable_sentinel: bool = True
allow_no_sentinel_for_debug: bool = False
long_run_sentinel_required_steps: int = 10_000_000
```

Add CLI:

```text
--enable-sentinel / --no-enable-sentinel
--allow-no-sentinel-for-debug
```

Add validation:

```python
def validate_long_run_safety(cfg):
    if (
        cfg.num_timesteps >= cfg.long_run_sentinel_required_steps
        and not cfg.enable_sentinel
        and not cfg.allow_no_sentinel_for_debug
    ):
        raise ValueError(...)
```

## Acceptance tests

### Test 2.1 — long run without sentinel fails

```python
cfg = CSNPPOConfig(num_timesteps=100_000_000, enable_sentinel=False)
assert raises(ValueError, validate_long_run_safety, cfg)
```

### Test 2.2 — smoke run without sentinel allowed

```python
cfg = CSNPPOConfig(num_timesteps=300_000, enable_sentinel=False)
validate_long_run_safety(cfg)  # no error
```

### Test 2.3 — explicit debug bypass works

```python
cfg = CSNPPOConfig(
    num_timesteps=100_000_000,
    enable_sentinel=False,
    allow_no_sentinel_for_debug=True,
)
validate_long_run_safety(cfg)  # no error
```

---

# Priority 3 — Adaptive per-cluster guard pressure

## Problem

The current guard strength is global/static:

```python
guard_lambda_mem
```

But forgetting is usually localized:

```text
cluster 0 regresses, cluster 2 does not
dynamic-obstacle behavior regresses, open exploration does not
collision rate regresses, coverage does not
```

The system should respond selectively.

## Desired behavior

Maintain a guard coefficient per cluster or per memory bucket:

```python
cluster_guard_lambda: float32[num_clusters]
```

On sentinel regression:

```math
\lambda_c \leftarrow \min(\lambda_{\max}, \gamma_{\text{up}} \lambda_c)
```

On stable recovery:

```math
\lambda_c \leftarrow \max(\lambda_{\min}, \gamma_{\text{down}} \lambda_c)
```

Recommended defaults:

```python
guard_lambda_min = 1.0
guard_lambda_base = 8.0
guard_lambda_max = 32.0
guard_lambda_up = 1.5
guard_lambda_down = 0.98
guard_recovery_patience = 3
```

## Files likely involved

```text
agent/csn_ppo/config.py
agent/csn_ppo/train.py
agent/csn_ppo/guarded_loss.py
agent/csn_ppo/gradient_projection.py
agent/csn_ppo/sentinel.py
```

## Required implementation

Add state:

```python
@flax.struct.dataclass
class GuardPressureState:
    cluster_lambda: jnp.ndarray       # [num_clusters]
    recovery_count: jnp.ndarray       # [num_clusters]
```

Update rule:

```python
def update_guard_pressure(state, regressions, recovered, cfg):
    regressed = regressions["regressed"]

    increased = jnp.minimum(
        cfg.guard_lambda_max,
        state.cluster_lambda * cfg.guard_lambda_up,
    )

    decayed = jnp.maximum(
        cfg.guard_lambda_min,
        state.cluster_lambda * cfg.guard_lambda_down,
    )

    next_lambda = jnp.where(regressed, increased, decayed)

    return state.replace(cluster_lambda=next_lambda)
```

When combining guard gradients, use per-bucket coefficients:

```python
coefs = coefficients_for_buckets(
    bucket_names=ACTIVE_MEMORY_BUCKETS,
    cluster_guard_lambda=guard_pressure.cluster_lambda,
    cfg=config,
)

g_total = combine_safe_and_guard_grads(g_safe, guard_grads, coefs)
```

## Regression response

When a sentinel regression is detected:

```text
1. mine failed states,
2. label them with champion teacher,
3. insert into slow memory,
4. increase guard pressure for that cluster,
5. freeze/slow curriculum,
6. optionally roll back to last non-regressed checkpoint if regression is severe.
```

Add severity rule:

```python
severe = (
    coverage_drop > 2 * sentinel_success_tolerance
    or collision_increase > 2 * sentinel_collision_tolerance
)

if severe:
    restore_last_safe_params_for_cluster_or_global()
```

## Acceptance tests

### Test 3.1 — regressed cluster lambda increases

```python
old = [8, 8, 8, 8]
regressed = [False, True, False, False]
new = update_guard_pressure(...)
assert new[1] > old[1]
assert new[0] <= old[0]
```

### Test 3.2 — lambda is capped

If repeated regressions occur, lambda must not exceed `guard_lambda_max`.

### Test 3.3 — guard gradient combine uses nonuniform coefs

Create bucket gradients with known scalar values. Assert final gradient changes according to bucket-specific coefficients.

---

# Priority 4 — Integrate curriculum mixture into the live environment

## Problem

The standalone curriculum module exists, but the live loop does not use it to sample reset distributions. This is a major gap.

Forgetting is partly caused by **distribution replacement**. If the world distribution drifts forward and stops sampling older regimes, the agent will forget no matter how good the pointwise memory guard is.

The training distribution should be:

```math
p_t(\text{world})
=
0.70 p_{\text{frontier}}
+
0.20 p_{\text{history}}
+
0.10 p_{\text{sentinel-failures}}
```

## Desired behavior

Every reset should sample a difficulty from the curriculum mixture:

```python
difficulty = sample_world_difficulties(
    curriculum_state,
    rng,
    num_envs,
    sentinel_failure_difficulty,
)
```

Then the environment reset should use that difficulty to shape obstacle speed, amplitude, moving fraction, and other continuous randomization parameters.

## Important API warning

`CoverEnv.reset(rng)` currently takes only an RNG key. The normal MuJoCo Playground wrapper may not directly support passing arbitrary per-env difficulty into reset.

Do **not** fake curriculum integration by creating an unused `curriculum.py` module. The reset distribution must actually change.

Acceptable implementation paths:

### Option A — Proper custom vectorized wrapper

Build a CSN-specific environment wrapper that supports:

```python
reset(keys, difficulties)
step(state, action)
```

and handles autoreset by sampling the next difficulty from `curriculum_state`.

This is the cleanest long-term solution.

### Option B — Difficulty-aware environment state

Add `difficulty` to `state.info` and ensure reset/autoreset samples it from the curriculum. This may require replacing or wrapping `wrap_for_brax_training`.

### Option C — Coarser global curriculum stage

If per-env difficulty is too invasive, implement a first-pass global curriculum:

```text
update 0-100: difficulty mixture frozen at stage 0.2
update 100-200: stage 0.3
...
```

This is weaker than the spec but better than no curriculum. It must still preserve historical sampling.

## Files likely involved

```text
agent/csn_ppo/curriculum.py
agent/csn_ppo/train.py
praxis/envs/cover_env.py
praxis/train_csn.py
```

## Required environment difficulty hook

Add a difficulty-dependent reset path in `CoverEnv`.

Pseudo-code:

```python
def _sample_obstacle_params(self, rng, difficulty):
    difficulty = jp.clip(difficulty, 0.0, 1.0)

    max_amp = lerp(0.1, self._config.obstacle.max_amplitude, difficulty)
    min_freq = lerp(0.05, self._config.obstacle.min_frequency, difficulty)
    max_freq = lerp(0.10, self._config.obstacle.max_frequency, difficulty)
    frac_moving = lerp(0.0, self._config.obstacle.frac_moving, difficulty)

    ...
```

Difficulty should affect at least:

```text
obstacle moving fraction,
obstacle speed/frequency,
obstacle amplitude,
spawn spread,
possibly collision margin or obstacle density if topology remains fixed.
```

Do **not** randomize topology by adding/removing geoms. Keep model topology fixed.

## Required training-loop integration

Add state:

```python
curriculum_state = init_curriculum_state(config)
```

Each rollout/update:

```python
difficulty_batch = sample_world_difficulties(
    curriculum_state,
    rng,
    config.num_envs,
    sentinel_failure_difficulty,
)
```

Pass `difficulty_batch` into reset/autoreset path.

On sentinel regression:

```python
curriculum_state = freeze_or_slow_curriculum(curriculum_state)
```

When current and historical sentinel slices pass:

```python
curriculum_state = maybe_advance_curriculum(curriculum_state, sentinel_metrics)
```

## Acceptance tests

### Test 4.1 — difficulty affects reset distribution

Run reset at difficulty 0.0 and 1.0 with same seed family.

Assert high difficulty produces larger average obstacle speed/amplitude.

### Test 4.2 — mixture fractions are approximately correct

Sample 100k difficulties from curriculum state.

Assert approximate component proportions:

```text
frontier: 70% ± 3%
history: 20% ± 3%
sentinel-failure: 10% ± 3%
```

### Test 4.3 — sentinel regression freezes curriculum

After regression:

```python
state = freeze_or_slow_curriculum(state)
assert state.frozen
```

### Test 4.4 — advancement requires current and historical pass

If current passes but history fails, difficulty must not advance.

If both pass, difficulty advances.

---

# Priority 5 — Expand synthetic probes and add analytic teachers

## Problem

The current synthetic probes are too sparse. They cover only a few broad coverage states. For robust long-horizon training, the agent needs dense protection over edge cases that are rare in rollouts but important for avoiding collapse.

Synthetic probes should protect task geometry, not merely clone old policy behavior.

## Desired behavior

Add synthetic probe families for:

```text
1. open frontier, low coverage
2. open frontier, high coverage
3. obstacle directly between agent and frontier
4. crossing dynamic obstacle left-to-right
5. crossing dynamic obstacle right-to-left
6. near wall, frontier points along wall
7. corner escape
8. obstacle behind agent, should mostly ignore
9. high-speed near-collision
10. near-complete final-cell state
11. stalled/oscillation state
12. dense obstacle cluster with one escape tangent
13. frontier behind agent
14. no-obstacle straight motion
15. padded/mask edge cases if masks become nontrivial again
```

## Analytic coverage teacher

Add an analytic teacher that gives a safe geometric action for simple cases.

Base idea:

```math
a_{\text{frontier}} = \frac{f}{\|f\| + \epsilon}
```

Obstacle avoidance:

```math
a =
a_{\text{frontier}}
+
\sum_i
\mathbf{1}[d_i < d_{\text{safe}}]
\cdot
w_i
\cdot
\left(
-\frac{r_i}{\|r_i\|+\epsilon}
+
\beta \cdot \text{tangent}(r_i)
\right)
```

where:

```text
f = frontier direction
r_i = relative obstacle vector
d_i = obstacle distance
w_i = strength based on proximity and whether obstacle is in front
```

Pseudo-code:

```python
def analytic_coverage_teacher(obs, cfg):
    frontier = obs[FRONTIER_SLICE[0]:FRONTIER_SLICE[0] + 2]
    frontier_norm = jnp.linalg.norm(frontier) + 1e-6
    desired = frontier / frontier_norm

    obstacles = obs[OBST_SLICE[0]:OBST_SLICE[1]].reshape(K, 4)

    avoid = jnp.zeros_like(desired)

    for ox, oy, ovx, ovy in obstacles:
        rel = jnp.array([ox, oy])
        dist = jnp.linalg.norm(rel) + 1e-6
        rel_dir = rel / dist

        in_front = jnp.dot(rel_dir, desired) > 0.3
        close = dist < cfg.synthetic_safe_dist

        repel = -rel_dir
        tangent = jnp.array([-rel_dir[1], rel_dir[0]])

        # Choose tangent direction that still has positive frontier progress.
        tangent = jnp.where(
            jnp.dot(tangent, desired) < 0.0,
            -tangent,
            tangent,
        )

        strength = jnp.clip((cfg.synthetic_safe_dist - dist) / cfg.synthetic_safe_dist, 0.0, 1.0)
        avoid += jnp.where(in_front & close, strength * (0.7 * repel + 0.3 * tangent), 0.0)

    action = desired + avoid
    action = action / (jnp.linalg.norm(action) + 1e-6)
    return jnp.clip(action, -1.0, 1.0)
```

## Teacher selection for probes

For each synthetic probe:

```text
policy_teacher = mosaic teacher action
analytic_teacher = analytic action
```

Choose:

```python
teacher = safer_of(policy_teacher, analytic_teacher)
```

Minimum viable version:

```python
if obstacle_collision_risk(policy_teacher) > obstacle_collision_risk(analytic_teacher):
    teacher = analytic_teacher
else:
    teacher = policy_teacher
```

## Files likely involved

```text
agent/csn_ppo/coverage_probes.py
agent/csn_ppo/rollout_mining.py
agent/csn_ppo/criticality_coverage.py
agent/csn_ppo/config.py
```

## Acceptance tests

### Test 5.1 — obstacle ahead probe avoids forward collision

For an obstacle directly on the frontier ray, analytic teacher should not point straight into the obstacle.

```python
assert dot(action, obstacle_direction) < 0.5
```

### Test 5.2 — open frontier probe moves toward frontier

Without nearby obstacles:

```python
assert dot(action, frontier_direction) > 0.8
```

### Test 5.3 — crossing obstacle probe produces lateral component

For crossing obstacle states, action should include a tangent/avoidance component.

### Test 5.4 — probes are valid contract observations

All generated probes must:

```text
have shape [28],
contain finite values,
respect normalized ranges where expected,
have valid obstacle layout,
have valid mask.
```

---

# Priority 6 — Add a fixed validation bank separate from rollout holdout

## Problem

The current train/holdout split is drawn from the same current rollout. It helps detect minibatch overfitting, but it does not fully detect closed-loop overfitting or forgetting.

For long runs, the system needs an independent validation bank:

```text
current frontier validation seeds,
historical validation seeds,
sentinel failure validation seeds,
synthetic contract probes.
```

## Desired behavior

Every N updates, evaluate:

```text
validation/current_score
validation/history_score
validation/sentinel_failure_score
validation/synthetic_guard_kl
```

If training improves but validation degrades, stop or roll back.

## Files likely involved

```text
agent/csn_ppo/train.py
agent/csn_ppo/sentinel.py
agent/csn_ppo/coverage_probes.py
agent/csn_ppo/metrics.py
```

## Required implementation

Add:

```python
@flax.struct.dataclass
class ValidationBank:
    current_keys: jnp.ndarray
    history_keys: jnp.ndarray
    sentinel_failure_keys: jnp.ndarray
    synthetic_obs: jnp.ndarray
    best_current: jnp.ndarray
    best_history: jnp.ndarray
    best_failure: jnp.ndarray
```

Evaluation:

```python
validation_metrics = evaluate_validation_bank(
    env,
    validation_bank,
    params,
    normalizer_params,
    make_policy,
    apply_policy_value,
    cfg,
)
```

Gate:

```python
def validation_regressed(metrics, best, cfg):
    return (
        metrics["validation/history_coverage"] < best["history_coverage"] - cfg.validation_tolerance
        or metrics["validation/synthetic_kl_p95"] > cfg.validation_kl_limit
    )
```

Use in update acceptance:

```python
if validation_regressed:
    params = best_safe_params
    opt_state = best_safe_opt_state
    increase_guard_pressure()
```

## Acceptance tests

### Test 6.1 — validation bank deterministic

Same params and same bank produce identical metrics.

### Test 6.2 — validation regression blocks acceptance

Mock candidate params with lower validation score. Assert training loop rejects/rolls back.

### Test 6.3 — synthetic validation KL catches drift

Create candidate policy with large action mean shift on synthetic probes. Assert `synthetic_kl_p95` exceeds threshold.

---

# Priority 7 — Scale and stratify memory for 100M+

## Problem

The current default memory sizes are small for 100M+ training. Also, uniform ring-buffer replacement can erase rare but important behaviors.

For long-run anti-forgetting, memory must be both large and stratified.

## Desired long-run defaults

For 100M-step runs:

```python
memory_size_fast = 1_048_576
memory_size_slow = 262_144
memory_batch_size = 4096
```

Add preset:

```text
--long-run
```

which sets safe long-run defaults.

## Stratified memory

Keep quotas by:

```text
cluster_id,
source_id,
criticality band,
age band,
sentinel-failure status.
```

Avoid memory being dominated by common states.

## Replacement score

Use:

```math
\text{score}(m)
=
c(m)
+
\lambda_{\text{rare}} \cdot \text{rarity}(m)
+
\lambda_{\text{sentinel}} \cdot \mathbf{1}_{\text{sentinel failure}}
-
\lambda_{\text{age}} \cdot \text{staleness}(m)
```

Evict the lowest score within the same quota bucket.

## Files likely involved

```text
agent/csn_ppo/memory.py
agent/csn_ppo/config.py
agent/csn_ppo/train.py
praxis/train_csn.py
```

## Required implementation

Minimum viable version:

```python
source_cluster_quota = {
    (SOURCE_SENTINEL_FAILURE, cluster): 0.25 / num_clusters,
    (SOURCE_SYNTHETIC_PROBE, cluster): 0.25 / num_clusters,
    (SOURCE_RECENT_CURRENT, cluster): 0.50 / num_clusters,
}
```

Add stratified sample:

```python
sample_memory_stratified(memory, rng, batch_size, quotas)
```

Do not sample memory uniformly once enough data exists.

## Acceptance tests

### Test 7.1 — sentinel failures cannot be evicted by common current states

Fill memory with sentinel failures. Insert many current atoms. Assert sentinel quota remains populated.

### Test 7.2 — stratified sampling respects quotas

Sample 100 batches and verify approximate proportions by source/cluster.

### Test 7.3 — long-run preset sets memory sizes

`--long-run` should set memory sizes and sentinel enabled.

---

# Priority 8 — Long-run safe defaults and launch configs

## Problem

The headline result used stronger guard settings than the config defaults. Long-run safety should not depend on remembering the right magic command.

## Desired behavior

Add a safe preset:

```text
--long-run
```

Recommended values:

```python
num_timesteps = 100_000_000
enable_sentinel = True
guard_warmup_steps = 1_500_000
guard_kl_budget = 0.002
guard_lambda_mem = 8.0
memory_size_fast = 1_048_576
memory_size_slow = 262_144
memory_batch_size = 4096
synthetic_probe_batch_size = 4096
sentinel_bank_size = 4096
sentinel_eval_interval = 25
enable_holdout_early_stop = True
enable_gradient_projection = True
```

## Files likely involved

```text
agent/csn_ppo/config.py
praxis/train_csn.py
README / CSN docs
```

## Required implementation

Add CLI:

```text
--long-run
--safe-long-run
```

When enabled, override config defaults unless explicitly set by user.

Example launch:

```bash
python -m praxis.train_csn \
  --long-run \
  --seed 0 \
  --run-name csn_100m_longrun
```

Ablation launch:

```bash
python -m praxis.train_csn \
  --long-run \
  --no-guard \
  --no-projection \
  --allow-no-sentinel-for-debug \
  --seed 0 \
  --run-name ppo_100m_ablation
```

## Acceptance tests

### Test 8.1 — long-run preset enables sentinel

```python
cfg = resolve_config(["--long-run"])
assert cfg.enable_sentinel
```

### Test 8.2 — long-run preset sets strong guard

```python
assert cfg.guard_lambda_mem >= 8.0
assert cfg.guard_kl_budget <= 0.003
```

### Test 8.3 — explicit user flags override preset

```bash
--long-run --guard-lambda-mem 4
```

must produce:

```python
cfg.guard_lambda_mem == 4
```

---

# Priority 9 — Documentation accuracy

## Problem

Some docs still describe the original goal-reaching navigation task, while the current implementation targets the 28-D coverage task.

This matters because Codex/Claude and future humans will otherwise optimize the wrong thing.

## Required doc updates

Update or add documentation that explicitly says:

```text
Current CSN-PPO implementation target:
- 28-D coverage/exploration task
- no goal-reaching reward
- collisions are non-terminal by default
- metric is coverage retention, not success-rate retention
```

Also document how to adapt CSN-PPO back to the original navigation task:

```text
To use CSN-PPO for 27-D navigation:
1. replace coverage criticality with nav criticality,
2. replace coverage probes with nav probes,
3. use success/collision sentinels instead of coverage/collision sentinels,
4. restore goal-relative observation contract,
5. label synthetic probes using goal-directed analytic teacher.
```

## Files likely involved

```text
README.md
agent/README.md
CSN_PPO_README.md
CSN_BUILD_PLAN.md
CSN_RESULTS.md
```

## Acceptance tests

No code test required, but docs must contain:

```text
"28-D coverage"
"not the original 27-D navigation contract"
"coverage retention"
"sentinel required for long runs"
```

---

# Priority 10 — 100M acceptance test protocol

## Problem

A 5M run that halves forgetting is promising but not enough.

The target claim is 100M+ steps without overfitting or catastrophic forgetting. That requires a stricter protocol.

## Required experiments

Run at least:

```text
3 seeds CSN-PPO long-run
3 seeds plain PPO/custom-loop ablation
3 seeds guard-only no-projection
3 seeds projection-only no-guard
```

Minimum commands:

```bash
python -m praxis.train_csn \
  --long-run \
  --seed 0 \
  --run-name csn_100m_s0

python -m praxis.train_csn \
  --long-run \
  --seed 1 \
  --run-name csn_100m_s1

python -m praxis.train_csn \
  --long-run \
  --seed 2 \
  --run-name csn_100m_s2

python -m praxis.train_csn \
  --long-run \
  --no-guard \
  --no-projection \
  --allow-no-sentinel-for-debug \
  --seed 0 \
  --run-name ppo_100m_s0
```

Repeat ablation for seeds 1 and 2.

## Required metrics

Track:

```text
eval/episode_coverage
eval/episode_collision
sentinel/coverage_mean
sentinel/coverage_min_cluster
sentinel/collision_rate_mean
sentinel/collision_rate_max_cluster
sentinel/regression_count
memory/kl_mean
memory/kl_p95
memory/policy_violation_frac
memory/value_violation_frac
ppo/train_surrogate
ppo/holdout_surrogate
ppo/generalization_gap
guard/cluster_lambda/*
curriculum/current_difficulty
curriculum/history_pass
curriculum/current_pass
validation/history_coverage
validation/synthetic_kl_p95
```

## Success criteria

Let:

```math
P = \text{peak evaluation coverage}
```

Let:

```math
F_T = \text{final evaluation coverage at } T \text{ steps}
```

Let:

```math
R_T = F_T / P
```

For 100M steps:

```text
CSN median retention R_100M >= 0.75
Plain PPO median retention R_100M <= CSN - 0.20 absolute
sentinel/coverage_min_cluster does not collapse below 0.60
sentinel/regression_count is corrected within 3 sentinel intervals
memory/kl_p95 remains below configured limit except during brief recovery
collision does not rise while coverage is preserved
```

Overfitting criterion:

```text
If train surrogate improves while validation/history coverage drops for >3 evals,
that is overfitting.
CSN must either prevent this or roll back/recover within 3 evals.
```

## Plots required

Generate:

```text
coverage vs steps
retention vs steps
sentinel min-cluster coverage vs steps
collision rate vs steps
memory KL p95 vs steps
guard cluster lambdas vs steps
generalization gap vs steps
```

Create a single comparison figure:

```text
runs/csn_100m_comparison.png
```

---

# Final implementation order

Build in this exact order:

```text
P0  Fix sentinel failure labeling with champion teachers.
P1  Use per-cluster mosaic teachers for all memory/probe labeling.
P2  Make sentinel mandatory for long runs.
P3  Add adaptive per-cluster guard pressure.
P4  Integrate curriculum into the live reset distribution.
P5  Expand probes and analytic teachers.
P6  Add fixed validation bank.
P7  Scale and stratify memory.
P8  Add long-run preset.
P9  Update documentation.
P10 Run 100M multi-seed acceptance protocol.
```

Do not start a 100M claim before P0, P1, P2, and P4 are complete.

---

# Non-negotiable invariants

These are correctness rules. Do not violate them.

## Invariant 1 — Old PPO data is not replayed as PPO data

Behavioral memory is for functional preservation only:

```text
policy KL guard
value consistency guard
gradient projection
```

Do not train PPO advantages from stale replay.

## Invariant 2 — Sentinel failure labels must come from a good teacher

Never full-strength-label sentinel failures with the current regressed policy.

## Invariant 3 — Guard loss must remain hinge-based

Do not turn the guard into unconditional behavior cloning.

Correct:

```math
L_{\text{guard-policy}}
=
E_m[w_m \max(0, KL_m - \delta_m)^2]
```

Wrong:

```math
E_m[w_m KL_m]
```

The hinge is what allows plasticity.

## Invariant 4 — Projection must remove only conflicting components

Correct:

```math
g_{\text{safe}}
=
g_{\text{ppo}}
-
\frac{
\min(0, g_{\text{ppo}}^\top g_{\text{mem}})
}{
\|g_{\text{mem}}\|^2+\epsilon
}
g_{\text{mem}}
```

Do not project out aligned/useful components.

## Invariant 5 — Curriculum must preserve old distributions

Do not use a one-way curriculum that replaces old worlds.

Correct:

```text
70% frontier
20% history
10% sentinel failures
```

## Invariant 6 — Long-run CSN requires closed-loop sentinels

Pointwise memory alone is not enough.

## Invariant 7 — Evaluation must use the current normalizer

Do not close over a stale initial normalizer in a jitted evaluator.

---

# Expected outcome after this build

After these priorities are implemented, the system should have all major defenses required for long-horizon PPO:

```text
local overfitting defense:
    current-rollout holdout + validation bank

long-horizon forgetting defense:
    behavioral memory + hinge-KL guard + nullspace projection

closed-loop regression defense:
    sentinel bank + champion-teacher failure mining

distribution replacement defense:
    70/20/10 curriculum mixture

bad-teacher defense:
    per-cluster mosaic teacher + analytic probe teacher

100M operational defense:
    long-run defaults + multi-seed acceptance protocol
```

The current implementation is a serious prototype. These changes are what move it toward a credible 100M+ anti-forgetting system.
