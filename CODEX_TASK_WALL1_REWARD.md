# Codex task — Praxis WALL 1 fix: remove the passive-optimum (reward/env shaping)

You are implementing a precise change. You CANNOT see the conversation that produced this
spec. Edit **`praxis/envs/cover_env.py`** (primary) and **`praxis/train.py`** (CLI plumbing
only). Do NOT touch `cover_scene.xml`. Keep everything JAX-traceable (no Python `if` on tracer
values; the new config values ARE static Python config, so `bool(cfg...)`/`float(cfg...)` at
trace-time is fine). READ both files first and adapt to the ACTUAL current code (line numbers
below are approximate).

## Why (diagnosis, confirmed)
`CoverEnv` pays `k_cov * new_cells` only the FIRST time each of 36 grid cells is entered. Once
all cells are covered, `new_cells==0` forever → zero coverage gradient for the rest of the
episode, while `-k_coll*collision` and `-k_time` are paid every step. So after coverage
completes, the only remaining gradient pushes toward "don't move" (avoid collisions). On-policy
PPO drifts to this passive optimum and FORGETS covering: coverage peaks ~0.82–0.99 at ~1.3M
steps then collapses to ~0.16–0.23 by 10M. Verified structural (5 PPO-hyperparameter sweeps all
collapse; at convergence v_loss≈2e-4, policy_loss≈0 → the advantage signal genuinely vanished).
We remove the passive optimum via opt-in reward/env changes. DEFAULTS must reproduce current
behavior exactly (the collapse must still reproduce on a no-flag run).

## INVARIANTS (must hold — these have bitten us before)
- **I1**: The coverage metric `metrics[METRIC_COVERAGE] = new_cells/N_CELLS` is a per-step DELTA
  that Brax SUMS over the episode → `eval/episode_coverage` = fraction of distinct cells visited.
  Do NOT change this definition. `info['visited']` stays the monotonic first-visit grid;
  `info['covered'] = visited.sum()`. Variant B's freshness uses a SEPARATE grid — never repurpose
  `visited`. This keeps coverage curves comparable across all runs.
- **I2 (time_out/bootstrap contract)**: `bootstrap_on_timeout=True`. A horizon cutoff is
  `done=1, time_out=1`. A NEW success terminal (full coverage, variant A) must be
  `done=1, time_out=0` so PPO does NOT bootstrap past a real terminal. If success and horizon
  coincide, success wins. Use `done = max(timeout, success)`, `time_out = timeout*(1-success)`,
  and set BOTH `info['truncation']` and `info['time_out']` to that `time_out`.
- **I3**: `step` must preserve incoming `state.metrics` (Brax EvalWrapper injects `'reward'`).
  Keep the existing `metrics = dict(state.metrics)` copy and only ADD keys.
- **I4**: Do NOT change the agent body/contacts/XML.
- **I5**: Any new metric key MUST also be in `_metrics_zero()` (reset) with the same scalar dtype
  `jp.zeros(())`, or Brax aggregation errors. New keys: `"completed"`, `"mean_freshness"`.

## STEP 1 — `default_config()`: add fields (after the `cfg.reward.k_time` line)
```python
    # --- WALL1 fix: opt-in reward shaping; defaults reproduce current behavior ---
    cfg.reward.terminate_on_full_coverage = False  # Variant A: true terminal when all cells covered
    cfg.reward.k_complete = 0.0                     # Variant A: completion bonus weight (0 => off)
    cfg.reward.collision_penalty_cap = 0.0          # Variant C.1: 0 => uncapped (current behavior)
    cfg.reward.patrol = False                       # Variant B: renewable freshness reward
    cfg.reward.k_fresh = 0.0                        # Variant B: weight on freshness restored/step
    cfg.reward.freshness_decay = 0.99               # Variant B: per-step freshness decay
```

## STEP 2 — `reset()`: add the freshness grid next to `info['visited']`
```python
    info["freshness"] = jp.zeros((contract.N_CELLS,))  # Variant B; inert unless patrol/k_fresh
```

## STEP 3 — `_metrics_zero()`: add two scalar keys to the returned dict
```python
    "completed": jp.zeros(()),
    "mean_freshness": jp.zeros(()),
```

## STEP 4 — `step()`: rewrite the tail (from the coverage update through the return)
Adapt variable names to the ACTUAL code. The edited tail should be equivalent to:
```python
        # --- coverage update (KEEP existing logic) ---
        prev_visited = info["visited"]
        cell = self._cell_index(agent_xy)             # use the existing cell-index call
        visited = prev_visited.at[cell].set(1.0)
        new_cells = visited.sum() - prev_visited.sum()

        # --- Variant B: renewable freshness (inert when patrol=False / k_fresh=0) ---
        decay = float(cfg.reward.freshness_decay) if bool(cfg.reward.patrol) else 1.0
        prev_fresh = info["freshness"] * decay
        gain = 1.0 - prev_fresh[cell]
        freshness = prev_fresh.at[cell].set(1.0)
        info["freshness"] = freshness
        r_fresh = float(cfg.reward.k_fresh) * gain

        # --- collision penalty (geometric, NON-terminal) + optional cap (Variant C.1) ---
        # KEEP the existing collision computation; just add the cap on r_coll:
        r_coll = -k_coll * collision
        _cap = float(cfg.reward.collision_penalty_cap)
        r_coll = jp.where(_cap > 0.0, jp.maximum(r_coll, -_cap), r_coll)

        # r_cover and r_time KEEP existing definitions:
        # r_cover = k_cov * new_cells ; r_time = -k_time

        # --- Variant A: terminate on full coverage + completion bonus ---
        timeout = (step_idx >= int(cfg.episode_length)).astype(jp.float32)
        fully_covered = (visited.sum() >= float(contract.N_CELLS)).astype(jp.float32)
        success = fully_covered * (1.0 if bool(cfg.reward.terminate_on_full_coverage) else 0.0)
        remaining_frac = jp.clip(
            (float(cfg.episode_length) - step_idx.astype(jp.float32)) / float(cfg.episode_length),
            0.0, 1.0)
        r_complete = float(cfg.reward.k_complete) * success * remaining_frac

        reward = r_cover + r_coll + r_time + r_complete + r_fresh

        done = jp.maximum(timeout, success)
        time_out = timeout * (1.0 - success)
        info["truncation"] = time_out
        info["time_out"] = time_out
        # ... KEEP the rest of info updates (step, time, visited, covered, rng) ...

        # --- metrics: KEEP all existing keys; ADD these two ---
        metrics["completed"] = success                                   # episode-sum => completion rate
        metrics["mean_freshness"] = freshness.mean() / float(cfg.episode_length)
```
KEEP `metrics[METRIC_COVERAGE] = new_cells/N_CELLS` and all other existing metric writes exactly.

## STEP 5 — `train.py`: CLI plumbing (additive, all defaults None => no override)
In `build_parser()` add (mirror existing flag style):
```python
    p.add_argument("--k-cov", type=float, default=None)
    p.add_argument("--k-coll", type=float, default=None)
    p.add_argument("--k-time", type=float, default=None)
    p.add_argument("--k-complete", type=float, default=None)
    p.add_argument("--k-fresh", type=float, default=None)
    p.add_argument("--freshness-decay", type=float, default=None)
    p.add_argument("--collision-penalty-cap", type=float, default=None)
    p.add_argument("--terminate-on-full-coverage", action=argparse.BooleanOptionalAction, default=None)
    p.add_argument("--patrol", action=argparse.BooleanOptionalAction, default=None)
```
Thread these into env construction: where the env config is built (the `default_config()` /
`build_env` path that already threads `episode_length`), apply each non-None arg onto the
matching `cfg.reward.*` field BEFORE the env is created. Only override when the arg is not None,
so a no-flag run is byte-identical to today.

## DEFAULT-PRESERVES-BEHAVIOR INVARIANT (verify)
With NO new flags: terminate_on_full_coverage=False → success=0 → done=timeout, time_out=timeout
(unchanged); k_complete=0 → r_complete=0; k_fresh=0 → r_fresh=0; collision_penalty_cap=0 →
r_coll uncapped. So `reward` and `done`/`time_out` are identical to today. Only two harmless
extra metric columns appear. The collapse must still reproduce on a default run.

## Verification (best effort — full run needs WSL GPU; the human runs that)
- AST check (no imports): `python -c "import ast; ast.parse(open('praxis/envs/cover_env.py',encoding='utf-8').read()); ast.parse(open('praxis/train.py',encoding='utf-8').read()); print('AST OK')"` (try `python`, `py`, `python3`).
- Re-read your diff: confirm the default path is unchanged per the invariant above.

## Deliverable
Apply edits to the two files. Print a summary: (1) regions changed with line refs, (2) AST OK or
no-python, (3) explicit confirmation the no-flag default path is behavior-identical (reward,
done, time_out unchanged), (4) list of the new CLI flags.
