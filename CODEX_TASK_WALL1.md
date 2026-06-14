# Codex task — Praxis WALL 1: PPO stability knobs (catastrophic-forgetting fix)

You are implementing a precise, self-contained change. You CANNOT see the conversation that
produced this spec, so follow it exactly. Edit **only** `praxis/train.py`. Do NOT modify
`praxis/config.py` or `praxis/envs/cover_env.py`.

## Background (why)
A 10M-step PPO run on this coverage task peaks at ~0.87 eval coverage around ~1.3M steps, then
collapses to ~0.32 ("catastrophic forgetting"). We are adding **opt-in** PPO-stability knobs so
long runs stay stable, plus diagnostics to confirm the cause. The training runs on a pinned
**brax 0.14.2**.

## HARD INVARIANT (do not violate)
A run with **no new flags** must produce **byte-identical** `train_kwargs` to today — every new
knob defaults to current behavior. The code already computes
`train_sig_params = inspect.signature(ppo.train).parameters` near `train.py:447`. Every new brax
kwarg must be injected inside a **signature-guard** block: only inject when **(a)** the user opted
in (value != sentinel default) **AND (b)** the param name is in `train_sig_params`; otherwise
`print("[ppo] NOTE: brax has no <name>; <flag> ignored.")`.

## CRITICAL brax 0.14.2 facts (verified — do not deviate)
- `learning_rate` is a **plain float only** — it does NOT accept an optax/callable schedule.
  Do NOT try to pass a cosine/linear callable into `learning_rate`; it will raise.
- The ONLY LR-scheduling mechanism is the kwarg `learning_rate_schedule`, whose legal values are
  the strings `"NONE"` (default) and `"ADAPTIVE_KL"`. Passing the **string** `"ADAPTIVE_KL"` is
  correct (brax does `LRSchedule("ADAPTIVE_KL")` internally; no enum import needed). Under
  ADAPTIVE_KL, brax multiplicatively shrinks LR toward `learning_rate_schedule_min_lr` when
  per-update KL > `2*desired_kl`, and grows it toward `learning_rate_schedule_max_lr` when
  KL < `desired_kl/2`. `learning_rate` remains the float STARTING lr the controller scales.
- These kwargs are ALL confirmed present in installed brax 0.14.2 `ppo.train`:
  `learning_rate_schedule, desired_kl, learning_rate_schedule_min_lr, learning_rate_schedule_max_lr,
  clipping_epsilon, clipping_epsilon_value, normalize_advantage, normalize_until_count, gae_lambda,
  vf_loss_coefficient, deterministic_eval`.

---

## STEP 1 — DEFAULTS dict (around `train.py:44-59`)
Add these keys (do NOT change existing keys):
```python
    lr_schedule="none",            # "none" | "adaptive_kl"
    desired_kl=0.01,               # target per-update KL (only used under adaptive_kl)
    lr_min=1e-5,                   # -> learning_rate_schedule_min_lr
    lr_max=1e-2,                   # -> learning_rate_schedule_max_lr
    clipping_epsilon=None,         # None => brax default (0.3); float => override
    clipping_epsilon_value=None,   # None => OFF (brax default); float => value-clip range
    normalize_advantage=None,      # None => brax default (True); bool => override
    normalize_until_count=None,    # None => off; int => freeze obs-normalizer stats after N obs
    gae_lambda=None,               # None => brax default 0.95; float => override
    vf_loss_coefficient=None,      # None => brax default 0.5; float => override
    deterministic_eval=False,      # False => current behavior (stochastic eval policy)
```

## STEP 2 — argparse flags in `build_parser` (around `train.py:299-338`, after `--reward-scaling`)
```python
    p.add_argument("--lr-schedule", type=str, choices=["none", "adaptive_kl"],
                   default=DEFAULTS["lr_schedule"],
                   help="LR controller. 'none' (default)=constant LR (current behavior). "
                        "'adaptive_kl'=brax KL-throttled adaptive LR. brax 0.14.2 has no cosine/linear schedule.")
    p.add_argument("--desired-kl", type=float, default=DEFAULTS["desired_kl"],
                   help="Target per-update KL for adaptive_kl. Inert unless --lr-schedule adaptive_kl.")
    p.add_argument("--lr-min", type=float, default=DEFAULTS["lr_min"], help="Floor LR for adaptive_kl.")
    p.add_argument("--lr-max", type=float, default=DEFAULTS["lr_max"], help="Ceiling LR for adaptive_kl.")
    p.add_argument("--clipping-epsilon", type=float, default=None,
                   help="Policy PPO clip eps. None => brax default 0.3. Lower (0.2) tightens trust region.")
    p.add_argument("--clipping-epsilon-value", type=float, default=None,
                   help="Value-function clip range. None => OFF (brax default). e.g. 0.2 enables clipped value loss.")
    p.add_argument("--normalize-advantage", action=argparse.BooleanOptionalAction, default=None,
                   help="Per-minibatch advantage standardization. None => brax default (True).")
    p.add_argument("--normalize-until-count", type=int, default=None,
                   help="Freeze obs-normalizer running stats after N observations. None => never freeze.")
    p.add_argument("--gae-lambda", type=float, default=None, help="GAE lambda. None => brax default 0.95.")
    p.add_argument("--vf-loss-coefficient", type=float, default=None,
                   help="Value loss coefficient. None => brax default 0.5.")
    p.add_argument("--deterministic-eval", action=argparse.BooleanOptionalAction, default=False,
                   help="Use the greedy (mean) policy at eval time. Default False = stochastic eval (current behavior). "
                        "Diagnostic: isolates eval-time action noise from real policy degradation.")
```

## STEP 3 — `resolve_config` (around `train.py:341-364`, add to the cfg dict; do NOT add to the `--smoke` override block)
```python
    cfg["lr_schedule"] = str(args.lr_schedule)
    cfg["desired_kl"] = float(args.desired_kl)
    cfg["lr_min"] = float(args.lr_min)
    cfg["lr_max"] = float(args.lr_max)
    cfg["clipping_epsilon"] = (None if args.clipping_epsilon is None else float(args.clipping_epsilon))
    cfg["clipping_epsilon_value"] = (None if args.clipping_epsilon_value is None else float(args.clipping_epsilon_value))
    cfg["normalize_advantage"] = args.normalize_advantage  # None | True | False
    cfg["normalize_until_count"] = (None if args.normalize_until_count is None else int(args.normalize_until_count))
    cfg["gae_lambda"] = (None if args.gae_lambda is None else float(args.gae_lambda))
    cfg["vf_loss_coefficient"] = (None if args.vf_loss_coefficient is None else float(args.vf_loss_coefficient))
    cfg["deterministic_eval"] = bool(args.deterministic_eval)
```

## STEP 4 — `train_kwargs` wiring (after the existing `max_grad_norm` block ~`train.py:480-489`, BEFORE the `ppo.train(...)` call)
Insert these signature-guarded, opt-in blocks:
```python
    # LR schedule (adaptive KL trust region). Only inject when opted in.
    if cfg["lr_schedule"] != "none":
        if "learning_rate_schedule" in train_sig_params:
            train_kwargs["learning_rate_schedule"] = "ADAPTIVE_KL"
            if "desired_kl" in train_sig_params:
                train_kwargs["desired_kl"] = float(cfg["desired_kl"])
            if "learning_rate_schedule_min_lr" in train_sig_params:
                train_kwargs["learning_rate_schedule_min_lr"] = float(cfg["lr_min"])
            if "learning_rate_schedule_max_lr" in train_sig_params:
                train_kwargs["learning_rate_schedule_max_lr"] = float(cfg["lr_max"])
            print(f"[ppo] ADAPTIVE_KL LR schedule ON: desired_kl={cfg['desired_kl']} "
                  f"lr in [{cfg['lr_min']},{cfg['lr_max']}] start={cfg['learning_rate']}")
        else:
            print("[ppo] NOTE: brax has no learning_rate_schedule; --lr-schedule ignored.")

    if cfg["clipping_epsilon"] is not None:
        if "clipping_epsilon" in train_sig_params:
            train_kwargs["clipping_epsilon"] = float(cfg["clipping_epsilon"])
        else:
            print("[ppo] NOTE: brax has no clipping_epsilon; --clipping-epsilon ignored.")

    if cfg["clipping_epsilon_value"] is not None:
        if "clipping_epsilon_value" in train_sig_params:
            train_kwargs["clipping_epsilon_value"] = float(cfg["clipping_epsilon_value"])
        else:
            print("[ppo] NOTE: brax has no clipping_epsilon_value; --clipping-epsilon-value ignored.")

    if cfg["normalize_advantage"] is not None:
        if "normalize_advantage" in train_sig_params:
            train_kwargs["normalize_advantage"] = bool(cfg["normalize_advantage"])
        else:
            print("[ppo] NOTE: brax has no normalize_advantage; --[no-]normalize-advantage ignored.")

    if cfg["normalize_until_count"] is not None:
        if "normalize_until_count" in train_sig_params:
            train_kwargs["normalize_until_count"] = int(cfg["normalize_until_count"])
        else:
            print("[ppo] NOTE: brax has no normalize_until_count; ignored.")

    if cfg["gae_lambda"] is not None:
        if "gae_lambda" in train_sig_params:
            train_kwargs["gae_lambda"] = float(cfg["gae_lambda"])
        else:
            print("[ppo] NOTE: brax has no gae_lambda; ignored.")

    if cfg["vf_loss_coefficient"] is not None:
        if "vf_loss_coefficient" in train_sig_params:
            train_kwargs["vf_loss_coefficient"] = float(cfg["vf_loss_coefficient"])
        else:
            print("[ppo] NOTE: brax has no vf_loss_coefficient; ignored.")

    # Deterministic eval is a real brax kwarg but its non-default (False) IS the current behavior,
    # so only inject when the user opts in to True, and still guard on the signature.
    if cfg["deterministic_eval"]:
        if "deterministic_eval" in train_sig_params:
            train_kwargs["deterministic_eval"] = True
            print("[ppo] deterministic_eval=True (greedy/mean policy at eval).")
        else:
            print("[ppo] NOTE: brax has no deterministic_eval; --deterministic-eval ignored.")
```

## STEP 5 — Fix training-metrics logging (IMPORTANT — the obvious approach is WRONG)
**Do NOT** add a "training branch" to `progress_fn`. In brax 0.14.2 with `log_training_metrics=True`,
`progress_fn` fires **once per eval**, and the metrics dict it receives is a **merge** of `eval/*`
**and** `training/*` keys (e.g. `training/learning_rate`, `training/kl_mean`, `training/policy_loss`,
`training/v_loss`, `training/total_loss`, `training/entropy`). The current code filters to `eval/`-only
and **silently drops** the `training/*` data.

Required change — make the controller observable from CSV:
- Inspect the existing eval-metrics CSV writer in `make_progress_fn`/`progress_fn` (around
  `train.py:161` and `train.py:206-223`). It currently writes eval metrics to `runs/<run>/metrics.csv`.
- In that **same eval branch**, also append a row to a sibling file `runs/<run>/train_metrics.csv`
  containing `step`, a wall-clock seconds column, and **every `training/*` scalar present** in the
  merged metrics dict (column name = the key, e.g. `training/learning_rate`). Reuse the existing
  lazy-header / `csv.DictWriter` pattern. NaN-fill any key missing in a given call. Do not change the
  existing `metrics.csv` contents/behavior.
- Also ensure the eval CSV captures `eval/episode_coverage_std` if brax emits it (it does — used to
  quantify the eval-noise band). If the current eval CSV already writes all `eval/*` keys, no change
  is needed there.

## Verification you should do locally (best effort)
- You will likely NOT have a working Python with jax/brax on this Windows host (those live in WSL),
  so a full `--help` run may fail on import. That is fine — the human will run the GPU verification.
- DO run a **syntax/AST check** that needs no imports:
  `python -c "import ast,sys; ast.parse(open('praxis/train.py',encoding='utf-8').read()); print('AST OK')"`
  (try `python`, then `py`, then `python3`). If none exist, skip and say so.
- Re-read your diff against the HARD INVARIANT: confirm that with all flags absent, none of the
  STEP-4 blocks inject anything and `train_kwargs` is unchanged from before.

## Deliverable
Apply the edits to `praxis/train.py`. At the end, print a concise summary:
(1) the exact lines/regions changed, (2) confirmation the AST check passed (or that no python was
available), (3) an explicit statement that the default (no-flag) path injects zero new train_kwargs.
