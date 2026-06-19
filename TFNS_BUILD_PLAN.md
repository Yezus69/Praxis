# TFNS Build Plan — Task-Free Synaptic Null-Space Continual PPO

Orchestration tracker for implementing `README_TASK_FREE_CONTINUAL_ATARI.md`.
Claude conducts + verifies; **all code is written by codex** (`codex exec`, gpt-5.5 xhigh).
Branch: `task-free-ns`.

## Design decisions (fixed)
- **Framework:** JAX 0.9.2 + flax 0.12.6 + optax 0.2.8 + envpool. Confirmed working on 2× RTX 4090.
  Functional style makes §13 (project the *applied* Adam delta + first moments) and §19 (atomic
  snapshot / rollback of params+optimizer+bases+memory) clean — params/opt-state are explicit pytrees.
- **Package:** fresh `tfns/` (clean, spec-aligned). Legacy `pmac/` is the old game-conditioned design
  the spec replaces — NOT reused except: `pmac/envs/atari_envpool.py` (envpool wrapper) and the
  behavior-distance / projection *math* as reference only.
- **No task identity anywhere** in the agent, memory keys, router, losses, or optimizer (§1, §4.1, §5).
- **Proof scope (user-chosen): "Core proof first"** — build every mechanism + pass all unit/integration
  tests, then 2-game Atari smoke, then 5-game sequential (seed 0[,1]) + plain-PPO baseline + retention
  report. Full multi-seed/order + 8-game + ablation sweep = follow-on (ask before launching).

## Runtime / commands
- GPU env: WSL distro `praxis`, `/opt/venv` (py3.11, jax-cuda). Repo at `/mnt/c/Users/Asav/source/repos/Praxis`.
- CPU unit tests (fast, deterministic): `JAX_PLATFORMS=cpu pytest -q tests/tfns/...`
- GPU runs: `CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1` (4090 #1) or `=2` (4090 #2);
  `XLA_PYTHON_CLIENT_PREALLOCATE=false`. Two 4090s = run two things in parallel (e.g. single-task refs).
- Codex: prompts live in `.codex/tfns/*.md` (codex sandbox blocks writing `.codex/`, so codex only
  *reads* them via stdin and writes the `tfns/` + `tests/tfns/` package). Invocation:
  `cat .codex/tfns/<M>.md | codex exec --sandbox workspace-write --skip-git-repo-check - `

## Module map (each part maps to a spec section — no slop)
- `tfns/config.py` — frozen global hyperparameters (§7.2 defaults, §10 tolerances, §9.4 weights). No per-game fields.
- `tfns/utils.py` — robust stats (median/MAD running normalizer), tree helpers, rng.
- `tfns/envs.py` — thin wrapper over the envpool helper + game lists (§21.1/21.2).
- `tfns/model/` — encoder (§6.2), explicit gru (§6.3), heads policy/value/key/aux (§6.4-6.6),
  adapters (§17.1), agent forward with NO game id (§6.1, §5).
- `tfns/protect/` — bases residual-SVD+QR (§12.6), projection dense/conv/bias (§12.2-12.5),
  optimizer-safe update (§13), constraints QP (§14), sentinel backtracking (§15).
- `tfns/memory/` — EpisodeSequence record + lossless obs compression (§9.2), byte-budget bank with
  admission (§9.4) / diversity (§9.5) / eviction (§9.6) / sentinels (§9.7), risk-balanced sampling + burn-in (§7.2, §16).
- `tfns/credit/` — return predictor F+Φ, causal innovation + eligibility, potential shaping (§11).
- `tfns/ppo/` — recurrent rollout + GAE (ppo-mask vs reset-mask) + sequence minibatch (§7), losses
  PPO+tubes+aux (§7.3, §10), per-update orchestration (§13 sequence wired end-to-end).
- `tfns/consolidate/` — ContinualState + atomic snapshot/restore + serialize (§4.7, §19), certify (§18.2),
  lifecycle (§18), plasticity + adapter activation (§17).
- `tfns/detect/` — task-free change detector PH/CUSUM + median/MAD + cooldown (§8).
- `tfns/train/` — synthetic integration envs (§24.1-3), atari smoke (§24.4), 5/8-game curriculum (§21),
  single-task refs (§21.3), eval + retention (§21.5-6), ablations (§22).

## Milestones (status: TODO / WIP / DONE)
- M1  scaffold + config + utils + envs + recurrent model + model tests .......... DONE (9/9 tests)
- M2  protect: bases + projection (dense/bias/conv) + tests (§23.2-4, §23.7) ..... DONE (7/7 tests)
- M3  protect: optimizer-safe update + GRU invariance + Adam safety (§23.5-6) ..... DONE (5/5 tests)
- M4  memory: record + bank + sampling + tests (§23.13, label-perm) .............. DONE (9/9 tests)
- M5  behavior tubes + constraints QP + sentinel backtracking (§23.8-9,17) ....... TODO
- M6  credit: predictor + shaping + tests (§23.12) ............................... TODO
- M7  ppo: rollout + GAE + losses + update orchestration (§23.10-11) ............. TODO
- M8  consolidate: state/snapshot + certify + lifecycle + plasticity + detect
      + atomic-rollback/gate-orientation/adapter tests (§23.15-17) .............. TODO
- M9  integration envs + integration tests (§24.1-3) ............................ TODO
- M10 Atari 2-game smoke (§24.4) + identity-leakage/no-mem-path (§23.1,14) ....... TODO
- M11 5-game sequential + single-task refs + retention + plain-PPO baseline (§21) . TODO

## Invariants checklist (must hold at the end — §28)
- [ ] policy forward has no task argument; label permutation changes nothing (§23.1)
- [ ] applied Adam delta (not just raw grad) is projected; first moments projected (§13, §23.6)
- [ ] multiple old constraints never averaged into one mean gradient (§14)
- [ ] every accepted update passes recurrent sentinel checks (§15)
- [ ] every accepted block passes closed-loop retention gate (§18.6)
- [ ] memory never produces actions; memory-disabled eval identical (§20, §23.14)
- [ ] fixed byte + residual-capacity budgets; honest capacity-exhaustion reporting (§9.3, §17, §26)
- [ ] all gates correctly oriented (max-loss / min-score); regression tests vs sign flips (§4.9, §23.17)
