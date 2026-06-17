# PMA-C Living Memory — Master Build Plan

Source of truth: `LIVING_MEMORY_PMA_C_FULL_ATARI_SPEC.md` (the spec). This file tracks
the module sequence, acceptance gates, and ops commands. All code is written by Codex
(gpt-5.5 xhigh); Claude conducts, specs each module, and verifies in WSL.

## RULE (user): build ALL spec parts BEFORE any larger training run. Match spec math exactly.
## Spec-completeness checklist (every §3 component + non-negotiable §32)
- [x] §4 live memory-conditioned agent (M2/M6a/M7b)
- [x] §5 stable key encoder + EMA (M2; EMA stepped in trainers). [ ] §5 key-drift penalty L_key WIRED (part of §12 below)
- [x] §6/§7 bounded compressed latent memory + tiers (M1; GPU hot bank M7b)
- [x] §9 retrieval reader + explicit blend (M2); eval-via-live+memory (M6b1)
- [x] §8 importance write rule (M3, jitted in M7b)
- [x] §11 conservation loss (M4) + WIRED guarded update (M6b2)
- [x] §15 tangent-cone projection (projection.py) + WIRED (M6b2)
- [x] §16 risk-normalized guard (M5) + WIRED (M6b2)
- [x] §17 synaptic stability (stability.py) + WIRED (M6b2)
- [x] §22 merge (M1) + §23 eviction/budget (M1)
- [x] §12 visual sentinel store + L_key + L_visual_beh WIRED into update/consolidation  (M6c)
- [x] §13 retrieval alignment loss WIRED (query/positive/hard-negatives)                (M6c)
- [x] §18 closed-loop old-game REVIEW rollouts wired into continual task loss           (M6d)
- [x] §19 sentinel eval + ROLLBACK GATE wired (accept/reject/restore + failure mem)     (M6d)
- [ ] §20 adapter growth (bank + sparse TopS router + L_adapter + plasticity trigger)   (M8)
- [ ] §21 slow consolidation phase (+ adapter distillation), accept iff sentinels pass  (M8)
- [ ] §24 memory deletion certification audit                                           (M8)
- [x] env-cleanup/OOM fix (close envpool envs; small eval envs) — needed to run anything (M6c)
- [ ] M9 proof harness: incremental JSON results (never lose data), both GPUs, run LAST
NO larger training runs until every [ ] above is [x].

## North star (the deployment invariant — spec §3, §29, §33)
After sequential training over Atari games, **old games must be played by the LIVE model +
bounded compressed memory + adapters — NOT by loading a per-game full checkpoint.** The
existing "champion routing" serves a frozen per-game full net at eval; that is the crutch the
spec forbids and that this build replaces with genuine **memory-conditioned inference + the
explicit memory-policy blend** `p_final = (1-b_t) p_net + b_t p_mem` (§9).

## Test bed — 5 Atari games
`Breakout, SpaceInvaders, BeamRider, Asterix, Qbert` (envpool `*-v5`, full_action_space=18,
obs uint8[4,84,84]) — the headline set, all proven to learn at ~4M steps with this PPO. Smoke game
(fast, clear signal): `SpaceInvaders` (clipped return 0.7→7.9 by 4M). NOTE: Pong dropped — it needs
>4M steps with this config (base PPO flat at -20 @4M), too slow for iteration.

## THROUGHPUT REALITY (measured)
Raw envpool ALE caps at **~22k FPS on this 24-core CPU** (env sim is CPU-bound; 256 vs 1024 envs both
~21-23k). So **100k+ SPS is NOT achievable for REAL ALE on this hardware** — that needs a GPU-emulated
Atari (not real ALE) or far more CPU cores. M7a fast envpool-XLA rollout does ~10.6k SPS (env↔GPU
serialization halves the ceiling) and LEARNS (SI verified). ~10-20k SPS makes the proof feasible
(5 games × 4M = 20M steps ≈ 30 min/seed). Optimizing toward the ~22k ceiling (async/Sebulba overlap) is
possible but diminishing returns; proof is feasible now.

## What already exists (faithful to spec math, keep it)
- Single shared Nature-CNN actor-critic conditioned on `(obs, game_onehot)` — `pmac/agents/atari_net.py`, `ppo_atari.py`.
- §11 conservation loss, §15 tangent-cone projection, §17 synaptic stability, §1 retention
  accounting, §19 sentinel STORE/eval — `pmac/{conservation,projection,stability,sentinels,evaluation}.py`, `rl_update.py`.
- These are the "guard-loss-only" system (= spec §30 baseline to beat).

## What's missing (this build) — additive, the existing guard path stays for ablation
The memory-conditioned side that makes the live model retain old games without champions.

## Module sequence (each = Codex task → WSL verify → commit)
- **M1 memory core** `pmac/memory/`: §6 MemoryAtom (latent key, p*, v*, importance, game_id,
  cluster, eps, source_flags), §7 hot(VRAM JAX arrays)/warm tiers fixed budget, §23 utility
  eviction + per-game B_g, §22 merge. Retrieval-ready as matmul. CPU tests.
- **M2 memory-conditioned agent**: §5 key encoder (EMA τ_key, normalized k_t), §9 retrieval
  (s_i, α_i, m_t, p_mem, v_mem, ρ_t, b_t), §4 F_θ(h_t,m_t,c_g), explicit blend, §11 latent head
  B_θ. Retrieval = GPU matmul vs hot bank. Reduces to base net when memory empty.
- **M3 write rule**: §8 I_t (wA1,wδ1,wN1.5,wH0.25,wL3,wQ2,wF3), top-pct + rare-cluster + quota;
  store p*=softmax(ℓ/T), v*=(V-μ_g)/σ_g.
- **M4 losses**: §11 latent conservation + Huber value, §12 visual sentinel L_key+L_visual-beh,
  §13 contrastive retrieval alignment L_retr.
- **M5 schedule/gate**: §16 risk-normalized λ_g (fixed Λ_total), §18 closed-loop review
  P_review(g) via live+memory, §19 full rollback gate.
- **M6 integration**: §27/§28 living-memory continual loop; eval = live+memory ONLY; champion
  demoted to short-term training rollback (§25). GPU smoke proves old-game retention beats
  no-memory-read.
- **M7 throughput**: ≥100k env steps/s on 2×4090 (more envs, GPU-resident obs+bank, async/multi-GPU).
- **M8 remaining**: §20 adapter growth + router, §21 slow consolidation + distillation,
  §22–24 certified merge/prune/deletion.
- **M9 proof**: §30 ablations (full vs guard-only, no-memory-read, no-conservation, no-projection,
  no-review…) + §29 5-game proof (mean norm retention ≥0.95, worst ≥0.90, new-game ≥90% baseline,
  checkpoint-free eval) + multi-seed report.

## Non-negotiables (spec §32) — every module must respect
No per-game full checkpoint as runtime policy · memory bounded & compressed · memory used at
inference AND training · old games evaluated closed-loop · unsafe updates rolled back · guard
pressure normalized as games grow · key-space drift controlled · deletion requires certification.

## Ops (see memory `praxis-infra-cheatsheet`)
- GPUs: 4090s = CUDA 1,2 with `CUDA_DEVICE_ORDER=PCI_BUS_ID`; never device 0 (2080Ti).
- Verify env: `wsl.exe -d praxis -- bash -lc "cd /mnt/c/Users/Asav/source/repos/Praxis && source /opt/venv/bin/activate && ..."` (jax 0.9.2, flax 0.12.6, envpool; 3 CUDA devices).
- Fast unit tests: `JAX_PLATFORMS=cpu python -m pytest tests/pmac -q` (CPU, ~8s for current suite).
- Codex: feed prompt via stdin+`-`, `codex exec --sandbox workspace-write -o <last>.txt`, run in
  background, read last-message file. Codex AST-validates only (no JAX/GPU on Windows host).
- Hard `timeout` on every train/test; long runs in background + react to completion; no forever loops.

## Status log (update as modules land)
- M0 done: infra confirmed, plan written, 5 games chosen.
- M1 done (5b39034): pmac/memory core; adversarial review SHIP; 42 tests.
- M2 done (817765d): memory-conditioned agent + reader + blend; 2 reviews SHIP;
  GPU jit smoke on 4090 (empty→b=0=base policy, populated→blend active); 49 tests.
  Note: inference-time hot bank kept modest (few-k atoms) for 100k SPS; warm bank
  serves slower training-time conservation. Reader is capacity-parametric.
- M3 done (8460433): write rule §8; review SHIP.
- M4 done (2046b06): latent conservation §11 + Huber, visual sentinel §12, retrieval alignment §13; review SHIP.
- M5 done: risk-normalized guard §16 + closed-loop review §18 + rollback gate §19/§26; self-reviewed.
- M6a built (UNCOMMITTED, validating): pmac/agents/ppo_living_memory.py + pmac/memory/runtime.py.
  Single-game living-memory trainer: mem-conditioned rollout + writes (M3) + PPO. 6 CPU tests pass.
  FIX applied: train/act on logits_net/v_net (b=0), NOT logits_final — the explicit blend (§9) is for
  OLD-game retention only; blending current-game stale teachers pinned Pong at -21. Validating Pong learns.
  THROUGHPUT: steady-state ~900-1000 SPS (~10x slower than base PPO) + ~45s compile startup. Profiled:
  not recompilation per step; it's mem_apply forward + per-segment host write path. M7 must fix (envpool
  XLA scan + batch writes) before the 5-game proof (M9) is feasible.
- M7a done (committed): fast base PPO envpool-XLA scan, ~10.6k SPS, learns SI.
- M7b done: fast MEM trainer ppo_living_memory_fast — XLA scan + GPU-resident hot bank (hot_insert =
  top-C by importance via lax.top_k) + jitted writes. ~10.3k SPS (15x over M6a's 700), mem fills, learns SI
  (clipped 0.7->8.3, beats base 7.0). The O(writes*bank) host bottleneck is GONE. Memory path now at env ceiling.
- M6b1 done: certify_protected_memories + build_protected_bank + eval_living_memory. PROVEN on GPU:
  SpaceInvaders played from live+memory (blend) = 285 true score vs net-only (b=0) = 140 — the explicit
  blend (§9) more than DOUBLES play. Deployment-invariant mechanism works + memory actively helps.
- M6 split: M6a (single train) -> M6b (continual: guard-aware update = latent conservation
  §11 + projection §15 + stability §17 + risk §16, eval=live+memory blend §9, ablations) -> M6c
  (review §18 + rollback gate §19 + visual §12/retrieval §13 losses). Then M7 throughput, M8, M9.
