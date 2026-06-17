# PMA-C Living Memory — Master Build Plan

Source of truth: `LIVING_MEMORY_PMA_C_FULL_ATARI_SPEC.md` (the spec). This file tracks
the module sequence, acceptance gates, and ops commands. All code is written by Codex
(gpt-5.5 xhigh); Claude conducts, specs each module, and verifies in WSL.

## North star (the deployment invariant — spec §3, §29, §33)
After sequential training over Atari games, **old games must be played by the LIVE model +
bounded compressed memory + adapters — NOT by loading a per-game full checkpoint.** The
existing "champion routing" serves a frozen per-game full net at eval; that is the crutch the
spec forbids and that this build replaces with genuine **memory-conditioned inference + the
explicit memory-policy blend** `p_final = (1-b_t) p_net + b_t p_mem` (§9).

## Test bed — 5 Atari games
`Pong, Breakout, SpaceInvaders, BeamRider, Qbert` (envpool `*-v5`, full_action_space=18,
obs uint8[4,84,84]). Smoke pair (fast, never gets stuck): `Pong, Breakout`.

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
- M6 split: M6a (single train, validating) -> M6b (continual: guard-aware update = latent conservation
  §11 + projection §15 + stability §17 + risk §16, eval=live+memory blend §9, ablations) -> M6c
  (review §18 + rollback gate §19 + visual §12/retrieval §13 losses). Then M7 throughput, M8, M9.
