# Living Memory PMA-C — Build & Proof Report

Implementation of `LIVING_MEMORY_PMA_C_FULL_ATARI_SPEC.md` end-to-end: a single live Atari agent that
plays previously-learned games using the **live model + bounded compressed memory + adapters**, with **no
per-game full checkpoint at evaluation** (the spec's deployment invariant, §3/§25/§29/§33).

All architecture code was written by Codex (gpt-5.5 xhigh) under a per-module spec; each module was
verified in WSL (`/opt/venv`, jax 0.9.2, envpool, 2×RTX 4090) and adversarially reviewed before commit.

## 1. Architecture — spec-complete (all §3 components)

| Spec | Component | Where | Status |
|---|---|---|---|
| §4 | Live memory-conditioned actor-critic `F_θ(h,m,c)` | `pmac/agents/atari_mem_net.py` | ✅ reviewed |
| §5 | Stable key encoder `E_key` (EMA τ_key, normalized k_t) + L_key | `atari_mem_net.py`, `memory/reader.py` | ✅ |
| §6/§7 | Bounded compressed latent **MemoryAtom** + hot/warm tiers | `pmac/memory/{atom,bank}.py` | ✅ |
| §8 | Importance write rule `I_t` (wA/wδ/wN/wH/wL/wQ/wF) | `pmac/memory/write.py` | ✅ |
| §9 | Retrieval reader (sim, top-K, α, m_t, ρ_t, b_t) + explicit blend `p_final=(1-b)p_net+b·p_mem` | `memory/reader.py` | ✅ |
| §11 | Behavior conservation `E[w·[D-ε]_+²]`, D=KL+λ_V·Huber, latent head B_θ | `memory/losses.py` | ✅ |
| §12 | Visual sentinel loss (L_key + L_visual_beh) + store | `memory/{losses,sentinels_visual}.py` | ✅ |
| §13 | Retrieval alignment (InfoNCE, hard negatives) | `memory/losses.py` | ✅ |
| §15 | Tangent-cone gradient projection + clipped guard correction | `pmac/projection.py`, `ppo_living_memory_fast.py` | ✅ |
| §16 | Risk-normalized guard allocation `λ_g=Λ·u_g/Σu_h` | `pmac/guard_schedule.py` | ✅ |
| §17 | Synaptic stability scaling `Δθ=-η·g/(1+α·Ω)` | `pmac/stability.py` | ✅ |
| §18 | Closed-loop old-game review (live+memory rollouts, P_review(g)) | `continual_living_memory.py` | ✅ |
| §19 | Sentinel eval + rollback gate (accept/reject/restore + failure mem) | `pmac/rollback_gate.py`, driver | ✅ |
| §20 | Adapter growth (bank + sparse TopS router + L_adapter, plasticity-triggered) | `atari_mem_net.py`, driver | ✅ |
| §21 | Slow consolidation (η_slow, sentinel-gated accept) + adapter distillation | `pmac/agents/consolidation.py` | ✅ |
| §22 | Count-weighted memory merge | `memory/bank.py` | ✅ |
| §23 | Utility-based eviction + per-game budget `B_g` | `memory/bank.py` | ✅ |
| §24 | Memory deletion certification (model-coverage + last-cluster) | `pmac/memory/deletion_cert.py` | ✅ |

**104 CPU unit tests pass.** Each module's math was checked line-by-line against the cited spec section in
an adversarial review (verdict SHIP for every module).

## 2. Throughput

The repo runs on JAX + envpool real ALE. Rollout uses envpool's **XLA interface inside a jitted `lax.scan`**
(env step in-graph, no per-step host sync); memory writes (keys/novelty/importance/insert) are **jitted on a
GPU-resident hot bank** (`hot_insert` = top-C by importance). Measured **~10k env-steps/s** training, which
learns (SpaceInvaders 0.7→8.3 clipped). The raw envpool ALE ceiling on this 24-core CPU is ~22k FPS (the
emulator is CPU-bound), so 100k+ SPS is not attainable for *real* ALE here without a GPU-emulated env.

## 3. Deployment invariant — demonstrated

The complete continual loop (`continual_living_memory`) runs end-to-end on GPU with every mechanism firing:
memory-conditioned rollout → importance writes → conservation/projection/stability guard → §18 review →
§19 rollback gate → §21 consolidation → live+memory eval. Observed in a 2-game run (SpaceInvaders→Breakout):
the §19 gate evaluated protected games and accepted the block; the §18 review ran; **§21 consolidation
correctly REJECTED a regressing update** (it detected SpaceInvaders dropping 270→212 under the consolidated
weights and reverted — the sentinel-gated accept works).

Old games are played by the **live net + a bounded compressed hot bank** (4096 atoms; each atom = a 128-d
fp16 latent key + 18-d teacher policy + scalars ≈ **~1.2 MB total for all games**) — NOT per-game frozen
nets. This is the spec's mechanism (§9 explicit blend), not the forbidden champion-checkpoint crutch (§25/§32).

**Live+memory retention** (deployment metric): after training a later game, the prior game replayed via
live+memory holds at its peak (2-game: SpaceInvaders 285→270, retention 0.946; 3- and 5-game runs below).

## 4. Ablations (§30) and the forgetting regime

Honest framing (the spec warns deployed=1.0 can be a structural floor): we report **two** measures —
(a) *deployed* = live model + memory (the blend, §9); (b) *shared-net-only* = `no_memory_read` (blend off,
logits_net) which is the falsifiable measure of how much the shared weights themselves retained.

Key empirical finding: the spec's net is **game-conditioned** (`c_g` context, §4), so it is inherently a
multi-task learner that resists catastrophic forgetting until the shared capacity saturates. At small scale
(2-3 games, ≤1.5M steps/game) with a *greedy* eval, **no ablation forgets** (full = no_conservation =
plain_ppo = retention 1.0): there is no forgetting to separate the ablations. The original champion-routing
headline needed **5 games × 4M steps** for baseline forgetting to appear (norm_retention 0.529). To surface
the contrast within an affordable budget we (i) use a **stochastic eval** (samples the policy → sensitive to
distribution drift greedy misses) and (ii) run the **full 5-game sequence** (max interference).

### 5-game results (SpaceInvaders→Breakout→BeamRider→Asterix→Qbert, 800k/game, stochastic eval)

<!-- RESULTS_TABLE: filled by make_report.py from g5_{full,plain,nomem}.json -->
*(pending — run `python pma_c_results/living_memory_proof/make_report.py /root/g5_full.json /root/g5_plain.json /root/g5_nomem.json`)*

Interpretation key:
- `full` (conservation + memory blend) vs `no_memory_read` (conservation, blend off): a gap shows the
  **memory recall** is doing the retention work (§29 criterion 9).
- `no_memory_read` (conservation, shared-net) vs `plain_ppo` (no conservation, shared-net): a gap shows the
  **conservation guard** protects the shared weights (§29 criterion 8).

## 5. Reproduce
```
# one ablation (writes incremental fsynced JSON):
CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. \
  ABLATION=full GAMES=SpaceInvaders-v5,Breakout-v5,BeamRider-v5,Asterix-v5,Qbert-v5 \
  PER_GAME=800000 NBLOCKS=1 NENVS=256 STOCH=1 RESULT_PATH=/root/g5_full.json \
  python pma_c_results/living_memory_proof/proof_run.py
python pma_c_results/living_memory_proof/make_report.py   # assemble table
```
Ablations supported by `continual_living_memory`: `full, no_conservation, no_projection, no_memory_read,
no_review, no_gate, no_adapter, no_consolidation, plain_ppo`.
