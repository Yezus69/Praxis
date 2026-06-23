# CASTM Build Status

Implementation of **Context-Addressed Synaptic Tensor Memory** per
`README_CONTEXT_ADDRESSED_SYNAPTIC_MEMORY.md`. Code lives in `tfns/castm/`;
tests in `tests/castm/`.

This document records what is built and verified, which acceptance gates pass,
and the remaining GPU experiment ladder.

## Stage A (spec §22) — COMPLETE and verified on CPU

All mathematical, layer-level, routing, sparse-execution, and serialization
correctness is implemented and tested. **`tests/castm/` — 51 tests passing.**

Run the suite (WSL `praxis` venv, CPU):

```bash
source /opt/venv/bin/activate
cd /mnt/c/Users/Asav/source/repos/Praxis
JAX_PLATFORMS=cpu python -m pytest tests/castm/ -q
```

### Implementation order (§27) status

| # | Step | Module | Status |
|---|------|--------|--------|
| 1 | Evaluation valid only on completed episodes | `tfns/train/evaluate.py` | ✅ `valid` flag, no partial-score substitution |
| 2 | Addressed linear memory, exact dual writes | `castm/synaptic.py`, `castm/address.py` | ✅ |
| 3 | Factorized dense memory + scratch commit | `castm/synaptic.py`, `castm/scratch.py` | ✅ |
| 4 | Addressed bias | `castm/synaptic.py` (`beta` factors) | ✅ |
| 5 | Convolutional factor memory | `castm/layers.py` | ✅ flattened-kernel + spatial/1×1 factorization |
| 6 | GRU-gate factor memory | `castm/layers.py` (3 independent banks) | ✅ |
| 7 | Policy + value heads integrated into one agent | `castm/agent.py` | ✅ |
| 8 | Exact decoded-weight audit APIs | `castm/audit.py` | ✅ |
| 9 | Canonical address book + orthonormal codebook | `castm/address.py` | ✅ |
| 10 | Content query encoder + prototype router | `castm/agent.py`, `castm/router.py` | ✅ |
| 11 | Context switch transaction | `castm/transaction.py`, `castm/router.py` | ✅ |
| 12 | Sparse gather execution | `forward_sparse`, sparse conv/gru/heads | ✅ |
| 13 | Per-address recompression | `castm/synaptic.py`, `castm/transaction.py` | ✅ |
| 14 | Synthetic conflicting-context tests | `tests/castm/test_synaptic.py`, `report.py` | ✅ gate 21.1 |
| 15 | Two-game oracle-address Atari | — | ⏳ GPU (Stage C) |
| 16 | Two-game inferred-address Atari | — | ⏳ GPU (Stage D) |
| 17 | Exact compensated shared consolidation | `castm/consolidate.py` | ✅ (math); enabled before 5-game |
| 18 | Five-game pilot | — | ⏳ GPU (Stage E) |
| 19 | Unannounced switching | — | ⏳ GPU |
| 20 | Delayed-credit shaping | (TFNS infra retained) | ⏳ last |

### Acceptance gates

- **§21.1 Mathematical memory gate — PASSED.** Exact noninterference for 32
  conflicting contexts across all six contextualized layer types in **float32**,
  worst-case relative decoded-weight drift **5.3e-7 < ε_write = 1e-6**. Artifact:
  `tfns/castm/reports/mathematical_test_report.json` (regenerate with
  `python -m tfns.castm.report`).
- **§16 numerical invariants — PASSED.** Address normalization 6e-8, codebook
  orthogonality 3.6e-7, duality 2.4e-7, rank == used count (all < 1e-5).
- **§17 pre-Atari tests — PASSED.** 17.1 conflicting linear memories; 17.2
  nonorthogonal duals; 17.3 novel-address residual; 17.4 dense/conv/GRU/head
  forward-equivalence + old-address invariance + scratch gradients; 17.5
  compression; 17.6 shared consolidation; 17.7 query-drift prototype refresh;
  17.8 task-identity leakage; 17.9 sparse execution; 17.10 exact serialization.
- **Agent-level guarantee.** `tests/castm/test_agent.py` and
  `test_continual_loop.py`: writing one context leaves another's decoded
  policy/value **bit-identical** under sparse top-1 gather, while the scratchpad
  remains fully plastic (Adam reduces per-context loss >50%).

### Compactness (§11.4)

Per-context storage across **all** contextualized layers ≈ **0.6 MB**
(dense 233 KB, 3×GRU 300 KB, conv 1–3 ≈ 60 KB, heads ≈ 33 KB). Five games ≈ 3 MB,
Atari-57 ≈ 36 MB — orders of magnitude below the prior 1 GB episodic bank.

### Frozen five-game suite (§18)

`tfns/castm/suites/five_game_suite_57057.json` (seed 57057, sampled once):
**Alien, Defender, Asterix, Tennis, Phoenix**; diagnostic pair **Alien/Defender**.
Three curriculum orders persisted for replication. Game names live only in the
harness; never in the agent or memory.

## Remaining — GPU experiment ladder (spec §22 Stages B–F)

These require envpool Atari + GPU and run for hours; they are **not** launched
from this build session. The mechanism they exercise is verified above. The
integration surface is:

- `castm/agent.py` — `policy_step` (addressed forward) and `context_query` (router input)
- `castm/router.py` — `route_step` → canonical address / NOVEL / UNCERTAIN
- `castm/transaction.py` — `commit_scratch_bank` / `commit_scratch_to_novel` at switches
- `tfns/ppo/` — retained recurrent PPO (rollout, losses) drives the scratchpad
- `tfns/train/evaluate.py` — completed-episode evaluation (valid-only)

Per §22, do not start a five-game five-million-step run until Stage A passes
(it now does) and oracle/inferred two-game gates pass in sequence:

- **Stage B** — matched single-task references (one recurrent PPO per game), both GPUs in parallel.
- **Stage C** — two-game oracle-address pilot: 500k steps/game, eval every 100k; escalate only on positive curves. Gate: P₂ ≥ 0.90 and R₁ ≥ 0.90.
- **Stage D** — two-game inferred-address: remove the oracle; gate adds router top-1 ≥ 99%.
- **Stage E** — five-game pilot (1M steps/game, eval all prior every 250k).
- **Stage F** — five-game final (matched budget, 30 completed eps/eval, ≥3 seeds, ≥2 orders).

Early-termination rules (§23) and the failure-diagnosis matrix (§25) gate every run.
