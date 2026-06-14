# AGENTS.md — build rules for Codex on this repo

## Mission
Build **CSN-PPO** (Contract-Sentinel Nullspace PPO) for the Praxis repo. The COMPLETE and
AUTHORITATIVE specification is **`CSN_PPO_README.md`** in the repo root. It is the single source
of truth.

## PRIME DIRECTIVE
Implement EXACTLY as `CSN_PPO_README.md` describes. **Every math formula in the code must match
the README's equations exactly** (Gaussian KL §7, hinge guard loss §6/§8, gradient projection
§9/§11, criticality §18/§19, KL/value budgets §18, PPO loss §28, constrained view §29). Do NOT
invent, "improve", simplify, or reinterpret the algorithm. If the README gives a code sketch,
follow it. If something is genuinely ambiguous, choose the most literal reading of the README and
note the assumption in your summary — do not silently deviate.

## Stack (pinned — do not change versions)
jax 0.9.2, jaxlib 0.9.2, brax 0.14.2, flax 0.12.6, optax 0.2.8, orbax 0.12.0, mujoco/mjx 3.9.0,
mujoco_playground 0.2.0. Python 3.11.

## Layout
- New package at the repo ROOT: **`agent/csn_ppo/`** (exactly per README §23). Imports are
  `agent.csn_ppo.*` (repo root is on PYTHONPATH at runtime).
- Tests in **`tests/`** (e.g. `tests/test_csn_memory.py`), runnable with pytest.
- Do NOT modify the existing `praxis/` coverage code unless a task explicitly says so.

## Observation/action contract (README §1)
27-D obs: `goal[0:4]=(dx,dy,dist,heading_err)`, `vel[4:7]=(vx,vy,omega)`,
`obstacles[7:23]` = 4×4 (px,py,vx,vy) sorted by distance, `mask[23:27]`. Action ∈ [-1,1]^2.

## JIT / JAX rules (README §35.6)
Fixed shapes only inside jitted/update paths. No Python lists or dynamic shapes there. Use the
JIT-safe variants the README calls out (percentile via `jnp.sort`+indexing, fixed-size `[4,4]`
probe generation, fixed-capacity ring-buffer memory).

## Execution environment
You (Codex) run on a **Windows host where jax/brax are NOT installed** — they live in a WSL distro
the human controls. **Do NOT attempt to run training or pytest** (imports will fail). Validate your
code with syntax/AST checks only:
`python -c "import ast; ast.parse(open('PATH', encoding='utf-8').read()); print('AST OK')"`
(try `python`, then `py`, then `python3`). The human runs pytest/training in WSL and reports back.

## After every task
Print a concise summary: files created/changed, and for each, a mapping of function → the README
section and formula it implements. Confirm AST checks passed.
