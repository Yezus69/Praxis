# Praxis — next-session prompt

Paste everything below (between the lines) as your first message in the new session.

---

You're continuing **Praxis** — an RL project where an agent learns to **explore and cover a
2D arena** with **real MuJoCo MJX physics** (it physically bounces off walls and moving
obstacles). The repo is `C:\Users\Asav\source\repos\Praxis`.

**FIRST: read `RESUME.md` in the repo and your project memory `praxis-build-state`.** They
have the full runbook, the (unusual) GPU/environment setup, the stack versions, the
reward/obs design, and the known gotchas. Don't skip — the environment is not standard Docker.

## Current state (working)
- The agent learns coverage: eval area-coverage rises **0.16 → 0.87**; individual rollout
  episodes hit **100%**. Trains in **~2 min** on one RTX 4090.
- **Runtime = a native WSL2 distro `praxis`** (NOT Docker — Docker Desktop on this box can't
  pass the modern split-driver GPU into containers). Python env at `/opt/venv`, code at
  `/root/praxis`. Run:
  `wsl -d praxis -u root -- bash -c 'cd /root/praxis; export PYTHONPATH=/root/praxis LD_LIBRARY_PATH=/usr/lib/wsl/lib CUDA_VISIBLE_DEVICES=1; /opt/venv/bin/python -m praxis.train ...'`
  Sync code in: `cp -r /mnt/c/Users/Asav/source/repos/Praxis/praxis /root/praxis/`.
  Copy artifacts OUT via UNC (WSL→/mnt/c writes don't flush reliably):
  `cp //wsl.localhost/praxis/root/praxis/runs/<run>/* C:/Users/Asav/source/repos/Praxis/runs/<run>/`.
- Stack: jax 0.9.2, brax 0.14.2, flax 0.12.6, mujoco/mjx 3.9.0, mujoco_playground 0.2.0.
  Env = `praxis/envs/cover_env.py` + `cover_scene.xml`.
- **Reproduce first:**
  `python -m praxis.train --num-timesteps 1300000 --num-envs 2048 --num-evals 9 --learning-rate 0.00015 --entropy-cost 0.005 --run-name cover`
  then `python -m praxis.eval_render --checkpoint-dir ckpts/cover/<biggest-step-dir> --camera topdown --episodes 8 --mujoco-gl osmesa --out runs/cover/rollout.mp4`
  then `python -m praxis.plot_curves --run-name cover`.

## Major walls to solve (priority order)

**WALL 1 — PPO catastrophic forgetting (THE big one).** Coverage peaks ~0.87 at ~1.3M steps,
then if training continues it COLLAPSES (to ~0.32 by 5M steps). Verified across learning_rate,
entropy_cost, and max_grad_norm. We currently just stop early. FIX IT so long runs stay stable
and average coverage exceeds 0.9. Ideas: LR decay schedule (`ppo.train` has
`learning_rate_schedule` / `_min_lr` / `_max_lr`), KL adaptive/early-stop (`desired_kl`), value
clipping (`clipping_epsilon_value`), fewer `num_updates_per_batch`, reward normalization.
Success = the coverage curve rises and STAYS high through a 10M-step run.

**WALL 2 — Occasional stuck episodes.** ~1 in 8 eval episodes the agent pins against an obstacle
and barely moves (covers ~11%, hundreds of collision-steps); the rest are 83–100%. Reduce it:
saturating/stronger collision penalty, a small "moving-while-in-contact" reward, better start
sampling (don't spawn next to an obstacle), or a brief back-off when stuck.

**WALL 3 — Push average coverage to ~95%+.** After 1–2, tune episode_length / agent speed /
grid resolution (now 6×6) / the frontier reward so the agent reliably covers nearly everything.

**WALL 4 (optional) — Throughput + runtime robustness.** Training is eval-dominated (~8k
env-steps/s; eval runs full episodes). Cut eval cost for faster iteration. Also the WSL-distro
runtime is a workaround — consider scripting its setup or revisiting Docker (needs a Docker
Desktop update).

## Gotchas (don't re-trip)
- **Coverage metric is a per-step DELTA** (Brax sums episode metrics → final fraction). Cumulative
  count is `info['covered']`. Reading the metric at episode end gives ~0 — use `info['covered']/N_CELLS`.
- The agent is deliberately 2-DOF (no yaw), FLOATS (no floor contact), with collision bitmasks.
  Don't reintroduce a grounded/yawing agent — it causes phantom MJX contacts that pin it.
- Obs (28-d) includes a **frontier vector** (direction to nearest unvisited cell) — the key signal
  that makes coverage learnable by an MLP.

Start by reproducing the current result, then tackle Wall 1.

---
