# Praxis — runbook (coverage/exploration, real physics, GPU)

**Updated:** 2026-06-13. The agent EXPLORES and COVERS the arena with REAL physics
(it physically bounces off walls + moving obstacles). Trains in minutes on a 4090.

## Results (cover6 — clean run, ~1.3M steps)
Area coverage **0.16 → 0.87**, episode reward −2.8 → +17.5, collision rate low.
Rollout episodes reach **100% coverage**. Artifacts in `runs/cover6/`
(rollout.mp4 = top-down sweep, curves.png, metrics.csv); checkpoints in `ckpts/cover6/`.

## The environment (why a WSL distro, not Docker)
JAX-CUDA can't run on native Windows. Docker Desktop 20.10.14 (2022) can't pass the
modern split-driver GPU into containers, so we run in a **native WSL2 distro `praxis`**
(exported from the built image). Driver 610.47 (a CLEAN reinstall was needed to fix the
WSL GPU-paravirt component). Python env at `/opt/venv`; code copied to `/root/praxis`.

### Run anything (template)
```
wsl -d praxis -u root -- bash -c '
  cd /root/praxis
  export PYTHONPATH=/root/praxis LD_LIBRARY_PATH=/usr/lib/wsl/lib \
         JAX_DEFAULT_MATMUL_PRECISION=highest CUDA_VISIBLE_DEVICES=1
  /opt/venv/bin/python <ARGS>'
```
`CUDA_VISIBLE_DEVICES=1` or `2` = an idle RTX 4090. Sync code after edits:
`cp -r /mnt/c/Users/Asav/source/repos/Praxis/praxis /root/praxis/`.
Copy artifacts OUT to Windows via UNC (9p writes from WSL don't flush reliably):
`cp //wsl.localhost/praxis/root/praxis/runs/<run>/* C:/Users/Asav/source/repos/Praxis/runs/<run>/`.

## Commands
- Smoke: `/opt/venv/bin/python -m pytest tests/test_smoke.py -q`
- **Train (fast, ~2 min):**
  `/opt/venv/bin/python -m praxis.train --num-timesteps 1300000 --num-envs 2048 --num-evals 9 --learning-rate 0.00015 --entropy-cost 0.005 --run-name cover`
- Video (top-down sweep): `/opt/venv/bin/python -m praxis.eval_render --checkpoint-dir ckpts/cover/<step> --camera topdown --episodes 8 --mujoco-gl osmesa --out runs/cover/rollout.mp4`
- Curves: `/opt/venv/bin/python -m praxis.plot_curves --run-name cover`

## Design notes
- **Real-physics agent**: 2 slide DOFs (x,y), NO yaw hinge, FLOATS above the floor,
  collision bitmasks (contype/conaffinity=2) so it bounces off walls+obstacles but has
  no floor contact. This avoids the phantom-contact bug that pins a grounded/yawing agent.
- **Coverage**: arena split into a 6x6 grid; reward = +new cells covered, −0.1·obstacle
  contact (non-terminal), −small time. Metrics are emitted as per-step DELTAS so Brax's
  episode-sum aggregation yields the final coverage FRACTION; cumulative count is `info['covered']`.
- **Observation (28-d)**: agent pos+vel, K nearest obstacles, and the FRONTIER vector
  (direction+distance to the nearest UNVISITED cell) + covered fraction. The frontier
  signal is the key — an MLP follows it directly (much easier than a flattened grid).
- **TRAINING TIP**: coverage peaks ~0.87 around ~1.3M steps then PPO over-optimizes and
  REGRESSES (catastrophic forgetting). Stop near the peak / use the best checkpoint. A
  learning-rate schedule or KL early-stopping would let it train longer stably.

## Stack
jax 0.9.2, brax 0.14.2, flax 0.12.6, mujoco/mjx 3.9.0, mujoco_playground 0.2.0.
(Prior goal-reaching MVP is in git history, commit f4bb3ed.)
