# Praxis — runbook (GPU working via WSL distro)

**Updated:** 2026-06-13. GPU training is WORKING. JAX sees all 3 GPUs.

## The environment story (important)
- JAX-CUDA can't run on native Windows, so it runs in Linux. **Docker Desktop (20.10.14, 2022)
  could NOT pass the modern split-driver GPU into containers** (it mounts an incomplete lib set →
  `cuInit CUDA_ERROR_NOT_FOUND`). The fix that worked: **export the built `praxis:gpu` image's
  filesystem and import it as a native WSL2 distro**, which gets native WSL GPU access.
- Driver was updated 561.09 → **610.47**, and a **clean reinstall** (`setup.exe -s -clean`) was
  required to fix a broken WSL GPU-paravirtualization component (`dxgkio_query_adapter_info Ioctl
  failed: -2`). WSL kernel is now modern (6.18.33.1, WSL 2.7.8).

## The working runtime: WSL distro `praxis`
Created once via:
```
docker create --name praxis_export praxis:gpu
docker export praxis_export -o C:\Users\Asav\praxis_fs.tar
wsl --import praxis C:\Users\Asav\praxis-wsl C:\Users\Asav\praxis_fs.tar --version 2
wsl -d praxis -u root -- bash -c 'echo /usr/lib/wsl/lib > /etc/ld.so.conf.d/ld.wsl.conf && ldconfig'
```
The Python env (jax 0.9.2 CUDA12 stack) lives at **/opt/venv** inside the distro. The repo code is
copied to **/root/praxis** (re-copy from /mnt/c after edits — see below).

### Run anything (template)
```
wsl -d praxis -u root -- bash -c '
  cd /root/praxis
  export PYTHONPATH=/root/praxis LD_LIBRARY_PATH=/usr/lib/wsl/lib \
         JAX_DEFAULT_MATMUL_PRECISION=highest CUDA_VISIBLE_DEVICES=1
  /opt/venv/bin/python <ARGS>
'
```
`CUDA_VISIBLE_DEVICES=1` or `2` pins to one idle RTX 4090 (0 is the 2080 Ti driving the display).

### Sync code after editing in the Windows repo
```
wsl -d praxis -u root -- bash -c 'cp -r /mnt/c/Users/Asav/source/repos/Praxis/praxis /root/praxis/'
```

## Commands
- GPU check: `/opt/venv/bin/python tools/verify_stack.py` → BACKEND: gpu, 3 CudaDevice.
- DoD-1 smoke (PASSED): `/opt/venv/bin/python -m pytest tests/test_smoke.py -q`
- DoD-2 train: `/opt/venv/bin/python -m praxis.train --num-timesteps 20000000 --num-envs 2048 --num-evals 20 --no-randomization --run-name gpu-dod2`
- DoD-3 video: `MUJOCO_GL=osmesa /opt/venv/bin/python -m praxis.eval_render --checkpoint-dir ckpts/gpu-dod2 --out runs/gpu-dod2/rollout.mp4 --mujoco-gl osmesa`
- Curves: `/opt/venv/bin/python -m praxis.plot_curves --run-name gpu-dod2`
- Copy artifacts back to Windows repo:
  `wsl -d praxis -u root -- cp -r /root/praxis/runs /mnt/c/Users/Asav/source/repos/Praxis/`

## Code status
All code written, reviewed, CPU- and GPU-validated. Pinned stack: jax 0.9.2, brax 0.14.2,
flax 0.12.6, mujoco/mjx 3.9.0, mujoco_playground 0.2.0. Key fixes baked in: MJX spheres (no
cylinder-box collision), preserve incoming `state.metrics` keys, emit `info['time_out']`,
`wrap_env_fn`+`bootstrap_on_timeout` in train, `network_factory` in eval load_policy.
