#!/usr/bin/env bash
# Reproducible setup of envpool full-ALE-Atari under the WSL `praxis` venv (jax 0.9.2).
# Idempotent. Run: bash pmac/envs/setup_envpool.sh   (with /opt/venv the target venv)
set -uo pipefail
PY=/opt/venv/bin/python
PIP=/opt/venv/bin/pip
SP=$("$PY" -c "import site,sys; print([p for p in site.getsitepackages() if p.endswith('site-packages')][0])")
echo "site-packages: $SP"

# 1. Install envpool + minimal runtime deps WITHOUT touching jax/numpy (--no-deps).
"$PIP" install --no-deps envpool dm-env dm-tree optree ale-py 2>&1 | tail -3

# 2. envpool 1.2.5 ships without Atari ROMs; ale-py bundles them. Copy them into envpool's expected dir.
ENVROM="$SP/envpool/atari/roms"
ALEROM="$SP/ale_py/roms"
mkdir -p "$ENVROM"
cp -n "$ALEROM"/*.bin "$ENVROM"/ 2>/dev/null || true
echo "roms in envpool: $(ls "$ENVROM"/*.bin 2>/dev/null | wc -l)"

# 3. envpool/entry.py eagerly imports ALL env families at import time; some (myosuite) ship without their
#    metadata assets and crash the import. Patch entry.py so each NON-atari family import is non-fatal.
"$PY" - "$SP" <<'PY'
import sys, os, io
sp = sys.argv[1]
p = os.path.join(sp, "envpool", "entry.py")
src = open(p, encoding="utf-8").read()
if "PMA-C patch" not in src:
    new = '''"""Entry point for all envs' registration. PMA-C patch: non-atari families are optional."""
import envpool.atari.registration  # noqa: F401  (required)
import importlib
for _m in ("envpool.box2d.registration","envpool.classic_control.registration",
           "envpool.gfootball.registration","envpool.highway.registration",
           "envpool.jumanji.registration","envpool.minigrid.registration",
           "envpool.mujoco.dmc.registration","envpool.mujoco.gym.registration",
           "envpool.mujoco.metaworld.registration","envpool.mujoco.myosuite.registration",
           "envpool.mujoco.playground.registration","envpool.mujoco.robotics.registration",
           "envpool.pgx.registration","envpool.procgen.registration",
           "envpool.toy_text.registration","envpool.vizdoom.registration"):
    try:
        importlib.import_module(_m)
    except Exception:
        pass
'''
    open(p, "w", encoding="utf-8").write(new)
    print("patched entry.py")
else:
    print("entry.py already patched")
PY

# 4. Verify.
"$PY" - <<'PY'
import numpy as np, envpool
e = envpool.make('Pong-v5', env_type='gymnasium', num_envs=4, full_action_space=True)
o = e.reset()[0]; e.step(np.zeros(4, np.int32))
print("envpool Atari OK: obs", o.shape, "act", e.action_space.n)
PY
echo "ENVPOOL_SETUP_DONE"
