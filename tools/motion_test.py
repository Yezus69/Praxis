"""Deep motion/contact diagnostic. Run inside the praxis WSL distro.
   python tools/motion_test.py [n_obstacles]
"""
import sys
import jax
import jax.numpy as jp
import numpy as np
from praxis.envs import NavEnv
from praxis.envs.nav_env import default_config
from praxis import contract

_n_obs = int(sys.argv[1]) if len(sys.argv) > 1 else 4
_cfg = default_config()
_cfg.n_active_obstacles = _n_obs
print(f"=== motion test n_active_obstacles={_n_obs} ===")
env = NavEnv(_cfg)
reset = jax.jit(env.reset)
step = jax.jit(env.step)

state = reset(jax.random.PRNGKey(0))
mp = np.asarray(state.data.mocap_pos)
print("mocap_pos (obstacles 0-3, goal=4):")
for i, row in enumerate(mp):
    print(f"   [{i}] {row}")
print(f"nq={state.data.qpos.shape}, nv={state.data.qvel.shape}, "
      f"start dist={float(state.obs[2]):.3f}")
print("step dist   vx     vy     omega  ncon  | agent_xy")
for i in range(120):
    dx, dy = state.obs[0], state.obs[1]
    norm = jp.sqrt(dx * dx + dy * dy) + 1e-6
    action = jp.array([dx / norm, dy / norm])
    state = step(state, action)
    qv = np.asarray(state.data.qvel)
    axy = np.asarray(state.data.qpos[:2])
    ncon = int(np.asarray(state.data.ncon)) if hasattr(state.data, "ncon") else -1
    if i < 26 or i % 15 == 0:
        print(f"{i:4d} {float(state.obs[2]):6.3f} {qv[0]:6.3f} {qv[1]:6.3f} "
              f"{qv[2]:6.3f} {ncon:4d}  | [{axy[0]:6.2f},{axy[1]:6.2f}]")
    if float(state.done) > 0.5:
        print(f">>> done at {i}: succ={float(state.metrics[contract.METRIC_SUCCESS]):.0f} "
              f"coll={float(state.metrics[contract.METRIC_COLLISION]):.0f}")
        break

# Inspect contacts at the stuck position.
print("\n=== contact details at current state ===")
try:
    c = state.data.contact
    dist = np.asarray(c.dist)
    g1 = np.asarray(c.geom1) if hasattr(c, "geom1") else np.asarray(c.geom)[:, 0]
    g2 = np.asarray(c.geom2) if hasattr(c, "geom2") else np.asarray(c.geom)[:, 1]
    names = [env.mj_model.geom(i).name for i in range(env.mj_model.ngeom)]
    for k in range(len(dist)):
        if dist[k] < 0.05:
            n1 = names[g1[k]] if 0 <= g1[k] < len(names) else g1[k]
            n2 = names[g2[k]] if 0 <= g2[k] < len(names) else g2[k]
            print(f"   contact {k}: {n1} <-> {n2}  dist={dist[k]:.4f}")
except Exception as e:
    print("   contact introspection failed:", e)
print("geom names:", [env.mj_model.geom(i).name for i in range(env.mj_model.ngeom)])
