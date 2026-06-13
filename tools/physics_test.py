"""Validate the real-physics agent body in cover_scene.xml (run in praxis distro).

Checks: free glide in open space (ncon=0, no floor contact), hard stop at a wall
(real collision), hard stop at an obstacle, and NO phantom pinning.
"""
import jax
import jax.numpy as jp
import numpy as np
import mujoco
from mujoco import mjx

m = mujoco.MjModel.from_xml_path("praxis/envs/cover_scene.xml")
mx = mjx.put_model(m)
N_SUB = 5


def make_data(xy, mocap):
    d = mjx.make_data(mx)
    qpos = d.qpos.at[0].set(xy[0]).at[1].set(xy[1])
    mp = d.mocap_pos.at[:].set(mocap)
    d = d.replace(qpos=qpos, qvel=jp.zeros_like(d.qvel), mocap_pos=mp)
    return mjx.forward(mx, d)


def step(d, ctrl):
    d = d.replace(ctrl=ctrl)
    return jax.lax.fori_loop(0, N_SUB, lambda _, dd: mjx.step(mx, dd), d)


jstep = jax.jit(step)
FAR = jp.array([[100.0, 100.0, 0.25]] * 4)


def run(title, start, mocap, ctrl, n=100):
    print(f"\n=== {title} ===")
    d = make_data(jp.array(start), mocap)
    for i in range(n):
        d = jstep(d, jp.array(ctrl))
        ncon = int(np.asarray(d._impl.ncon))
        if i < 4 or i % 20 == 0 or i == n - 1:
            print(f"  step {i:3d}: xy=[{float(d.qpos[0]):6.3f},{float(d.qpos[1]):6.3f}] "
                  f"v=[{float(d.qvel[0]):6.3f},{float(d.qvel[1]):6.3f}] ncon={ncon}")
    return float(d.qpos[0]), float(d.qpos[1])


x, _ = run("drive +x into wall (inner face 3.0; expect stop ~2.85)", [-2.5, 0.0], FAR, [2.0, 0.0])
print(f"  -> stopped at x={x:.3f} (good if ~2.80-2.90)")

# obstacle at origin, drive +x from left: expect stop at ~-0.40 (0 - (r_obs0.25 + r_agent0.15))
obs1 = FAR.at[0].set(jp.array([0.0, 0.0, 0.25]))
x, _ = run("drive +x into obstacle at origin (expect stop ~-0.40)", [-2.5, 0.0], obs1, [2.0, 0.0])
print(f"  -> stopped at x={x:.3f} (good if ~-0.45 to -0.35)")

x, y = run("glide diagonally across OPEN space (expect free motion, ncon=0)", [-2.5, -2.5], FAR, [1.5, 1.5], n=60)
print(f"  -> reached xy=[{x:.3f},{y:.3f}] (good if it traveled freely, ~ +x+y)")
