import importlib.metadata as m
for p in ['jax', 'jaxlib', 'jax-cuda12-plugin', 'brax', 'flax', 'optax',
          'orbax-checkpoint', 'mujoco', 'mujoco-mjx']:
    try:
        print(f"{p}=={m.version(p)}")
    except Exception as e:
        print(p, 'ERR', e)
import jax
print("BACKEND:", jax.default_backend())
print("DEVICES:", jax.devices())
