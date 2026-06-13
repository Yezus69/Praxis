"""Ground-truth introspection of installed APIs (run inside the container).

Prints versions, GPU backend, and the exact signatures/attributes the env+trainer
depend on, so the orchestrator can validate the coding agents' assumptions.
"""
import importlib, inspect


def sig(obj):
    try:
        return str(inspect.signature(obj))
    except (ValueError, TypeError):
        return "<no signature>"


def hr(t):
    print("\n" + "=" * 70 + f"\n{t}\n" + "=" * 70)


hr("VERSIONS")
for m in ["jax", "jaxlib", "brax", "mujoco", "mujoco_mjx", "mujoco_playground",
          "flax", "optax", "orbax.checkpoint", "ml_collections", "mediapy"]:
    try:
        mod = importlib.import_module(m)
        print(f"{m:22s} {getattr(mod, '__version__', '?')}")
    except Exception as e:
        print(f"{m:22s} IMPORT FAIL: {type(e).__name__}: {e}")

hr("JAX BACKEND")
import jax
print("default_backend:", jax.default_backend())
print("devices:", jax.devices())

hr("mjx_env: MjxEnv / State / helpers")
try:
    from mujoco_playground import mjx_env
    print("mjx_env module:", mjx_env.__file__)
except Exception:
    from mujoco_playground._src import mjx_env
    print("mjx_env (._src) module:", mjx_env.__file__)
print("MjxEnv.__init__:", sig(mjx_env.MjxEnv.__init__))
print("MjxEnv abstract methods:", getattr(mjx_env.MjxEnv, "__abstractmethods__", None))
print("MjxEnv members:", [n for n in dir(mjx_env.MjxEnv) if not n.startswith("__")])
State = mjx_env.State
print("State type:", State)
try:
    print("State fields:", [f.name for f in State.__dataclass_fields__.values()] if hasattr(State, "__dataclass_fields__") else "n/a")
except Exception as e:
    print("State fields err:", e)
print("State.__init__:", sig(State))
print("mjx_env has 'make_data':", hasattr(mjx_env, "make_data"),
      "| 'step':", hasattr(mjx_env, "step"),
      "| 'init':", hasattr(mjx_env, "init"))
for h in ["make_data", "step", "init"]:
    if hasattr(mjx_env, h):
        print(f"  mjx_env.{h}:", sig(getattr(mjx_env, h)))

hr("wrapper.wrap_for_brax_training")
from mujoco_playground import wrapper
print("wrap_for_brax_training:", sig(wrapper.wrap_for_brax_training))
print("wrapper members:", [n for n in dir(wrapper) if not n.startswith("_")])

hr("brax ppo.train / networks")
from brax.training.agents.ppo import train as ppo
from brax.training.agents.ppo import networks as ppo_networks
print("ppo.train params:", list(inspect.signature(ppo.train).parameters.keys()))
print("make_ppo_networks:", sig(ppo_networks.make_ppo_networks))
print("make_inference_fn:", sig(ppo_networks.make_inference_fn))

hr("brax ppo.checkpoint")
try:
    from brax.training.agents.ppo import checkpoint as ppo_ckpt
    print("checkpoint members:", [n for n in dir(ppo_ckpt) if not n.startswith("_")])
    for fn in ["load_policy", "load_config", "save"]:
        if hasattr(ppo_ckpt, fn):
            print(f"  {fn}:", sig(getattr(ppo_ckpt, fn)))
except Exception as e:
    print("ppo.checkpoint import fail:", type(e).__name__, e)

print("\n>>> INTROSPECTION DONE")
