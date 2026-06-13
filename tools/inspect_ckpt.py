import sys
import orbax.checkpoint as ocp
import numpy as np

ckpt = sys.argv[1]
p = ocp.PyTreeCheckpointer().restore(ckpt)
print("top type:", type(p).__name__,
      "keys/len:", list(p.keys()) if isinstance(p, dict) else len(p))


def show(x, prefix=""):
    if isinstance(x, dict):
        for k, v in x.items():
            show(v, prefix + f"/{k}")
    elif isinstance(x, (list, tuple)):
        for i, v in enumerate(x):
            show(v, prefix + f"[{i}]")
    else:
        print(f"  {prefix}: {type(x).__name__} shape={getattr(x, 'shape', None)}")


show(p)
