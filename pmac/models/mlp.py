"""Plain-pytree MLP with no-op growth adapters."""

from __future__ import annotations

import jax
import jax.numpy as jnp


def _init_weight(key, in_dim: int, out_dim: int, std: float):
    return jax.random.normal(key, (in_dim, out_dim), dtype=jnp.float32) * std


def init_mlp(key, layer_sizes: list[int], scale=None):
    """Initialize params = {layers: [{w,b}, ...], adapters: []}."""
    if len(layer_sizes) < 2:
        raise ValueError("layer_sizes must include input and output sizes")
    keys = jax.random.split(key, len(layer_sizes) - 1)
    layers = []
    last = len(layer_sizes) - 2
    for i, (in_dim, out_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        if scale is None:
            std = (2.0 / float(in_dim)) ** 0.5 if i < last else (1.0 / float(in_dim)) ** 0.5
        else:
            std = float(scale)
        layers.append(
            {
                "w": _init_weight(keys[i], int(in_dim), int(out_dim), std),
                "b": jnp.zeros((int(out_dim),), dtype=jnp.float32),
            }
        )
    return {"layers": layers, "adapters": []}


def _adapter_apply(adapter, h):
    z = jnp.maximum(h @ adapter["down"]["w"] + adapter["down"]["b"], 0.0)
    return z @ adapter["up"]["w"] + adapter["up"]["b"]


def mlp_apply(params, x):
    """Apply an MLP: ReLU hidden layers, optional residual adapters, linear head."""
    layers = params["layers"]
    h = jnp.asarray(x)
    for layer in layers[:-1]:
        h = jnp.maximum(h @ layer["w"] + layer["b"], 0.0)
    for adapter in params.get("adapters", []):
        h = h + _adapter_apply(adapter, h)
    final = layers[-1]
    return h @ final["w"] + final["b"]


def grow_adapter(key, params, hidden_dim, rank=64):
    """Append a zero-output residual adapter, preserving current behavior."""
    hidden_dim = int(hidden_dim)
    rank = int(rank)
    key_down, _ = jax.random.split(key)
    down_std = (2.0 / float(hidden_dim)) ** 0.5
    adapter = {
        "down": {
            "w": _init_weight(key_down, hidden_dim, rank, down_std),
            "b": jnp.zeros((rank,), dtype=jnp.float32),
        },
        "up": {
            "w": jnp.zeros((rank, hidden_dim), dtype=jnp.float32),
            "b": jnp.zeros((hidden_dim,), dtype=jnp.float32),
        },
    }
    new_params = dict(params)
    new_params["layers"] = list(params["layers"])
    new_params["adapters"] = list(params.get("adapters", [])) + [adapter]
    return new_params


def num_params(params) -> int:
    """Count scalar parameters in a pytree."""
    return sum(int(leaf.size) for leaf in jax.tree_util.tree_leaves(params))


__all__ = ["init_mlp", "mlp_apply", "grow_adapter", "num_params"]
