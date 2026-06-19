"""Protected-subspace projection for TFNS affine updates."""

from __future__ import annotations

import dataclasses
from collections.abc import Mapping
from typing import Any

from flax.core import FrozenDict, freeze, unfreeze
import jax.numpy as jnp


Path = tuple[str, ...]
GatePaths = Mapping[str, tuple[Path, Path, Path]]


@dataclasses.dataclass(frozen=True)
class ProtectedModule:
    """Descriptor for one protected affine operator family."""

    name: str
    kind: str
    d_aug: int
    kernel_path: Path | None = None
    bias_path: Path | None = None
    gate_paths: GatePaths | None = None
    kh: int | None = None
    kw: int | None = None
    stride: tuple[int, int] | None = None
    c_in: int | None = None


def _left_project(Kbar: jnp.ndarray, U: jnp.ndarray) -> jnp.ndarray:
    U = jnp.asarray(U, dtype=Kbar.dtype)
    if int(U.shape[1]) == 0:
        return Kbar
    return Kbar - U @ (U.T @ Kbar)


def project_affine(
    dW: jnp.ndarray,
    db: jnp.ndarray,
    U: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Project a dense affine update with the bias as the final row."""

    dW = jnp.asarray(dW)
    db = jnp.asarray(db, dtype=dW.dtype)
    Kbar = jnp.concatenate([dW, db[None, :]], axis=0)
    safe = _left_project(Kbar, U)
    return safe[:-1, :], safe[-1, :]


def project_gru_gate(
    dW: jnp.ndarray,
    dU: jnp.ndarray,
    db: jnp.ndarray,
    U: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """Project one GRU gate update against the shared ``[x; h; 1]`` basis."""

    dW = jnp.asarray(dW)
    dU = jnp.asarray(dU, dtype=dW.dtype)
    db = jnp.asarray(db, dtype=dW.dtype)
    input_dim = int(dW.shape[0])
    hidden = int(dU.shape[0])
    Kbar = jnp.concatenate([dW, dU, db[None, :]], axis=0)
    safe = _left_project(Kbar, U)
    return safe[:input_dim, :], safe[input_dim : input_dim + hidden, :], safe[-1, :]


def _normalize_stride(stride: int | tuple[int, int]) -> tuple[int, int]:
    if isinstance(stride, tuple):
        if len(stride) != 2:
            raise ValueError(f"expected 2D stride, got {stride!r}")
        return int(stride[0]), int(stride[1])
    return int(stride), int(stride)


def conv_patches(
    x: jnp.ndarray,
    kh: int,
    kw: int,
    stride: int | tuple[int, int],
    c_in: int,
) -> jnp.ndarray:
    """Extract VALID NHWC im2col patches matching ``kernel.reshape`` order."""

    x = jnp.asarray(x)
    if x.ndim != 4:
        raise ValueError(f"expected NHWC input, got rank {x.ndim}")
    kh = int(kh)
    kw = int(kw)
    c_in = int(c_in)
    stride_h, stride_w = _normalize_stride(stride)
    batch, height, width, channels = x.shape
    if int(channels) != c_in:
        raise ValueError(f"expected c_in={c_in}, got {channels}")
    out_h = (int(height) - kh) // stride_h + 1
    out_w = (int(width) - kw) // stride_w + 1
    if out_h <= 0 or out_w <= 0:
        raise ValueError("kernel is larger than the VALID convolution input")

    offsets = []
    for row in range(kh):
        for col in range(kw):
            offsets.append(
                x[
                    :,
                    row : row + out_h * stride_h : stride_h,
                    col : col + out_w * stride_w : stride_w,
                    :,
                ]
            )
    patch_grid = jnp.stack(offsets, axis=-2)
    return patch_grid.reshape((int(batch) * out_h * out_w, kh * kw * c_in))


def project_conv(
    dK: jnp.ndarray,
    db: jnp.ndarray,
    U: jnp.ndarray,
) -> tuple[jnp.ndarray, jnp.ndarray]:
    """Project a convolution update after C-order patch/kernel flattening."""

    dK = jnp.asarray(dK)
    db = jnp.asarray(db, dtype=dK.dtype)
    kh, kw, c_in, c_out = dK.shape
    flat = dK.reshape((int(kh) * int(kw) * int(c_in), int(c_out)))
    Kbar = jnp.concatenate([flat, db[None, :]], axis=0)
    safe = _left_project(Kbar, U)
    return safe[:-1, :].reshape(dK.shape), safe[-1, :]


def collect_conv_basis_columns(
    x: jnp.ndarray,
    kh: int,
    kw: int,
    stride: int | tuple[int, int],
    c_in: int,
) -> jnp.ndarray:
    """Return augmented VALID patch columns for convolution basis construction."""

    patches = conv_patches(x, kh, kw, stride, c_in)
    ones = jnp.ones((patches.shape[0], 1), dtype=patches.dtype)
    return jnp.concatenate([patches, ones], axis=-1).T


def _params_root(params: Mapping[str, Any]) -> Mapping[str, Any]:
    if isinstance(params, FrozenDict):
        params = unfreeze(params)
    if "params" in params and isinstance(params["params"], Mapping):
        return params["params"]
    return params


def _has_path(root: Mapping[str, Any], path: Path) -> bool:
    cur: Any = root
    for part in path:
        if not isinstance(cur, Mapping) or part not in cur:
            return False
        cur = cur[part]
    return True


def _get_path(root: Mapping[str, Any], path: Path) -> Any:
    cur: Any = root
    for part in path:
        cur = cur[part]
    return cur


def _to_mutable_tree(tree: Any) -> Any:
    if isinstance(tree, FrozenDict):
        return unfreeze(tree)
    if isinstance(tree, Mapping):
        return {key: _to_mutable_tree(value) for key, value in tree.items()}
    return tree


def _effective_path(root: Mapping[str, Any], path: Path) -> Path | None:
    if _has_path(root, path):
        return path
    params_path = ("params",) + path
    if _has_path(root, params_path):
        return params_path
    return None


def _set_path(root: dict[str, Any], path: Path, value: Any) -> None:
    cur: Any = root
    for part in path[:-1]:
        cur = cur[part]
    cur[path[-1]] = value


def build_protected_modules(
    params: Mapping[str, Any],
    model_config: Any,
) -> dict[str, ProtectedModule]:
    """Build descriptors for M1 protected affine operators from params."""

    root = _params_root(params)
    modules: dict[str, ProtectedModule] = {}

    conv_strides = tuple(getattr(model_config, "conv_strides", (1, 1, 1)))
    for idx in range(1, 4):
        module = f"conv{idx}"
        kernel_path = ("encoder", module, "kernel")
        bias_path = ("encoder", module, "bias")
        if not (_has_path(root, kernel_path) and _has_path(root, bias_path)):
            continue
        kernel = _get_path(root, kernel_path)
        kh, kw, c_in, _ = kernel.shape
        stride = _normalize_stride(conv_strides[idx - 1])
        modules[f"encoder_{module}"] = ProtectedModule(
            name=f"encoder_{module}",
            kind="conv",
            d_aug=int(kh) * int(kw) * int(c_in) + 1,
            kernel_path=kernel_path,
            bias_path=bias_path,
            kh=int(kh),
            kw=int(kw),
            stride=stride,
            c_in=int(c_in),
        )

    dense_kernel = ("encoder", "dense", "kernel")
    dense_bias = ("encoder", "dense", "bias")
    if _has_path(root, dense_kernel) and _has_path(root, dense_bias):
        kernel = _get_path(root, dense_kernel)
        modules["encoder_dense"] = ProtectedModule(
            name="encoder_dense",
            kind="dense",
            d_aug=int(kernel.shape[0]) + 1,
            kernel_path=dense_kernel,
            bias_path=dense_bias,
        )

    gate_paths = {
        gate: (("gru", f"W_{gate}"), ("gru", f"U_{gate}"), ("gru", f"b_{gate}"))
        for gate in ("z", "r", "n")
    }
    if all(_has_path(root, path) for paths in gate_paths.values() for path in paths):
        W_z = _get_path(root, gate_paths["z"][0])
        U_z = _get_path(root, gate_paths["z"][1])
        modules["gru"] = ProtectedModule(
            name="gru",
            kind="gru_gate",
            d_aug=int(W_z.shape[0]) + int(U_z.shape[0]) + 1,
            gate_paths=gate_paths,
        )

    for head in ("policy_head", "value_head", "key_head"):
        kernel_path = (head, "affine", "kernel")
        bias_path = (head, "affine", "bias")
        if _has_path(root, kernel_path) and _has_path(root, bias_path):
            kernel = _get_path(root, kernel_path)
            modules[head] = ProtectedModule(
                name=head,
                kind="dense",
                d_aug=int(kernel.shape[0]) + 1,
                kernel_path=kernel_path,
                bias_path=bias_path,
            )

    # Adapter bases are introduced only after adapters contribute to certified
    # behavior in a later milestone, so dormant M1 adapter params are skipped.
    return modules


def project_update(
    update_tree: Mapping[str, Any],
    bases: Mapping[str, jnp.ndarray],
    modules: Mapping[str, ProtectedModule],
) -> Mapping[str, Any]:
    """Project protected update leaves in an optax-style update pytree."""

    was_frozen = isinstance(update_tree, FrozenDict)
    result = _to_mutable_tree(update_tree)

    for name, module in modules.items():
        U = bases.get(name)
        if U is None or int(jnp.asarray(U).shape[1]) == 0:
            continue

        if module.kind in {"dense", "conv"}:
            if module.kernel_path is None or module.bias_path is None:
                continue
            kernel_path = _effective_path(result, module.kernel_path)
            bias_path = _effective_path(result, module.bias_path)
            if kernel_path is None or bias_path is None:
                continue
            d_kernel = _get_path(result, kernel_path)
            d_bias = _get_path(result, bias_path)
            if module.kind == "dense":
                safe_kernel, safe_bias = project_affine(d_kernel, d_bias, U)
            else:
                safe_kernel, safe_bias = project_conv(d_kernel, d_bias, U)
            _set_path(result, kernel_path, safe_kernel)
            _set_path(result, bias_path, safe_bias)
            continue

        if module.kind == "gru_gate" and module.gate_paths is not None:
            for paths in module.gate_paths.values():
                W_path = _effective_path(result, paths[0])
                U_path = _effective_path(result, paths[1])
                b_path = _effective_path(result, paths[2])
                if W_path is None or U_path is None or b_path is None:
                    continue
                dW = _get_path(result, W_path)
                dU = _get_path(result, U_path)
                db = _get_path(result, b_path)
                safe_dW, safe_dU, safe_db = project_gru_gate(dW, dU, db, U)
                _set_path(result, W_path, safe_dW)
                _set_path(result, U_path, safe_dU)
                _set_path(result, b_path, safe_db)

    return freeze(result) if was_frozen else result


__all__ = [
    "ProtectedModule",
    "build_protected_modules",
    "collect_conv_basis_columns",
    "conv_patches",
    "project_affine",
    "project_conv",
    "project_gru_gate",
    "project_update",
]
