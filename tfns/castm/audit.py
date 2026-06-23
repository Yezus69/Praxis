"""Exact decoded-weight audit APIs (spec sections 16, 15.3, 14.5).

These functions measure the numerical invariants that guard every commit,
recompression, and consolidation transaction. They operate on
:class:`tfns.castm.synaptic.SynapticMemory` and
:class:`tfns.castm.address.AddressBook`.

The central invariant (spec 16.4): a committed write for context ``i`` must not
change the decoded layer weights for any protected context ``j != i``, measured
as relative Frobenius drift. Section 16.5 verifies the intended write landed at
context ``i``.
"""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp
import numpy as np

from tfns.castm import address as addr
from tfns.castm.synaptic import SynapticMemory, decode_bias, decode_weight


EPS = 1e-12


def decoded_weights(mem: SynapticMemory, book: addr.AddressBook) -> dict[int, jnp.ndarray]:
    """Decode effective weights at every used canonical address."""

    used = np.where(np.asarray(book.used))[0]
    return {int(i): decode_weight(mem, addr.code(book, int(i))) for i in used}


def decoded_fingerprints(mem: SynapticMemory, book: addr.AddressBook) -> dict[int, dict[str, float]]:
    """Per-address decoded-weight fingerprints (spec 14.5 audit state).

    Fingerprint = (Frobenius norm, sum, a cheap hashed checksum) so two states
    can be compared for exact-restore (spec 17.10) without storing full matrices.
    """

    out: dict[int, dict[str, float]] = {}
    for i, W in decoded_weights(mem, book).items():
        Wn = np.asarray(W, dtype=np.float64)
        b = np.asarray(decode_bias(mem, addr.code(book, int(i))), dtype=np.float64)
        out[int(i)] = {
            "w_fro": float(np.linalg.norm(Wn)),
            "w_sum": float(np.sum(Wn)),
            "b_fro": float(np.linalg.norm(b)),
            "checksum": float(np.sum(Wn * np.cos(np.arange(Wn.size).reshape(Wn.shape) + 1.0))),
        }
    return out


def _rel_fro(delta: jnp.ndarray, ref: jnp.ndarray) -> float:
    num = float(jnp.linalg.norm(delta))
    den = float(jnp.linalg.norm(ref)) + EPS
    return num / den


def noninterference_error(
    mem_before: SynapticMemory,
    mem_after: SynapticMemory,
    book: addr.AddressBook,
    write_ctx: int,
) -> dict[int, float]:
    """Relative decoded-weight drift at every protected context ``j != write_ctx``.

    Implements spec 16.4. Returns a dict ``{ctx_index: rel_fro_drift}``. The
    write passes the noninterference invariant when ``max(values) < eps_write``.
    """

    used = np.where(np.asarray(book.used))[0]
    errors: dict[int, float] = {}
    for j in [int(x) for x in used]:
        if j == int(write_ctx):
            continue
        k_j = addr.code(book, j)
        w_before = decode_weight(mem_before, k_j)
        w_after = decode_weight(mem_after, k_j)
        b_before = decode_bias(mem_before, k_j)
        b_after = decode_bias(mem_after, k_j)
        w_err = _rel_fro(w_after - w_before, w_before)
        b_err = _rel_fro(b_after - b_before, b_before) if float(jnp.linalg.norm(b_before)) > 0 else float(
            jnp.linalg.norm(b_after - b_before)
        )
        errors[j] = float(max(w_err, b_err))
    return errors


def max_noninterference_error(
    mem_before: SynapticMemory,
    mem_after: SynapticMemory,
    book: addr.AddressBook,
    write_ctx: int,
) -> float:
    errs = noninterference_error(mem_before, mem_after, book, write_ctx)
    return max(errs.values()) if errs else 0.0


def intended_write_error(
    mem_before: SynapticMemory,
    mem_after: SynapticMemory,
    book: addr.AddressBook,
    write_ctx: int,
    delta_expected: Any,
    *,
    bias_expected: Any | None = None,
) -> float:
    """Relative error of the realized vs intended write at context ``write_ctx`` (spec 16.5)."""

    k_i = addr.code(book, int(write_ctx))
    realized = decode_weight(mem_after, k_i) - decode_weight(mem_before, k_i)
    delta_expected = jnp.asarray(delta_expected, dtype=realized.dtype)
    w_err = _rel_fro(realized - delta_expected, delta_expected)
    if bias_expected is None:
        return float(w_err)
    realized_b = decode_bias(mem_after, k_i) - decode_bias(mem_before, k_i)
    bias_expected = jnp.asarray(bias_expected, dtype=realized_b.dtype)
    den = float(jnp.linalg.norm(bias_expected)) + EPS
    b_err = float(jnp.linalg.norm(realized_b - bias_expected)) / den
    return float(max(w_err, b_err))


def reconstruction_error(
    mem_before: SynapticMemory,
    mem_after: SynapticMemory,
    book: addr.AddressBook,
    ctx_id: int,
) -> dict[str, float]:
    """Max and relative decoded-weight change at one context (recompression audit)."""

    k = addr.code(book, int(ctx_id))
    w_before = decode_weight(mem_before, k)
    w_after = decode_weight(mem_after, k)
    diff = w_after - w_before
    return {
        "max_abs": float(jnp.max(jnp.abs(diff))),
        "rel_fro": _rel_fro(diff, w_before),
    }


def fingerprints_match(
    a: dict[int, dict[str, float]],
    b: dict[int, dict[str, float]],
    tol: float = 1e-6,
) -> bool:
    """Compare two fingerprint dicts within ``tol`` (relative on norms)."""

    if set(a.keys()) != set(b.keys()):
        return False
    for key in a:
        for field in ("w_fro", "w_sum", "b_fro", "checksum"):
            va, vb = a[key][field], b[key][field]
            denom = max(abs(va), abs(vb), 1.0)
            if abs(va - vb) / denom > tol:
                return False
    return True


__all__ = [
    "decoded_fingerprints",
    "decoded_weights",
    "fingerprints_match",
    "intended_write_error",
    "max_noninterference_error",
    "noninterference_error",
    "reconstruction_error",
]
