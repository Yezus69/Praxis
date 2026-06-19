"""Basis construction utilities for protected TFNS subspaces."""

from __future__ import annotations

from typing import Any

import jax.numpy as jnp


def empty_basis(d_aug: int) -> jnp.ndarray:
    """Return an empty augmented activation basis."""

    return jnp.zeros((int(d_aug), 0), dtype=jnp.float32)


def _working_dtype(U: jnp.ndarray, A: jnp.ndarray) -> jnp.dtype:
    return jnp.result_type(U, A, jnp.float32)


def _zero_energy_tolerance(dtype: jnp.dtype, reference_energy: jnp.ndarray) -> float:
    eps = float(jnp.finfo(dtype).eps)
    scale = max(float(reference_energy), 1.0)
    return 1000.0 * eps * scale


def expand_basis(
    U: jnp.ndarray,
    A: jnp.ndarray,
    energy: float = 0.995,
    max_rank: int | None = None,
) -> tuple[jnp.ndarray, dict[str, Any]]:
    """Append residual activation directions to an orthonormal basis.

    ``A`` contains augmented activation columns, already weighted by the caller.
    ``info["residual_energy_captured"]`` is the captured fraction of residual
    energy; ``info["captured_energy"]`` and ``info["discarded_energy"]`` are
    absolute squared-Frobenius energies.
    """

    if not 0.0 < float(energy) <= 1.0:
        raise ValueError("energy must be in (0, 1]")

    U = jnp.asarray(U)
    A = jnp.asarray(A)
    if U.ndim != 2 or A.ndim != 2:
        raise ValueError("U and A must both be rank-2 arrays")
    if U.shape[0] != A.shape[0]:
        raise ValueError(f"U and A disagree on d_aug: {U.shape[0]} vs {A.shape[0]}")

    dtype = _working_dtype(U, A)
    U = U.astype(dtype)
    A = A.astype(dtype)
    d_aug = int(U.shape[0])
    rank = int(U.shape[1])
    max_allowed = d_aug if max_rank is None else min(int(max_rank), d_aug)
    allowed_add = max(0, max_allowed - rank)

    base_info: dict[str, Any] = {
        "added_rank": 0,
        "residual_energy_captured": 0.0,
        "captured_energy": 0.0,
        "discarded_energy": 0.0,
        "capacity_hit": False,
    }
    if A.shape[1] == 0:
        return U, base_info

    R = A - U @ (U.T @ A)
    left, singular_values, _ = jnp.linalg.svd(R, full_matrices=False)
    squared = singular_values * singular_values
    total_energy = jnp.sum(squared)
    reference_energy = jnp.sum(A * A)
    if float(total_energy) <= _zero_energy_tolerance(dtype, reference_energy):
        return U, base_info

    cumulative = jnp.cumsum(squared)
    threshold = float(energy) * float(total_energy)
    k_needed = int(jnp.searchsorted(cumulative, threshold, side="left")) + 1
    k_needed = min(k_needed, int(singular_values.shape[0]), max(0, d_aug - rank))
    k_append = min(k_needed, allowed_add)

    if k_append > 0:
        combined = jnp.concatenate([U, left[:, :k_append]], axis=1)
        U_new, _ = jnp.linalg.qr(combined, mode="reduced")
        U_new = U_new[:, : rank + k_append].astype(dtype)
    else:
        U_new = U

    captured_energy = jnp.sum(squared[:k_append])
    captured_fraction = captured_energy / total_energy
    capacity_hit = bool(k_append < k_needed)
    discarded_energy = total_energy - captured_energy if capacity_hit else jnp.asarray(0.0, dtype)
    info = {
        "added_rank": int(k_append),
        "residual_energy_captured": float(captured_fraction),
        "captured_energy": float(captured_energy),
        "discarded_energy": float(discarded_energy),
        "capacity_hit": capacity_hit,
    }
    return U_new, info


def free_rank_fraction(U: jnp.ndarray, d_aug: int) -> float:
    """Return the unoccupied activation rank fraction for a module."""

    rank = int(jnp.asarray(U).shape[1])
    return 1.0 - float(rank) / float(d_aug)


def represented_energy(U: jnp.ndarray, A: jnp.ndarray) -> jnp.ndarray:
    """Return ``||U U.T A||_F^2 / ||A||_F^2``."""

    U = jnp.asarray(U)
    A = jnp.asarray(A)
    dtype = _working_dtype(U, A)
    U = U.astype(dtype)
    A = A.astype(dtype)
    projection = U @ (U.T @ A)
    numerator = jnp.sum(projection * projection)
    denominator = jnp.sum(A * A)
    return jnp.where(denominator > 0.0, numerator / denominator, jnp.asarray(0.0, dtype))


def residual_norm(U: jnp.ndarray, x: jnp.ndarray) -> jnp.ndarray:
    """Return the norm of the component of ``x`` outside ``span(U)``."""

    U = jnp.asarray(U)
    x = jnp.asarray(x)
    dtype = _working_dtype(U, x)
    U = U.astype(dtype)
    x = x.astype(dtype)
    residual = x - U @ (U.T @ x)
    return jnp.linalg.norm(residual)


def orthonormality_error(U: jnp.ndarray) -> jnp.ndarray:
    """Return ``||U.T U - I||_max`` for an activation basis."""

    U = jnp.asarray(U)
    rank = int(U.shape[1])
    if rank == 0:
        return jnp.asarray(0.0, dtype=U.dtype)
    gram = U.T @ U
    return jnp.max(jnp.abs(gram - jnp.eye(rank, dtype=U.dtype)))


def to_storage(U: jnp.ndarray, fp16: bool = False) -> jnp.ndarray:
    """Convert a basis to its persisted dtype."""

    dtype = jnp.float16 if fp16 else jnp.float32
    return jnp.asarray(U, dtype=dtype)


def from_storage(stored: jnp.ndarray) -> jnp.ndarray:
    """Reload a stored basis for projection math."""

    return jnp.asarray(stored, dtype=jnp.float32)


__all__ = [
    "empty_basis",
    "expand_basis",
    "free_rank_fraction",
    "from_storage",
    "orthonormality_error",
    "represented_energy",
    "residual_norm",
    "to_storage",
]
