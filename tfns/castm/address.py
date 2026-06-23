"""Canonical address book for Context-Addressed Synaptic Tensor Memory.

This module implements the address algebra of spec sections 5.2, 5.4, 9.3 and
the numerical invariants of section 16 (address normalization, codebook
orthogonality, duality, rank).

Design notes
------------
* The codebook ``K`` has shape ``(d_k, n_max)`` with one canonical address per
  column. A preallocated orthonormal codebook is the preferred implementation
  (``K^T K = I``); the general nonorthogonal path is supported for tests and
  future dynamic address growth.
* ``used`` marks allocated columns. ``rank`` (section 16.6) is the number of
  allocated addresses.
* The dual matrix ``D = K (K^T K + lam I)^{-1}`` satisfies ``D^T K = I`` at full
  rank with ``lam = 0``. For an orthonormal codebook ``D = K`` and ``d_i = k_i``.
* No game / task / label identity ever enters this module. Addresses are
  internal memory coordinates allocated from experience.

All public state is a JAX pytree (``flax.struct.dataclass``) so it jits and
serializes exactly.
"""

from __future__ import annotations

from typing import Any

from flax import struct
import jax
import jax.numpy as jnp
import numpy as np


@struct.dataclass
class AddressBook:
    """Canonical codebook plus allocation state.

    Attributes
    ----------
    K : ``(d_k, n_max)`` codebook columns (the canonical addresses).
    used : ``(n_max,)`` boolean allocation mask.
    """

    K: jnp.ndarray
    used: jnp.ndarray

    @property
    def d_k(self) -> int:
        return int(self.K.shape[0])

    @property
    def n_max(self) -> int:
        return int(self.K.shape[1])


def orthonormal_codebook(d_k: int, n_max: int, seed: int = 0) -> jnp.ndarray:
    """Return a ``(d_k, n_max)`` matrix with orthonormal columns.

    Built from a QR factorization of a seeded Gaussian and re-orthogonalized
    once (modified Gram-Schmidt style via a second QR) to drive ``K^T K - I``
    down to floating-point round-off. ``n_max`` must not exceed ``d_k``.
    """

    d_k = int(d_k)
    n_max = int(n_max)
    if n_max > d_k:
        raise ValueError(f"n_max={n_max} cannot exceed d_k={d_k} for an orthonormal codebook")
    rng = np.random.default_rng(int(seed))
    g = rng.standard_normal((d_k, d_k)).astype(np.float64)
    q, _ = np.linalg.qr(g)
    # Second pass tightens orthogonality after the float cast.
    q, _ = np.linalg.qr(q)
    cols = q[:, :n_max].astype(np.float32)
    return jnp.asarray(cols)


def empty_address_book(d_k: int = 128, n_max: int = 64, seed: int = 0) -> AddressBook:
    """Create an address book with a preallocated orthonormal codebook, none used."""

    K = orthonormal_codebook(d_k, n_max, seed)
    used = jnp.zeros((int(n_max),), dtype=bool)
    return AddressBook(K=K, used=used)


def address_book_from_matrix(K: Any) -> AddressBook:
    """Wrap an explicit (possibly nonorthogonal) ``(d_k, n)`` matrix, all used.

    Columns are normalized to unit length (section 16.1). Used for the
    nonorthogonal dual-address test (spec 17.2).
    """

    K = jnp.asarray(K, dtype=jnp.float32)
    norms = jnp.linalg.norm(K, axis=0, keepdims=True)
    K = K / jnp.maximum(norms, 1e-12)
    used = jnp.ones((int(K.shape[1]),), dtype=bool)
    return AddressBook(K=K, used=used)


def num_used(book: AddressBook) -> int:
    return int(jnp.sum(book.used.astype(jnp.int32)))


def code(book: AddressBook, i: int) -> jnp.ndarray:
    """Return canonical address ``k_i`` (column ``i``)."""

    return book.K[:, int(i)]


def used_codes(book: AddressBook) -> jnp.ndarray:
    """Return ``(d_k, n_used)`` matrix of currently allocated addresses."""

    idx = np.where(np.asarray(book.used))[0]
    return book.K[:, jnp.asarray(idx)] if idx.size else book.K[:, :0]


def dual_matrix(K_used: Any, lam: float = 0.0) -> jnp.ndarray:
    """Return ``D = K (K^T K + lam I)^{-1}`` (section 5.2).

    With full column rank and ``lam = 0`` this satisfies ``D^T K = I``. For an
    orthonormal codebook ``D = K``.
    """

    K_used = jnp.asarray(K_used, dtype=jnp.float32)
    n = int(K_used.shape[1])
    if n == 0:
        return K_used
    gram = K_used.T @ K_used + float(lam) * jnp.eye(n, dtype=K_used.dtype)
    return K_used @ jnp.linalg.inv(gram)


def dual(book: AddressBook, i: int, lam: float = 0.0, orthonormal: bool = True) -> jnp.ndarray:
    """Return the dual address ``d_i`` for context ``i``.

    For an orthonormal codebook ``d_i = k_i`` exactly. Otherwise the dual is
    computed from the full used set so that ``d_i^T k_j = delta_ij``.
    """

    if orthonormal:
        return code(book, i)
    idx = np.where(np.asarray(book.used))[0]
    pos = int(np.where(idx == int(i))[0][0])
    D = dual_matrix(book.K[:, jnp.asarray(idx)], lam=lam)
    return D[:, pos]


def allocate_canonical(book: AddressBook) -> tuple[AddressBook, int]:
    """Allocate the next unused canonical (orthonormal) address.

    Returns the updated book and the allocated index. Raises if the codebook is
    exhausted (section 23.9 — capacity exhaustion must be detected, not silently
    wrapped).
    """

    free = np.where(~np.asarray(book.used))[0]
    if free.size == 0:
        raise RuntimeError("address codebook exhausted; recompression/merge required")
    idx = int(free[0])
    used = book.used.at[idx].set(True)
    return book.replace(used=used), idx


def allocate_novel_from_candidate(
    book: AddressBook,
    u: Any,
    eps_novel: float = 1e-3,
) -> tuple[AddressBook, int, float]:
    """Allocate a genuinely novel address from a candidate query ``u``.

    Implements the residual construction of spec 5.4::

        P_perp = I - K K^+ ,  r = P_perp u ,  k_new = r / ||r||

    The new column is written into the next free slot. Returns the updated book,
    the allocated index, and ``||r||`` (the novelty residual norm). If
    ``||r|| <= eps_novel`` the candidate lies in the span of existing addresses
    and no allocation occurs (returns index ``-1``).
    """

    u = jnp.asarray(u, dtype=jnp.float32)
    K_used = used_codes(book)
    if int(K_used.shape[1]) == 0:
        r = u
    else:
        # P_perp u = u - K (K^+ u); K^+ = (K^T K)^-1 K^T for full column rank.
        Kp = jnp.linalg.pinv(K_used)
        r = u - K_used @ (Kp @ u)
    r_norm = float(jnp.linalg.norm(r))
    if r_norm <= float(eps_novel):
        return book, -1, r_norm
    k_new = r / r_norm
    free = np.where(~np.asarray(book.used))[0]
    if free.size == 0:
        raise RuntimeError("address codebook exhausted; recompression/merge required")
    idx = int(free[0])
    K = book.K.at[:, idx].set(k_new)
    used = book.used.at[idx].set(True)
    return book.replace(K=K, used=used), idx, r_norm


def effective_address(book: AddressBook, posterior: Any, indices: Any | None = None) -> jnp.ndarray:
    """Return the effective address ``z = sum_i pi_i k_i`` (spec 9.3).

    ``posterior`` is a weight per address. If ``indices`` is given, ``posterior``
    is aligned to those columns; otherwise it must cover all ``n_max`` columns.
    """

    posterior = jnp.asarray(posterior, dtype=jnp.float32)
    if indices is None:
        return book.K @ posterior
    indices = jnp.asarray(indices, dtype=jnp.int32)
    cols = book.K[:, indices]
    return cols @ posterior


def compensation_vector(book: AddressBook, orthonormal: bool = True) -> jnp.ndarray:
    """Return ``g`` satisfying ``K^T g = 1`` (spec 8.2) for shared consolidation.

    Minimum-norm solution ``g = K (K^T K)^{-1} 1``. For an orthonormal codebook
    ``g = K 1 = sum_i k_i``.
    """

    K_used = used_codes(book)
    n = int(K_used.shape[1])
    if n == 0:
        return jnp.zeros((book.d_k,), dtype=jnp.float32)
    ones = jnp.ones((n,), dtype=jnp.float32)
    if orthonormal:
        return K_used @ ones
    gram = K_used.T @ K_used
    return K_used @ jnp.linalg.solve(gram, ones)


# --- Numerical invariants (spec section 16) -------------------------------------


def address_norm_error(book: AddressBook) -> float:
    """Max ``| ||k_i||_2 - 1 |`` over used addresses (spec 16.1)."""

    K_used = used_codes(book)
    if int(K_used.shape[1]) == 0:
        return 0.0
    norms = jnp.linalg.norm(K_used, axis=0)
    return float(jnp.max(jnp.abs(norms - 1.0)))


def orthogonality_error(book: AddressBook) -> float:
    """``||K^T K - I||_max`` over used addresses (spec 16.2)."""

    K_used = used_codes(book)
    n = int(K_used.shape[1])
    if n == 0:
        return 0.0
    gram = K_used.T @ K_used
    return float(jnp.max(jnp.abs(gram - jnp.eye(n, dtype=gram.dtype))))


def duality_error(book: AddressBook, lam: float = 0.0) -> float:
    """``||D^T K - I||_max`` over used addresses (spec 16.3)."""

    K_used = used_codes(book)
    n = int(K_used.shape[1])
    if n == 0:
        return 0.0
    D = dual_matrix(K_used, lam=lam)
    prod = D.T @ K_used
    return float(jnp.max(jnp.abs(prod - jnp.eye(n, dtype=prod.dtype))))


def address_rank(book: AddressBook) -> int:
    """Numerical rank of the used codebook (must equal ``num_used``, spec 16.6)."""

    K_used = used_codes(book)
    if int(K_used.shape[1]) == 0:
        return 0
    s = jnp.linalg.svd(K_used, compute_uv=False)
    tol = float(jnp.max(s)) * max(K_used.shape) * jnp.finfo(K_used.dtype).eps
    return int(jnp.sum((s > tol).astype(jnp.int32)))


def condition_number(book: AddressBook) -> float:
    """Condition number of the used codebook (routing/dual stability)."""

    K_used = used_codes(book)
    if int(K_used.shape[1]) == 0:
        return 1.0
    s = jnp.linalg.svd(K_used, compute_uv=False)
    s_min = float(jnp.min(s))
    if s_min <= 0.0:
        return float("inf")
    return float(jnp.max(s) / s_min)


def has_nan(book: AddressBook) -> bool:
    """True if any address contains NaN/Inf (spec 16.7)."""

    return bool(~jnp.all(jnp.isfinite(book.K)))


__all__ = [
    "AddressBook",
    "address_book_from_matrix",
    "address_norm_error",
    "address_rank",
    "allocate_canonical",
    "allocate_novel_from_candidate",
    "code",
    "compensation_vector",
    "condition_number",
    "duality_error",
    "dual",
    "dual_matrix",
    "effective_address",
    "empty_address_book",
    "has_nan",
    "num_used",
    "orthogonality_error",
    "orthonormal_codebook",
    "used_codes",
]
