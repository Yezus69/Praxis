"""Address-book invariant tests (spec 16.1-16.3, 16.6; 5.4; 9.3; 17.2, 17.3)."""

from __future__ import annotations

import jax.numpy as jnp
import numpy as np
import pytest

from tfns.castm import address as addr


def test_orthonormal_codebook_invariants():
    book = addr.empty_address_book(d_k=128, n_max=64, seed=1)
    # Allocate all 64 addresses.
    for _ in range(64):
        book, _ = addr.allocate_canonical(book)
    assert addr.num_used(book) == 64
    assert addr.address_norm_error(book) < 1e-5      # spec 16.1
    assert addr.orthogonality_error(book) < 1e-5     # spec 16.2
    assert addr.duality_error(book) < 1e-5           # spec 16.3
    assert addr.address_rank(book) == 64             # spec 16.6
    assert addr.condition_number(book) < 1.0 + 1e-4
    assert not addr.has_nan(book)


def test_orthonormal_dual_equals_code():
    book = addr.empty_address_book(d_k=64, n_max=16, seed=2)
    for _ in range(16):
        book, _ = addr.allocate_canonical(book)
    for i in range(16):
        d_i = addr.dual(book, i, orthonormal=True)
        k_i = addr.code(book, i)
        np.testing.assert_allclose(np.asarray(d_i), np.asarray(k_i), atol=1e-6)


def test_codebook_exhaustion_raises():
    book = addr.empty_address_book(d_k=8, n_max=4, seed=3)
    for _ in range(4):
        book, _ = addr.allocate_canonical(book)
    with pytest.raises(RuntimeError):
        addr.allocate_canonical(book)


def test_nonorthogonal_dual_duality(spec_17_2=True):
    # Spec 17.2: deliberately nonorthogonal addresses; verify D^T K = I exactly.
    rng = np.random.default_rng(7)
    K = rng.standard_normal((12, 6)).astype(np.float32)
    book = addr.address_book_from_matrix(K)
    assert addr.duality_error(book, lam=0.0) < 1e-4
    # Dual columns satisfy d_i . k_j = delta_ij.
    K_used = np.asarray(addr.used_codes(book))
    D = np.asarray(addr.dual_matrix(addr.used_codes(book), lam=0.0))
    prod = D.T @ K_used
    np.testing.assert_allclose(prod, np.eye(6), atol=1e-4)


def test_novel_address_residual_orthogonality():
    # Spec 17.3 / 5.4: allocate from r = (I - K K^+) u; verify orthogonality.
    book = addr.empty_address_book(d_k=32, n_max=16, seed=11)
    for _ in range(4):
        book, _ = addr.allocate_canonical(book)
    rng = np.random.default_rng(5)
    u = jnp.asarray(rng.standard_normal((32,)).astype(np.float32))
    book2, idx, r_norm = addr.allocate_novel_from_candidate(book, u, eps_novel=1e-3)
    assert idx >= 0 and r_norm > 1e-3
    k_new = addr.code(book2, idx)
    # New address is orthogonal to all previously used addresses.
    for j in range(4):
        dot = float(jnp.dot(k_new, addr.code(book2, j)))
        assert abs(dot) < 1e-5
    assert addr.orthogonality_error(book2) < 1e-4


def test_novel_candidate_in_span_not_allocated():
    book = addr.empty_address_book(d_k=16, n_max=8, seed=13)
    for _ in range(3):
        book, _ = addr.allocate_canonical(book)
    # A candidate inside the span of used addresses yields ~0 residual.
    u = addr.code(book, 0) * 2.0 + addr.code(book, 1) * 0.5
    book2, idx, r_norm = addr.allocate_novel_from_candidate(book, u, eps_novel=1e-3)
    assert idx == -1
    assert r_norm < 1e-3
    assert addr.num_used(book2) == 3


def test_compensation_vector_satisfies_ones():
    book = addr.empty_address_book(d_k=32, n_max=8, seed=17)
    for _ in range(8):
        book, _ = addr.allocate_canonical(book)
    g = addr.compensation_vector(book, orthonormal=True)
    K_used = addr.used_codes(book)
    prod = np.asarray(K_used.T @ g)
    np.testing.assert_allclose(prod, np.ones(8), atol=1e-5)
