"""Exact serialization / restore tests for the continual state (spec 14, 17.10)."""

from __future__ import annotations

import jax
import jax.numpy as jnp
import numpy as np

from tfns.castm import address as addr
from tfns.castm import router as rt
from tfns.castm import scratch as scr
from tfns.castm import state as st
from tfns.castm import synaptic as syn
from tfns.castm import transaction as tx


def _build_state(seed=0):
    rng = np.random.default_rng(seed)
    book = addr.empty_address_book(d_k=64, n_max=16, seed=seed)
    banks = {}
    for name, (o, i) in {"dense": (12, 10), "policy": (18, 12), "value": (1, 12)}.items():
        W0 = jnp.asarray(rng.standard_normal((o, i)).astype(np.float32))
        b0 = jnp.asarray(rng.standard_normal((o,)).astype(np.float32))
        banks[name] = syn.empty_synaptic_memory(W0, b0, comp_rank=8, n_slots=16, d_k=64)
    # Discover and commit two contexts.
    for _ in range(2):
        book, _ = addr.allocate_canonical(book)
    for ctx in (0, 1):
        sbank = {}
        for name, mem in banks.items():
            A = jnp.asarray(rng.standard_normal((4, mem.in_dim)).astype(np.float32) * 0.3)
            B = jnp.asarray(rng.standard_normal((mem.out_dim, 4)).astype(np.float32) * 0.3)
            beta = jnp.asarray(rng.standard_normal((mem.out_dim,)).astype(np.float32) * 0.3)
            sbank[name] = scr.ScratchDelta(A_s=A, B_s=B, beta_s=beta)
        banks, rep = tx.commit_scratch_bank(banks, sbank, book, ctx)
        assert rep["accepted"]
    proto = rt.empty_prototype_index(16, 8, 32)
    qw = rng.standard_normal((20, 32)).astype(np.float32)
    proto, _ = rt.allocate_context(proto, qw, rt.RouterConfig())
    router_state = rt.init_router_state(batch=4, max_contexts=16)
    return st.ContinualState(banks=banks, book=book, proto_index=proto, router_state=router_state,
                             meta={"seed": seed})


def test_save_load_byte_identical():
    state = _build_state(seed=1)
    data = st.save_bytes(state)
    restored = st.load_bytes(state, data)
    assert st.states_identical(state, restored)


def test_decoded_weights_identical_after_restore():
    state = _build_state(seed=2)
    data = st.save_bytes(state)
    restored = st.load_bytes(state, data)
    for name in state.banks:
        for ctx in range(2):
            w0 = np.asarray(syn.decode_weight(state.banks[name], addr.code(state.book, ctx)))
            w1 = np.asarray(syn.decode_weight(restored.banks[name], addr.code(restored.book, ctx)))
            np.testing.assert_array_equal(w0, w1)


def test_state_to_file_roundtrip(tmp_path):
    state = _build_state(seed=3)
    path = str(tmp_path / "castm_state.msgpack")
    st.save_file(state, path)
    restored = st.load_file(state, path)
    assert st.states_identical(state, restored)


def test_state_byte_accounting():
    state = _build_state(seed=4)
    nb = st.state_nbytes(state)
    assert nb["synaptic_factors"] > 0
    assert nb["address_book"] > 0
    assert nb["content_prototypes"] > 0
    assert set(nb["per_layer"].keys()) == set(state.banks.keys())
    # Total synaptic bytes is the sum of shared + factors + address factors.
    assert nb["total_synaptic"] == nb["shared_params"] + nb["synaptic_factors"] + nb["address_factors"]
