"""Persistent continual state container and exact serialization (spec 14, 17.10).

Bundles the persistent CASTM state — shared/synaptic memory per contextualized
layer, the canonical address book, the content-prototype index, and the router
state — into one pytree that serializes and restores *exactly* (byte-identical
addresses, factors, routing state). Scratch and optimizer state are included when
present so a run can resume mid-segment.

Serialization uses flax msgpack, which preserves array dtypes and values exactly.
"""

from __future__ import annotations

from typing import Any, Mapping

from flax import serialization, struct
import jax.numpy as jnp
import numpy as np

from tfns.castm.address import AddressBook
from tfns.castm.router import PrototypeIndex, RouterState
from tfns.castm.synaptic import SynapticMemory, nbytes


@struct.dataclass
class ContinualState:
    """The full persistent CASTM continual state (spec 14)."""

    banks: dict           # dict[str, SynapticMemory]  (14.2)
    book: AddressBook     # canonical address book      (14.4)
    proto_index: PrototypeIndex  # content prototypes    (14.4)
    router_state: RouterState | None = None  # per-stream routing state (14.4)
    shared_params: Any = None  # shared policy params (14.1), optional pytree
    meta: dict = struct.field(default_factory=dict)  # audit/experiment meta (14.5)


def save_bytes(state: ContinualState) -> bytes:
    """Serialize the continual state to msgpack bytes (exact)."""

    return serialization.to_bytes(state)


def load_bytes(template: ContinualState, data: bytes) -> ContinualState:
    """Restore a continual state from bytes into a matching ``template`` structure."""

    return serialization.from_bytes(template, data)


def save_file(state: ContinualState, path: str) -> str:
    with open(path, "wb") as f:
        f.write(save_bytes(state))
    return path


def load_file(template: ContinualState, path: str) -> ContinualState:
    with open(path, "rb") as f:
        return load_bytes(template, f.read())


def _leaves_equal(a: Any, b: Any) -> bool:
    """Recursively check two pytrees have byte-identical array leaves."""

    import jax

    la = jax.tree_util.tree_leaves(a)
    lb = jax.tree_util.tree_leaves(b)
    if len(la) != len(lb):
        return False
    for xa, xb in zip(la, lb):
        xa = np.asarray(xa)
        xb = np.asarray(xb)
        if xa.shape != xb.shape or xa.dtype != xb.dtype:
            return False
        if not np.array_equal(xa, xb):
            return False
    return True


def states_identical(a: ContinualState, b: ContinualState) -> bool:
    """True if two continual states are byte-identical across all array leaves (17.10)."""

    return _leaves_equal(a, b)


def state_nbytes(state: ContinualState) -> dict[str, Any]:
    """Per-component byte breakdown of the continual state (spec 11.3, 14)."""

    per_layer: dict[str, dict] = {}
    factors = 0
    shared = 0
    address_factors = 0
    for name, mem in state.banks.items():
        nb = nbytes(mem)
        per_layer[name] = nb
        factors += nb["factors"]
        shared += nb["shared"]
        address_factors += nb["address_factors"]
    book_bytes = int(np.asarray(state.book.K).nbytes + np.asarray(state.book.used).nbytes)
    proto_bytes = int(
        np.asarray(state.proto_index.prototypes).nbytes
        + np.asarray(state.proto_index.count).nbytes
        + np.asarray(state.proto_index.used).nbytes
    )
    return {
        "shared_params": shared,
        "synaptic_factors": factors,
        "address_book": book_bytes,
        "address_factors": address_factors,
        "content_prototypes": proto_bytes,
        "per_layer": per_layer,
        "total_synaptic": shared + factors + address_factors,
    }


__all__ = [
    "ContinualState",
    "load_bytes",
    "load_file",
    "save_bytes",
    "save_file",
    "state_nbytes",
    "states_identical",
]
