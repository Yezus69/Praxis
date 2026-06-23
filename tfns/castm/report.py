"""Self-contained mathematical test report generator (spec deliverables 1, 2).

Directly measures the numerical invariants of spec section 16 over >=32
conflicting contexts for every contextualized layer type, plus the address-book
invariants (16.1-16.3, 16.6) and a synthetic conflicting-linear-memory check
(17.1). Writes a JSON artifact; the headline acceptance is the mathematical
memory gate (21.1): exact noninterference for at least 32 conflicting contexts.

Run:
    python -m tfns.castm.report
"""

from __future__ import annotations

import json
import os
from typing import Any

import jax.numpy as jnp
import numpy as np

from tfns.castm import address as addr
from tfns.castm import audit
from tfns.castm import layers as L
from tfns.castm import synaptic as syn


def _book(n: int, d_k: int = 128, seed: int = 57) -> addr.AddressBook:
    book = addr.empty_address_book(d_k=d_k, n_max=n, seed=seed)
    for _ in range(n):
        book, _ = addr.allocate_canonical(book)
    return book


def _sequential_conflicting(out_dim: int, in_dim: int, rank: int, n_ctx: int, seed: int):
    rng = np.random.default_rng(seed)
    W0 = jnp.asarray(rng.standard_normal((out_dim, in_dim)).astype(np.float32))
    b0 = jnp.asarray(rng.standard_normal((out_dim,)).astype(np.float32))
    mem = syn.empty_synaptic_memory(W0, b0, comp_rank=rank, n_slots=n_ctx + 4, d_k=128)
    book = _book(n_ctx, d_k=128, seed=seed + 1)
    max_ni = 0.0
    max_iw = 0.0
    targets = {}
    for i in range(n_ctx):
        A = jnp.asarray(rng.standard_normal((rank, in_dim)).astype(np.float32) * 0.5)
        B = jnp.asarray(rng.standard_normal((out_dim, rank)).astype(np.float32) * 0.5)
        beta = jnp.asarray(rng.standard_normal((out_dim,)).astype(np.float32) * 0.5)
        delta = B @ A
        before = mem
        mem, slot = syn.append_component(mem, A, B, beta, addr.code(book, i), i)
        assert slot >= 0
        max_ni = max(max_ni, audit.max_noninterference_error(before, mem, book, i))
        max_iw = max(max_iw, audit.intended_write_error(before, mem, book, i, delta, bias_expected=beta))
        targets[i] = np.asarray(delta) + np.asarray(W0)
    max_decoded_drift = 0.0
    for i in range(n_ctx):
        W = np.asarray(syn.decode_weight(mem, addr.code(book, i)))
        max_decoded_drift = max(max_decoded_drift, float(np.max(np.abs(W - targets[i]))))
    bytes_per_ctx = syn.active_factor_bytes(mem) / max(n_ctx, 1)
    return {
        "n_contexts": int(n_ctx),
        "rank": int(rank),
        "max_noninterference": float(max_ni),
        "max_intended_write_error": float(max_iw),
        "max_decoded_drift": float(max_decoded_drift),
        "bytes_per_context": float(bytes_per_ctx),
    }


def build_report(n_ctx: int = 32) -> dict[str, Any]:
    book = _book(64, d_k=128, seed=99)
    address_invariants = {
        "norm_error": addr.address_norm_error(book),
        "orthogonality_error": addr.orthogonality_error(book),
        "duality_error": addr.duality_error(book),
        "rank": addr.address_rank(book),
        "num_used": addr.num_used(book),
        "condition_number": addr.condition_number(book),
    }
    layer_cases = {
        # name: (out, in, rank) using the flattened-matrix view (conv uses kh*kw*c_in).
        "dense_encoder": (512, 3136, 16),
        "gru_gate": (512, 1058, 16),
        "policy_head": (18, 512, 8),
        "value_head": (1, 512, 8),
        "conv2_flat": (64, 512, 8),
        "conv3_flat": (64, 576, 8),
    }
    layers_report = {
        name: _sequential_conflicting(o, i, r, n_ctx, seed=1000 + idx)
        for idx, (name, (o, i, r)) in enumerate(layer_cases.items())
    }

    eps_write = 1e-6
    gate_pass = all(v["max_noninterference"] < eps_write for v in layers_report.values())
    report = {
        "spec": "README_CONTEXT_ADDRESSED_SYNAPTIC_MEMORY.md",
        "precision": "float32",
        "eps_write": eps_write,
        "address_invariants": address_invariants,
        "address_invariants_pass": (
            address_invariants["norm_error"] < 1e-5
            and address_invariants["orthogonality_error"] < 1e-5
            and address_invariants["duality_error"] < 1e-5
            and address_invariants["rank"] == address_invariants["num_used"]
        ),
        "layers": layers_report,
        "mathematical_memory_gate_21_1": {
            "min_contexts_required": 32,
            "contexts_tested": int(n_ctx),
            "exact_noninterference_eps_write": eps_write,
            "passed": bool(gate_pass and n_ctx >= 32),
        },
    }
    return report


def main():
    report = build_report(n_ctx=32)
    out_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "reports")
    os.makedirs(out_dir, exist_ok=True)
    path = os.path.join(out_dir, "mathematical_test_report.json")
    with open(path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    gate = report["mathematical_memory_gate_21_1"]
    print(f"mathematical memory gate (21.1) passed: {gate['passed']} "
          f"({gate['contexts_tested']} contexts, eps_write={gate['exact_noninterference_eps_write']})")
    worst = max(v["max_noninterference"] for v in report["layers"].values())
    print(f"worst-layer max noninterference: {worst:.2e}")
    print(f"report written -> {path}")


if __name__ == "__main__":
    main()
