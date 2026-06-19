from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import flax.linen as nn
import jax

import jax.numpy as jnp
import numpy as np

from tfns.model.agent import RecurrentAgent
from tfns.protect.bases import (
    empty_basis,
    expand_basis,
    free_rank_fraction,
    from_storage,
    orthonormality_error,
    represented_energy,
    residual_norm,
    to_storage,
)
from tfns.protect.projection import (
    build_protected_modules,
    collect_conv_basis_columns,
    project_affine,
    project_conv,
    project_gru_gate,
    project_update,
)


def _orthonormal(key, d_aug: int, rank: int, dtype=jnp.float32) -> jnp.ndarray:
    raw = jax.random.normal(key, (d_aug, rank), dtype=dtype)
    q, _ = jnp.linalg.qr(raw, mode="reduced")
    return q[:, :rank]


def _max_abs(x: jnp.ndarray) -> float:
    return float(jnp.max(jnp.abs(x)))


def _tree_get(tree, path: tuple[str, ...]):
    cur = tree
    for part in path:
        cur = cur[part]
    return cur


def _random_update_like(params, key):
    leaves, treedef = jax.tree_util.tree_flatten(params)
    keys = jax.random.split(key, len(leaves))
    update_leaves = [
        jax.random.normal(k, leaf.shape, dtype=leaf.dtype)
        for k, leaf in zip(keys, leaves, strict=True)
    ]
    return jax.tree_util.tree_unflatten(treedef, update_leaves)


def test_affine_projection_annihilates_augmented_protected_span():
    key = jax.random.PRNGKey(1)
    k_u, k_w, k_b, k_c = jax.random.split(key, 4)
    d_in, d_out, rank = 5, 4, 3
    U = _orthonormal(k_u, d_in + 1, rank)
    dW = jax.random.normal(k_w, (d_in, d_out), dtype=jnp.float32)
    db = jax.random.normal(k_b, (d_out,), dtype=jnp.float32)

    dW_safe, db_safe = project_affine(dW, db, U)
    Kbar_safe = jnp.concatenate([dW_safe, db_safe[None, :]], axis=0)
    assert _max_abs(U.T @ Kbar_safe) <= 1e-4

    c = jax.random.normal(k_c, (rank,), dtype=jnp.float32)
    xbar = U @ c
    col = int(jnp.argmax(jnp.abs(U[-1, :])))
    xbar = jnp.where(jnp.abs(xbar[-1]) > 1e-6, xbar, U[:, col])
    xbar = xbar / xbar[-1]
    x = xbar[:-1]
    np.testing.assert_allclose(
        np.asarray(x @ dW_safe + db_safe),
        np.zeros((d_out,), dtype=np.float32),
        rtol=0.0,
        atol=1e-4,
    )
    np.testing.assert_allclose(
        np.asarray(xbar @ Kbar_safe),
        np.zeros((d_out,), dtype=np.float32),
        rtol=0.0,
        atol=1e-4,
    )


def test_bias_projection_co_adapts_weight_and_bias_rows():
    dW = jnp.zeros((2, 3), dtype=jnp.float32)
    db = jnp.array([1.0, -2.0, 3.0], dtype=jnp.float32)
    U = jnp.array([[1.0], [0.0], [1.0]], dtype=jnp.float32) / jnp.sqrt(2.0)

    dW_safe, db_safe = project_affine(dW, db, U)
    xbar = jnp.array([1.0, 0.0, 1.0], dtype=jnp.float32)
    np.testing.assert_allclose(
        np.asarray(xbar[:-1] @ dW_safe + xbar[-1] * db_safe),
        np.zeros((3,), dtype=np.float32),
        rtol=0.0,
        atol=1e-4,
    )
    assert float(jnp.linalg.norm(db_safe)) > 0.0


def test_affine_projection_float32_runtime_precision():
    key = jax.random.PRNGKey(2)
    k_u, k_w, k_b = jax.random.split(key, 3)
    U = _orthonormal(k_u, 8, 3, dtype=jnp.float32)
    dW = jax.random.normal(k_w, (7, 5), dtype=jnp.float32)
    db = jax.random.normal(k_b, (5,), dtype=jnp.float32)

    dW_safe, db_safe = project_affine(dW, db, U)
    Kbar_safe = jnp.concatenate([dW_safe, db_safe[None, :]], axis=0)
    assert _max_abs(U.T @ Kbar_safe) <= 1e-4


def test_convolution_projection_preserves_flax_conv_outputs_on_basis_input():
    key = jax.random.PRNGKey(3)
    k_x, k_init, k_update = jax.random.split(key, 3)
    batch, height, width, c_in = 2, 5, 5, 2
    kh, kw, c_out, stride = 3, 3, 4, 1
    x = jax.random.normal(k_x, (batch, height, width, c_in), dtype=jnp.float32)
    conv = nn.Conv(
        features=c_out,
        kernel_size=(kh, kw),
        strides=(stride, stride),
        padding="VALID",
        dtype=jnp.float32,
        param_dtype=jnp.float32,
    )
    variables = conv.init(k_init, x)
    kernel = variables["params"]["kernel"]
    bias = variables["params"]["bias"]

    columns = collect_conv_basis_columns(x, kh, kw, stride, c_in)
    U, _ = jnp.linalg.qr(columns, mode="reduced")
    k_dk, k_db = jax.random.split(k_update)
    dK = 0.05 * jax.random.normal(k_dk, kernel.shape, dtype=jnp.float32)
    db = 0.05 * jax.random.normal(k_db, bias.shape, dtype=jnp.float32)
    dK_safe, db_safe = project_conv(dK, db, U)

    out_before = conv.apply({"params": {"kernel": kernel, "bias": bias}}, x)
    out_after = conv.apply(
        {"params": {"kernel": kernel + dK_safe, "bias": bias + db_safe}},
        x,
    )
    np.testing.assert_allclose(
        np.asarray(out_after), np.asarray(out_before), rtol=0.0, atol=1e-4
    )


def test_gru_gate_projection_uses_shared_augmented_xi_basis():
    key = jax.random.PRNGKey(4)
    k_u, k_w, k_r, k_b, k_c = jax.random.split(key, 5)
    input_dim, hidden, rank = 4, 3, 3
    U = _orthonormal(k_u, input_dim + hidden + 1, rank)
    dW = jax.random.normal(k_w, (input_dim, hidden), dtype=jnp.float32)
    dU = jax.random.normal(k_r, (hidden, hidden), dtype=jnp.float32)
    db = jax.random.normal(k_b, (hidden,), dtype=jnp.float32)

    dW_safe, dU_safe, db_safe = project_gru_gate(dW, dU, db, U)
    stacked_safe = jnp.concatenate([dW_safe, dU_safe, db_safe[None, :]], axis=0)
    assert _max_abs(U.T @ stacked_safe) <= 1e-4

    c = jax.random.normal(k_c, (rank,), dtype=jnp.float32)
    xi = U @ c
    np.testing.assert_allclose(
        np.asarray(xi @ stacked_safe),
        np.zeros((hidden,), dtype=np.float32),
        rtol=0.0,
        atol=1e-4,
    )


def test_basis_construction_properties_and_storage_round_trip():
    key = jax.random.PRNGKey(5)
    k_a, k_b, k_x, k_kbar = jax.random.split(key, 4)
    d_aug = 8
    U0 = empty_basis(d_aug).astype(jnp.float32)
    A = jax.random.normal(k_a, (d_aug, 4), dtype=jnp.float32)
    U1, info1 = expand_basis(U0, A, energy=1.0)
    assert info1["added_rank"] > 0
    assert _max_abs(orthonormality_error(U1)) <= 1e-5

    before = represented_energy(U1, A)
    B = jax.random.normal(k_b, (d_aug, 3), dtype=jnp.float32)
    U2, _ = expand_basis(U1, B, energy=1.0)
    after = represented_energy(U2, A)
    assert float(after) + 1e-4 >= float(before)
    assert U2.shape[1] <= d_aug

    U_dup, info_dup = expand_basis(U1, A, energy=1.0)
    assert info_dup["added_rank"] == 0
    assert U_dup.shape[1] == U1.shape[1]

    x = U1 @ jax.random.normal(k_x, (U1.shape[1],), dtype=jnp.float32)
    assert float(residual_norm(U1, x)) <= 1e-4
    assert free_rank_fraction(U1, d_aug) == 1.0 - U1.shape[1] / d_aug

    U_cap, cap_info = expand_basis(
        empty_basis(6).astype(jnp.float32),
        jnp.eye(6, dtype=jnp.float32),
        energy=1.0,
        max_rank=3,
    )
    assert U_cap.shape == (6, 3)
    assert cap_info["capacity_hit"] is True
    assert cap_info["discarded_energy"] > 0.0

    U32 = U1.astype(jnp.float32)
    np.testing.assert_array_equal(np.asarray(from_storage(to_storage(U32))), np.asarray(U32))
    U16 = from_storage(to_storage(U32, fp16=True))
    Kbar = 0.01 * jax.random.normal(k_kbar, (d_aug, 3), dtype=jnp.float32)
    Kbar_safe = Kbar - U16 @ (U16.T @ Kbar)
    assert _max_abs(U16.T @ Kbar_safe) < 1e-2


def test_project_update_projects_policy_and_gru_leaves_only():
    key = jax.random.PRNGKey(6)
    agent = RecurrentAgent()
    obs = jnp.ones((1, 84, 84, 4), dtype=jnp.float32)
    prev_action = jnp.array([0], dtype=jnp.int32)
    prev_reward = jnp.array([0.0], dtype=jnp.float32)
    reset = jnp.array([False])
    hidden = agent.init_hidden(1)
    params = agent.init(key, obs, prev_action, prev_reward, reset, hidden)["params"]
    modules = build_protected_modules(params, agent.model_config)
    assert "policy_head" in modules
    assert "gru" in modules

    update = _random_update_like(params, jax.random.PRNGKey(7))
    policy_basis = _orthonormal(
        jax.random.PRNGKey(8),
        modules["policy_head"].d_aug,
        4,
        dtype=jnp.float32,
    )
    gru_basis = _orthonormal(
        jax.random.PRNGKey(9),
        modules["gru"].d_aug,
        5,
        dtype=jnp.float32,
    )
    safe = project_update(update, {"policy_head": policy_basis, "gru": gru_basis}, modules)

    policy_kernel_path = modules["policy_head"].kernel_path
    policy_bias_path = modules["policy_head"].bias_path
    assert policy_kernel_path is not None
    assert policy_bias_path is not None
    assert np.linalg.norm(
        np.asarray(_tree_get(safe, policy_kernel_path) - _tree_get(update, policy_kernel_path))
    ) > 1e-6
    assert np.linalg.norm(np.asarray(safe["gru"]["W_z"] - update["gru"]["W_z"])) > 1e-6

    aux_path = ("next_feat_head", "hidden", "kernel")
    np.testing.assert_allclose(
        np.asarray(_tree_get(safe, aux_path)),
        np.asarray(_tree_get(update, aux_path)),
        rtol=0.0,
        atol=0.0,
    )

    Kbar_safe = jnp.concatenate(
        [
            _tree_get(safe, policy_kernel_path),
            _tree_get(safe, policy_bias_path)[None, :],
        ],
        axis=0,
    )
    assert _max_abs(policy_basis.T @ Kbar_safe) <= 1e-4
