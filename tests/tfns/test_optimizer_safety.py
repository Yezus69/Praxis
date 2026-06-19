from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax

from flax.core import FrozenDict, freeze, unfreeze
import jax.numpy as jnp
import numpy as np
import optax

from tfns.config import AdapterConfig, ModelConfig
from tfns.model.agent import RecurrentAgent
from tfns.protect.bases import empty_basis, expand_basis, orthonormality_error
from tfns.protect.optimizer import optimizer_safe_step, project_first_moments
from tfns.protect.projection import build_protected_modules, project_update
from tfns.utils import tree_add_scaled, tree_global_norm


BATCH = 2
TIME = 6
MODEL_CONFIG = ModelConfig(
    conv_channels=(4, 8, 8),
    dense_dim=16,
    action_embed_dim=4,
    gru_hidden=16,
    key_dim=8,
)
ADAPTER_CONFIG = AdapterConfig(num_adapters=2, rank=4, top_k=1)


def _agent_and_params(key=jax.random.PRNGKey(0)):
    agent = RecurrentAgent(model_config=MODEL_CONFIG, adapter_config=ADAPTER_CONFIG)
    obs = jnp.zeros((BATCH, 84, 84, 4), dtype=jnp.float32)
    prev_action = jnp.zeros((BATCH,), dtype=jnp.int32)
    prev_reward = jnp.zeros((BATCH,), dtype=jnp.float32)
    reset = jnp.ones((BATCH,), dtype=bool)
    hidden = agent.init_hidden(BATCH)
    params = agent.init(key, obs, prev_action, prev_reward, reset, hidden)["params"]
    return agent, params


def _sequence(key=jax.random.PRNGKey(1)):
    obs_pattern = jax.random.uniform(key, (3, BATCH, 84, 84, 4), dtype=jnp.float32)
    obs_seq = jnp.concatenate([obs_pattern, obs_pattern], axis=0)
    act_pattern = jnp.array([[0, 1], [2, 3], [4, 5]], dtype=jnp.int32)
    act_seq = jnp.concatenate([act_pattern, act_pattern], axis=0)
    rew_pattern = jnp.array([[-1.0, 0.5], [0.0, 1.0], [1.0, -0.5]], dtype=jnp.float32)
    rew_seq = jnp.concatenate([rew_pattern, rew_pattern], axis=0)
    reset_seq = jnp.zeros((TIME, BATCH), dtype=bool)
    reset_seq = reset_seq.at[0, :].set(True)
    reset_seq = reset_seq.at[3, :].set(True)
    return obs_seq, act_seq, rew_seq, reset_seq


def _orthonormal(key, d_aug: int, rank: int, dtype=jnp.float32) -> jnp.ndarray:
    raw = jax.random.normal(key, (d_aug, rank), dtype=dtype)
    q, _ = jnp.linalg.qr(raw, mode="reduced")
    return q[:, :rank]


def _random_update_like(tree, key, scale: float = 1.0):
    leaves, treedef = jax.tree_util.tree_flatten(tree)
    keys = jax.random.split(key, len(leaves))
    update_leaves = [
        scale * jax.random.normal(k, leaf.shape, dtype=leaf.dtype)
        for k, leaf in zip(keys, leaves, strict=True)
    ]
    return jax.tree_util.tree_unflatten(treedef, update_leaves)


def _only_roots(update, roots: tuple[str, ...]):
    zeros = jax.tree_util.tree_map(jnp.zeros_like, update)
    update_mut = unfreeze(update) if isinstance(update, FrozenDict) else update
    zeros_mut = unfreeze(zeros) if isinstance(zeros, FrozenDict) else zeros
    for root in roots:
        zeros_mut[root] = update_mut[root]
    return freeze(zeros_mut) if isinstance(update, FrozenDict) else zeros_mut


def _only_gru_zr(update):
    gru_only = _only_roots(update, ("gru",))
    mutable = unfreeze(gru_only) if isinstance(gru_only, FrozenDict) else gru_only
    for name in ("W_n", "U_n", "b_n"):
        mutable["gru"][name] = jnp.zeros_like(mutable["gru"][name])
    return freeze(mutable) if isinstance(gru_only, FrozenDict) else mutable


def _tree_get(tree, path: tuple[str, ...]):
    cur = tree
    for part in path:
        cur = cur[part]
    return cur


def _dense_kbar(tree, module) -> jnp.ndarray:
    assert module.kernel_path is not None
    assert module.bias_path is not None
    kernel = _tree_get(tree, module.kernel_path)
    bias = _tree_get(tree, module.bias_path)
    return jnp.concatenate([kernel, bias[None, :]], axis=0)


def _gru_kbar(tree, module, gate: str) -> jnp.ndarray:
    assert module.gate_paths is not None
    W_path, U_path, b_path = module.gate_paths[gate]
    return jnp.concatenate(
        [_tree_get(tree, W_path), _tree_get(tree, U_path), _tree_get(tree, b_path)[None, :]],
        axis=0,
    )


def _null_norm(U: jnp.ndarray, Kbar: jnp.ndarray) -> float:
    U = U.astype(Kbar.dtype)
    return float(jnp.linalg.norm(U.T @ Kbar))


def _adam_mu(opt_state):
    tree_utils = getattr(optax, "tree_utils", None)
    tree_get = getattr(tree_utils, "tree_get", None)
    if tree_get is not None:
        try:
            return tree_get(opt_state, "mu")
        except (AttributeError, KeyError, TypeError, ValueError):
            pass
    return opt_state[0].mu


def _trees_allclose(a, b, atol: float = 0.0) -> bool:
    return bool(
        jax.tree_util.tree_all(
            jax.tree_util.tree_map(
                lambda x, y: jnp.allclose(x, y, rtol=0.0, atol=atol),
                a,
                b,
            )
        )
    )


def _tree_delta(new_tree, old_tree):
    return jax.tree_util.tree_map(lambda new, old: new - old, new_tree, old_tree)


def test_gru_sequence_invariance_under_projected_update():
    key = jax.random.PRNGKey(10)
    k_init, k_seq, k_delta = jax.random.split(key, 3)
    agent, params = _agent_and_params(k_init)
    obs_seq, act_seq, rew_seq, reset_seq = _sequence(k_seq)
    h0 = agent.init_hidden(BATCH)

    outputs, _ = agent.unroll(
        params,
        obs_seq,
        act_seq,
        rew_seq,
        reset_seq,
        h0,
        collect_presyn=True,
    )
    xi = outputs.presyn["gru_xi"]
    d_aug = int(xi.shape[-1])
    A = xi.reshape((-1, d_aug)).T.astype(jnp.float32)
    U, _ = expand_basis(empty_basis(d_aug).astype(jnp.float32), A, energy=0.99999)
    assert float(orthonormality_error(U)) <= 1e-5

    modules = build_protected_modules(params, MODEL_CONFIG)
    bases = {"gru": U}
    delta = _only_gru_zr(_random_update_like(params, k_delta, scale=0.5))
    safe = project_update(delta, bases, modules)
    params2 = tree_add_scaled(params, safe, 1.0)

    outputs2, _ = agent.unroll(
        params2,
        obs_seq,
        act_seq,
        rew_seq,
        reset_seq,
        h0,
        collect_presyn=True,
    )
    proj_h_delta = float(jnp.linalg.norm(outputs2.h_next - outputs.h_next))
    assert proj_h_delta <= 1e-3
    np.testing.assert_allclose(
        np.asarray(outputs2.logits), np.asarray(outputs.logits), rtol=0.0, atol=1e-3
    )
    np.testing.assert_allclose(
        np.asarray(outputs2.value), np.asarray(outputs.value), rtol=0.0, atol=1e-3
    )
    np.testing.assert_allclose(
        np.asarray(outputs2.q_key), np.asarray(outputs.q_key), rtol=0.0, atol=1e-3
    )

    params_bad = tree_add_scaled(params, delta, 1.0)
    outputs_bad, _ = agent.unroll(params_bad, obs_seq, act_seq, rew_seq, reset_seq, h0)
    bad_h_delta = float(jnp.linalg.norm(outputs_bad.h_next - outputs.h_next))
    assert bad_h_delta > 1e-2
    assert bad_h_delta > 50.0 * proj_h_delta


def test_adam_delta_projection_is_necessary_for_policy_head():
    key = jax.random.PRNGKey(20)
    k_init, k_u, *grad_keys = jax.random.split(key, 8)
    _, params = _agent_and_params(k_init)
    modules = build_protected_modules(params, MODEL_CONFIG)
    U = _orthonormal(k_u, modules["policy_head"].d_aug, rank=7)
    bases = {"policy_head": U}
    tx = optax.adam(1.0e-3)
    opt_state = tx.init(params)

    for grad_key in grad_keys[:4]:
        warm_grad = _random_update_like(params, grad_key, scale=1.0e-2)
        _, opt_state = tx.update(warm_grad, opt_state, params)

    raw_grad = _random_update_like(params, grad_keys[4], scale=1.0e-2)
    g_proj = project_update(raw_grad, bases, modules)
    assert _null_norm(U, _dense_kbar(g_proj, modules["policy_head"])) <= 1e-4

    updates, cand_state = tx.update(g_proj, opt_state, params)
    assert _null_norm(U, _dense_kbar(updates, modules["policy_head"])) > 1e-3

    updates_safe = project_update(updates, bases, modules)
    assert _null_norm(U, _dense_kbar(updates_safe, modules["policy_head"])) <= 1e-4

    safe_state = project_first_moments(cand_state, bases, modules)
    mu_safe = _adam_mu(safe_state)
    assert _null_norm(U, _dense_kbar(mu_safe, modules["policy_head"])) <= 1e-4


def test_reject_restores_parameters_and_optimizer_state():
    key = jax.random.PRNGKey(30)
    k_init, k_grad, k_u = jax.random.split(key, 3)
    _, params = _agent_and_params(k_init)
    modules = build_protected_modules(params, MODEL_CONFIG)
    U = _orthonormal(k_u, modules["policy_head"].d_aug, rank=5)
    bases = {"policy_head": U}
    tx = optax.adam(1.0e-3)
    opt_state = tx.init(params)
    grad = _random_update_like(params, k_grad)

    new_params, new_opt_state, info = optimizer_safe_step(
        params,
        opt_state,
        grad,
        tx,
        bases,
        modules,
        accept_fn=lambda _: False,
    )

    assert _trees_allclose(new_params, params)
    assert _trees_allclose(new_opt_state, opt_state)
    assert info["accepted"] is False
    assert info["applied_scale"] == 0.0


def test_accept_path_norm_bound_and_committed_delta_null_spaces():
    key = jax.random.PRNGKey(40)
    k_init, k_grad, k_policy, k_gru = jax.random.split(key, 4)
    _, params = _agent_and_params(k_init)
    modules = build_protected_modules(params, MODEL_CONFIG)
    policy_U = _orthonormal(k_policy, modules["policy_head"].d_aug, rank=5)
    gru_U = _orthonormal(k_gru, modules["gru"].d_aug, rank=6)
    bases = {"policy_head": policy_U, "gru": gru_U}
    tx = optax.adam(1.0e-2)
    opt_state = tx.init(params)
    grad = _random_update_like(params, k_grad)
    max_update_norm = 1.0e-3

    new_params, _, info = optimizer_safe_step(
        params,
        opt_state,
        grad,
        tx,
        bases,
        modules,
        max_update_norm=max_update_norm,
        accept_fn=None,
    )

    committed = _tree_delta(new_params, params)
    assert float(tree_global_norm(committed)) <= max_update_norm + 1e-6
    assert float(info["applied_norm"]) <= max_update_norm + 1e-6
    assert info["accepted"] is True
    assert info["applied_scale"] == 1.0
    assert _null_norm(policy_U, _dense_kbar(committed, modules["policy_head"])) <= 1e-4
    for gate in ("z", "r", "n"):
        assert _null_norm(gru_U, _gru_kbar(committed, modules["gru"], gate)) <= 1e-4
