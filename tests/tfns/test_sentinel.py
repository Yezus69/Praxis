from __future__ import annotations

import copy
import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

from flax.core import FrozenDict, freeze, unfreeze
import jax
import jax.numpy as jnp
import numpy as np
import optax

from tfns.config import AdapterConfig, ModelConfig
from tfns.model.agent import RecurrentAgent
from tfns.protect.optimizer import optimizer_safe_step
from tfns.protect.projection import build_protected_modules
from tfns.protect.sentinel import make_sentinel_acceptor, sentinel_cluster_metrics
from tfns.utils import tree_add_scaled


BATCH = 1
TIME = 4
MODEL_CONFIG = ModelConfig(
    conv_channels=(4, 4, 4),
    dense_dim=16,
    action_embed_dim=4,
    gru_hidden=16,
    key_dim=8,
)
ADAPTER_CONFIG = AdapterConfig(num_adapters=2, rank=2, top_k=1)


def _agent_params_and_cluster(key=jax.random.PRNGKey(0)):
    agent = RecurrentAgent(model_config=MODEL_CONFIG, adapter_config=ADAPTER_CONFIG)
    obs0 = jnp.zeros((BATCH, 84, 84, 4), dtype=jnp.float32)
    act0 = jnp.zeros((BATCH,), dtype=jnp.int32)
    rew0 = jnp.zeros((BATCH,), dtype=jnp.float32)
    reset0 = jnp.ones((BATCH,), dtype=bool)
    h0 = agent.init_hidden(BATCH)
    params = agent.init(key, obs0, act0, rew0, reset0, h0)["params"]

    obs_seq = jnp.linspace(
        0.0,
        1.0,
        TIME * BATCH * 84 * 84 * 4,
        dtype=jnp.float32,
    ).reshape((TIME, BATCH, 84, 84, 4))
    act_seq = (jnp.arange(TIME * BATCH, dtype=jnp.int32) % MODEL_CONFIG.act_dim).reshape(
        (TIME, BATCH)
    )
    rew_seq = jnp.linspace(-1.0, 1.0, TIME * BATCH, dtype=jnp.float32).reshape(
        (TIME, BATCH)
    )
    reset_seq = jnp.zeros((TIME, BATCH), dtype=bool).at[0, :].set(True)
    outputs, _ = agent.unroll(params, obs_seq, act_seq, rew_seq, reset_seq, h0)
    cluster = {
        "obs_seq": obs_seq,
        "act_seq": act_seq,
        "rew_seq": rew_seq,
        "reset_seq": reset_seq,
        "h0": h0,
        "burn_in": 1,
        "teacher_logits": outputs.logits,
        "teacher_value": outputs.value,
        "teacher_key": outputs.q_key,
        "teacher_router_weights": outputs.router_weights,
    }
    return agent, params, cluster


def _mutable_copy(tree):
    return unfreeze(tree) if isinstance(tree, FrozenDict) else copy.deepcopy(tree)


def _restore_type(template, mutable):
    return freeze(mutable) if isinstance(template, FrozenDict) else mutable


def _set_path(tree, path: tuple[str, ...], value):
    mutable = _mutable_copy(tree)
    cur = mutable
    for part in path[:-1]:
        cur = cur[part]
    cur[path[-1]] = value
    return _restore_type(tree, mutable)


def _get_path(tree, path: tuple[str, ...]):
    cur = tree
    for part in path:
        cur = cur[part]
    return cur


def _leaf_delta(params, path: tuple[str, ...], delta):
    update = jax.tree_util.tree_map(jnp.zeros_like, params)
    return _set_path(update, path, delta)


def _policy_bias_delta(params, amount: float):
    path = ("policy_head", "affine", "bias")
    bias = _get_path(params, path)
    delta = jnp.zeros_like(bias).at[0].set(float(amount))
    return _leaf_delta(params, path, delta)


def _value_bias_delta(params, amount: float):
    path = ("value_head", "affine", "bias")
    bias = _get_path(params, path)
    return _leaf_delta(params, path, jnp.ones_like(bias) * float(amount))


def _neg_tree(tree):
    return jax.tree_util.tree_map(lambda x: -x, tree)


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


def test_gate_orientation_policy_and_value_max_losses_fail_when_above_tol():
    agent, params, cluster = _agent_params_and_cluster()

    policy_delta = _policy_bias_delta(params, amount=2.0)
    bad_policy_params = tree_add_scaled(params, policy_delta, 1.0)
    policy_metrics = sentinel_cluster_metrics(
        agent,
        bad_policy_params,
        cluster,
        burn_in=cluster["burn_in"],
    )
    policy_tol = float(policy_metrics["kl"]) * 0.5
    policy_tols = SimpleNamespace(
        kl_tol=policy_tol,
        value_tol=1.0e6,
        key_cos_tol=1.0e6,
        router_tol=1.0e6,
    )
    policy_accept = make_sentinel_acceptor(agent, [cluster], policy_tols)

    assert float(policy_metrics["kl"]) > policy_tol
    assert policy_accept(params) is True
    assert policy_accept(bad_policy_params) is False

    value_delta = _value_bias_delta(params, amount=2.0)
    bad_value_params = tree_add_scaled(params, value_delta, 1.0)
    value_metrics = sentinel_cluster_metrics(
        agent,
        bad_value_params,
        cluster,
        burn_in=cluster["burn_in"],
    )
    value_tol = float(value_metrics["value_err"]) * 0.5
    value_tols = SimpleNamespace(
        kl_tol=1.0e6,
        value_tol=value_tol,
        key_cos_tol=1.0e6,
        router_tol=1.0e6,
    )
    value_accept = make_sentinel_acceptor(agent, [cluster], value_tols)

    assert float(value_metrics["value_err"]) > value_tol
    assert value_accept(params) is True
    assert value_accept(bad_value_params) is False


def test_optimizer_backtracking_accepts_smaller_sentinel_safe_scale():
    agent, params, cluster = _agent_params_and_cluster(jax.random.PRNGKey(10))
    update = _policy_bias_delta(params, amount=2.0)
    full_params = tree_add_scaled(params, update, 1.0)
    half_params = tree_add_scaled(params, update, 0.5)

    full_kl = float(
        sentinel_cluster_metrics(agent, full_params, cluster, burn_in=cluster["burn_in"])["kl"]
    )
    half_kl = float(
        sentinel_cluster_metrics(agent, half_params, cluster, burn_in=cluster["burn_in"])["kl"]
    )
    assert full_kl > half_kl

    tols = SimpleNamespace(
        kl_tol=(full_kl + half_kl) * 0.5,
        value_tol=1.0e6,
        key_cos_tol=1.0e6,
        router_tol=1.0e6,
    )
    accept_fn = make_sentinel_acceptor(agent, [cluster], tols)
    assert accept_fn(full_params) is False
    assert accept_fn(half_params) is True

    tx = optax.sgd(learning_rate=1.0)
    opt_state = tx.init(params)
    modules = build_protected_modules(params, MODEL_CONFIG)
    new_params, _, info = optimizer_safe_step(
        params,
        opt_state,
        _neg_tree(update),
        tx,
        {},
        modules,
        accept_fn=accept_fn,
        backtrack_scales=(1.0, 0.5, 0.25),
    )

    assert info["accepted"] is True
    assert 0.0 < info["applied_scale"] < 1.0
    assert not _trees_allclose(new_params, params)


def test_optimizer_rejects_all_scales_and_restores_state():
    agent, params, cluster = _agent_params_and_cluster(jax.random.PRNGKey(20))
    update = _value_bias_delta(params, amount=100.0)
    tols = SimpleNamespace(
        kl_tol=1.0e6,
        value_tol=1.0e-4,
        key_cos_tol=1.0e6,
        router_tol=1.0e6,
    )
    accept_fn = make_sentinel_acceptor(agent, [cluster], tols)

    tx = optax.sgd(learning_rate=1.0)
    opt_state = tx.init(params)
    modules = build_protected_modules(params, MODEL_CONFIG)
    new_params, new_opt_state, info = optimizer_safe_step(
        params,
        opt_state,
        _neg_tree(update),
        tx,
        {},
        modules,
        accept_fn=accept_fn,
        backtrack_scales=(1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125),
    )

    assert info["accepted"] is False
    assert info["applied_scale"] == 0.0
    assert _trees_allclose(new_params, params)
    assert _trees_allclose(new_opt_state, opt_state)
