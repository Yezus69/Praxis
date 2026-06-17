from types import SimpleNamespace

import numpy as np
import jax
import jax.numpy as jnp
from flax.core import freeze, unfreeze

from pmac.agents.atari_mem_net import adapter_reg, mem_apply, mem_init
from pmac.agents.continual_living_memory import _maybe_grow_adapter


def _bank(capacity=2, d_k=8, d_c=4, act_dim=5, valid=False):
    return {
        "keys": jnp.zeros((capacity, d_k), dtype=jnp.float32),
        "context": jnp.zeros((capacity, d_c), dtype=jnp.float32),
        "teacher_policy": jnp.zeros((capacity, act_dim), dtype=jnp.float32),
        "teacher_value": jnp.zeros((capacity,), dtype=jnp.float32),
        "importance": jnp.zeros((capacity,), dtype=jnp.float32),
        "game_id": jnp.zeros((capacity,), dtype=jnp.int32),
        "source5": jnp.zeros((capacity, 5), dtype=jnp.float32),
        "age": jnp.zeros((capacity,), dtype=jnp.float32),
        "valid": jnp.full((capacity,), bool(valid), dtype=bool),
    }


def _params(n_adapters=3, adapter_rank=4, top_s=2):
    return mem_init(
        jax.random.PRNGKey(0),
        2,
        2,
        d_k=8,
        d_c=4,
        d_m=6,
        act_dim=5,
        top_k=1,
        n_adapters=n_adapters,
        adapter_rank=adapter_rank,
        top_s=top_s,
    )


def test_adapter_inert_none_and_zero_mask_are_identical():
    params = _params()
    obs = jnp.zeros((2, 84, 84, 4), dtype=jnp.float32)
    game_id = jnp.asarray([0, 1], dtype=jnp.int32)
    bank = _bank()
    hp = {"tau_r": 1.0, "beta_c": 0.0, "beta_I": 0.0, "beta_a": 0.0, "top_k": 1, "w_rho": 1.0, "w_c": 1.0, "b0": 0.0}

    out_none = mem_apply(params, obs, game_id, bank, hp, active_mask=None)
    out_zero = mem_apply(params, obs, game_id, bank, hp, active_mask=jnp.zeros((3,), dtype=jnp.float32))

    for name in ("logits_net", "v_net", "logits_final", "v_final", "m", "k"):
        np.testing.assert_allclose(np.asarray(out_zero[name]), np.asarray(out_none[name]), atol=1.0e-5)
    np.testing.assert_allclose(np.asarray(out_zero["adapter_delta"]), 0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(out_zero["adapter_gamma"]), 0.0, atol=0.0)


def test_active_adapter_can_change_forward_and_reg_is_finite_sparse():
    params = _params()
    mutable = unfreeze(params)
    mutable["adapter_down_0"]["kernel"] = jnp.zeros_like(mutable["adapter_down_0"]["kernel"])
    mutable["adapter_down_0"]["bias"] = jnp.ones_like(mutable["adapter_down_0"]["bias"])
    mutable["adapter_up_0"]["kernel"] = jnp.ones_like(mutable["adapter_up_0"]["kernel"]) * 0.01
    params = freeze(mutable)

    obs = jnp.ones((2, 84, 84, 4), dtype=jnp.float32) * 0.25
    game_id = jnp.asarray([0, 1], dtype=jnp.int32)
    bank = _bank()
    hp = {"tau_r": 1.0, "beta_c": 0.0, "beta_I": 0.0, "beta_a": 0.0, "top_k": 1, "w_rho": 1.0, "w_c": 1.0, "b0": 0.0}

    inert = mem_apply(params, obs, game_id, bank, hp, active_mask=None)
    active = mem_apply(params, obs, game_id, bank, hp, active_mask=jnp.asarray([1.0, 0.0, 0.0]))
    routed = mem_apply(params, obs, game_id, bank, hp, active_mask=jnp.asarray([1.0, 1.0, 1.0]))

    assert not np.allclose(np.asarray(active["logits_net"]), np.asarray(inert["logits_net"]))
    assert np.all(np.count_nonzero(np.asarray(routed["adapter_gamma"]) > 1.0e-6, axis=-1) <= 2)
    reg = adapter_reg(
        params,
        routed["adapter_gamma"],
        lambda_sparse=1.0e-3,
        lambda_load=1.0e-3,
        lambda_norm=1.0e-4,
    )
    assert np.isfinite(float(reg))


def test_growth_controller_patience_recovery_and_no_adapter():
    cfg = SimpleNamespace(adapter_r_min=0.5, adapter_patience=2)
    mask = np.zeros((3,), dtype=np.float32)

    mask, streak, grew = _maybe_grow_adapter(mask, 0, None, 1.0, 0.1, cfg, ablation="full")
    assert streak == 0
    assert not grew

    mask, streak, grew = _maybe_grow_adapter(mask, streak, 1.0, 1.0, 0.1, cfg, ablation="full")
    assert streak == 1
    assert not grew

    mask, streak, grew = _maybe_grow_adapter(mask, streak, 1.0, 0.9, 0.1, cfg, ablation="full")
    assert grew
    assert streak == 0
    np.testing.assert_array_equal(mask, np.asarray([1.0, 0.0, 0.0], dtype=np.float32))

    mask, streak, grew = _maybe_grow_adapter(mask, 1, 0.9, 1.1, 0.9, cfg, ablation="full")
    assert streak == 0
    assert not grew

    no_mask, no_streak, no_grew = _maybe_grow_adapter(
        np.zeros((2,), dtype=np.float32),
        2,
        1.0,
        0.5,
        0.1,
        cfg,
        ablation="no_adapter",
    )
    np.testing.assert_array_equal(no_mask, np.zeros((2,), dtype=np.float32))
    assert no_streak == 0
    assert not no_grew
