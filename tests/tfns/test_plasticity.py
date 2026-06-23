from __future__ import annotations

import os
from types import SimpleNamespace

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from tfns.config import AdapterConfig, ModelConfig, TFNSConfig
from tfns.consolidate.plasticity import (
    activate_adapter,
    plasticity_report,
    should_activate_adapter,
)
from tfns.model.agent import RecurrentAgent
from tfns.protect.bases import empty_basis


def _blocked_block(score: float, rho: float = 0.05):
    return {
        "median_rho": rho,
        "score": score,
        "replay_loss": 0.2,
        "conservation_loss": 0.1,
        "raw_grad_norm": 1.0,
        "candidate_delta_norm": 10.0,
        "projected_delta_norm": 0.2,
    }


def test_plasticity_report_includes_free_rank_and_median_rho():
    state = SimpleNamespace(bases={"dense": empty_basis(4)})
    modules = {"dense": SimpleNamespace(d_aug=4)}
    history = [
        {"modules": {"dense": {"applied_norm": 1.0, "candidate_delta_norm": 4.0}}},
        {"modules": {"dense": {"applied_norm": 2.0, "candidate_delta_norm": 4.0}}},
    ]

    report = plasticity_report(state, modules, history)

    assert report["modules"]["dense"]["free_rank"] == 1.0
    np.testing.assert_allclose(report["modules"]["dense"]["rho"], 0.375)
    np.testing.assert_allclose(report["median_rho"], 0.375)


def test_should_activate_adapter_requires_low_rho_stagnation_losses_and_protection_blockage():
    cfg = TFNSConfig(adapter=AdapterConfig(patience_blocks=3, plasticity_ratio_thresh=0.1))
    blocked = [_blocked_block(1.0), _blocked_block(1.0), _blocked_block(0.99)]

    assert should_activate_adapter(blocked, cfg) is True

    improving = [_blocked_block(1.0), _blocked_block(1.05), _blocked_block(1.10)]
    assert should_activate_adapter(improving, cfg) is False

    high_rho = [_blocked_block(1.0), _blocked_block(1.0, rho=0.2), _blocked_block(0.99)]
    assert should_activate_adapter(high_rho, cfg) is False

    broken_optimizer = [_blocked_block(1.0), _blocked_block(1.0), _blocked_block(0.99)]
    broken_optimizer[-1]["raw_grad_norm"] = 0.0
    assert should_activate_adapter(broken_optimizer, cfg) is False


def test_maybe_activate_adapter_fires_on_sustained_protection_obstruction():
    from tfns.train.block import _maybe_activate_adapter

    cfg = TFNSConfig(adapter=AdapterConfig(patience_blocks=3, plasticity_ratio_thresh=0.1))
    obstructed = {
        "candidate_delta_norm": 10.0,
        "projected_delta_norm": 0.2,
        "replay_tube_total": 0.2,
        "raw_grad_norm": 1.0,
    }

    def fresh_state():
        return SimpleNamespace(
            bases={"encoder_conv1": empty_basis(4)},
            adapter_dormant=np.array([True, True], dtype=np.bool_),
            robust_stats={},
        )

    # Stagnating score across the patience window -> activate on the third block.
    state = fresh_state()
    scores = [1.0, 1.0, 0.99]
    idx = None
    for score in scores:
        idx = _maybe_activate_adapter(
            state, obstructed, cfg, score=score, detector_changed=False
        )
    assert idx == 0
    assert int(np.sum(~state.adapter_dormant)) == 1

    # Improving score never triggers activation.
    state = fresh_state()
    for score in [1.0, 1.05, 1.10]:
        idx = _maybe_activate_adapter(
            state, obstructed, cfg, score=score, detector_changed=False
        )
    assert idx is None
    assert bool(np.all(state.adapter_dormant))

    # With no protected bases (plain regime) activation never fires.
    plain = SimpleNamespace(bases={}, adapter_dormant=np.array([True], dtype=np.bool_), robust_stats={})
    for score in [1.0, 1.0, 0.99]:
        assert _maybe_activate_adapter(plain, obstructed, cfg, score=score, detector_changed=False) is None


def test_activate_adapter_flips_lowest_dormant_and_exhaustion_returns_none():
    state = SimpleNamespace(adapter_dormant=np.array([True, True], dtype=np.bool_))

    state, idx0 = activate_adapter(state)
    assert idx0 == 0
    np.testing.assert_array_equal(state.adapter_dormant, np.array([False, True]))

    state, idx1 = activate_adapter(state)
    assert idx1 == 1
    np.testing.assert_array_equal(state.adapter_dormant, np.array([False, False]))

    state, idx_none = activate_adapter(state)
    assert idx_none is None


def test_zero_init_adapter_activation_changes_no_output_initially():
    model_cfg = ModelConfig(
        act_dim=4,
        conv_channels=(2, 2, 2),
        dense_dim=8,
        action_embed_dim=4,
        gru_hidden=8,
        key_dim=8,
    )
    adapter_cfg = AdapterConfig(num_adapters=2, rank=2, top_k=1)
    agent = RecurrentAgent(model_config=model_cfg, adapter_config=adapter_cfg)
    obs = jnp.zeros((1, 84, 84, 4), dtype=jnp.float32)
    prev_action = jnp.zeros((1,), dtype=jnp.int32)
    prev_reward = jnp.zeros((1,), dtype=jnp.float32)
    reset = jnp.ones((1,), dtype=bool)
    hidden = agent.init_hidden(1)
    params = agent.init(jax.random.PRNGKey(0), obs, prev_action, prev_reward, reset, hidden)[
        "params"
    ]

    dormant = jnp.array([True, True])
    active_one = jnp.array([False, True])
    out_dormant = agent.apply(
        {"params": params},
        obs,
        prev_action,
        prev_reward,
        reset,
        hidden,
        adapter_dormant=dormant,
    )
    out_active = agent.apply(
        {"params": params},
        obs,
        prev_action,
        prev_reward,
        reset,
        hidden,
        adapter_dormant=active_one,
    )

    np.testing.assert_allclose(np.asarray(out_active.logits), np.asarray(out_dormant.logits), atol=1e-6)
    np.testing.assert_allclose(np.asarray(out_active.value), np.asarray(out_dormant.value), atol=1e-6)
