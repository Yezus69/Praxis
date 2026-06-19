from __future__ import annotations

import os

os.environ.setdefault("JAX_PLATFORMS", "cpu")

import jax
import jax.numpy as jnp
import numpy as np

from tfns.credit.credit import (
    causal_decomposition,
    eligibility_trace,
    potential_shaping,
    shaping_enabled,
    shaping_eta,
    telescoping_residual,
)
from tfns.credit.predictor import (
    ReturnPredictor,
    discounted_returns,
    make_predictor_optimizer,
    predictor_loss,
    train_step,
)


def test_contribution_indexing_assigns_credit_to_causal_action():
    t_star = 2
    reward = 7.0
    F_seq = jnp.array([0.0, 0.0, 0.0, reward, reward, reward], dtype=jnp.float32)

    parts = causal_decomposition(F_seq, jnp.asarray(reward, dtype=jnp.float32))

    assert int(jnp.argmax(parts["C"])) == t_star
    non_causal = np.delete(np.asarray(parts["c"]), t_star)
    np.testing.assert_allclose(non_causal, np.zeros_like(non_causal), rtol=0.0, atol=1e-6)


def test_eligibility_propagates_backward_and_respects_boundaries():
    gamma = 0.9
    lambda_c = 0.8
    decay = gamma * lambda_c
    t_star = 4
    C = jnp.zeros((6,), dtype=jnp.float32).at[t_star].set(1.0)

    trace = eligibility_trace(C, gamma=gamma, lambda_c=lambda_c)
    expected = np.zeros((6,), dtype=np.float32)
    for s in range(t_star + 1):
        expected[s] = decay ** (t_star - s)
    np.testing.assert_allclose(np.asarray(trace), expected, rtol=1e-6, atol=1e-6)

    ends = jnp.array([False, False, True, False, False, False])
    reset_trace = eligibility_trace(C, gamma=gamma, lambda_c=lambda_c, episode_end_mask=ends)
    expected_reset = np.array([0.0, 0.0, 0.0, decay, 1.0, 0.0], dtype=np.float32)
    np.testing.assert_allclose(np.asarray(reset_trace), expected_reset, rtol=1e-6, atol=1e-6)


def test_decomposition_sums_to_G0():
    rng = np.random.default_rng(3)
    rewards = jnp.asarray(rng.normal(size=(8,)).astype(np.float32))
    episode_end = jnp.array([False, False, False, False, False, False, False, True])
    G0_broadcast, _ = discounted_returns(rewards, gamma=0.93, episode_end_mask=episode_end)
    F_seq = jnp.asarray(rng.normal(size=(9,)).astype(np.float32))

    parts = causal_decomposition(F_seq, G0_broadcast[0])
    total = parts["c_init"] + jnp.sum(parts["c"]) + parts["c_term"]

    np.testing.assert_allclose(
        np.asarray(total),
        np.asarray(G0_broadcast[0]),
        rtol=1e-5,
        atol=1e-5,
    )


def test_potential_shaping_uses_phi_telescope_and_zero_terminal_potential():
    rewards = jnp.array([1.0, 0.5, -1.0], dtype=jnp.float32)
    Phi_seq = jnp.array([2.0, 1.0, 4.0, -3.0], dtype=jnp.float32)
    episode_end = jnp.array([False, True, True])
    gamma = 0.9
    eta = 0.25

    shaped = potential_shaping(rewards, Phi_seq, gamma, eta, episode_end)
    expected = rewards + eta * jnp.array(
        [
            gamma * Phi_seq[1] - Phi_seq[0],
            gamma * 0.0 - Phi_seq[1],
            gamma * 0.0 - Phi_seq[2],
        ],
        dtype=jnp.float32,
    )
    np.testing.assert_allclose(np.asarray(shaped), np.asarray(expected), rtol=1e-6, atol=1e-6)
    np.testing.assert_allclose(
        np.asarray(telescoping_residual(Phi_seq, gamma, episode_end)),
        np.asarray(jnp.sum((expected - rewards) / eta)),
        rtol=1e-6,
        atol=1e-6,
    )

    prefix_diffs = jnp.diff(jnp.array([0.0, 0.25, 0.25, 1.0], dtype=jnp.float32))
    assert not np.allclose(np.asarray(shaped), np.asarray(prefix_diffs))


def test_eta_formula_and_auto_disable():
    np.testing.assert_allclose(np.asarray(shaping_eta(0.0, 1.0)), 0.5, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(shaping_eta(0.75, 1.0)), 0.25, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(shaping_eta(1.0, 1.0)), 0.0, rtol=0.0, atol=1e-6)
    np.testing.assert_allclose(np.asarray(shaping_eta(2.0, 1.0)), 0.0, rtol=0.0, atol=1e-6)

    assert not shaping_enabled([0.1, 0.2, 1.0], [1.0, 1.0, 1.0], windows=2)
    assert shaping_enabled([1.0, 0.4, 0.3], [1.0, 1.0, 1.0], windows=2)
    assert not shaping_enabled([0.2], [1.0], windows=2)


def test_predictor_isolation_and_trivial_learning():
    model = ReturnPredictor(act_dim=2, action_embed_dim=4, hidden=8)
    T, B, feat_dim = 4, 3, 5
    base_features = jnp.ones((T, B, feat_dim), dtype=jnp.float32)
    actions = jnp.zeros((T, B), dtype=jnp.int32)
    rewards = jnp.zeros((T, B), dtype=jnp.float32).at[-1].set(1.0)
    resets = jnp.zeros((T, B), dtype=bool)
    episode_end = jnp.zeros((T, B), dtype=bool).at[-1].set(True)
    h0 = model.init_hidden(B)

    variables = model.init(
        jax.random.PRNGKey(0),
        base_features,
        actions,
        rewards,
        resets,
        h0,
    )
    params = variables["params"]
    batch = {
        "model": model,
        "features": base_features,
        "actions": actions,
        "rewards": rewards,
        "resets": resets,
        "episode_end_mask": episode_end,
        "gamma": 1.0,
        "h0": h0,
    }

    policy_scale = jnp.asarray(2.0, dtype=jnp.float32)

    def loss_from_policy_feature(scale):
        scaled_batch = dict(batch)
        scaled_batch["features"] = base_features * scale
        loss, _ = predictor_loss(params, scaled_batch)
        return loss

    policy_grad = jax.grad(loss_from_policy_feature)(policy_scale)
    np.testing.assert_allclose(np.asarray(policy_grad), np.asarray(0.0, dtype=np.float32), atol=1e-7)

    _, aux0 = predictor_loss(params, batch)
    tx = make_predictor_optimizer({"lr": 3e-2})
    opt_state = tx.init(params)
    for _ in range(60):
        params, opt_state, _ = train_step(params, opt_state, batch, tx)
    _, aux1 = predictor_loss(params, batch)

    assert float(aux1["mse_F"]) < float(aux0["mse_F"])
