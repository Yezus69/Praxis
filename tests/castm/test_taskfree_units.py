"""Task-free trainer unit guards: forced-FIRE PPO mask (14), no head re-init (7),
completed-episode gate (15)."""

from __future__ import annotations

import pathlib

import jax
import jax.numpy as jnp
import numpy as np
import pytest

import baseline_ppo as bp
from tfns.castm import train_castm as tc
from tfns.castm import train_taskfree as tf


def _synth_batch(n, a_dim, seed=0):
    rng = np.random.default_rng(seed)
    actions = jnp.asarray(rng.integers(0, a_dim, size=n).astype(np.int32))
    return tc.Batch(
        obs=jnp.zeros((n, 1)),  # unused by _ppo_terms
        actions=actions,
        logprobs=jnp.asarray(rng.standard_normal(n).astype(np.float32)),
        advantages=jnp.asarray(rng.standard_normal(n).astype(np.float32)),
        returns=jnp.asarray(rng.standard_normal(n).astype(np.float32)),
        values=jnp.asarray(rng.standard_normal(n).astype(np.float32)),
    )


def test_forced_fire_zeroes_policy_grad_at_forced_steps():
    """Required test 14: forced transitions contribute zero policy-ratio/entropy grad."""
    n, a_dim = 12, 6
    batch = _synth_batch(n, a_dim, seed=1)
    rng = np.random.default_rng(2)
    logits = jnp.asarray(rng.standard_normal((n, a_dim)).astype(np.float32))
    values = jnp.asarray(rng.standard_normal(n).astype(np.float32))
    forced = np.zeros(n, np.float32)
    forced[[1, 4, 7, 9]] = 1.0          # these steps were environment-forced (FIRE)
    free_mask = jnp.asarray(1.0 - forced)

    def policy_only_loss(lg, fm):
        loss, m = tc._ppo_terms(lg, values, batch, 0.1, 0.5, 0.01, free_mask=fm)
        # isolate the logits-dependent part (pg - ent*entropy); value loss has no logits dep
        return m[1] - 0.01 * m[3]

    g_masked = np.asarray(jax.grad(policy_only_loss)(logits, free_mask))
    g_unmasked = np.asarray(jax.grad(policy_only_loss)(logits, jnp.ones(n)))
    forced_idx = np.flatnonzero(forced)
    free_idx = np.flatnonzero(1.0 - forced)
    assert np.allclose(g_masked[forced_idx], 0.0, atol=1e-7), "forced steps leaked into policy grad"
    assert np.linalg.norm(g_masked[free_idx]) > 1e-4, "free steps have no gradient (mask too aggressive)"
    assert np.linalg.norm(g_unmasked[forced_idx]) > 1e-4, "without the mask forced steps DO contribute (control)"


def test_value_loss_still_uses_forced_steps():
    """Value learning keeps all transitions (architecture §6)."""
    n, a_dim = 10, 5
    batch = _synth_batch(n, a_dim, seed=3)
    rng = np.random.default_rng(4)
    logits = jnp.asarray(rng.standard_normal((n, a_dim)).astype(np.float32))
    values = jnp.asarray(rng.standard_normal(n).astype(np.float32))
    free_mask = jnp.zeros(n)  # ALL steps forced -> policy grad fully masked, value loss survives
    _, m = tc._ppo_terms(logits, values, batch, 0.1, 0.5, 0.01, free_mask=free_mask)
    assert float(m[2]) > 0.0, "value loss vanished when all steps forced"
    # value loss grad w.r.t. values is non-zero even with all policy steps masked
    def vloss(v):
        _, mm = tc._ppo_terms(logits, v, batch, 0.1, 0.5, 0.01, free_mask=free_mask)
        return mm[2]
    assert np.linalg.norm(np.asarray(jax.grad(vloss)(values))) > 1e-4


def test_no_head_reinitialization_in_source():
    """Required test 7: the task-free trainer never re-initialises policy/value heads."""
    src = pathlib.Path(tf.__file__).read_text(encoding="utf-8")
    low = src.lower()
    # The boundary head-reset shortcut (present in train_plastic) must be absent here.
    assert "initializers.orthogonal" not in src, "head re-init present in task-free trainer"
    assert ".replace(w0=" not in low, "weight re-initialisation present in task-free trainer"
    # The continuous mechanism (online resolve) must be wired in instead.
    assert "online_resolve" in src
    assert "reset_opt_on_alloc" in src  # regime-change handled via optimizer moments, not weights


class _FakeEnv:
    def __init__(self, num_envs, done_each_step):
        self.n = int(num_envs)
        self.done = bool(done_each_step)

    def reset(self):
        return np.zeros((self.n, 8, 8, 4), np.uint8), {}

    def step(self, a):
        obs = np.zeros((self.n, 8, 8, 4), np.uint8)
        reward = np.ones((self.n,), np.float32)
        term = np.full((self.n,), self.done, bool)
        trunc = np.zeros((self.n,), bool)
        return obs, reward, term, trunc, {}

    def close(self):
        pass


def test_completed_episode_gate(monkeypatch):
    """Required test 15: an evaluation that cannot complete enough episodes is invalid."""
    def fake_sample(banks, obs, k, ctx_id, fire, rng):
        return jnp.zeros((obs.shape[0],), jnp.int32), rng

    # (a) env that never terminates within the step budget -> invalid.
    monkeypatch.setattr(bp, "make_env", lambda *a, **k: _FakeEnv(4, done_each_step=False))
    ev_invalid = tc.evaluate_game(fake_sample, None, "X", jnp.zeros((4,)), 0,
                                  num_envs=4, n_episodes=8, seed=0, fire_reset=False, max_steps=40)
    assert ev_invalid["valid"] is False and ev_invalid["n"] < 8

    # (b) env that terminates every step -> completes quickly -> valid.
    monkeypatch.setattr(bp, "make_env", lambda *a, **k: _FakeEnv(4, done_each_step=True))
    ev_valid = tc.evaluate_game(fake_sample, None, "X", jnp.zeros((4,)), 0,
                                num_envs=4, n_episodes=8, seed=0, fire_reset=False, max_steps=10_000)
    assert ev_valid["valid"] is True and ev_valid["n"] >= 8
