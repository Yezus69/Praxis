"""tests/test_smoke.py — coverage env smoke test (the orchestrator gate).

Verifies the FROZEN contract shapes, jit/vmap traceability, finite outputs, valid
done flags, reset determinism, and that coverage actually accrues as the agent moves.
Passes on CPU or GPU.
"""

from __future__ import annotations

import jax
import jax.numpy as jp
import numpy as np
import pytest

from praxis import contract
from praxis.envs import CoverEnv

N_VMAP = 4096


@pytest.fixture(scope="module")
def env():
    return CoverEnv()


def test_sizes_match_contract(env):
    assert env.action_size == contract.ACT_DIM == 2
    assert env.observation_size == contract.OBS_DIM == 28


def test_jit_reset_step_single(env):
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    state = jit_reset(jax.random.PRNGKey(0))
    assert state.obs.shape == (contract.OBS_DIM,)
    assert jp.isfinite(state.obs).all()

    state2 = jit_step(state, jp.zeros((contract.ACT_DIM,)))
    assert state2.obs.shape == (contract.OBS_DIM,)
    assert jp.isfinite(state2.obs).all()
    assert jp.isfinite(state2.reward).all()
    assert float(state2.done) in (0.0, 1.0)
    assert float(state2.info["time_out"]) in (0.0, 1.0)


def test_metric_keys_consistent(env):
    state = env.reset(jax.random.PRNGKey(3))
    state2 = env.step(state, jp.zeros((contract.ACT_DIM,)))
    assert set(state.metrics.keys()) == set(state2.metrics.keys())
    expected = {contract.METRIC_COVERAGE, contract.METRIC_COLLISION,
                *contract.METRIC_REWARD_COMPONENTS}
    assert set(state2.metrics.keys()) == expected


def test_vmap_reset_step_batch(env):
    n = N_VMAP
    keys = jax.random.split(jax.random.PRNGKey(42), n)
    states = jax.jit(jax.vmap(env.reset))(keys)
    assert states.obs.shape == (n, contract.OBS_DIM)
    assert jp.isfinite(states.obs).all()

    actions = jp.zeros((n, contract.ACT_DIM))
    states2 = jax.jit(jax.vmap(env.step))(states, actions)
    assert states2.obs.shape == (n, contract.OBS_DIM)
    assert states2.reward.shape == (n,)
    assert states2.done.shape == (n,)
    assert jp.isfinite(states2.obs).all()
    assert jp.isfinite(states2.reward).all()
    done = np.asarray(states2.done)
    assert np.all((done == 0.0) | (done == 1.0))


def test_coverage_accrues(env):
    """Driving the agent should cover new cells (coverage rises, reward positive)."""
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    state = jit_reset(jax.random.PRNGKey(7))
    covered0 = float(state.info["covered"])  # cumulative cells (0 at reset)
    action = jp.array([1.0, 0.7])  # drive across the arena
    total_r = 0.0
    for _ in range(60):
        state = jit_step(state, action)
        total_r += float(state.reward)
        assert jp.isfinite(state.obs).all()
    covered1 = float(state.info["covered"])
    assert covered1 > covered0, f"coverage did not increase: {covered0} -> {covered1}"
    assert total_r > 0.0, "moving across fresh cells should net positive reward"


def test_reset_determinism(env):
    jit_reset = jax.jit(env.reset)
    a = jit_reset(jax.random.PRNGKey(123))
    b = jit_reset(jax.random.PRNGKey(123))
    c = jit_reset(jax.random.PRNGKey(124))
    np.testing.assert_array_equal(np.asarray(a.obs), np.asarray(b.obs))
    assert not np.array_equal(np.asarray(a.obs), np.asarray(c.obs))


def test_step_determinism(env):
    jit_reset = jax.jit(env.reset)
    jit_step = jax.jit(env.step)
    action = jp.array([0.5, -0.3])
    s0 = jit_reset(jax.random.PRNGKey(9))
    s1a = jit_step(s0, action)
    s1b = jit_step(s0, action)
    np.testing.assert_array_equal(np.asarray(s1a.obs), np.asarray(s1b.obs))
    np.testing.assert_array_equal(np.asarray(s1a.reward), np.asarray(s1b.reward))
