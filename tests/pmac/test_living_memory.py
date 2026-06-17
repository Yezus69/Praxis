import numpy as np
import jax
import jax.numpy as jnp

from pmac.agents.atari_mem_net import mem_init
from pmac.agents.ppo_living_memory import lm_policy_step
from pmac.memory.atom import SourceFlag
from pmac.memory.bank import MemoryBank
from pmac.memory.reader import expand_source_flags
from pmac.memory.runtime import RunningValueNorm, default_retrieval_hp, pad_bank


def _policy(n, act_dim):
    policy = np.zeros((n, act_dim), dtype=np.float32)
    policy[:, 0] = 0.7
    policy[:, 1:] = 0.3 / max(1, act_dim - 1)
    return policy


def test_pad_bank_empty_shapes_and_invalid_rows():
    bank = MemoryBank(capacity=4, d_k=3, d_c=2, act_dim=5)
    arrays = pad_bank(bank, 6, d_k=3, d_c=2, act_dim=5)

    assert arrays["keys"].shape == (6, 3)
    assert arrays["context"].shape == (6, 2)
    assert arrays["teacher_policy"].shape == (6, 5)
    assert arrays["source5"].shape == (6, 5)
    assert arrays["valid"].shape == (6,)
    assert not bool(np.any(np.asarray(arrays["valid"])))


def test_pad_bank_fills_valid_rows_and_expands_source_flags():
    bank = MemoryBank(capacity=4, d_k=2, d_c=2, act_dim=3)
    flags = np.asarray([int(SourceFlag.HIGH_RETURN), int(SourceFlag.HIGH_RETURN | SourceFlag.NOVELTY)])
    bank.insert(
        [[3.0, 4.0], [0.0, 5.0]],
        [[1.0, 0.0], [0.0, 1.0]],
        _policy(2, 3),
        [0.5, -0.5],
        [1.0, 2.0],
        [7, 7],
        eps_policy=0.01,
        eps_value=0.02,
        source_flags=flags,
    )

    arrays = pad_bank(bank, 4, d_k=2, d_c=2, act_dim=3)

    np.testing.assert_array_equal(np.asarray(arrays["valid"]), [True, True, False, False])
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(arrays["keys"])[:2], axis=1),
        [1.0, 1.0],
        atol=1.0e-2,
    )
    np.testing.assert_allclose(
        np.asarray(arrays["source5"])[:2],
        np.asarray(expand_source_flags(flags)),
        atol=0.0,
    )


def test_pad_bank_over_capacity_keeps_top_importance():
    bank = MemoryBank(capacity=5, d_k=2, d_c=1, act_dim=3)
    bank.insert(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0], [0.0, -1.0]],
        np.zeros((5, 1), dtype=np.float32),
        _policy(5, 3),
        np.zeros((5,), dtype=np.float32),
        [0.1, 10.0, 5.0, 8.0, 1.0],
        [0, 0, 0, 0, 0],
        eps_policy=0.01,
        eps_value=0.02,
    )

    arrays = pad_bank(bank, 3, d_k=2, d_c=1, act_dim=3)

    np.testing.assert_array_equal(np.asarray(arrays["valid"]), [True, True, True])
    np.testing.assert_allclose(np.sort(np.asarray(arrays["importance"])), [5.0, 8.0, 10.0])


def test_running_value_norm_moves_and_stays_positive():
    norm = RunningValueNorm(momentum=0.9, sigma_floor=0.05)
    norm.update(np.asarray([10.0, 14.0], dtype=np.float32))

    assert 0.0 < norm.mu() < 14.0
    assert norm.sigma() >= 0.05
    z = (np.asarray([10.0, 14.0], dtype=np.float32) - norm.mu()) / norm.sigma()
    assert np.all(np.isfinite(z))


def test_default_retrieval_hp_documented_keys_and_top_k():
    hp = default_retrieval_hp(7)

    assert hp == {
        "tau_r": 0.5,
        "beta_c": 1.0,
        "beta_I": 0.25,
        "beta_a": 0.1,
        "top_k": 7,
        "w_rho": 4.0,
        "w_c": 1.0,
        "b0": 1.0,
    }


def test_lm_policy_step_empty_bank_smoke_cpu():
    n_games = 2
    d_k = 8
    d_c = 4
    d_m = 6
    act_dim = 5
    capacity = 4
    params = mem_init(
        jax.random.PRNGKey(0),
        n_games,
        capacity,
        d_k=d_k,
        d_c=d_c,
        d_m=d_m,
        act_dim=act_dim,
        top_k=2,
    )
    bank = MemoryBank(capacity=8, d_k=d_k, d_c=d_c, act_dim=act_dim)
    bank_arrays = pad_bank(bank, capacity, d_k=d_k, d_c=d_c, act_dim=act_dim)
    obs = np.zeros((2, 4, 84, 84), dtype=np.uint8)
    game_id = jnp.asarray([0, 1], dtype=jnp.int32)

    actions, logprobs, values, _ = lm_policy_step(
        params,
        obs,
        game_id,
        bank_arrays,
        0.0,
        1.0,
        jax.random.PRNGKey(1),
        hp=default_retrieval_hp(2),
    )

    assert actions.shape == (2,)
    assert logprobs.shape == (2,)
    assert values.shape == (2,)
    assert np.all(np.isfinite(np.asarray(logprobs)))
    assert np.all(np.isfinite(np.asarray(values)))
