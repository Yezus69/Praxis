import numpy as np
import jax.numpy as jnp

from pmac.agents.ppo_living_memory_fast import (
    empty_hot_bank,
    hot_insert,
    hot_novelty,
    write_importance,
)


def _norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1.0e-8)


def _atoms(keys, importances, *, d_c=2, act_dim=4, game_id=0, valid=None):
    keys = np.asarray(keys, dtype=np.float32)
    n = int(keys.shape[0])
    if valid is None:
        valid = np.ones((n,), dtype=bool)
    policy = np.zeros((n, act_dim), dtype=np.float32)
    policy[:, 0] = 1.0
    return {
        "keys": jnp.asarray(_norm(keys)),
        "context": jnp.zeros((n, d_c), dtype=jnp.float32),
        "teacher_policy": jnp.asarray(policy),
        "teacher_value": jnp.zeros((n,), dtype=jnp.float32),
        "importance": jnp.asarray(importances, dtype=jnp.float32),
        "game_id": jnp.full((n,), int(game_id), dtype=jnp.int32),
        "source5": jnp.zeros((n, 5), dtype=jnp.float32),
        "age": jnp.zeros((n,), dtype=jnp.float32),
        "valid": jnp.asarray(valid, dtype=bool),
    }


def test_empty_hot_bank_shapes_and_invalid_rows():
    bank = empty_hot_bank(5, 3, 2, 4)

    assert bank["keys"].shape == (5, 3)
    assert bank["context"].shape == (5, 2)
    assert bank["teacher_policy"].shape == (5, 4)
    assert bank["source5"].shape == (5, 5)
    assert bank["valid"].shape == (5,)
    assert not bool(np.any(np.asarray(bank["valid"])))


def test_hot_insert_keeps_top_capacity_by_importance_and_ignores_invalid():
    bank = _atoms(
        keys=[[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]],
        importances=[1.0, 5.0, 2.0],
    )
    new_atoms = _atoms(
        keys=[[1.0, -1.0], [-1.0, 0.0], [0.5, 0.5], [0.25, 1.0]],
        importances=[4.0, 10.0, 99.0, 6.0],
        valid=[True, True, False, True],
    )

    updated = hot_insert(bank, new_atoms)

    assert updated["keys"].shape == (3, 2)
    np.testing.assert_array_equal(np.asarray(updated["valid"]), [True, True, True])
    np.testing.assert_allclose(
        np.sort(np.asarray(updated["importance"])),
        [5.0, 6.0, 10.0],
        atol=0.0,
    )


def test_hot_insert_empty_bank_pads_remaining_slots_invalid():
    bank = empty_hot_bank(4, 2, 2, 3)
    new_atoms = _atoms(
        keys=[[1.0, 0.0], [0.0, 1.0]],
        importances=[3.0, 1.0],
        act_dim=3,
    )

    updated = hot_insert(bank, new_atoms)

    assert int(np.sum(np.asarray(updated["valid"]))) == 2
    np.testing.assert_allclose(
        np.sort(np.asarray(updated["importance"])[np.asarray(updated["valid"])]),
        [1.0, 3.0],
        atol=0.0,
    )
    np.testing.assert_allclose(
        np.asarray(updated["importance"])[~np.asarray(updated["valid"])],
        np.zeros((2,), dtype=np.float32),
        atol=0.0,
    )


def test_hot_novelty_same_game_mask_shape_and_finite():
    bank = {
        "keys": jnp.asarray(_norm([[1.0, 0.0], [0.0, 1.0]])),
        "context": jnp.zeros((2, 1), dtype=jnp.float32),
        "teacher_policy": jnp.zeros((2, 2), dtype=jnp.float32),
        "teacher_value": jnp.zeros((2,), dtype=jnp.float32),
        "importance": jnp.ones((2,), dtype=jnp.float32),
        "game_id": jnp.asarray([7, 8], dtype=jnp.int32),
        "source5": jnp.zeros((2, 5), dtype=jnp.float32),
        "age": jnp.zeros((2,), dtype=jnp.float32),
        "valid": jnp.ones((2,), dtype=bool),
    }
    keys = jnp.asarray(_norm([[1.0, 0.0], [0.0, 1.0], [1.0, 1.0]]))

    scores = hot_novelty(keys, bank, jnp.asarray([7, 7, 9], dtype=jnp.int32))

    assert scores.shape == (3,)
    assert np.all(np.isfinite(np.asarray(scores)))
    np.testing.assert_allclose(np.asarray(scores), [0.0, 1.0, 1.0], atol=1.0e-6)


def test_write_importance_jitted_shape_finite_and_weights():
    abs_adv_hat = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    abs_delta_hat = jnp.asarray([0.5, 1.5], dtype=jnp.float32)
    novelty = jnp.asarray([0.2, 0.8], dtype=jnp.float32)
    entropy = jnp.asarray([0.4, 0.6], dtype=jnp.float32)
    life = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
    ret_hat = jnp.asarray([1.5, -0.5], dtype=jnp.float32)
    forget = jnp.asarray([0.25, 0.25], dtype=jnp.float32)

    scores = write_importance(
        abs_adv_hat,
        abs_delta_hat,
        novelty,
        entropy,
        life,
        ret_hat,
        forget,
    )

    expected = (
        np.asarray(abs_adv_hat)
        + np.asarray(abs_delta_hat)
        + 1.5 * np.asarray(novelty)
        + 0.25 * np.asarray(entropy)
        + 3.0 * np.asarray(life)
        + 2.0 * np.asarray(ret_hat)
        + 3.0 * np.asarray(forget)
    )
    assert scores.shape == (2,)
    assert np.all(np.isfinite(np.asarray(scores)))
    np.testing.assert_allclose(np.asarray(scores), expected, atol=1.0e-6)
