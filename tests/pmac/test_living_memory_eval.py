import numpy as np
import jax.numpy as jnp

from pmac.agents.living_memory_eval import _select_greedy_actions, build_protected_bank
from pmac.memory import SourceFlag
from pmac.memory.reader import expand_source_flags


def _policy(n, act_dim):
    out = np.zeros((n, act_dim), dtype=np.float32)
    out[:, 0] = 0.75
    out[:, 1:] = 0.25 / max(1, act_dim - 1)
    return out


def _protected(keys, importance, game_id, *, d_c=3, act_dim=4):
    keys = np.asarray(keys, dtype=np.float32)
    n = int(keys.shape[0])
    flags = np.full(
        (n,),
        int(SourceFlag.SENTINEL | SourceFlag.HIGH_RETURN),
        dtype=np.int32,
    )
    return {
        "keys": keys,
        "context": np.full((n, d_c), float(game_id), dtype=np.float32),
        "teacher_policy": _policy(n, act_dim),
        "teacher_value": np.linspace(0.0, 1.0, n, dtype=np.float32),
        "importance": np.asarray(importance, dtype=np.float32),
        "game_id": np.full((n,), int(game_id), dtype=np.int32),
        "source_flags": flags,
        "age": np.zeros((n,), dtype=np.float32),
    }


def test_build_protected_bank_shapes_valid_mask_key_norm_and_source5():
    first = _protected([[3.0, 4.0], [0.0, 2.0]], [2.0, 1.0], 7)
    second = _protected([[5.0, 0.0]], [3.0], 8)

    bank = build_protected_bank([first, second], capacity=5, d_k=2, d_c=3, A=4)

    assert bank["keys"].shape == (5, 2)
    assert bank["context"].shape == (5, 3)
    assert bank["teacher_policy"].shape == (5, 4)
    assert bank["teacher_value"].shape == (5,)
    assert bank["importance"].shape == (5,)
    assert bank["game_id"].shape == (5,)
    assert bank["source5"].shape == (5, 5)
    assert bank["age"].shape == (5,)
    assert bank["valid"].shape == (5,)
    np.testing.assert_array_equal(np.asarray(bank["valid"]), [True, True, True, False, False])
    np.testing.assert_allclose(
        np.linalg.norm(np.asarray(bank["keys"])[:3], axis=1),
        np.ones((3,), dtype=np.float32),
        atol=1.0e-6,
    )

    flags = np.asarray(first["source_flags"].tolist() + second["source_flags"].tolist(), dtype=np.int32)
    np.testing.assert_allclose(
        np.asarray(bank["source5"])[:3],
        np.asarray(expand_source_flags(flags)),
        atol=0.0,
    )


def test_build_protected_bank_over_capacity_keeps_even_per_game_floor():
    first = _protected(
        [[1.0, 0.0], [2.0, 0.0], [3.0, 0.0], [4.0, 0.0]],
        [10.0, 9.0, 1.0, 0.5],
        0,
        d_c=2,
        act_dim=3,
    )
    second = _protected(
        [[0.0, 1.0], [0.0, 2.0], [0.0, 3.0], [0.0, 4.0]],
        [8.0, 7.0, 6.0, 5.0],
        1,
        d_c=2,
        act_dim=3,
    )

    bank = build_protected_bank([first, second], capacity=4, d_k=2, d_c=2, A=3)
    valid = np.asarray(bank["valid"], dtype=bool)
    game_id = np.asarray(bank["game_id"])[valid]
    importance = np.asarray(bank["importance"])[valid]

    assert int(np.sum(game_id == 0)) == 2
    assert int(np.sum(game_id == 1)) == 2
    np.testing.assert_allclose(np.sort(importance[game_id == 0]), [9.0, 10.0], atol=0.0)
    np.testing.assert_allclose(np.sort(importance[game_id == 1]), [7.0, 8.0], atol=0.0)


def test_select_greedy_actions_uses_blend_logits_or_net_logits():
    out = {
        "logits_final": jnp.asarray([[0.0, 3.0, 1.0], [5.0, 1.0, 0.0]], dtype=jnp.float32),
        "logits_net": jnp.asarray([[4.0, 0.0, 1.0], [0.0, 2.0, 3.0]], dtype=jnp.float32),
    }

    np.testing.assert_array_equal(np.asarray(_select_greedy_actions(out, blend=True)), [1, 0])
    np.testing.assert_array_equal(np.asarray(_select_greedy_actions(out, blend=False)), [0, 2])
