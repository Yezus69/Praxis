import numpy as np
import jax
import jax.numpy as jnp

from pmac.agents.ppo_living_memory_fast import (
    combine_latent_guard_aux_grads,
    combine_latent_guard_grads,
)
from pmac.memory.sentinels_visual import VisualSentinelStore, build_align_batch


def _tree_all_zero(tree, atol=1e-6):
    return all(np.allclose(np.asarray(leaf), 0.0, atol=atol) for leaf in jax.tree_util.tree_leaves(tree))


def _norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1.0e-8)


def test_visual_sentinel_store_capacity_and_batch_shapes():
    store = VisualSentinelStore(per_game=2)
    for i in range(3):
        obs = np.full((4, 84, 84), i, dtype=np.uint8)
        key = np.asarray([1.0 + i, 0.0], dtype=np.float32)
        policy = np.asarray([0.75, 0.25], dtype=np.float32)
        store.add(7, obs, key, policy, np.asarray(float(i), dtype=np.float32))

    assert len(store) == 2
    batch = store.batch()

    assert batch["obs"].shape == (2, 4, 84, 84)
    assert batch["obs"].dtype == np.uint8
    assert batch["game_id"].shape == (2,)
    assert batch["key_star"].shape == (2, 2)
    assert batch["key_star"].dtype == np.float16
    assert batch["teacher_policy"].shape == (2, 2)
    assert batch["teacher_value"].shape == (2,)
    np.testing.assert_array_equal(batch["obs"][:, 0, 0, 0], [1, 2])

    fixed = store.batch(4, seed=0)
    assert fixed["obs"].shape == (4, 4, 84, 84)
    assert fixed["key_star"].shape == (4, 2)


def test_build_align_batch_uses_different_game_hard_negatives():
    sent = {
        "obs": np.zeros((2, 4, 84, 84), dtype=np.uint8),
        "game_id": np.asarray([0, 1], dtype=np.int32),
        "key_star": _norm([[1.0, 0.0], [0.0, 1.0]]).astype(np.float16),
        "teacher_policy": np.full((2, 3), 1.0 / 3.0, dtype=np.float16),
        "teacher_value": np.zeros((2,), dtype=np.float16),
    }
    bank = {
        "keys": jnp.asarray(_norm([[1.0, 0.0], [0.9, 0.1], [0.0, 1.0], [0.1, 0.9]])),
        "game_id": jnp.asarray([0, 2, 1, 2], dtype=jnp.int32),
        "valid": jnp.asarray([True, True, True, True]),
    }

    align = build_align_batch(sent, bank, n_neg=2, batch_size=2, seed=1)

    assert align["obs"].shape == (2, 4, 84, 84)
    assert align["pos_key"].shape == (2, 2)
    assert align["neg_keys"].shape == (2, 2, 2)
    assert align["neg_game_id"].shape == (2, 2)
    assert np.all(align["neg_game_id"] != align["game_id"][:, None])


def test_aux_guard_combine_adds_visual_retr_before_stability_and_skips_nonfinite():
    params = {"w": jnp.asarray([1.0, 1.0], dtype=jnp.float32)}
    omega = {"w": jnp.asarray([1.0, 3.0], dtype=jnp.float32)}
    g_task = {"w": jnp.asarray([1.0, 1.0], dtype=jnp.float32)}
    g_guard = {"w": jnp.asarray([0.5, 0.0], dtype=jnp.float32)}
    g_visual = {"w": jnp.asarray([3.0, 0.0], dtype=jnp.float32)}
    g_retr = {"w": jnp.asarray([0.0, 4.0], dtype=jnp.float32)}

    no_aux, _, _ = combine_latent_guard_grads(
        params,
        g_task,
        g_guard,
        omega,
        lambda_total=2.0,
        kappa=10.0,
        stability_alpha=1.0,
        project=False,
    )
    with_aux, _, metrics = combine_latent_guard_aux_grads(
        params,
        g_task,
        g_guard,
        g_visual,
        g_retr,
        omega,
        lambda_total=2.0,
        kappa=10.0,
        stability_alpha=1.0,
        project=False,
    )

    np.testing.assert_allclose(np.asarray(no_aux["w"]), [1.0, 0.25], atol=1e-6)
    np.testing.assert_allclose(np.asarray(with_aux["w"]), [2.5, 1.25], atol=1e-6)
    assert not bool(np.asarray(metrics.nonfinite))

    nonfinite, omega_after_nan, nan_metrics = combine_latent_guard_aux_grads(
        params,
        g_task,
        g_guard,
        {"w": jnp.asarray([jnp.nan, 0.0], dtype=jnp.float32)},
        g_retr,
        omega,
        project=False,
    )
    assert bool(np.asarray(nan_metrics.nonfinite))
    assert _tree_all_zero(nonfinite)
    np.testing.assert_allclose(np.asarray(omega_after_nan["w"]), np.asarray(omega["w"]), atol=0.0)
