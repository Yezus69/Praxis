import numpy as np

from pmac.memory import MemoryBank
from pmac.memory.atom import SourceFlag
from pmac.memory.write import (
    DEFAULT_WRITE_WEIGHTS,
    RunningStats,
    build_insert_kwargs,
    importance,
    novelty,
    policy_entropy,
    select_writes,
    td_error,
    teacher_targets,
    write_source_flags,
)


def _prob_entropy(p):
    p = np.asarray(p, dtype=np.float32)
    return -np.sum(p * np.log(p + 1.0e-8), axis=-1)


def test_importance_uses_exact_spec_weights():
    assert DEFAULT_WRITE_WEIGHTS == {
        "adv": 1.0,
        "delta": 1.0,
        "novelty": 1.5,
        "entropy": 0.25,
        "life": 3.0,
        "ret": 2.0,
        "forget": 3.0,
    }
    abs_adv_hat = np.asarray([1.0, 2.0], dtype=np.float32)
    abs_delta_hat = np.asarray([0.5, 1.5], dtype=np.float32)
    nov = np.asarray([0.2, 0.8], dtype=np.float32)
    ent = np.asarray([0.4, 0.6], dtype=np.float32)
    life = np.asarray([0.0, 1.0], dtype=np.float32)
    ret_hat = np.asarray([1.5, -0.5], dtype=np.float32)
    forget = np.asarray([0.25, 0.25], dtype=np.float32)

    expected = (
        abs_adv_hat
        + abs_delta_hat
        + 1.5 * nov
        + 0.25 * ent
        + 3.0 * life
        + 2.0 * ret_hat
        + 3.0 * forget
    )
    np.testing.assert_allclose(
        importance(abs_adv_hat, abs_delta_hat, nov, ent, life, ret_hat, forget),
        expected,
        atol=1.0e-7,
    )


def test_teacher_targets_softmax_temperature_and_value_normalization():
    logits = np.asarray([[3.0, 0.0, -3.0], [1.0, 1.0, 1.0]], dtype=np.float32)
    values = np.asarray([12.0, 8.0], dtype=np.float32)

    p1, v1 = teacher_targets(logits, values, mu_g=10.0, sigma_g=2.0, temperature=1.0)
    p2, _ = teacher_targets(logits, values, mu_g=10.0, sigma_g=2.0, temperature=2.0)

    np.testing.assert_allclose(np.sum(p1, axis=1), np.ones((2,), dtype=np.float32), atol=1.0e-6)
    assert _prob_entropy(p2)[0] > _prob_entropy(p1)[0]
    np.testing.assert_allclose(v1, (values - 10.0) / (2.0 + 1.0e-8), atol=1.0e-6)


def test_td_error_matches_one_step_bootstrap():
    rewards = np.asarray([1.0, -1.0, 0.25], dtype=np.float32)
    values = np.asarray([0.5, 2.0, -0.5], dtype=np.float32)
    next_values = np.asarray([0.75, -1.0, 0.5], dtype=np.float32)

    np.testing.assert_allclose(
        td_error(rewards, values, next_values, gamma=0.9),
        rewards + 0.9 * next_values - values,
        atol=1.0e-7,
    )


def test_novelty_same_game_cosine_and_empty_game_case():
    bank_keys = np.asarray([[1.0, 0.0]], dtype=np.float32)
    bank_valid = np.asarray([True])
    bank_game_id = np.asarray([7], dtype=np.int32)
    keys = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)

    scores = novelty(keys, bank_keys, bank_valid, bank_game_id, cur_game=7)
    np.testing.assert_allclose(scores, [0.0, 1.0], atol=1.0e-6)
    np.testing.assert_allclose(
        novelty(keys[:1], bank_keys, bank_valid, bank_game_id, cur_game=8),
        [1.0],
        atol=0.0,
    )


def test_select_writes_top_masks_quota_and_all():
    scores = np.arange(10, dtype=np.float32)
    rare = np.zeros((10,), dtype=bool)
    sentinel = np.zeros((10,), dtype=bool)
    rare[1] = True
    sentinel[2] = True

    mask = select_writes(scores, 0.2, rare_mask=rare, sentinel_mask=sentinel, min_quota=5)
    expected = np.zeros((10,), dtype=bool)
    expected[[1, 2, 7, 8, 9]] = True
    np.testing.assert_array_equal(mask, expected)
    np.testing.assert_array_equal(select_writes(scores, 1.0), np.ones((10,), dtype=bool))


def test_running_stats_normalizes_scale_and_constants_are_stable():
    stats = RunningStats()
    x = np.asarray([100.0, 200.0, 300.0], dtype=np.float32)
    stats.update(x)
    z = stats.normalize(x)

    np.testing.assert_allclose(np.mean(z), 0.0, atol=1.0e-6)
    np.testing.assert_allclose(np.mean(np.abs(z)), 1.0, atol=1.0e-5)

    const = RunningStats()
    const.update(np.asarray([5.0, 5.0, 5.0], dtype=np.float32))
    z_const = const.normalize(np.asarray([5.0, 5.0], dtype=np.float32))
    np.testing.assert_allclose(z_const, [0.0, 0.0], atol=0.0)
    assert np.all(np.isfinite(z_const))


def test_build_insert_kwargs_round_trips_through_memory_bank():
    bank = MemoryBank(capacity=4, d_k=2, d_c=2, act_dim=3)
    keys = np.asarray([[2.0, 0.0], [0.0, 3.0]], dtype=np.float32)
    contexts = np.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=np.float32)
    logits = np.asarray([[2.0, 0.0, -1.0], [0.0, 2.0, -1.0]], dtype=np.float32)
    values = np.asarray([7.0, 3.0], dtype=np.float32)
    importances = np.asarray([4.0, 5.0], dtype=np.float32)
    rarity = np.asarray([0.2, 0.8], dtype=np.float32)
    flags = write_source_flags(
        high_return=[True, False],
        near_life_loss=[False, True],
        novelty_hi=[False, True],
        failure_recovery=[False, False],
    )

    inserted = bank.insert(
        **build_insert_kwargs(
            keys,
            contexts,
            logits,
            values,
            game_ids=np.asarray([3, 3], dtype=np.int32),
            importances=importances,
            mu_g=5.0,
            sigma_g=2.0,
            temperature=1.0,
            novelty=rarity,
            eps_policy=0.01,
            eps_value=0.02,
            source_flags=flags,
        )
    )

    assert inserted.shape == (2,)
    np.testing.assert_allclose(
        np.sum(bank.teacher_policy[: len(bank)].astype(np.float32), axis=1),
        [1.0, 1.0],
        atol=1.0e-3,
    )
    np.testing.assert_allclose(bank.importance[: len(bank)], importances, atol=1.0e-7)
    np.testing.assert_allclose(bank.rarity[: len(bank)], rarity, atol=1.0e-7)
    assert int(bank.source_flags[0]) & int(SourceFlag.HIGH_RETURN)
    assert int(bank.source_flags[1]) & int(SourceFlag.NEAR_LIFE_LOSS)
    assert int(bank.source_flags[1]) & int(SourceFlag.NOVELTY)


def test_policy_entropy_is_log_softmax_stable():
    logits = np.asarray([[1000.0, 1000.0], [1000.0, -1000.0]], dtype=np.float32)
    ent = policy_entropy(logits)

    np.testing.assert_allclose(ent[0], np.log(2.0), atol=1.0e-6)
    np.testing.assert_allclose(ent[1], 0.0, atol=1.0e-6)
