import numpy as np

from pmac.memory import MemoryBank, SourceFlag, allocate_budgets


def _policies(n, act_dim):
    policy = np.zeros((n, act_dim), dtype=np.float32)
    policy[:, 0] = 0.7
    policy[:, 1] = 0.2
    policy[:, 2:] = 0.1 / max(1, act_dim - 2)
    return policy


def test_insert_normalizes_keys_and_counts_games():
    bank = MemoryBank(capacity=4, d_k=3, d_c=2, act_dim=4)
    bank.insert(
        [[3.0, 0.0, 0.0], [0.0, 4.0, 0.0]],
        [[1.0, 0.0], [0.0, 1.0]],
        _policies(2, 4),
        [0.1, 0.2],
        [1.0, 2.0],
        [7, 8],
        eps_policy=0.01,
        eps_value=0.02,
    )

    norms = np.linalg.norm(bank.key[: len(bank)].astype(np.float32), axis=1)
    np.testing.assert_allclose(norms, np.ones((2,), dtype=np.float32), atol=1e-2)
    assert len(bank) == 2
    assert bank.per_game_counts() == {7: 1, 8: 1}


def test_capacity_eviction_keeps_high_utility_and_uncovered_sentinel():
    bank = MemoryBank(capacity=3, d_k=2, d_c=1, act_dim=3)
    bank.insert(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]],
        np.zeros((4, 1), dtype=np.float32),
        _policies(4, 3),
        np.zeros((4,), dtype=np.float32),
        [0.1, 10.0, 9.0, 0.2],
        [1, 1, 1, 1],
        eps_policy=0.01,
        eps_value=0.02,
        source_flags=[int(SourceFlag.SENTINEL), 0, 0, 0],
    )

    survivors = bank.importance[: len(bank)].astype(np.float32)
    assert len(survivors) == 3
    np.testing.assert_allclose(np.sort(survivors), [0.1, 9.0, 10.0], atol=1e-3)
    assert np.any((bank.source_flags[: len(bank)] & int(SourceFlag.SENTINEL)) != 0)


def test_b_min_floor_prefers_evicting_games_above_floor():
    bank = MemoryBank(capacity=3, d_k=2, d_c=1, act_dim=3, b_min=1)
    bank.insert(
        [[1.0, 0.0], [0.0, 1.0], [1.0, 1.0], [-1.0, 0.0]],
        np.zeros((4, 1), dtype=np.float32),
        _policies(4, 3),
        np.zeros((4,), dtype=np.float32),
        [0.0, 1.0, 2.0, 3.0],
        [0, 1, 1, 1],
        eps_policy=0.01,
        eps_value=0.02,
    )

    assert bank.per_game_counts() == {0: 1, 1: 2}


def test_merge_weighted_same_game_and_never_cross_game():
    bank = MemoryBank(capacity=4, d_k=2, d_c=1, act_dim=3)
    idx = bank.insert(
        [[1.0, 0.0], [1.0, 0.0]],
        np.zeros((2, 1), dtype=np.float32),
        [[0.7, 0.2, 0.1], [0.7, 0.2, 0.1]],
        [0.2, 0.4],
        [1.0, 2.0],
        [5, 5],
        eps_policy=[0.02, 0.01],
        eps_value=[0.03, 0.02],
    )

    assert bank.merge_new(idx, r_merge=1e-3, eps_pi_merge=1e-6, eps_v_merge=0.3) == 1
    assert len(bank) == 1
    np.testing.assert_allclose(bank.key[0].astype(np.float32), [1.0, 0.0], atol=1e-2)
    np.testing.assert_allclose(bank.teacher_policy[0].astype(np.float32), [0.7, 0.2, 0.1], atol=1e-2)
    np.testing.assert_allclose(float(bank.teacher_value[0]), 0.3, atol=1e-2)
    np.testing.assert_allclose(bank.importance[0], 2.0 + np.log(3.0), atol=1e-6)
    assert int(bank.count[0]) == 2
    np.testing.assert_allclose(float(bank.eps_policy[0]), 0.01, atol=1e-7)
    np.testing.assert_allclose(float(bank.eps_value[0]), 0.02, atol=1e-7)

    cross_game = MemoryBank(capacity=4, d_k=2, d_c=1, act_dim=3)
    idx = cross_game.insert(
        [[1.0, 0.0], [1.0, 0.0]],
        np.zeros((2, 1), dtype=np.float32),
        [[0.7, 0.2, 0.1], [0.7, 0.2, 0.1]],
        [0.2, 0.2],
        [1.0, 2.0],
        [5, 6],
        eps_policy=0.01,
        eps_value=0.02,
    )
    assert cross_game.merge_new(idx, r_merge=1e-3, eps_pi_merge=1e-6, eps_v_merge=0.1) == 0
    assert len(cross_game) == 2


def test_allocate_budgets_floor_order_and_degenerate_floor():
    budgets = allocate_budgets([0, 1, 2], [1.0, 3.0, 6.0], b_total=10, b_min=1)
    assert sum(budgets.values()) <= 10
    assert min(budgets.values()) >= 1
    assert budgets[2] > budgets[1] > budgets[0]

    tight = allocate_budgets([0, 1, 2], [1.0, 1.0, 1.0], b_total=2, b_min=1)
    assert sum(tight.values()) <= 2
    assert min(tight.values()) >= 0


def test_to_retrieval_arrays_shapes_dtypes_and_policy_sums():
    bank = MemoryBank(capacity=3, d_k=2, d_c=2, act_dim=3)
    bank.insert(
        [[3.0, 4.0], [0.0, 2.0]],
        [[1.0, 2.0], [3.0, 4.0]],
        [[0.7, 0.2, 0.1], [0.1, 0.8, 0.1]],
        [0.5, 0.6],
        [1.0, 2.0],
        [0, 1],
        eps_policy=0.01,
        eps_value=0.02,
    )

    arrays = bank.to_retrieval_arrays()
    assert arrays["keys"].shape == (2, 2)
    assert arrays["context"].shape == (2, 2)
    assert arrays["teacher_policy"].shape == (2, 3)
    assert arrays["teacher_value"].dtype == np.float32
    assert arrays["importance"].dtype == np.float32
    np.testing.assert_allclose(np.linalg.norm(arrays["keys"], axis=1), [1.0, 1.0], atol=1e-2)
    np.testing.assert_allclose(np.sum(arrays["teacher_policy"], axis=1), [1.0, 1.0], atol=1e-2)
