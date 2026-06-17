import numpy as np
import jax
import jax.numpy as jnp

from pmac.agents.atari_mem_net import MemAtariActorCritic, mem_init
from pmac.memory import MemoryBank, SourceFlag
from pmac.memory import deletion_cert
from pmac.memory.losses import _latent_behavior_from_bank


EPS = 1e-8


def _dims(n_games=2, d_k=8, d_c=4, d_m=6, act_dim=4):
    return {
        "n_games": n_games,
        "d_k": d_k,
        "d_c": d_c,
        "d_m": d_m,
        "act_dim": act_dim,
    }


def _hp(top_k=1):
    return {
        "tau_r": 0.7,
        "beta_c": 0.0,
        "beta_I": 0.0,
        "beta_a": 0.0,
        "top_k": int(top_k),
        "w_rho": 1.0,
        "w_c": 1.0,
        "b0": 0.0,
    }


def _norm(keys):
    keys = np.asarray(keys, dtype=np.float32)
    return keys / (np.linalg.norm(keys, axis=-1, keepdims=True) + EPS)


def _policy(n, act_dim):
    policy = np.zeros((n, act_dim), dtype=np.float32)
    policy[:, 0] = 0.7
    policy[:, 1] = 0.2
    policy[:, 2:] = 0.1 / max(1, act_dim - 2)
    return policy


def _reader_bank(keys, dims):
    keys = _norm(keys)
    n = int(keys.shape[0])
    return {
        "keys": jnp.asarray(keys, dtype=jnp.float32),
        "context": jnp.zeros((n, int(dims["d_c"])), dtype=jnp.float32),
        "teacher_policy": jnp.asarray(_policy(n, int(dims["act_dim"]))),
        "teacher_value": jnp.zeros((n,), dtype=jnp.float32),
        "importance": jnp.ones((n,), dtype=jnp.float32),
        "game_id": jnp.zeros((n,), dtype=jnp.int32),
        "source5": jnp.zeros((n, 5), dtype=jnp.float32),
        "age": jnp.zeros((n,), dtype=jnp.float32),
        "valid": jnp.ones((n,), dtype=bool),
    }


def _memory_bank():
    bank = MemoryBank(capacity=4, d_k=3, d_c=2, act_dim=3)
    bank.insert(
        [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
        np.zeros((3, 2), dtype=np.float32),
        _policy(3, 3),
        [0.0, 0.0, -1.0],
        [1.0, 1.0, 1.0],
        [0, 0, 1],
        eps_policy=0.01,
        eps_value=0.01,
        source_flags=[0, 0, int(SourceFlag.SENTINEL)],
    )
    bank.cluster_id[:3] = np.asarray([10, 11, 20], dtype=np.int32)
    return bank


def _true_coverage(params, atoms, bank, hp, *, lambda_v=1.0):
    del params, bank, hp, lambda_v
    return np.ones((np.asarray(atoms["keys"]).shape[0],), dtype=bool)


def test_model_coverage_current_latent_behavior_and_far_teacher():
    dims = _dims()
    hp = _hp(top_k=1)
    params = mem_init(jax.random.PRNGKey(0), capacity=1, top_k=1, **dims)
    bank = _reader_bank([[1.0, 2.0, 3.0, 4.0, -1.0, -2.0, 0.5, 0.25]], dims)
    key = bank["keys"]
    game_id = jnp.asarray([0], dtype=jnp.int32)
    net = MemAtariActorCritic(**dims)
    logits, value = _latent_behavior_from_bank(net, params, key, game_id, bank, hp)

    atoms = {
        "keys": key,
        "game_id": game_id,
        "teacher_policy": jax.nn.softmax(logits, axis=-1),
        "teacher_value": value,
        "eps_policy": jnp.asarray([1.0e-5], dtype=jnp.float32),
    }
    assert bool(deletion_cert.model_coverage(params, atoms, bank, hp)[0])

    far_atoms = dict(
        atoms,
        teacher_policy=jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32),
        teacher_value=value + 5.0,
        eps_policy=jnp.zeros((1,), dtype=jnp.float32),
    )
    assert not bool(deletion_cert.model_coverage(params, far_atoms, bank, hp)[0])


def test_certify_deletion_requires_all_spec24_conditions(monkeypatch):
    bank = _memory_bank()
    monkeypatch.setattr(
        deletion_cert,
        "model_coverage",
        lambda params, atoms, bank, hp, *, lambda_v=1.0: np.zeros(
            (np.asarray(atoms["keys"]).shape[0],), dtype=bool
        ),
    )
    assert not deletion_cert.certify_deletion(
        bank, [0], None, _hp(), protected_game_min_clusters={0: 1}
    )

    monkeypatch.setattr(deletion_cert, "model_coverage", _true_coverage)
    assert not deletion_cert.certify_deletion(
        bank, [0, 1], None, _hp(), protected_game_min_clusters={0: 1}
    )
    assert not deletion_cert.certify_deletion(
        bank, [0], None, _hp(), protected_game_min_clusters={0: 1}, retrieval_ok=False
    )
    assert not deletion_cert.certify_deletion(
        bank, [0], None, _hp(), protected_game_min_clusters={0: 1}, sentinel_ok=False
    )
    assert not deletion_cert.certify_deletion(
        bank, [0], None, _hp(), protected_game_min_clusters={0: 1}, review_ok=False
    )
    assert deletion_cert.certify_deletion(
        bank,
        [0],
        None,
        _hp(),
        protected_game_min_clusters={0: 1, 1: 1},
        retrieval_ok=True,
        sentinel_ok=True,
        review_ok=True,
    )


def test_certify_and_prune_is_incremental_and_keeps_bank_valid(monkeypatch):
    bank = _memory_bank()

    def coverage_by_teacher_value(params, atoms, bank, hp, *, lambda_v=1.0):
        del params, bank, hp, lambda_v
        return np.asarray(atoms["teacher_value"], dtype=np.float32) >= 0.0

    monkeypatch.setattr(deletion_cert, "model_coverage", coverage_by_teacher_value)
    pruned = deletion_cert.certify_and_prune(
        bank,
        [[0], [1], [2]],
        None,
        _hp(),
        protected_game_min_clusters={0: 1, 1: 1},
    )

    assert pruned == [0]
    assert len(bank) == 2
    assert bank.per_game_counts() == {0: 1, 1: 1}
    np.testing.assert_array_equal(bank.cluster_id[: len(bank)], [11, 20])
    assert int(bank.source_flags[1]) & int(SourceFlag.SENTINEL)
