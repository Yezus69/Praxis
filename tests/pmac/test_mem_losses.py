import numpy as np
import jax
import jax.numpy as jnp

from pmac.agents.atari_mem_net import MemAtariActorCritic, mem_apply, mem_init
from pmac.behavior_distance import huber
from pmac.memory.losses import (
    _latent_behavior_from_bank,
    latent_conservation_loss,
    retrieval_alignment_loss,
    visual_sentinel_loss,
)


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


def _norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + EPS)


def _bank(keys, dims, valid=None):
    keys = np.asarray(keys, dtype=np.float32)
    capacity = keys.shape[0]
    act_dim = int(dims["act_dim"])
    if valid is None:
        valid = np.ones((capacity,), dtype=bool)
    policy = np.full((capacity, act_dim), 1.0 / act_dim, dtype=np.float32)
    return {
        "keys": jnp.asarray(_norm(keys)),
        "context": jnp.zeros((capacity, int(dims["d_c"])), dtype=jnp.float32),
        "teacher_policy": jnp.asarray(policy),
        "teacher_value": jnp.zeros((capacity,), dtype=jnp.float32),
        "importance": jnp.ones((capacity,), dtype=jnp.float32),
        "game_id": jnp.zeros((capacity,), dtype=jnp.int32),
        "source5": jnp.zeros((capacity, 5), dtype=jnp.float32),
        "age": jnp.zeros((capacity,), dtype=jnp.float32),
        "valid": jnp.asarray(valid),
    }


def _finite_tree(tree):
    return all(np.all(np.isfinite(np.asarray(leaf))) for leaf in jax.tree_util.tree_leaves(tree))


def _orthogonal_to(key):
    key = np.asarray(key, dtype=np.float32)
    idx = int(np.argmin(np.abs(key)))
    basis = np.zeros_like(key)
    basis[idx] = 1.0
    vec = basis - np.dot(basis, key) * key
    norm = np.linalg.norm(vec)
    if norm < 1e-6:
        basis = np.zeros_like(key)
        basis[(idx + 1) % key.shape[0]] = 1.0
        vec = basis - np.dot(basis, key) * key
        norm = np.linalg.norm(vec)
    return vec / (norm + EPS)


def test_huber_quadratic_linear_and_symmetric():
    x = jnp.asarray([0.0, 0.5, 3.0, -3.0], dtype=jnp.float32)
    y = huber(x, delta=1.0)

    np.testing.assert_allclose(np.asarray(y[0]), 0.0, atol=0.0)
    np.testing.assert_allclose(np.asarray(y[1]), 0.125, atol=1e-7)
    np.testing.assert_allclose(np.asarray(y[2]), 2.5, atol=1e-7)
    np.testing.assert_allclose(np.asarray(y[3]), np.asarray(y[2]), atol=1e-7)


def test_latent_conservation_zero_positive_and_finite_grad():
    dims = _dims()
    params = mem_init(jax.random.PRNGKey(0), capacity=1, top_k=1, **dims)
    hp = _hp(top_k=1)
    key = jnp.asarray(_norm([[1.0, 2.0, 3.0, 4.0, -1.0, -2.0, 0.5, 0.25]]))
    bank = _bank(key, dims)
    game_id = jnp.asarray([0], dtype=jnp.int32)
    net = MemAtariActorCritic(**dims)
    logits, value = _latent_behavior_from_bank(net, params, key, game_id, bank, hp)
    teacher_policy = jax.nn.softmax(logits, axis=-1)

    batch = {
        "keys": key,
        "game_id": game_id,
        "teacher_policy": teacher_policy,
        "teacher_value": value,
        "eps": jnp.asarray([1e-5], dtype=jnp.float32),
        "weight": jnp.asarray([1.0], dtype=jnp.float32),
    }
    same_loss = latent_conservation_loss(params, batch, bank, hp, dims=dims)
    np.testing.assert_allclose(np.asarray(same_loss), 0.0, atol=1e-8)

    far_policy = jnp.asarray([[1.0, 0.0, 0.0, 0.0]], dtype=jnp.float32)
    far_batch = dict(batch, teacher_policy=far_policy, teacher_value=value + 5.0, eps=jnp.zeros((1,)))
    far_loss = latent_conservation_loss(params, far_batch, bank, hp, dims=dims)
    assert float(far_loss) > 0.0

    grad = jax.grad(lambda p: latent_conservation_loss(p, far_batch, bank, hp, dims=dims))(params)
    assert _finite_tree(grad)


def test_visual_sentinel_key_behavior_and_finite_values():
    dims = _dims()
    params = mem_init(jax.random.PRNGKey(1), capacity=1, top_k=1, **dims)
    hp = _hp(top_k=1)
    bank = _bank(np.ones((1, dims["d_k"]), dtype=np.float32), dims, valid=np.zeros((1,), dtype=bool))
    obs = jax.random.uniform(jax.random.PRNGKey(2), (1, 84, 84, 4), dtype=jnp.float32)
    game_id = jnp.asarray([0], dtype=jnp.int32)
    net = MemAtariActorCritic(**dims)
    _, key_star = net.apply({"params": params}, obs, method=MemAtariActorCritic.encode)
    out = mem_apply(params, obs, game_id, bank, hp)
    teacher_policy = jax.nn.softmax(out["logits_net"], axis=-1)
    batch = {
        "obs": obs,
        "game_id": game_id,
        "key_star": key_star,
        "teacher_policy": teacher_policy,
        "teacher_value": out["v_net"],
    }

    l_key, l_beh = visual_sentinel_loss(params, batch, bank, hp, dims=dims)
    np.testing.assert_allclose(np.asarray(l_key), 0.0, atol=1e-5)
    np.testing.assert_allclose(np.asarray(l_beh), 0.0, atol=1e-7)

    orth_key = jnp.asarray([_orthogonal_to(np.asarray(key_star[0]))], dtype=jnp.float32)
    orth_batch = dict(batch, key_star=orth_key)
    l_key_orth, l_beh_orth = visual_sentinel_loss(params, orth_batch, bank, hp, dims=dims)
    np.testing.assert_allclose(np.asarray(l_key_orth), 1.0, atol=1e-4)
    assert np.all(np.isfinite(np.asarray([l_key_orth, l_beh_orth])))


def test_retrieval_alignment_prefers_positive_and_has_finite_grad():
    dims = _dims()
    params = mem_init(jax.random.PRNGKey(3), capacity=1, top_k=1, **dims)
    obs = jax.random.uniform(jax.random.PRNGKey(4), (1, 84, 84, 4), dtype=jnp.float32)
    net = MemAtariActorCritic(**dims)
    _, query = net.apply({"params": params}, obs, method=MemAtariActorCritic.encode)
    orth = jnp.asarray([_orthogonal_to(np.asarray(query[0]))], dtype=jnp.float32)
    neg = jnp.stack([orth[0], -orth[0]], axis=0)[None, :, :]

    good_batch = {"obs": obs, "pos_key": query, "neg_keys": neg}
    good_loss = retrieval_alignment_loss(params, good_batch, tau=0.1, dims=dims)
    assert float(good_loss) < 1e-3

    bad_neg = jnp.concatenate([query[:, None, :], neg[:, :1, :]], axis=1)
    bad_batch = {"obs": obs, "pos_key": orth, "neg_keys": bad_neg}
    bad_loss = retrieval_alignment_loss(params, bad_batch, tau=0.1, dims=dims)
    assert float(bad_loss) > float(good_loss) + 1.0

    grad = jax.grad(lambda p: retrieval_alignment_loss(p, good_batch, tau=0.1, dims=dims))(params)
    assert _finite_tree(grad)


def test_all_memory_losses_return_finite_scalars():
    dims = _dims()
    params = mem_init(jax.random.PRNGKey(5), capacity=1, top_k=1, **dims)
    hp = _hp(top_k=1)
    key = jnp.asarray(_norm([[0.5, 1.0, -0.25, 0.75, 2.0, -1.0, 0.3, 0.9]]))
    bank = _bank(key, dims)
    atom_batch = {
        "keys": key,
        "game_id": jnp.asarray([0], dtype=jnp.int32),
        "teacher_policy": jnp.asarray([[0.7, 0.1, 0.1, 0.1]], dtype=jnp.float32),
        "teacher_value": jnp.asarray([0.5], dtype=jnp.float32),
        "eps": jnp.asarray([0.0], dtype=jnp.float32),
        "weight": jnp.asarray([1.0], dtype=jnp.float32),
    }
    obs = jax.random.uniform(jax.random.PRNGKey(6), (1, 84, 84, 4), dtype=jnp.float32)
    net = MemAtariActorCritic(**dims)
    _, q = net.apply({"params": params}, obs, method=MemAtariActorCritic.encode)
    out = mem_apply(params, obs, atom_batch["game_id"], bank, hp)
    sent_batch = {
        "obs": obs,
        "game_id": atom_batch["game_id"],
        "key_star": q,
        "teacher_policy": jax.nn.softmax(out["logits_net"], axis=-1),
        "teacher_value": out["v_net"],
    }
    align_batch = {
        "obs": obs,
        "pos_key": q,
        "neg_keys": jnp.asarray([[_orthogonal_to(np.asarray(q[0]))]], dtype=jnp.float32),
    }

    values = (
        latent_conservation_loss(params, atom_batch, bank, hp, dims=dims),
        *visual_sentinel_loss(params, sent_batch, bank, hp, dims=dims),
        retrieval_alignment_loss(params, align_batch, dims=dims),
    )
    assert np.all(np.isfinite(np.asarray(values)))
