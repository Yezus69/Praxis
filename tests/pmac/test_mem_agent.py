import numpy as np
import jax
import jax.numpy as jnp

from pmac.agents.atari_mem_net import MemAtariActorCritic, mem_apply, mem_init
from pmac.memory import reader


EPS = 1e-8


def _hp(top_k=3, **overrides):
    hp = {
        "tau_r": 0.7,
        "beta_c": 0.4,
        "beta_I": 0.2,
        "beta_a": 0.3,
        "top_k": int(top_k),
        "w_rho": 1.5,
        "w_c": 1.2,
        "b0": 0.1,
    }
    hp.update(overrides)
    return hp


def _norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + EPS)


def _bank(c=6, d_k=4, d_c=3, act_dim=5, valid=None):
    keys = _norm(np.arange(1, c * d_k + 1, dtype=np.float32).reshape(c, d_k))
    context = np.arange(1, c * d_c + 1, dtype=np.float32).reshape(c, d_c) / 7.0
    policy = np.zeros((c, act_dim), dtype=np.float32)
    for i in range(c):
        policy[i, i % act_dim] = 0.65
        policy[i] += 0.35 / act_dim
    if valid is None:
        valid = np.ones((c,), dtype=bool)
    return {
        "keys": jnp.asarray(keys),
        "context": jnp.asarray(context),
        "teacher_policy": jnp.asarray(policy),
        "teacher_value": jnp.linspace(-0.5, 0.5, c),
        "importance": jnp.linspace(0.5, 2.0, c),
        "game_id": jnp.asarray([0, 1, 0, 2, 1, 0][:c], dtype=jnp.int32),
        "source5": reader.expand_source_flags(jnp.arange(c, dtype=jnp.int32)),
        "age": jnp.linspace(0.0, 5.0, c),
        "valid": jnp.asarray(valid),
    }


def test_retrieve_topk_matches_bruteforce_and_alpha_rho():
    bank = _bank()
    k = jnp.asarray(_norm([[2.0, 1.0, 0.5, 0.25], [0.25, 0.5, 1.0, 2.0]]))
    c_g = jnp.asarray([[1.0, 0.0, 0.5], [0.0, 1.0, 0.5]], dtype=jnp.float32)
    hp = _hp(top_k=3)

    out = reader.retrieve(k, c_g, jnp.asarray([0, 1], dtype=jnp.int32), bank, hp)

    keys = np.asarray(bank["keys"])
    context = np.asarray(bank["context"])
    valid = np.asarray(bank["valid"])
    sim_key = np.asarray(k) @ keys.T
    ctx_sim = _norm(np.asarray(c_g)) @ _norm(context).T
    age = np.asarray(bank["age"])
    age_pen = age / (np.max(age * valid.astype(np.float32)) + EPS)
    s = (
        sim_key / (hp["tau_r"] + EPS)
        + hp["beta_c"] * ctx_sim
        + hp["beta_I"] * np.log(np.asarray(bank["importance"]) + EPS)
        - hp["beta_a"] * age_pen[None, :]
    )
    s = np.where(valid[None, :], s, -1e30)
    expected_idx = np.argsort(-s, axis=1)[:, : hp["top_k"]]

    selected_keys = np.asarray(out.atom_feats[..., : keys.shape[-1]])
    np.testing.assert_allclose(selected_keys, keys[expected_idx], atol=1e-6)
    np.testing.assert_allclose(np.asarray(out.alpha).sum(axis=-1), np.ones((2,)), atol=1e-6)
    expected_rho = np.maximum(np.max(np.take_along_axis(sim_key, expected_idx, axis=1), axis=1), 0.0)
    np.testing.assert_allclose(np.asarray(out.rho), expected_rho, atol=1e-6)


def test_all_invalid_bank_blend_is_base_and_b_zero():
    act_dim = 4
    bank = _bank(act_dim=act_dim, valid=np.zeros((6,), dtype=bool))
    p_net = jnp.asarray([[0.1, 0.2, 0.3, 0.4]], dtype=jnp.float32)
    v_net = jnp.asarray([1.25], dtype=jnp.float32)
    out = reader.retrieve(
        jnp.asarray(_norm([[1.0, 0.0, 0.0, 0.0]])),
        jnp.ones((1, 3), dtype=jnp.float32),
        jnp.asarray([0], dtype=jnp.int32),
        bank,
        _hp(top_k=3),
    )
    p_final, _, v_final = reader.blend(p_net, v_net, out.p_mem, out.v_mem, out.b, 10.0, 2.0)

    np.testing.assert_allclose(np.asarray(out.b), [0.0], atol=0.0)
    np.testing.assert_allclose(np.asarray(p_final), np.asarray(p_net), atol=0.0)
    np.testing.assert_allclose(np.asarray(v_final), np.asarray(v_net), atol=0.0)


def test_blend_identities_and_logits_are_log_probability():
    p_net = jnp.asarray([[0.1, 0.9], [0.4, 0.6]], dtype=jnp.float32)
    v_net = jnp.asarray([1.0, 2.0], dtype=jnp.float32)
    p_mem = jnp.asarray([[0.7, 0.3], [0.25, 0.75]], dtype=jnp.float32)
    v_mem = jnp.asarray([3.0, 4.0], dtype=jnp.float32)

    p_final, logits_final, v_final = reader.blend(p_net, v_net, p_mem, v_mem, jnp.ones((2,)), 5.0, 2.0)
    np.testing.assert_allclose(np.asarray(p_final), np.asarray(p_mem), atol=1e-7)
    np.testing.assert_allclose(np.asarray(v_final), np.asarray(2.0 * v_mem + 5.0), atol=1e-7)
    np.testing.assert_allclose(np.asarray(logits_final), np.log(np.asarray(p_final) + EPS), atol=1e-7)

    p_final0, _, v_final0 = reader.blend(p_net, v_net, p_mem, v_mem, jnp.zeros((2,)), 5.0, 2.0)
    np.testing.assert_allclose(np.asarray(p_final0), np.asarray(p_net), atol=1e-7)
    np.testing.assert_allclose(np.asarray(v_final0), np.asarray(v_net), atol=1e-7)


def test_context_bias_prefers_same_game_and_raises_blend():
    bank = {
        "keys": jnp.asarray(_norm([[1.0, 0.0], [1.0, 0.0]])),
        "context": jnp.asarray([[1.0, 0.0], [0.0, 1.0]], dtype=jnp.float32),
        "teacher_policy": jnp.asarray([[0.9, 0.1], [0.1, 0.9]], dtype=jnp.float32),
        "teacher_value": jnp.asarray([1.0, -1.0], dtype=jnp.float32),
        "importance": jnp.ones((2,), dtype=jnp.float32),
        "game_id": jnp.asarray([7, 8], dtype=jnp.int32),
        "source5": jnp.zeros((2, 5), dtype=jnp.float32),
        "age": jnp.zeros((2,), dtype=jnp.float32),
        "valid": jnp.ones((2,), dtype=bool),
    }
    query_k = jnp.asarray(_norm([[1.0, 0.0]]))
    query_c = jnp.asarray([[1.0, 0.0]], dtype=jnp.float32)
    with_context = reader.retrieve(
        query_k, query_c, jnp.asarray([7], dtype=jnp.int32), bank, _hp(top_k=2, beta_c=2.0, w_c=3.0)
    )
    no_context_blend = reader.retrieve(
        query_k, query_c, jnp.asarray([7], dtype=jnp.int32), bank, _hp(top_k=2, beta_c=2.0, w_c=0.0)
    )

    np.testing.assert_allclose(np.asarray(with_context.atom_feats[0, 0, :2]), np.asarray(bank["keys"][0]))
    assert float(with_context.context_match[0]) > 0.5
    assert float(with_context.b[0]) > float(no_context_blend.b[0])


def test_mem_atari_shapes_key_norm_summary_and_latent_behavior():
    n_games = 3
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
    net = MemAtariActorCritic(n_games=n_games, d_k=d_k, d_c=d_c, d_m=d_m, act_dim=act_dim)
    obs = jax.random.uniform(jax.random.PRNGKey(0), (2, 84, 84, 4), dtype=jnp.float32)
    game_id = jnp.asarray([0, 1], dtype=jnp.int32)
    h, k = net.apply({"params": params}, obs, method=MemAtariActorCritic.encode)
    c_embed = net.apply({"params": params}, game_id, method=MemAtariActorCritic.context)
    atom_feats = jnp.ones((2, 2, d_k + d_c + act_dim + 1 + 5), dtype=jnp.float32)
    alpha = jnp.asarray([[0.25, 0.75], [1.0, 0.0]], dtype=jnp.float32)
    m = net.apply({"params": params}, atom_feats, alpha, method=MemAtariActorCritic.mem_summary)
    logits, value = net.apply({"params": params}, h, m, c_embed, method=MemAtariActorCritic.policy_value)
    latent_logits, latent_value = net.apply(
        {"params": params}, k, c_embed, m, method=MemAtariActorCritic.latent_behavior
    )

    np.testing.assert_allclose(np.asarray(jnp.linalg.norm(k, axis=-1)), np.ones((2,)), atol=1e-3)
    assert logits.shape == (2, act_dim)
    assert value.shape == (2,)
    assert latent_logits.shape == (2, act_dim)
    assert latent_value.shape == (2,)
    assert m.shape == (2, d_m)


def test_memory_empty_deployed_policy_matches_base_distribution():
    n_games = 2
    d_k = 8
    d_c = 4
    d_m = 6
    act_dim = 5
    capacity = 3
    top_k = 2
    params = mem_init(
        jax.random.PRNGKey(1),
        n_games,
        capacity,
        d_k=d_k,
        d_c=d_c,
        d_m=d_m,
        act_dim=act_dim,
        top_k=top_k,
    )
    bank = {
        "keys": jnp.zeros((capacity, d_k), dtype=jnp.float32),
        "context": jnp.zeros((capacity, d_c), dtype=jnp.float32),
        "teacher_policy": jnp.zeros((capacity, act_dim), dtype=jnp.float32),
        "teacher_value": jnp.zeros((capacity,), dtype=jnp.float32),
        "importance": jnp.zeros((capacity,), dtype=jnp.float32),
        "game_id": jnp.zeros((capacity,), dtype=jnp.int32),
        "source5": jnp.zeros((capacity, 5), dtype=jnp.float32),
        "age": jnp.zeros((capacity,), dtype=jnp.float32),
        "valid": jnp.zeros((capacity,), dtype=bool),
    }
    hp = _hp(top_k=top_k)
    obs = jnp.zeros((2, 84, 84, 4), dtype=jnp.float32)
    game_id = jnp.asarray([0, 1], dtype=jnp.int32)
    out = mem_apply(params, obs, game_id, bank, hp)

    net = MemAtariActorCritic(n_games=n_games, d_k=d_k, d_c=d_c, d_m=d_m, act_dim=act_dim)
    h, _ = net.apply({"params": params}, obs, method=MemAtariActorCritic.encode)
    c_embed = net.apply({"params": params}, game_id, method=MemAtariActorCritic.context)
    m0 = jnp.zeros((2, d_m), dtype=jnp.float32)
    base_logits, base_v = net.apply(
        {"params": params}, h, m0, c_embed, method=MemAtariActorCritic.policy_value
    )

    np.testing.assert_allclose(np.asarray(out["b"]), np.zeros((2,)), atol=0.0)
    np.testing.assert_allclose(np.asarray(out["m"]), np.zeros((2, d_m)), atol=1e-7)
    np.testing.assert_allclose(
        np.asarray(jax.nn.softmax(out["logits_final"], axis=-1)),
        np.asarray(jax.nn.softmax(base_logits, axis=-1)),
        atol=1e-6,
    )
    np.testing.assert_allclose(np.asarray(out["logits_net"]), np.asarray(base_logits), atol=1e-6)
    np.testing.assert_allclose(np.asarray(out["v_final"]), np.asarray(base_v), atol=1e-6)


def test_expand_source_flags_and_ema_update():
    flags = jnp.asarray([0, 1, 3, 16, 31], dtype=jnp.int32)
    expanded = reader.expand_source_flags(flags)
    expected = np.asarray(
        [
            [0, 0, 0, 0, 0],
            [1, 0, 0, 0, 0],
            [1, 1, 0, 0, 0],
            [0, 0, 0, 0, 1],
            [1, 1, 1, 1, 1],
        ],
        dtype=np.float32,
    )
    np.testing.assert_allclose(np.asarray(expanded), expected, atol=0.0)

    target = {"x": jnp.asarray([1.0, 2.0])}
    online = {"x": jnp.asarray([3.0, 4.0])}
    np.testing.assert_allclose(np.asarray(reader.ema_update(target, online, 0.0)["x"]), [1.0, 2.0])
    np.testing.assert_allclose(np.asarray(reader.ema_update(target, online, 1.0)["x"]), [3.0, 4.0])
