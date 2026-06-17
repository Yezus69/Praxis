from types import SimpleNamespace

import numpy as np
import pytest
import jax
import jax.numpy as jnp

from pmac.agents import continual_living_memory as clm
from pmac.agents.ppo_living_memory_fast import (
    FastLMConfig,
    combine_latent_guard_grads,
)
from pmac.evaluation import normalized_retention
from pmac.stability import zeros_omega_like
from pmac.tree_utils import tree_dot, tree_norm


def _tree_all_zero(tree, atol=1e-6):
    return all(np.allclose(np.asarray(leaf), 0.0, atol=atol) for leaf in jax.tree_util.tree_leaves(tree))


def _cfg():
    return FastLMConfig(
        total_timesteps=1,
        num_envs=1,
        num_steps=1,
        num_minibatches=1,
        update_epochs=1,
        hot_capacity=4,
        top_k=1,
        d_k=2,
        d_c=2,
        d_m=3,
        guard_sample_atoms=4,
    )


def _driver_cfg():
    return SimpleNamespace(
        total_timesteps=1,
        n_blocks=1,
        review_steps_frac=0.0,
        audit_every_blocks=1,
        gate_r_min=0.0,
        gate_delta_frac=1.0,
        hot_capacity=4,
        top_k=1,
        d_k=2,
        d_c=2,
        d_m=3,
        guard_sample_atoms=4,
        visual_sentinels_per_game=64,
    )


def _policy(n, act_dim=3):
    out = np.zeros((n, act_dim), dtype=np.float32)
    out[:, 0] = 1.0
    return out


def _protected(game, game_id, keys, importance, *, act_dim=3):
    keys = np.asarray(keys, dtype=np.float32)
    n = int(keys.shape[0])
    return {
        "game": str(game),
        "keys": keys,
        "context": np.full((n, 2), float(game_id), dtype=np.float32),
        "teacher_policy": _policy(n, act_dim),
        "teacher_value": np.linspace(0.0, 1.0, n, dtype=np.float32),
        "importance": np.asarray(importance, dtype=np.float32),
        "game_id": np.full((n,), int(game_id), dtype=np.int32),
        "source5": np.zeros((n, 5), dtype=np.float32),
        "age": np.zeros((n,), dtype=np.float32),
        "valid": np.ones((n,), dtype=bool),
    }


def test_latent_guard_combine_projects_scales_and_skips_nonfinite():
    params = {"w": jnp.asarray([1.0, 1.0], dtype=jnp.float32)}
    omega = zeros_omega_like(params)
    g_task = {"w": jnp.asarray([-1.0, 0.0], dtype=jnp.float32)}
    g_guard = {"w": jnp.asarray([1.0, 0.0], dtype=jnp.float32)}

    projected, next_omega, metrics = combine_latent_guard_grads(
        params,
        g_task,
        g_guard,
        omega,
        lambda_total=0.0,
        kappa=1.0,
        stability_alpha=0.0,
        rho_omega=0.0,
        project=True,
    )

    assert float(tree_dot(projected, g_guard)) >= -1e-6
    assert float(metrics.projection_ratio) < 1.0
    assert float(tree_norm(next_omega)) > 0.0

    plain, _, _ = combine_latent_guard_grads(
        params,
        {"w": jnp.asarray([2.0, -2.0], dtype=jnp.float32)},
        g_guard,
        omega,
        lambda_total=0.0,
        stability_alpha=0.0,
        project=False,
    )
    stable, _, _ = combine_latent_guard_grads(
        params,
        {"w": jnp.asarray([2.0, -2.0], dtype=jnp.float32)},
        g_guard,
        {"w": jnp.asarray([9.0, 9.0], dtype=jnp.float32)},
        lambda_total=0.0,
        stability_alpha=1.0,
        project=False,
    )
    assert float(tree_norm(stable)) < float(tree_norm(plain))

    nonfinite, omega_after_nan, nan_metrics = combine_latent_guard_grads(
        params,
        g_task,
        {"w": jnp.asarray([jnp.nan, 0.0], dtype=jnp.float32)},
        omega,
        project=True,
    )
    assert bool(np.asarray(nan_metrics.nonfinite))
    assert _tree_all_zero(nonfinite)
    assert _tree_all_zero(omega_after_nan)


def test_make_guard_shapes_lambda_allocation_and_padding():
    cfg = _cfg()
    protected = [
        _protected("A", 0, [[1.0, 0.0], [0.0, 1.0]], [2.0, 1.0]),
        _protected("B", 1, [[1.0, 1.0]], [3.0]),
    ]

    guard = clm.make_guard(
        protected,
        {"A": 1.0, "B": 1.0},
        cfg,
        project=False,
        sample_atoms=4,
    )

    assert guard["project"] is False
    assert guard["atoms"]["keys"].shape == (4, 2)
    assert guard["atoms"]["teacher_policy"].shape == (4, 3)
    assert guard["bank"]["keys"].shape == (4, 2)
    assert sum(guard["lambda_by_game"].values()) == pytest.approx(cfg.guard_lambda_total, abs=1e-6)
    assert guard["lambda_by_game"]["A"] == pytest.approx(guard["lambda_by_game"]["B"])
    np.testing.assert_allclose(np.asarray(guard["atoms"]["weight"]), [1.0, 1.0, 1.0, 0.0])


def _install_driver_stubs(monkeypatch):
    calls = {"guards": [], "blends": []}
    trained_so_far = set()
    scores = {
        "random": 1.0,
        "A_current": 10.0,
        "A_forgotten": 4.0,
        "B_current": 9.0,
    }

    def fake_mem_init(*_args, **_kwargs):
        return {"kind": "random"}

    def fake_bank(protected_sets, capacity, d_k, d_c, act_dim):
        return {
            "keys": jnp.zeros((int(capacity), int(d_k)), dtype=jnp.float32),
            "context": jnp.zeros((int(capacity), int(d_c)), dtype=jnp.float32),
            "teacher_policy": jnp.zeros((int(capacity), int(act_dim)), dtype=jnp.float32),
            "teacher_value": jnp.zeros((int(capacity),), dtype=jnp.float32),
            "importance": jnp.zeros((int(capacity),), dtype=jnp.float32),
            "game_id": jnp.zeros((int(capacity),), dtype=jnp.int32),
            "source5": jnp.zeros((int(capacity), 5), dtype=jnp.float32),
            "age": jnp.zeros((int(capacity),), dtype=jnp.float32),
            "valid": jnp.zeros((int(capacity),), dtype=bool),
            "n_sets": len(protected_sets),
        }

    def fake_train(
        game,
        game_id,
        n_games,
        cfg,
        seed,
        *,
        init_params=None,
        hot_bank=None,
        ema_params=None,
        value_norm=None,
        guard=None,
        aux=None,
    ):
        del game_id, n_games, cfg, seed, init_params, hot_bank, ema_params, value_norm, aux
        trained_so_far.add(str(game))
        calls["guards"].append(guard)
        conserves_prior = guard is not None
        return {
            "params": {
                "kind": "trained",
                "game": str(game),
                "trained_so_far": frozenset(trained_so_far),
                "conserves_prior": conserves_prior,
            },
            "ema_params": {"kind": "ema", "game": str(game)},
            "value_norm": {"mu": 0.0, "sigma": 1.0},
            "final_return": 0.0,
        }

    def fake_certify(params, ema_params, value_norm, game, game_id, *, cfg, seed):
        del params, ema_params, value_norm, cfg, seed
        return _protected(str(game), int(game_id), [[1.0, 0.0]], [1.0])

    def fake_collect(params, ema_params, value_norm, game, game_id, *, cfg, seed, n=64):
        del params, ema_params, value_norm, game, cfg, seed, n
        return {
            "obs": np.zeros((1, 4, 84, 84), dtype=np.uint8),
            "game_id": np.asarray([int(game_id)], dtype=np.int32),
            "key_star": np.asarray([[1.0, 0.0]], dtype=np.float16),
            "teacher_policy": np.full((1, 18), 1.0 / 18.0, dtype=np.float16),
            "teacher_value": np.zeros((1,), dtype=np.float16),
        }

    def fake_eval(params, game, game_id, protected_bank, *, cfg, seed, episodes=12, blend=True):
        del game_id, protected_bank, cfg, seed, episodes
        calls["blends"].append(bool(blend))
        if params.get("kind") == "random":
            return scores["random"]
        game = str(game)
        trained = params.get("trained_so_far", frozenset())
        if game == "A":
            if "B" not in trained or params.get("conserves_prior", False):
                return scores["A_current"]
            return scores["A_forgotten"]
        return scores["B_current"]

    monkeypatch.setattr(clm, "mem_init", fake_mem_init)
    monkeypatch.setattr(clm, "build_protected_bank", fake_bank)
    monkeypatch.setattr(clm, "train_living_memory_fast", fake_train)
    monkeypatch.setattr(clm, "certify_protected_memories", fake_certify)
    monkeypatch.setattr(clm, "collect_visual_sentinels", fake_collect)
    monkeypatch.setattr(clm, "eval_living_memory", fake_eval)
    monkeypatch.setattr(clm, "_audit_violation_rate", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(clm, "_audit_retrieval_alignment", lambda *_args, **_kwargs: 1.0)
    return calls


def _assert_retention_matches_scores(result):
    for game in result["games"]:
        assert result["retention"][game] == pytest.approx(
            normalized_retention(
                result["final_scores"][game],
                result["best_scores"][game],
                result["random_scores"][game],
            )
        )


def test_continual_driver_retention_and_ablation_routing(monkeypatch):
    calls = _install_driver_stubs(monkeypatch)
    full = clm.continual_living_memory(["A", "B"], 2, _driver_cfg(), 0, ablation="full")

    assert calls["guards"][0] is None
    assert calls["guards"][1] is not None
    assert calls["guards"][1]["project"] is True
    assert calls["blends"][:2] == [False, False]
    assert all(calls["blends"][2:])
    _assert_retention_matches_scores(full)

    calls = _install_driver_stubs(monkeypatch)
    no_conservation = clm.continual_living_memory(
        ["A", "B"], 2, _driver_cfg(), 0, ablation="no_conservation"
    )
    assert calls["guards"] == [None, None]
    assert all(calls["blends"][2:])
    _assert_retention_matches_scores(no_conservation)
    assert no_conservation["retention"]["A"] < full["retention"]["A"]

    calls = _install_driver_stubs(monkeypatch)
    no_memory_read = clm.continual_living_memory(["A", "B"], 2, _driver_cfg(), 0, ablation="no_memory_read")
    assert calls["guards"][1] is not None
    assert not any(calls["blends"])
    _assert_retention_matches_scores(no_memory_read)

    calls = _install_driver_stubs(monkeypatch)
    no_projection = clm.continual_living_memory(["A", "B"], 2, _driver_cfg(), 0, ablation="no_projection")
    assert calls["guards"][1]["project"] is False
    _assert_retention_matches_scores(no_projection)

    calls = _install_driver_stubs(monkeypatch)
    plain_ppo = clm.continual_living_memory(["A", "B"], 2, _driver_cfg(), 0, ablation="plain_ppo")
    assert calls["guards"] == [None, None]
    assert calls["blends"] == [False, False, False, False, False]
    _assert_retention_matches_scores(plain_ppo)
