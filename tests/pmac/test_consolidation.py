from types import SimpleNamespace

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from pmac.agents import continual_living_memory as clm
from pmac.agents.atari_mem_net import mem_init
from pmac.agents.consolidation import consolidate
from pmac.memory.losses import latent_conservation_loss
from pmac.memory.runtime import default_retrieval_hp


def _dims(n_games=2, d_k=8, d_c=4, d_m=6, act_dim=4):
    return {
        "n_games": n_games,
        "d_k": d_k,
        "d_c": d_c,
        "d_m": d_m,
        "act_dim": act_dim,
    }


def _norm(x):
    x = np.asarray(x, dtype=np.float32)
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + 1.0e-8)


def _params(seed=0):
    dims = _dims()
    return mem_init(jax.random.PRNGKey(seed), capacity=2, top_k=1, **dims)


def _protected_bank():
    dims = _dims()
    keys = _norm(
        [
            [1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
        ]
    )
    policy = np.zeros((2, dims["act_dim"]), dtype=np.float32)
    policy[:, 0] = 1.0
    return {
        "keys": jnp.asarray(keys),
        "context": jnp.zeros((2, dims["d_c"]), dtype=jnp.float32),
        "teacher_policy": jnp.asarray(policy),
        "teacher_value": jnp.asarray([5.0, 4.0], dtype=jnp.float32),
        "importance": jnp.ones((2,), dtype=jnp.float32),
        "game_id": jnp.asarray([0, 1], dtype=jnp.int32),
        "source5": jnp.zeros((2, 5), dtype=jnp.float32),
        "age": jnp.zeros((2,), dtype=jnp.float32),
        "valid": jnp.asarray([True, True]),
    }


def _atom_batch(bank):
    return {
        "keys": bank["keys"][:1],
        "game_id": bank["game_id"][:1],
        "teacher_policy": bank["teacher_policy"][:1],
        "teacher_value": bank["teacher_value"][:1],
        "eps": jnp.zeros((1,), dtype=jnp.float32),
        "weight": jnp.ones((1,), dtype=jnp.float32),
    }


def _cfg(**overrides):
    base = {
        "lr": 0.1,
        "consolidate_steps": 8,
        "consolidate_lr_frac": 0.1,
        "consolidate_tol": 0.0,
        "guard_sample_atoms": 1,
        "top_k": 1,
        "guard_lambda_v": 1.0,
        "lambda_visual": 0.0,
        "lambda_key": 1.0,
        "lambda_retr": 0.0,
        "visual_lambda_v": 1.0,
        "retr_tau": 0.1,
        "visual_sentinel_batch": 1,
        "retr_n_neg": 1,
        "lambda_distill": 0.0,
        "eps_policy": 0.0,
        "tau_key": 0.1,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


def _tree_changed(a, b, atol=1.0e-8):
    return any(
        not np.allclose(np.asarray(x), np.asarray(y), atol=atol)
        for x, y in zip(jax.tree_util.tree_leaves(a), jax.tree_util.tree_leaves(b))
    )


def test_consolidate_reduces_latent_conservation_loss_on_protected_atoms():
    params = _params(0)
    bank = _protected_bank()
    cfg = _cfg()
    hp = default_retrieval_hp(1)
    batch = _atom_batch(bank)
    pre = latent_conservation_loss(params, batch, bank, hp, dims=_dims())

    result = consolidate(
        params,
        params,
        bank,
        None,
        cfg=cfg,
        n_steps=cfg.consolidate_steps,
        slow_lr=cfg.lr * cfg.consolidate_lr_frac,
        sentinel_eval_fn=lambda _p: 1.0,
    )

    assert result["accepted"] is True
    post = latent_conservation_loss(result["params"], batch, bank, hp, dims=_dims())
    assert float(post) < float(pre)
    assert result["loss_terms"]["L_cons"] > 0.0


def test_consolidate_acceptance_gate_returns_original_or_candidate_params():
    params = _params(1)
    bank = _protected_bank()
    cfg = _cfg(consolidate_steps=2)

    regressing_scores = iter([1.0, 0.0])
    rejected = consolidate(
        params,
        params,
        bank,
        None,
        cfg=cfg,
        n_steps=cfg.consolidate_steps,
        slow_lr=cfg.lr * cfg.consolidate_lr_frac,
        sentinel_eval_fn=lambda _p: next(regressing_scores),
    )
    assert rejected["accepted"] is False
    assert rejected["params"] is params
    assert rejected["ema_params"] is params

    accepted = consolidate(
        params,
        params,
        bank,
        None,
        cfg=cfg,
        n_steps=cfg.consolidate_steps,
        slow_lr=cfg.lr * cfg.consolidate_lr_frac,
        sentinel_eval_fn=lambda _p: 1.0,
    )
    assert accepted["accepted"] is True
    assert accepted["params"] is not params
    assert _tree_changed(accepted["params"], params)


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
        act_dim=3,
        guard_sample_atoms=1,
        visual_sentinels_per_game=1,
        visual_sentinel_batch=1,
        retr_n_neg=1,
        lr=0.1,
        consolidate_steps=1,
        consolidate_lr_frac=0.1,
        consolidate_tol=0.0,
        consolidate_every_games=1,
        lambda_distill=0.0,
    )


def _policy(n, act_dim=3):
    out = np.zeros((n, act_dim), dtype=np.float32)
    out[:, 0] = 1.0
    return out


def _protected(game, game_id):
    return {
        "game": str(game),
        "keys": np.asarray([[1.0, 0.0]], dtype=np.float32),
        "context": np.full((1, 2), float(game_id), dtype=np.float32),
        "teacher_policy": _policy(1),
        "teacher_value": np.asarray([1.0], dtype=np.float32),
        "importance": np.ones((1,), dtype=np.float32),
        "game_id": np.asarray([int(game_id)], dtype=np.int32),
        "source5": np.zeros((1, 5), dtype=np.float32),
        "age": np.zeros((1,), dtype=np.float32),
        "valid": np.ones((1,), dtype=bool),
    }


def _install_driver_stubs(monkeypatch, *, accepted=False):
    calls = {"consolidate": [], "eval_params": []}

    def fake_mem_init(*_args, **_kwargs):
        return {"tag": "random"}

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
        active_mask=None,
    ):
        del n_games, cfg, seed, init_params, hot_bank, ema_params, value_norm, guard, aux, active_mask
        return {
            "params": {"tag": f"trained-{game}", "game_id": int(game_id)},
            "ema_params": {"tag": f"ema-{game}", "game_id": int(game_id)},
            "value_norm": {"mu": 0.0, "sigma": 1.0},
            "final_return": 0.0,
            "r_plastic": 1.0,
        }

    def fake_certify(params, ema_params, value_norm, game, game_id, *, cfg, seed):
        del params, ema_params, value_norm, cfg, seed
        return _protected(game, game_id)

    def fake_collect(params, ema_params, value_norm, game, game_id, *, cfg, seed, n=64):
        del params, ema_params, value_norm, game, cfg, seed, n
        return {
            "obs": np.zeros((1, 4, 84, 84), dtype=np.uint8),
            "game_id": np.asarray([int(game_id)], dtype=np.int32),
            "key_star": np.asarray([[1.0, 0.0]], dtype=np.float16),
            "teacher_policy": _policy(1).astype(np.float16),
            "teacher_value": np.zeros((1,), dtype=np.float16),
        }

    def fake_eval(params, game, game_id, protected_bank, *, cfg, seed, episodes=12, blend=True, active_mask=None):
        del game, game_id, protected_bank, cfg, seed, episodes, blend, active_mask
        calls["eval_params"].append(params)
        if params.get("tag") == "random":
            return 0.0
        return 10.0

    def fake_consolidate(params, ema_params, protected_bank, visual_store, **kwargs):
        score = kwargs["sentinel_eval_fn"](params)
        calls["consolidate"].append(
            {
                "params": params,
                "protected_bank": protected_bank,
                "visual_store_len": len(visual_store),
                "sentinel_score": score,
                "slow_lr": kwargs["slow_lr"],
                "n_steps": kwargs["n_steps"],
            }
        )
        return {
            "params": {"tag": "bad-consolidated"},
            "ema_params": {"tag": "bad-ema"},
            "accepted": bool(accepted),
            "pre_score": score,
            "post_score": score if accepted else score - 1.0,
            "slow_lr": kwargs["slow_lr"],
            "steps": kwargs["n_steps"],
            "adapter_distill_active": False,
            "reason": "accepted" if accepted else "sentinel_regression",
        }

    monkeypatch.setattr(clm, "mem_init", fake_mem_init)
    monkeypatch.setattr(clm, "build_protected_bank", fake_bank)
    monkeypatch.setattr(clm, "train_living_memory_fast", fake_train)
    monkeypatch.setattr(clm, "certify_protected_memories", fake_certify)
    monkeypatch.setattr(clm, "collect_visual_sentinels", fake_collect)
    monkeypatch.setattr(clm, "eval_living_memory", fake_eval)
    monkeypatch.setattr(clm, "consolidate", fake_consolidate)
    monkeypatch.setattr(clm, "_audit_violation_rate", lambda *_args, **_kwargs: 0.0)
    monkeypatch.setattr(clm, "_audit_retrieval_alignment", lambda *_args, **_kwargs: 1.0)
    return calls


def test_driver_wires_consolidation_and_no_consolidation_ablation(monkeypatch):
    cfg = _driver_cfg()
    calls = _install_driver_stubs(monkeypatch, accepted=False)

    rejected = clm.continual_living_memory(["A", "B"], 2, cfg, 0, ablation="no_gate")

    assert len(calls["consolidate"]) == 1
    call = calls["consolidate"][0]
    assert call["protected_bank"]["n_sets"] == 2
    assert call["visual_store_len"] == 2
    assert call["sentinel_score"] == pytest.approx(10.0)
    assert call["slow_lr"] == pytest.approx(cfg.lr * cfg.consolidate_lr_frac)
    assert call["n_steps"] == cfg.consolidate_steps
    assert rejected["consolidation_rejections"] == 1
    assert all(params.get("tag") != "bad-consolidated" for params in calls["eval_params"])

    calls = _install_driver_stubs(monkeypatch, accepted=False)
    skipped = clm.continual_living_memory(["A", "B"], 2, cfg, 0, ablation="no_consolidation")

    assert calls["consolidate"] == []
    assert skipped["consolidation_decisions"] == []
