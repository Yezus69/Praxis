from types import SimpleNamespace

import numpy as np
import pytest

from pmac.agents import continual_living_memory as clm
from pmac.memory import SourceFlag


def _cfg(**overrides):
    base = {
        "total_timesteps": 8,
        "n_blocks": 2,
        "review_steps_frac": 0.0,
        "audit_every_blocks": 1,
        "gate_r_min": 0.9,
        "gate_delta_frac": 0.1,
        "lambda_review": 0.5,
        "hot_capacity": 4,
        "d_k": 2,
        "d_c": 2,
        "d_m": 3,
        "act_dim": 3,
        "top_k": 1,
        "guard_sample_atoms": 2,
        "visual_sentinels_per_game": 1,
        "visual_sentinel_batch": 1,
        "retr_n_neg": 1,
    }
    base.update(overrides)
    return SimpleNamespace(**base)


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
        "importance": np.asarray([1.0], dtype=np.float32),
        "game_id": np.asarray([int(game_id)], dtype=np.int32),
        "source_flags": np.asarray([int(SourceFlag.SENTINEL | SourceFlag.HIGH_RETURN)], dtype=np.int32),
        "age": np.zeros((1,), dtype=np.float32),
        "valid": np.ones((1,), dtype=bool),
    }


def _install_stubs(monkeypatch, *, a_scores):
    calls = {"train": [], "certify_params": [], "sample_u": []}
    scores = {
        "A": list(a_scores),
        "B": [5.0, 6.0, 8.0, 8.0],
    }

    def fake_mem_init(*_args, **_kwargs):
        return {"step": np.asarray(-1, dtype=np.int32), "random": np.asarray(True)}

    def fake_bank(protected_sets, capacity, d_k, d_c, act_dim):
        return {
            "keys": np.zeros((int(capacity), int(d_k)), dtype=np.float32),
            "context": np.zeros((int(capacity), int(d_c)), dtype=np.float32),
            "teacher_policy": np.zeros((int(capacity), int(act_dim)), dtype=np.float32),
            "teacher_value": np.zeros((int(capacity),), dtype=np.float32),
            "importance": np.zeros((int(capacity),), dtype=np.float32),
            "game_id": np.zeros((int(capacity),), dtype=np.int32),
            "source5": np.zeros((int(capacity), 5), dtype=np.float32),
            "age": np.zeros((int(capacity),), dtype=np.float32),
            "valid": np.zeros((int(capacity),), dtype=bool),
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
        del n_games, seed, ema_params, value_norm, guard, aux
        prev = 0 if init_params is None else int(np.asarray(init_params["step"]))
        step = prev + 1
        calls["train"].append(
            {
                "game": str(game),
                "game_id": int(game_id),
                "steps": int(cfg.total_timesteps),
                "init_step": None if init_params is None else prev,
                "hot_bank": hot_bank,
            }
        )
        return {
            "params": {
                "step": np.asarray(step, dtype=np.int32),
                "last_game_id": np.asarray(int(game_id), dtype=np.int32),
            },
            "ema_params": {"step": np.asarray(step, dtype=np.int32)},
            "hot_bank": {"version": np.asarray(step, dtype=np.int32)},
            "value_norm": {"version": np.asarray(step, dtype=np.int32)},
            "final_return": float(step),
        }

    def fake_certify(params, ema_params, value_norm, game, game_id, *, cfg, seed):
        del ema_params, value_norm, cfg, seed
        calls["certify_params"].append(
            {"game": str(game), "step": int(np.asarray(params["step"]))}
        )
        return _protected(str(game), int(game_id))

    def fake_collect(params, ema_params, value_norm, game, game_id, *, cfg, seed, n=64):
        del params, ema_params, value_norm, game, cfg, seed, n
        return {
            "obs": np.zeros((1, 4, 84, 84), dtype=np.uint8),
            "game_id": np.asarray([int(game_id)], dtype=np.int32),
            "key_star": np.asarray([[1.0, 0.0]], dtype=np.float16),
            "teacher_policy": np.asarray([[1.0, 0.0, 0.0]], dtype=np.float16),
            "teacher_value": np.zeros((1,), dtype=np.float16),
        }

    def fake_eval(params, game, game_id, protected_bank, *, cfg, seed, episodes=12, blend=True):
        del game_id, protected_bank, cfg, seed, episodes, blend
        if bool(np.asarray(params.get("random", False))):
            return 0.0
        values = scores[str(game)]
        if len(values) > 1:
            return float(values.pop(0))
        return float(values[0])

    def fake_sample(u, n, rng):
        del rng
        calls["sample_u"].append(dict(u))
        return ["A"][: int(n)]

    monkeypatch.setattr(clm, "mem_init", fake_mem_init)
    monkeypatch.setattr(clm, "build_protected_bank", fake_bank)
    monkeypatch.setattr(clm, "train_living_memory_fast", fake_train)
    monkeypatch.setattr(clm, "certify_protected_memories", fake_certify)
    monkeypatch.setattr(clm, "collect_visual_sentinels", fake_collect)
    monkeypatch.setattr(clm, "eval_living_memory", fake_eval)
    monkeypatch.setattr(clm, "_audit_violation_rate", lambda *args, **kwargs: 0.0)
    monkeypatch.setattr(clm, "_audit_retrieval_alignment", lambda *args, **kwargs: float("inf"))
    monkeypatch.setattr(clm, "sample_review_games", fake_sample)
    return calls


def test_gate_reject_restores_snapshot_and_applies_reject_actions(monkeypatch):
    calls = _install_stubs(monkeypatch, a_scores=[10.0, 10.0, 5.0, 10.0])

    out = clm.continual_living_memory(["A", "B"], 2, _cfg(), 0, ablation="no_review")

    assert out["gate_rejections"] == 1
    assert out["gate_decisions"][1]["accept"] is False
    assert out["gate_decisions"][1]["regressed_games"] == ["A"]
    assert calls["certify_params"][-1] == {"game": "B", "step": 3}
    assert out["risk_scores"]["A"] > 0.0
    assert out["review_boosts"]["A"] > 0.0
    assert out["failure_memory_writes"] == 1


def test_gate_accept_keeps_candidate_state(monkeypatch):
    calls = _install_stubs(monkeypatch, a_scores=[10.0, 10.0, 10.0, 10.0])

    out = clm.continual_living_memory(["A", "B"], 2, _cfg(), 0, ablation="no_review")

    assert out["gate_rejections"] == 0
    assert [decision["accept"] for decision in out["gate_decisions"]] == [True, True]
    assert calls["certify_params"][-1] == {"game": "B", "step": 4}
    assert out["failure_memory_writes"] == 0


def test_review_sampling_and_ablation_routing(monkeypatch):
    calls = _install_stubs(monkeypatch, a_scores=[10.0, 10.0, 10.0])

    full = clm.continual_living_memory(
        ["A", "B"],
        2,
        _cfg(total_timesteps=10, n_blocks=1, review_steps_frac=0.5),
        0,
        ablation="full",
    )

    assert calls["sample_u"] == [{"A": 1.0}]
    assert [(call["game"], call["game_id"], call["steps"], call["init_step"]) for call in calls["train"]] == [
        ("A", 0, 10, None),
        ("B", 1, 10, 1),
        ("A", 0, 5, 2),
    ]
    assert calls["train"][0]["hot_bank"] is None
    assert int(np.asarray(calls["train"][1]["hot_bank"]["version"])) == 1
    assert int(np.asarray(calls["train"][2]["hot_bank"]["version"])) == 2
    assert full["review_counts"] == {"A": 1}
    assert len(full["gate_decisions"]) == 1

    calls = _install_stubs(monkeypatch, a_scores=[10.0, 10.0, 10.0])
    no_review = clm.continual_living_memory(
        ["A", "B"],
        2,
        _cfg(total_timesteps=10, n_blocks=1, review_steps_frac=0.5),
        0,
        ablation="no_review",
    )
    assert calls["sample_u"] == []
    assert [call["game"] for call in calls["train"]] == ["A", "B"]
    assert no_review["review_counts"] == {}

    calls = _install_stubs(monkeypatch, a_scores=[10.0, 10.0, 10.0])
    no_gate = clm.continual_living_memory(
        ["A", "B"],
        2,
        _cfg(total_timesteps=10, n_blocks=1, review_steps_frac=0.5),
        0,
        ablation="no_gate",
    )
    assert calls["sample_u"] == [{"A": 1.0}]
    assert no_gate["review_counts"] == {"A": 1}
    assert no_gate["gate_decisions"] == []
