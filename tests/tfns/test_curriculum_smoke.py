from __future__ import annotations

import json
from types import SimpleNamespace

import numpy as np
import pytest

from tfns.train import curriculum


pytestmark = pytest.mark.slow


class _DummyHandle:
    def close(self) -> None:
        pass


class _DummyEnvStep:
    def __init__(self, num_envs: int):
        self.obs = np.zeros((int(num_envs), 84, 84, 4), dtype=np.uint8)
        self.current_obs = self.obs

    def get_obs(self):
        return self.obs

    def reset(self):
        return self.obs

    def __call__(self, action):
        extra = {"episode_returns": [float(np.mean(np.asarray(action)))]}
        n = int(self.obs.shape[0])
        return (
            self.obs,
            np.zeros((n,), dtype=np.float32),
            np.zeros((n,), dtype=np.bool_),
            np.zeros((n,), dtype=np.bool_),
            extra,
        )


class _DummyMemory:
    def __init__(self):
        self.count = 0

    def __len__(self):
        return int(self.count)

    def bytes_used(self):
        return int(self.count) * 128

    def clusters(self):
        return {idx: [idx] for idx in range(int(self.count))}


class _DummyState:
    def __init__(self):
        self.params = {"w": np.array([0.0], dtype=np.float32)}
        self.bases = {"policy_head": np.zeros((8, 2), dtype=np.float32)}
        self.memory = _DummyMemory()
        self.protected_clusters = []
        self.adapter_dormant = np.array([False, True, True], dtype=np.bool_)
        self.robust_stats = {}
        self.rollout_carry = None
        self.block_index = 0


def test_smoke_curriculum_driver_persists_results_and_progress(tmp_path, monkeypatch):
    games = ["SpaceInvaders-v5", "Breakout-v5"]
    consolidation_calls = []

    monkeypatch.setattr(
        curriculum.atari_env,
        "make_atari_env_step",
        lambda game, num_envs, seed, training=True: (_DummyEnvStep(num_envs), _DummyHandle()),
    )
    monkeypatch.setattr(
        curriculum,
        "_init_agent_state",
        lambda cfg, game, seed: (object(), object(), _DummyState()),
    )
    monkeypatch.setattr(
        curriculum.evaluate,
        "random_score",
        lambda game, *, num_envs, n_episodes, seed, max_steps=None: {
            "mean": 0.0,
            "sem": 0.0,
            "n": int(n_episodes),
            "returns": [0.0] * int(n_episodes),
            "capped": False,
        },
    )

    def fake_run_blocks(state, agent, tx, env_step, cfg, n_blocks, **kwargs):
        state.block_index += 1
        state.memory.count += 1
        env_step.block_returns.append(10.0 + state.block_index)
        env_step.recent_returns.append(10.0 + state.block_index)
        return state, [
            {
                "block_index": state.block_index,
                "accept_count": 1,
                "reject_count": 0,
                "memory_count": state.memory.count,
                "memory_bytes": state.memory.bytes_used(),
                "memory_clusters": len(state.memory.clusters()),
            }
        ]

    monkeypatch.setattr(curriculum.loop, "run_blocks", fake_run_blocks)

    def fake_eval_game(
        agent,
        params,
        game,
        *,
        num_envs,
        n_episodes,
        seed,
        greedy=False,
        adapter_dormant=None,
        max_steps=None,
    ):
        score = 12.0 if game == games[0] else 9.0
        if greedy:
            score -= 1.0
        return {
            "mean": score,
            "sem": 0.0,
            "n": int(n_episodes),
            "returns": [score] * int(n_episodes),
            "capped": False,
        }

    monkeypatch.setattr(curriculum.evaluate, "evaluate_game", fake_eval_game)

    def fake_consolidate_skill(state, agent, tx, eval_fn, candidate_records, cfg, **kwargs):
        consolidation_calls.append(SimpleNamespace(score_windows=list(kwargs["score_windows"])))
        state.protected_clusters.append({"cluster_id": len(state.protected_clusters)})
        return state, True, {"reason": "accepted", "gate": {"accepted": True}}

    monkeypatch.setattr(curriculum.loop, "consolidate_skill", fake_consolidate_skill)

    result = curriculum.main(
        [
            "--mode",
            "curriculum",
            "--smoke",
            "--games",
            *games,
            "--num-envs",
            "4",
            "--rollout-len",
            "2",
            "--steps-per-game",
            "16",
            "--eval-episodes",
            "2",
            "--out-dir",
            str(tmp_path),
        ]
    )

    result_path = tmp_path / "curriculum_seed0.json"
    progress_path = tmp_path / "progress.jsonl"
    assert result_path.exists()
    assert progress_path.exists()

    persisted = json.loads(result_path.read_text(encoding="utf-8"))
    assert persisted["mode"] == "curriculum"
    assert len(persisted["retention_matrix"]) == 2
    assert [len(row["retention"]) for row in persisted["retention_matrix"]] == [1, 2]
    assert all(persisted["consolidation"][game]["ran"] for game in games)
    assert len(consolidation_calls) == 2

    progress_lines = progress_path.read_text(encoding="utf-8").strip().splitlines()
    assert len(progress_lines) == 4
    first_progress = json.loads(progress_lines[0])
    assert {
        "game",
        "steps_done",
        "recent_score",
        "env_SPS",
        "memory_count",
        "accept_count",
        "reject_count",
        "consolidation_status",
    } <= set(first_progress)
    assert result["final"]["retention"]["per_game_retention"]
