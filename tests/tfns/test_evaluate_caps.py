from __future__ import annotations

from types import SimpleNamespace

import jax.numpy as jnp
import numpy as np

from tfns.train import evaluate


class _DummyHandle:
    def close(self) -> None:
        pass


class _NeverDoneEnv:
    def __init__(self, num_envs: int, reward: np.ndarray):
        self.obs = np.zeros((int(num_envs), 84, 84, 4), dtype=np.uint8)
        self.reward = np.asarray(reward, dtype=np.float32)
        self.calls = 0

    def __call__(self, action):
        self.calls += 1
        n = int(self.obs.shape[0])
        reset = np.zeros((n,), dtype=np.bool_)
        return (
            self.obs,
            np.clip(self.reward, -1.0, 1.0).astype(np.float32),
            reset.copy(),
            reset,
            {"reward_raw": self.reward.copy(), "episode_returns": []},
        )


def test_evaluate_game_caps_to_invalid_without_partial_substitution(monkeypatch):
    # Spec section 19/26: a capped evaluation with no completed episodes must be
    # invalid and must NOT substitute the in-progress (partial) mean as a score.
    env = _NeverDoneEnv(2, np.array([1.0, 2.0], dtype=np.float32))

    monkeypatch.setattr(
        evaluate,
        "make_atari_env_step",
        lambda game, num_envs, seed, training=False: (env, _DummyHandle()),
    )
    monkeypatch.setattr(
        evaluate,
        "_policy_action",
        lambda agent, params, obs, prev_action, prev_reward, reset, hidden, rng, greedy, dormant: (
            jnp.zeros((int(obs.shape[0]),), dtype=jnp.int32),
            hidden,
        ),
    )
    agent = SimpleNamespace(
        adapter_config=SimpleNamespace(num_adapters=1),
        init_hidden=lambda num_envs, dtype: jnp.zeros((int(num_envs), 4), dtype=dtype),
    )

    result = evaluate.evaluate_game(
        agent,
        {},
        "Breakout-v5",
        num_envs=2,
        n_episodes=1,
        seed=0,
        max_steps=4,
    )

    assert env.calls == 2
    assert result["capped"] is True
    assert result["valid"] is False
    assert result["n"] == 0
    assert result["returns"] == []
    assert np.isnan(result["mean"])  # never the partial 3.0
    assert result["partial_mean"] == 3.0  # diagnostic only
    assert result["total_transitions"] == 4


def test_random_score_caps_to_invalid_without_partial_substitution(monkeypatch):
    env = _NeverDoneEnv(3, np.array([1.0, 2.0, 3.0], dtype=np.float32))

    monkeypatch.setattr(
        evaluate,
        "make_atari_env_step",
        lambda game, num_envs, seed, training=False: (env, _DummyHandle()),
    )

    result = evaluate.random_score(
        "Breakout-v5",
        num_envs=3,
        n_episodes=2,
        seed=0,
        max_steps=6,
    )

    assert env.calls == 2
    assert result["capped"] is True
    assert result["valid"] is False
    assert result["n"] == 0
    assert result["returns"] == []
    assert np.isnan(result["mean"])  # never the partial 4.0
    assert result["partial_mean"] == 4.0  # diagnostic only


def test_eval_result_valid_when_all_episodes_complete():
    # When enough episodes complete, the score is the true completed mean.
    result = evaluate._eval_result(
        [10.0, 20.0, 30.0],
        np.zeros((3,), dtype=np.float32),
        n_episodes=2,
        capped=False,
        total_transitions=128,
    )
    assert result["valid"] is True
    assert result["n"] == 2  # truncated to n_requested completed episodes
    assert result["mean"] == 15.0  # mean of first n_requested=2 completed
    assert result["total_transitions"] == 128


def test_closed_loop_eval_threads_max_steps(monkeypatch):
    seen: dict[str, list[int | None]] = {"eval": [], "random": []}

    def fake_evaluate_game(
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
        seen["eval"].append(max_steps)
        return {"mean": 1.0, "sem": 0.0, "n": 1, "returns": [1.0], "capped": True}

    def fake_random_score(game, *, num_envs, n_episodes, seed, max_steps=None):
        seen["random"].append(max_steps)
        return {"mean": 0.0, "sem": 0.0, "n": 1, "returns": [0.0], "capped": True}

    monkeypatch.setattr(evaluate, "evaluate_game", fake_evaluate_game)
    monkeypatch.setattr(evaluate, "random_score", fake_random_score)

    eval_fn = evaluate.make_closed_loop_eval_fn(
        {"game-a": {"game": "Breakout-v5", "S_best": 2.0, "S_single": 2.0}},
        object(),
        num_envs=2,
        n_episodes=3,
        seed=0,
        max_steps=123,
    )
    result = eval_fn({})

    assert seen == {"eval": [123], "random": [123]}
    assert result["game-a"]["capped"] is True
    assert result["game-a"]["random_capped"] is True
