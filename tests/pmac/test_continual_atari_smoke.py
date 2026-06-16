import numpy as np
import pytest

pytest.importorskip("envpool")

from pmac.experiments.continual_atari import (  # noqa: E402
    AtariAnchorBuffer,
    ContinualAtariConfig,
    _guard_from_buffers,
    run_atari_baseline,
    run_atari_pmac,
)


def _dummy_buffer(game_id: int, n_games: int, n: int = 2) -> AtariAnchorBuffer:
    onehot = np.eye(n_games, dtype=np.float32)[game_id]
    return AtariAnchorBuffer(
        obs_uint8=np.zeros((n, 4, 84, 84), dtype=np.uint8),
        game_onehot=np.broadcast_to(onehot, (n, n_games)).copy(),
        teacher_logits=np.zeros((n, 18), dtype=np.float32),
        teacher_value=np.zeros((n,), dtype=np.float32),
    )


def test_guard_from_buffers_length_normalizes_by_prior_count():
    buffers = [_dummy_buffer(i, 3) for i in range(3)]
    cfg = ContinualAtariConfig(guard_coef=3.0, guard_norm="length", guard_batch=6)
    guard = _guard_from_buffers(buffers, cfg, ablation=None)

    assert guard is not None
    assert guard["guard_coef"] == 1.0
    np.testing.assert_array_equal(
        guard["prior_offsets"],
        np.asarray([0, 2, 4, 6], dtype=np.int32),
    )

    cfg_none = ContinualAtariConfig(guard_coef=3.0, guard_norm="none", guard_batch=6)
    unnormalized = _guard_from_buffers(buffers, cfg_none, ablation=None)
    no_replay = _guard_from_buffers(buffers, cfg, ablation="no_replay")
    assert unnormalized is not None
    assert no_replay is not None
    assert unnormalized["guard_coef"] == 3.0
    assert no_replay["guard_coef"] == 3.0


def _tiny_cfg():
    return ContinualAtariConfig(
        per_game_steps=4,
        num_envs=1,
        num_steps=4,
        update_epochs=1,
        num_minibatches=1,
        eval_episodes=1,
        eval_envs=1,
        eval_max_steps_per_episode=64,
        eval_steps_cap=64,
        anchor_buffer_per_game=2,
        guard_batch=2,
        guard_coef=1.0,
        guard_norm="length",
        anneal_lr=False,
    )


def test_continual_atari_tiny_runs_baseline_no_cons_and_pmac():
    games = ["Pong-v5", "Breakout-v5"]
    cfg = _tiny_cfg()

    baseline = run_atari_baseline(games, cfg, seed=0)
    no_cons = run_atari_pmac(games, cfg, seed=0, ablation="no_conservation")
    pmac = run_atari_pmac(games, cfg, seed=0)

    assert baseline["return_matrix"].shape == (2, 2)
    assert no_cons["return_matrix"].shape == (2, 2)
    assert pmac["return_matrix"].shape == (2, 2)
    np.testing.assert_allclose(baseline["return_matrix"], no_cons["return_matrix"])
    np.testing.assert_allclose(baseline["random_scores"], no_cons["random_scores"])
    assert no_cons["extra"]["guard_enabled"] is False
    assert no_cons["extra"]["guard_source"] == "none"
    assert pmac["extra"]["guard_enabled"] is True
    assert pmac["extra"]["guard_norm"] == "length"
    assert pmac["extra"]["guard_effective_coefs"] == [0.0, 1.0]

    for result in (baseline, no_cons, pmac):
        for key in (
            "mean_norm_retention",
            "worst_norm_retention",
            "norm_forgetting",
            "mean_final_return",
        ):
            assert key in result["metrics"]
        assert np.all(np.isfinite(result["return_matrix"]))
