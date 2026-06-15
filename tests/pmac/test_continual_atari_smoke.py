import numpy as np
import pytest

pytest.importorskip("envpool")

from pmac.experiments.continual_atari import (  # noqa: E402
    ContinualAtariConfig,
    run_atari_baseline,
    run_atari_pmac,
)


def _tiny_cfg():
    return ContinualAtariConfig(
        per_game_steps=4,
        num_envs=1,
        num_steps=4,
        update_epochs=1,
        num_minibatches=1,
        eval_episodes=1,
        eval_max_steps_per_episode=64,
        anchor_buffer_per_game=2,
        guard_batch=2,
        guard_coef=1.0,
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

    for result in (baseline, no_cons, pmac):
        for key in (
            "mean_norm_retention",
            "worst_norm_retention",
            "norm_forgetting",
            "mean_final_return",
        ):
            assert key in result["metrics"]
        assert np.all(np.isfinite(result["return_matrix"]))
