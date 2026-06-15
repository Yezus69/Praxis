import numpy as np

from pmac.envs.minatar_gymnax import GAMES, make_games
from pmac.experiments.continual_minatar import (
    ContinualRLConfig,
    run_minatar_baseline,
    run_minatar_pmac,
)


def _tiny_cfg():
    return ContinualRLConfig(
        per_game_steps=4,
        num_envs=1,
        num_steps=4,
        update_epochs=1,
        num_minibatches=1,
        eval_episodes=2,
        eval_horizon=8,
        anchor_buffer_per_game=4,
        guard_batch=2,
        guard_coef=1.0,
        anneal_lr=False,
    )


def test_minatar_continual_tiny_matched_and_pmac_finite():
    specs = make_games(GAMES[:2])
    cfg = _tiny_cfg()

    baseline = run_minatar_baseline(specs, cfg, seed=0)
    no_cons = run_minatar_pmac(specs, cfg, seed=0, ablation="no_conservation")
    pmac = run_minatar_pmac(specs, cfg, seed=0)

    assert baseline["return_matrix"].shape == (2, 2)
    assert no_cons["return_matrix"].shape == (2, 2)
    assert pmac["return_matrix"].shape == (2, 2)

    for result in (baseline, no_cons, pmac):
        for key in ("mean_final", "forgetting", "mean_retention", "worst_retention"):
            assert key in result["metrics"]

    np.testing.assert_allclose(
        no_cons["return_matrix"],
        baseline["return_matrix"],
        rtol=1e-6,
        atol=1e-6,
    )
    assert np.all(np.isfinite(pmac["return_matrix"]))
