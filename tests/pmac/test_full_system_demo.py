from pmac.experiments.full_system_demo import DemoConfig, run_demo


def test_full_system_demo_tiny_config_passes():
    cfg = DemoConfig(
        seed=11,
        n_train=192,
        n_test=96,
        anchor_count=32,
        sentinel_count=32,
        eval_count=64,
        train0_steps=140,
        train1_steps=110,
        consolidation_steps=18,
    )

    verdicts = run_demo(cfg)

    assert verdicts == {"A": True, "B": True, "C": True, "D": True}
