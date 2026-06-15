from pmac.growth import GrowthController, should_grow


def test_should_grow_false_before_patience():
    assert not should_grow([0.01, 0.01], patience=3, min_ratio=0.1)


def test_should_grow_true_when_recent_mean_below_min_ratio():
    assert should_grow([0.9, 0.05, 0.04, 0.03], patience=3, min_ratio=0.1)
    assert not should_grow([0.01, 0.2, 0.2], patience=3, min_ratio=0.1)


def test_growth_controller_reset_clears_history_and_counts_growth():
    controller = GrowthController(patience=2, min_ratio=0.1)
    controller.observe(0.01)
    controller.observe(0.02)

    assert controller.should_grow()
    controller.reset()
    assert controller.state.history == []
    assert controller.state.grown == 1
    assert not controller.should_grow()
