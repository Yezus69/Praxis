from praxis.train_csn import resolve_config


def test_8_1_long_run_preset_enables_sentinel():
    cfg = resolve_config(["--long-run"])
    assert cfg.enable_sentinel is True


def test_8_2_long_run_preset_sets_strong_guard():
    cfg = resolve_config(["--long-run"])
    assert cfg.guard_lambda_mem >= 8.0
    assert cfg.guard_kl_budget <= 0.003


def test_8_3_explicit_user_flags_override_preset():
    cfg = resolve_config(["--long-run", "--guard-lambda-mem", "4"])
    assert cfg.guard_lambda_mem == 4
