import pytest

from agent.csn_ppo.config import CSNPPOConfig, validate_long_run_safety


def test_long_run_without_sentinel_fails():
    cfg = CSNPPOConfig(num_timesteps=100_000_000, enable_sentinel=False)

    with pytest.raises(ValueError):
        validate_long_run_safety(cfg)


def test_smoke_run_without_sentinel_allowed():
    cfg = CSNPPOConfig(num_timesteps=300_000, enable_sentinel=False)

    validate_long_run_safety(cfg)


def test_debug_bypass_allows_long_run_without_sentinel():
    cfg = CSNPPOConfig(
        num_timesteps=100_000_000,
        enable_sentinel=False,
        allow_no_sentinel_for_debug=True,
    )

    validate_long_run_safety(cfg)
