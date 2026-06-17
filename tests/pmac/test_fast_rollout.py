import pmac.agents.ppo_atari_fast as ppo_atari_fast
import pmac.envs.atari_envpool as atari_envpool
from pmac.agents.ppo_atari_fast import FastPPOConfig, train_ppo_atari_fast


def test_fast_config_defaults_to_large_xla_batch():
    cfg = FastPPOConfig()

    assert cfg.num_envs == 256


def test_fast_training_public_surface_is_importable():
    assert ppo_atari_fast.FastPPOConfig is FastPPOConfig
    assert ppo_atari_fast.train_ppo_atari_fast is train_ppo_atari_fast
    assert callable(train_ppo_atari_fast)


def test_make_train_env_xla_public_surface_exists():
    assert callable(atari_envpool.make_train_env_xla)
