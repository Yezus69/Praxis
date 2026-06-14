"""CSN-PPO configuration."""

from dataclasses import dataclass


@dataclass
class CSNPPOConfig:
    # PPO baseline
    num_timesteps: int = int(1e8)
    num_envs: int = 2048
    episode_length: int = 1000
    unroll_length: int = 20
    batch_size: int = 256
    num_minibatches: int = 32
    max_updates_per_batch: int = 4
    learning_rate: float = 3e-4
    entropy_cost: float = 1e-2
    discounting: float = 0.97
    reward_scaling: float = 1.0
    normalize_observations: bool = True
    seed: int = 0

    # Behavioral memory
    memory_size_fast: int = 1_048_576
    memory_size_slow: int = 262_144
    memory_batch_size: int = 4096
    min_memory_size_before_guard: int = 16_384

    # Guard budgets
    guard_policy_coef: float = 1.0
    guard_value_coef: float = 0.25
    guard_kl_budget: float = 0.02
    critical_kl_budget: float = 0.005
    value_budget: float = 0.25
    critical_value_budget: float = 0.05
    memory_kl_limit_p95: float = 0.05

    # Gradient projection
    enable_gradient_projection: bool = True
    projection_eps: float = 1e-8

    # Sentinel evaluation
    sentinel_eval_interval: int = 25
    sentinel_bank_size: int = 4096
    sentinel_success_tolerance: float = 0.05
    sentinel_collision_tolerance: float = 0.03
    sentinel_patience: int = 3

    # Synthetic probes
    synthetic_probe_batch_size: int = 4096
    synthetic_probe_insert_interval: int = 1

    # Holdout overfit control
    holdout_fraction: float = 0.2
    holdout_eps: float = 1e-4
    target_kl: float = 0.03

    # Curriculum mixture
    frontier_fraction: float = 0.70
    history_fraction: float = 0.20
    sentinel_failure_fraction: float = 0.10

    # Mosaic teacher
    enable_mosaic_teacher: bool = True
    champion_min_margin: float = 0.02
    champion_patience: int = 3
