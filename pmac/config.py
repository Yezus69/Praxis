"""Configuration defaults for PMA-C experiments."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PMAConfig:
    """Recommended PMA-C defaults from spec section 26, scaled for MNIST."""

    anchor_memory_per_skill: int = 2_000
    episodic_memory_per_skill: int = 100_000
    sentinel_count_per_skill: int = 512
    challenge_count_per_skill: int = 512

    drift_budget: float = 0.0
    drift_budget_kl: float = 0.0
    drift_budget_policy_kl: float = 0.01
    drift_budget_value: float = 0.05

    guard_lambda: float = 1.0
    guard_lambda_max: float = 64.0

    projection_enabled: bool = True
    stability_enabled: bool = True
    consolidation_enabled: bool = True
    gate_enabled: bool = True
    growth_enabled: bool = True

    growth_min_ratio: float = 0.10
    growth_patience: int = 100

    slow_lr_multiplier: float = 0.01
    stability_alpha: float = 10.0
    stability_decay: float = 0.99

    consolidation_interval: int = 10_000
    consolidation_epochs: int = 3

    old_skill_sample_fraction: float = 0.30
    sentinel_eval_interval: int = 1_000
    full_audit_interval: int = 5_000
    audit_interval: int = 100

    delta_current: float = 0.05
    # Gate leniency: the principled regression signal is the sentinel-accuracy check
    # (allowed_regression). The raw guard-loss epsilon must be loose or it vetoes nearly
    # all new-task learning (an absolute 1e-4 caused ~83% of gate checks to roll back,
    # collapsing plasticity). Keep it well above the typical per-step guard-loss increase.
    delta_cons: float = 0.25
    allowed_regression: float = 0.05
    skill_solved_threshold: float = 0.95
    num_guard_nodes: int = 4


@dataclass
class ExperimentConfig:
    """Small-scale continual supervised experiment configuration."""

    hidden_sizes: tuple[int, ...] = (256, 256)
    epochs_per_task: int = 1
    batch_size: int = 128
    lr: float = 0.1
    optimizer: str = "sgd"
    temperature: float = 2.0
    replay_batch: int = 64
    num_guard_nodes: int = 4
    seed: int = 0
    max_eval: int = 2_000
    use_jit: bool = True


__all__ = ["PMAConfig", "ExperimentConfig"]
