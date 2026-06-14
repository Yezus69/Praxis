"""CSN-PPO configuration."""

from dataclasses import dataclass

from praxis import contract


@dataclass
class CSNPPOConfig:
    # Coverage env / obs dims
    obs_dim: int = contract.OBS_DIM
    action_dim: int = contract.ACT_DIM
    cov_collision_thresh_norm: float = (
        contract.AGENT_RADIUS + contract.OBSTACLE_RADIUS + contract.COLLISION_MARGIN
    ) / contract.ARENA_HALF
    cov_frontier_norm: float = 2.0 * contract.ARENA_HALF

    # PPO baseline, matched to praxis/train.py defaults for coverage.
    num_timesteps: int = int(1e7)
    num_envs: int = 2048
    episode_length: int = contract.EPISODE_LENGTH
    unroll_length: int = 20
    batch_size: int = 256
    num_minibatches: int = 32
    max_updates_per_batch: int = 4
    learning_rate: float = 3e-4
    entropy_cost: float = 1e-2
    discounting: float = 0.97
    reward_scaling: float = 1.0
    gae_lambda: float = 0.95
    clipping_epsilon: float = 0.3
    vf_coefficient: float = 0.5
    normalize_advantage: bool = True
    normalize_observations: bool = True
    max_grad_norm: float = 1.0
    bootstrap_on_timeout: bool = True
    policy_hidden_layer_sizes: tuple[int, ...] = (256, 256, 256)
    value_hidden_layer_sizes: tuple[int, ...] = (256, 256, 256)
    seed: int = 0

    # Behavioral memory
    memory_size_fast: int = 131_072
    memory_size_slow: int = 32_768
    memory_batch_size: int = 2048
    min_memory_size_before_guard: int = 4096
    guard_warmup_steps: int = 0

    # Guard budgets
    guard_policy_coef: float = 1.0
    guard_value_coef: float = 0.25
    guard_lambda_mem: float = 1.0
    guard_lambda_min: float = 1.0
    guard_lambda_base: float = 8.0
    guard_lambda_max: float = 32.0
    guard_lambda_up: float = 1.5
    guard_lambda_down: float = 0.98
    guard_recovery_patience: int = 3
    guard_kl_budget: float = 0.02
    max_atom_kl: float = 50.0
    guard_mean_clip: float = 3.0
    guard_min_logstd: float = -2.3
    critical_kl_budget: float = 0.005
    value_budget: float = 0.25
    critical_value_budget: float = 0.05
    value_budget_beta: float = 1.0
    memory_kl_limit_p95: float = 0.05

    # Criticality weights, README §19 form with coverage-derived features.
    crit_w_advantage: float = 1.0
    crit_w_collision: float = 3.0
    crit_w_frontier: float = 2.0
    crit_w_dynamic: float = 1.0
    crit_w_novelty: float = 1.0
    crit_clip_min: float = 0.1
    crit_clip_max: float = 10.0
    slow_memory_threshold: float = 3.0
    atoms_per_rollout: int = 1024
    teacher_logstd_floor: float = -6.9
    # Analytic teacher is a geometric direction, not a tight target; guard should
    # keep the policy roughly aligned, not pin it.
    analytic_teacher_logstd: float = -1.6

    # Gradient projection
    enable_gradient_projection: bool = True
    projection_eps: float = 1e-8
    enable_guard: bool = True

    # Sentinel evaluation
    enable_sentinel: bool = True
    allow_no_sentinel_for_debug: bool = False
    long_run_sentinel_required_steps: int = 10_000_000
    sentinel_eval_interval: int = 25
    sentinel_bank_size: int = 4096
    sentinel_success_tolerance: float = 0.05
    sentinel_collision_tolerance: float = 0.03
    sentinel_patience: int = 3

    # Synthetic probes
    synthetic_probe_batch_size: int = 512
    synthetic_probe_insert_interval: int = 1
    synthetic_safe_dist: float = 0.45

    # Holdout overfit control
    enable_holdout_early_stop: bool = True
    enable_kl_early_stop: bool = False
    holdout_fraction: float = 0.2
    holdout_eps: float = 1e-4
    target_kl: float = 0.03

    # Fixed validation bank
    validation_tolerance: float = 0.05
    validation_kl_limit: float = 1.0
    validation_patience: int = 3
    validation_kl_margin: float = 1.0
    validation_eval_interval: int = 25

    # Curriculum mixture
    frontier_fraction: float = 0.70
    history_fraction: float = 0.20
    sentinel_failure_fraction: float = 0.10

    # Mosaic teacher
    enable_mosaic_teacher: bool = True
    num_clusters: int = 4
    champion_min_margin: float = 0.02
    champion_patience: int = 2
    champion_eval_interval: int = 0

    # Evaluation/logging
    num_evals: int = 20
    num_eval_envs: int = 128
    eval_deterministic: bool = False
    log_interval: int = 1

    def __post_init__(self) -> None:
        lhs = int(self.num_envs) * int(self.unroll_length)
        rhs = int(self.batch_size) * int(self.num_minibatches)
        if rhs <= 0 or lhs % rhs != 0:
            raise ValueError(
                "CSN-PPO shape constraint violated: "
                f"(num_envs * unroll_length) = {self.num_envs}*{self.unroll_length} = {lhs} "
                f"must be divisible by (batch_size * num_minibatches) = "
                f"{self.batch_size}*{self.num_minibatches} = {rhs}."
            )
        if rhs % int(self.num_envs) != 0:
            raise ValueError(
                "Brax PPO rollout constraint violated: "
                f"(batch_size * num_minibatches) = {rhs} must be divisible by "
                f"num_envs = {self.num_envs}."
            )


def validate_long_run_safety(cfg):
    if (cfg.num_timesteps >= cfg.long_run_sentinel_required_steps
            and not cfg.enable_sentinel
            and not cfg.allow_no_sentinel_for_debug):
        raise ValueError(
            "Long CSN-PPO runs require --enable-sentinel. "
            "Use --allow-no-sentinel-for-debug only for ablations."
        )
