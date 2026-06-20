"""TFNS global configuration.

All fields are global; no game identity ever enters config.
"""

from __future__ import annotations

import dataclasses


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    act_dim: int = 18
    frame_stack: int = 4
    obs_hw: int = 84
    conv_channels: tuple[int, int, int] = (32, 64, 64)
    conv_kernels: tuple[int, int, int] = (8, 4, 3)
    conv_strides: tuple[int, int, int] = (4, 2, 1)
    dense_dim: int = 512
    action_embed_dim: int = 32
    gru_hidden: int = 512
    key_dim: int = 128
    activation: str = "relu"
    ema_decay: float = 0.995
    key_eps: float = 1e-6


@dataclasses.dataclass(frozen=True)
class AuxConfig:
    aux_coef: float = 0.1
    next_feat_coef: float = 1.0
    reward_cat_coef: float = 1.0
    terminal_coef: float = 1.0


@dataclasses.dataclass(frozen=True)
class AdapterConfig:
    num_adapters: int = 8
    rank: int = 32
    top_k: int = 2
    plasticity_ratio_thresh: float = 0.1
    patience_blocks: int = 3


@dataclasses.dataclass(frozen=True)
class PPOConfig:
    num_envs: int = 64
    rollout_len: int = 128
    gamma: float = 0.99
    gae_lambda: float = 0.95
    clip_coef: float = 0.2
    vf_clip: float = 0.2
    ent_coef: float = 0.01
    anneal_ent: bool = True
    ent_coef_final: float = 0.0
    vf_coef: float = 0.5
    max_grad_norm: float = 0.5
    lr: float = 2.5e-4
    update_epochs: int = 4
    num_minibatches: int = 4
    seq_chunk: int = 32
    anneal_lr: bool = True


@dataclasses.dataclass(frozen=True)
class ReplayConfig:
    seq_len: int = 64
    burn_in: int = 16
    protected_region: int = 48
    replay_frac_start: float = 0.25
    batch_size: int | None = None


@dataclasses.dataclass(frozen=True)
class MemoryConfig:
    byte_budget: int = 1 << 30
    w_adv: float = 1.0
    w_td: float = 1.0
    w_causal: float = 2.0
    w_novelty: float = 1.5
    w_surprise: float = 1.0
    w_failure: float = 2.0
    w_entropy: float = 0.25
    w_drift: float = 3.0
    lam_risk: float = 1.0
    lam_cover: float = 1.0
    lam_causal: float = 1.0
    lam_red: float = 1.0
    lam_age: float = 0.5
    min_per_cluster: int = 2
    score_mean_w: float = 0.34
    score_quantile_w: float = 0.33
    score_max_w: float = 0.33
    score_quantile: float = 0.9
    cluster_sim_thresh: float = 0.9
    cluster_merge_thresh: float = 0.97
    max_clusters: int = 64
    max_admit_per_block: int = 32
    max_records: int = 4000


@dataclasses.dataclass(frozen=True)
class BehaviorConfig:
    teacher_temp: float = 1.0
    kl_tol: float = 0.01
    value_tol: float = 0.1
    key_cos_tol: float = 0.02
    lambda_v: float = 1.0
    lambda_q: float = 1.0
    tail_frac: float = 0.10


@dataclasses.dataclass(frozen=True)
class CreditConfig:
    lambda_c: float = 0.9
    eta_max: float = 0.5
    predictor_val_windows: int = 2


@dataclasses.dataclass(frozen=True)
class ProtectConfig:
    residual_energy: float = 0.995
    max_rank_frac: float = 0.95
    constraint_max_clusters: int = 8
    constraint_ridge: float = 1e-3
    backtrack_scales: tuple[float, ...] = (1.0, 0.5, 0.25, 0.125, 0.0625, 0.03125)
    float32_basis: bool = True


@dataclasses.dataclass(frozen=True)
class RiskConfig:
    rho_0: float = 0.1
    lam_D: float = 1.0
    lam_Q: float = 1.0
    lam_R: float = 1.0
    lam_A: float = 1.0


@dataclasses.dataclass(frozen=True)
class ConsolidateConfig:
    learned_threshold: float = 0.9
    stable_windows: int = 2
    retention_accept: float = 0.90
    slow_replay_steps: int = 4
    slow_replay_max_update_norm: float = 0.05
    replay_risk_raise_on_gate_fail: float = 0.1


@dataclasses.dataclass(frozen=True)
class DetectConfig:
    ph_delta: float = 0.05
    ph_lambda: float = 5.0
    cooldown_blocks: int = 2


@dataclasses.dataclass(frozen=True)
class TFNSConfig:
    model: ModelConfig = dataclasses.field(default_factory=ModelConfig)
    aux: AuxConfig = dataclasses.field(default_factory=AuxConfig)
    adapter: AdapterConfig = dataclasses.field(default_factory=AdapterConfig)
    ppo: PPOConfig = dataclasses.field(default_factory=PPOConfig)
    replay: ReplayConfig = dataclasses.field(default_factory=ReplayConfig)
    memory: MemoryConfig = dataclasses.field(default_factory=MemoryConfig)
    behavior: BehaviorConfig = dataclasses.field(default_factory=BehaviorConfig)
    credit: CreditConfig = dataclasses.field(default_factory=CreditConfig)
    protect: ProtectConfig = dataclasses.field(default_factory=ProtectConfig)
    risk: RiskConfig = dataclasses.field(default_factory=RiskConfig)
    consolidate: ConsolidateConfig = dataclasses.field(default_factory=ConsolidateConfig)
    detect: DetectConfig = dataclasses.field(default_factory=DetectConfig)


__all__ = [
    "AdapterConfig",
    "AuxConfig",
    "BehaviorConfig",
    "ConsolidateConfig",
    "CreditConfig",
    "DetectConfig",
    "MemoryConfig",
    "ModelConfig",
    "PPOConfig",
    "ProtectConfig",
    "ReplayConfig",
    "RiskConfig",
    "TFNSConfig",
]
