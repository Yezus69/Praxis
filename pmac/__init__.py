"""PMA-C foundational package."""

from pmac.adapter import DomainAdapter
from pmac.behavior_distance import (
    DISTANCES,
    cosine_distance,
    kl_categorical,
    mean_distance,
    mse,
    value_abs,
)
from pmac.conservation import AnchorBatch, anchor_loss, conservation_loss, hinge_violation
from pmac.growth import GrowthController, GrowthState, should_grow
from pmac.projection import plasticity_ratio, project_conflicts
from pmac.stability import scale_by_stability, update_omega, zeros_omega_like
from pmac.tree_utils import (
    tree_add,
    tree_add_scaled,
    tree_dot,
    tree_l2sq,
    tree_norm,
    tree_scale,
    tree_sub,
    tree_zeros_like,
)

__version__ = "0.1.0"

__all__ = [
    "__version__",
    "DomainAdapter",
    "DISTANCES",
    "kl_categorical",
    "mse",
    "cosine_distance",
    "value_abs",
    "mean_distance",
    "AnchorBatch",
    "hinge_violation",
    "anchor_loss",
    "conservation_loss",
    "project_conflicts",
    "plasticity_ratio",
    "scale_by_stability",
    "update_omega",
    "zeros_omega_like",
    "GrowthState",
    "GrowthController",
    "should_grow",
    "tree_dot",
    "tree_norm",
    "tree_add",
    "tree_sub",
    "tree_scale",
    "tree_add_scaled",
    "tree_zeros_like",
    "tree_l2sq",
]
