"""Protected activation subspace API."""

from tfns.protect.bases import (
    empty_basis,
    expand_basis,
    free_rank_fraction,
    from_storage,
    orthonormality_error,
    represented_energy,
    residual_norm,
    to_storage,
)
from tfns.protect.projection import (
    ProtectedModule,
    build_protected_modules,
    collect_conv_basis_columns,
    conv_patches,
    project_affine,
    project_conv,
    project_gru_gate,
    project_update,
)

__all__ = [
    "ProtectedModule",
    "build_protected_modules",
    "collect_conv_basis_columns",
    "conv_patches",
    "empty_basis",
    "expand_basis",
    "free_rank_fraction",
    "from_storage",
    "orthonormality_error",
    "project_affine",
    "project_conv",
    "project_gru_gate",
    "project_update",
    "represented_energy",
    "residual_norm",
    "to_storage",
]
