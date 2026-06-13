"""praxis.envs — the Phase-0 COVERAGE / exploration environment package.

Public surface:
    CoverEnv         — the MJX area-coverage env (functional State API, real physics).
    default_config   — its default ml_collections config.
"""

from praxis.envs.cover_env import CoverEnv, default_config

__all__ = ["CoverEnv", "default_config"]
