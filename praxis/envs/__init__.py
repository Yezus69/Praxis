"""praxis.envs — the Phase-0 navigation environment package.

Public surface:
    NavEnv            — the MJX / Playground navigation env (functional State API).
    default_config    — its default ml_collections config.
    domain_randomize  — Brax-style continuous domain randomization fn.
"""

from praxis.envs.nav_env import NavEnv, default_config
from praxis.envs.randomize import domain_randomize

__all__ = ["NavEnv", "default_config", "domain_randomize"]
