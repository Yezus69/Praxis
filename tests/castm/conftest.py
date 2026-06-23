from __future__ import annotations

import jax
import pytest


@pytest.fixture(autouse=True)
def assert_x64_disabled():
    # CASTM production precision is FP32 (spec 16 uses eps_write=1e-6 in FP32
    # unit tests). Keep x64 disabled so tests reflect deployed precision.
    assert not jax.config.jax_enable_x64
