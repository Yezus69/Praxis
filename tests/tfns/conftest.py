from __future__ import annotations

import jax
import pytest


@pytest.fixture(autouse=True)
def assert_x64_disabled():
    assert not jax.config.jax_enable_x64
