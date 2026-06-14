import jax
import jax.numpy as jnp

from agent.csn_ppo.sentinel import (
    SentinelSeed,
    create_sentinel_bank,
    detect_sentinel_regressions,
)


def test_create_sentinel_bank_shapes_and_dtypes():
    bank = create_sentinel_bank(jax.random.PRNGKey(0), size=8, num_clusters=4)

    assert bank.reset_rng.shape == (8, 2)
    assert bank.cluster_id.shape == (8,)
    assert bank.difficulty.shape == (8,)
    assert bank.best_coverage.shape == (4,)
    assert bank.best_collision_rate.shape == (4,)
    assert bank.champion_policy_id.shape == (4,)
    assert bank.reset_rng.dtype == jnp.uint32
    assert bank.cluster_id.dtype == jnp.int32
    assert bank.difficulty.dtype == jnp.float32
    assert bank.best_coverage.dtype == jnp.float32
    assert bank.best_collision_rate.dtype == jnp.float32
    assert bank.champion_policy_id.dtype == jnp.int32


def test_detect_sentinel_regressions_uses_exact_section_13_rule():
    bank = SentinelSeed(
        reset_rng=jax.random.split(jax.random.PRNGKey(1), 3),
        cluster_id=jnp.arange(3, dtype=jnp.int32),
        difficulty=jnp.zeros((3,), dtype=jnp.float32),
        best_coverage=jnp.array([0.80, 0.70, 0.60], dtype=jnp.float32),
        best_collision_rate=jnp.array([0.10, 0.20, 0.30], dtype=jnp.float32),
        champion_policy_id=jnp.full((3,), -1, dtype=jnp.int32),
    )
    current_metrics = {
        "coverage": jnp.array([0.74, 0.66, 0.55], dtype=jnp.float32),
        "collision_rate": jnp.array([0.10, 0.24, 0.33], dtype=jnp.float32),
    }

    regressions = detect_sentinel_regressions(
        current_metrics,
        bank,
        success_tol=0.05,
        collision_tol=0.03,
    )

    assert bool(regressions["success_bad"][0])
    assert not bool(regressions["success_bad"][1])
    assert bool(regressions["collision_bad"][1])
    assert not bool(regressions["collision_bad"][2])
    assert bool(regressions["regressed"][0])
    assert bool(regressions["regressed"][1])
    assert not bool(regressions["regressed"][2])


def test_detect_sentinel_regressions_false_inside_tolerance():
    bank = SentinelSeed(
        reset_rng=jax.random.split(jax.random.PRNGKey(2), 2),
        cluster_id=jnp.arange(2, dtype=jnp.int32),
        difficulty=jnp.zeros((2,), dtype=jnp.float32),
        best_coverage=jnp.array([0.80, 0.70], dtype=jnp.float32),
        best_collision_rate=jnp.array([0.10, 0.20], dtype=jnp.float32),
        champion_policy_id=jnp.full((2,), -1, dtype=jnp.int32),
    )
    current_metrics = {
        "coverage": jnp.array([0.75, 0.6501], dtype=jnp.float32),
        "collision_rate": jnp.array([0.13, 0.2299], dtype=jnp.float32),
    }

    regressions = detect_sentinel_regressions(
        current_metrics,
        bank,
        success_tol=0.05,
        collision_tol=0.03,
    )

    assert not bool(jnp.any(regressions["success_bad"]))
    assert not bool(jnp.any(regressions["collision_bad"]))
    assert not bool(jnp.any(regressions["regressed"]))


def test_sentinel_seed_is_jax_pytree():
    bank = create_sentinel_bank(jax.random.PRNGKey(3), size=5, num_clusters=2)

    leaves = jax.tree_util.tree_leaves(bank)

    assert len(leaves) == 6
    assert all(isinstance(leaf, jax.Array) for leaf in leaves)
