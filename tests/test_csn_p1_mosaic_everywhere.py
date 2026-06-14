from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from agent.csn_ppo.config import CSNPPOConfig
from agent.csn_ppo.coverage_probes import pack_cover_obs
from agent.csn_ppo.mosaic_teacher import (
    ChampionState,
    ClusterChampion,
    MosaicChampions,
    init_champion,
    init_mosaic_champions,
)
from agent.csn_ppo.rollout_mining import (
    label_atoms_with_mosaic_teacher,
    mine_atoms,
)
from praxis import contract


@dataclass(frozen=True)
class DummyParams:
    mean: jnp.ndarray


CFG = CSNPPOConfig(atoms_per_rollout=4, num_clusters=4)


def _params(mean):
    return DummyParams(mean=jnp.asarray(mean, dtype=jnp.float32))


def _normalizer(value: float):
    return {"mean": jnp.asarray([value], dtype=jnp.float32)}


def _apply_policy_value(params, normalizer_params, obs):
    del normalizer_params
    n = obs.shape[0]
    mean = jnp.broadcast_to(params.mean, (n, contract.ACT_DIM))
    logstd = jnp.zeros((n, contract.ACT_DIM), dtype=jnp.float32)
    value = jnp.zeros((n,), dtype=jnp.float32)
    return mean, logstd, value


def _cluster_champions(cluster_means):
    champions = list(init_mosaic_champions(CFG.num_clusters).champions)
    for cluster_id, mean in cluster_means.items():
        champions[cluster_id] = ClusterChampion(
            param_snapshot=_params(mean),
            normalizer_snapshot=_normalizer(float(cluster_id)),
            best_coverage=1.0,
            best_collision_rate=0.0,
            best_return=1.0,
            consecutive_wins=0,
            policy_id=f"cluster-{cluster_id}",
            cluster_id=cluster_id,
        )
    return MosaicChampions(champions=tuple(champions))


def _global_champion(mean):
    return ChampionState(
        normalizer_params=_normalizer(100.0),
        params=_params(mean),
        champion_best_coverage=1.0,
        champion_wins=0,
    )


def _obs_for_cluster(cluster_id: int):
    agent = jnp.zeros((contract.AGENT_DIM,), dtype=jnp.float32)
    far = jnp.asarray([0.9, 0.9, 0.0, 0.0], dtype=jnp.float32)
    obstacles = jnp.repeat(far[None, :], contract.K, axis=0)
    if cluster_id == 0:
        obstacles = obstacles.at[0].set(jnp.asarray([0.05, 0.0, 0.0, 0.0], dtype=jnp.float32))
    elif cluster_id == 2:
        obstacles = obstacles.at[0].set(jnp.asarray([0.40, 0.0, 0.30, 0.0], dtype=jnp.float32))
    elif cluster_id == 3:
        obstacles = obstacles.at[0].set(jnp.asarray([0.40, 0.0, 0.0, 0.0], dtype=jnp.float32))
    frontier = jnp.asarray([1.0, 0.0, 0.5], dtype=jnp.float32)
    covered = jnp.asarray(0.2, dtype=jnp.float32)
    return pack_cover_obs(agent, obstacles, frontier, covered)


def test_1_1_cluster_specific_teacher_selection():
    teacher_means = {
        0: jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        1: jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        2: jnp.asarray([-1.0, 0.0], dtype=jnp.float32),
        3: jnp.asarray([0.0, -1.0], dtype=jnp.float32),
    }
    obs = jnp.stack([_obs_for_cluster(cid) for cid in range(CFG.num_clusters)])
    cluster_id = jnp.arange(CFG.num_clusters, dtype=jnp.int32)

    atoms = label_atoms_with_mosaic_teacher(
        obs=obs,
        cluster_id=cluster_id,
        champions=_cluster_champions(teacher_means),
        global_champion=init_champion(),
        current_params=_params([0.25, 0.25]),
        current_normalizer=_normalizer(0.0),
        apply_policy_value=_apply_policy_value,
        cfg=CFG,
    )

    expected = jnp.stack([teacher_means[cid] for cid in range(CFG.num_clusters)])
    np.testing.assert_array_equal(np.asarray(atoms.cluster_id), np.arange(CFG.num_clusters))
    np.testing.assert_allclose(np.asarray(atoms.mean), np.asarray(expected))


def test_1_2_fallback_order_global_then_current_before_any_champion():
    global_mean = jnp.asarray([0.5, 0.5], dtype=jnp.float32)
    current_mean = jnp.asarray([-0.25, 0.25], dtype=jnp.float32)
    obs = jnp.stack([_obs_for_cluster(1)])
    cluster_id = jnp.asarray([1], dtype=jnp.int32)

    missing_cluster_atoms = label_atoms_with_mosaic_teacher(
        obs=obs,
        cluster_id=cluster_id,
        champions=_cluster_champions({0: [1.0, 0.0]}),
        global_champion=_global_champion(global_mean),
        current_params=_params(current_mean),
        current_normalizer=_normalizer(0.0),
        apply_policy_value=_apply_policy_value,
        cfg=CFG,
    )
    np.testing.assert_allclose(np.asarray(missing_cluster_atoms.mean[0]), np.asarray(global_mean))

    no_champion_atoms = label_atoms_with_mosaic_teacher(
        obs=obs,
        cluster_id=cluster_id,
        champions=init_mosaic_champions(CFG.num_clusters),
        global_champion=init_champion(),
        current_params=_params(current_mean),
        current_normalizer=_normalizer(0.0),
        apply_policy_value=_apply_policy_value,
        cfg=CFG,
    )
    np.testing.assert_allclose(np.asarray(no_champion_atoms.mean[0]), np.asarray(current_mean))


def test_1_3_mine_atoms_uses_mosaic_not_single_global_teacher():
    teacher_means = {
        0: jnp.asarray([1.0, 0.0], dtype=jnp.float32),
        1: jnp.asarray([0.0, 1.0], dtype=jnp.float32),
        2: jnp.asarray([-1.0, 0.0], dtype=jnp.float32),
        3: jnp.asarray([0.0, -1.0], dtype=jnp.float32),
    }
    global_mean = jnp.asarray([0.5, 0.5], dtype=jnp.float32)
    obs = jnp.stack([_obs_for_cluster(cid) for cid in range(CFG.num_clusters)])
    adv_abs = jnp.asarray([4.0, 3.0, 2.0, 1.0], dtype=jnp.float32)

    atoms, _, _ = mine_atoms(
        obs_flat=obs,
        adv_abs=adv_abs,
        params=_params([0.0, 0.0]),
        normalizer_params=_normalizer(0.0),
        apply_policy_value=_apply_policy_value,
        cfg=CFG,
        champions=_cluster_champions(teacher_means),
        global_champion=_global_champion(global_mean),
    )

    cluster_ids = np.asarray(atoms.cluster_id)
    assert set(cluster_ids.tolist()) == set(range(CFG.num_clusters))
    for mean, cluster_id in zip(np.asarray(atoms.mean), cluster_ids):
        np.testing.assert_allclose(mean, np.asarray(teacher_means[int(cluster_id)]))
    assert not np.allclose(
        np.asarray(atoms.mean),
        np.broadcast_to(np.asarray(global_mean), np.asarray(atoms.mean).shape),
    )
