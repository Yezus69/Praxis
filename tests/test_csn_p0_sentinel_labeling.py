from __future__ import annotations

from dataclasses import dataclass

import jax.numpy as jnp
import numpy as np

from agent.csn_ppo.config import CSNPPOConfig
from agent.csn_ppo.memory import SOURCE_SENTINEL_FAILURE_UNTRUSTED
from agent.csn_ppo.mosaic_teacher import (
    ChampionState,
    ClusterChampion,
    MosaicChampions,
    init_champion,
    init_mosaic_champions,
)
from agent.csn_ppo.sentinel import (
    SentinelTrajectories,
    label_failed_sentinel_atoms_with_best_teacher,
)
from praxis import contract


@dataclass(frozen=True)
class DummyParams:
    mean: jnp.ndarray


CFG = CSNPPOConfig(atoms_per_rollout=2)


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


def _trajectories(cluster_id: int = 0):
    obs = jnp.zeros((1, 2, contract.OBS_DIM), dtype=jnp.float32)
    return SentinelTrajectories(
        obs=obs,
        reward=jnp.zeros((1, 2), dtype=jnp.float32),
        coverage=jnp.zeros((1,), dtype=jnp.float32),
        collision_rate=jnp.zeros((1,), dtype=jnp.float32),
        cluster_id=jnp.asarray([cluster_id], dtype=jnp.int32),
        active=jnp.ones((1, 2), dtype=jnp.float32),
    )


def _regressions(num_clusters: int = 1, cluster_id: int = 0):
    regressed = np.zeros((num_clusters,), dtype=bool)
    regressed[cluster_id] = True
    return {"regressed": jnp.asarray(regressed)}


def _cluster_champions(cluster_id: int, params, normalizer, num_clusters: int = 1):
    champions = list(init_mosaic_champions(num_clusters).champions)
    champions[cluster_id] = ClusterChampion(
        param_snapshot=params,
        normalizer_snapshot=normalizer,
        best_coverage=1.0,
        best_collision_rate=0.0,
        best_return=1.0,
        consecutive_wins=0,
        policy_id="cluster-champion",
        cluster_id=cluster_id,
    )
    return MosaicChampions(champions=tuple(champions))


def _global_champion(params, normalizer):
    return ChampionState(
        normalizer_params=normalizer,
        params=params,
        champion_best_coverage=1.0,
        champion_wins=0,
    )


def test_0_1_failed_sentinel_labels_do_not_use_current_policy():
    current_mean = jnp.asarray([0.0, 0.0], dtype=jnp.float32)
    champion_mean = jnp.asarray([1.0, 0.0], dtype=jnp.float32)
    current_params = _params(current_mean)
    champion_params = _params(champion_mean)

    failed_atoms = label_failed_sentinel_atoms_with_best_teacher(
        sentinel_trajectories=_trajectories(cluster_id=0),
        regressions=_regressions(),
        champions=_cluster_champions(0, champion_params, _normalizer(1.0)),
        global_champion=init_champion(),
        current_params=current_params,
        current_normalizer=_normalizer(0.0),
        apply_policy_value=_apply_policy_value,
        cfg=CFG,
    )

    assert failed_atoms is not None
    assert np.allclose(failed_atoms.mean, champion_mean)
    assert not np.allclose(failed_atoms.mean, current_mean)


def test_0_2_missing_cluster_champion_falls_back_to_global_champion():
    current_mean = jnp.asarray([0.0, 0.0], dtype=jnp.float32)
    global_mean = jnp.asarray([0.0, 1.0], dtype=jnp.float32)
    current_params = _params(current_mean)
    global_params = _params(global_mean)

    failed_atoms = label_failed_sentinel_atoms_with_best_teacher(
        sentinel_trajectories=_trajectories(cluster_id=0),
        regressions=_regressions(),
        champions=init_mosaic_champions(num_clusters=1),
        global_champion=_global_champion(global_params, _normalizer(2.0)),
        current_params=current_params,
        current_normalizer=_normalizer(0.0),
        apply_policy_value=_apply_policy_value,
        cfg=CFG,
    )

    assert failed_atoms is not None
    assert np.allclose(failed_atoms.mean, global_mean)
    assert not np.allclose(failed_atoms.mean, current_mean)


def test_0_3_no_teacher_skips_or_inserts_only_zero_weight_untrusted_atoms():
    failed_atoms = label_failed_sentinel_atoms_with_best_teacher(
        sentinel_trajectories=_trajectories(cluster_id=0),
        regressions=_regressions(),
        champions=init_mosaic_champions(num_clusters=1),
        global_champion=init_champion(),
        current_params=_params([0.0, 0.0]),
        current_normalizer=_normalizer(0.0),
        apply_policy_value=_apply_policy_value,
        cfg=CFG,
    )

    if failed_atoms is None:
        return

    assert np.allclose(failed_atoms.weight, 0.0)
    assert np.all(failed_atoms.source_id == SOURCE_SENTINEL_FAILURE_UNTRUSTED)
