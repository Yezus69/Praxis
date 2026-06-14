from __future__ import annotations

from dataclasses import dataclass
from types import SimpleNamespace

import numpy as np

from agent.csn_ppo.mosaic_teacher import (
    get_cluster_teacher,
    init_mosaic_champions,
    maybe_update_champions,
)


@dataclass(frozen=True)
class DummyParams:
    policy: dict[str, np.ndarray]
    value: dict[str, np.ndarray]

    def replace(self, **kwargs):
        return DummyParams(
            policy=kwargs.get("policy", self.policy),
            value=kwargs.get("value", self.value),
        )


CFG = SimpleNamespace(champion_min_margin=0.02, champion_patience=2)


def _params(value: float) -> DummyParams:
    return DummyParams(
        policy={"w": np.asarray([value], dtype=np.float32)},
        value={"w": np.asarray([value + 100.0], dtype=np.float32)},
    )


def _normalizer(value: float) -> dict[str, np.ndarray]:
    return {"mean": np.asarray([value], dtype=np.float32)}


def _metrics(coverage: float, collision_rate: float, mean_return: float):
    return {
        "coverage": coverage,
        "collision_rate": collision_rate,
        "mean_return": mean_return,
    }


def _promote_cluster(champions, cluster_id: int = 0):
    champions = maybe_update_champions(
        {cluster_id: _metrics(0.60, 0.00, 10.0)},
        _params(1.0),
        _normalizer(1.0),
        champions,
        CFG,
        "policy-1",
    )
    return maybe_update_champions(
        {cluster_id: _metrics(0.60, 0.00, 10.0)},
        _params(2.0),
        _normalizer(2.0),
        champions,
        CFG,
        "policy-2",
    )


def test_maybe_update_champions_promotes_only_after_patience():
    champions = init_mosaic_champions(num_clusters=2)

    champions = maybe_update_champions(
        {0: _metrics(0.60, 0.00, 10.0)},
        _params(1.0),
        _normalizer(1.0),
        champions,
        CFG,
        "policy-1",
    )
    assert champions.champions[0].consecutive_wins == 1
    assert champions.champions[0].param_snapshot is None
    assert champions.champions[0].normalizer_snapshot is None

    champions = maybe_update_champions(
        {0: _metrics(0.60, 0.00, 10.0)},
        _params(2.0),
        _normalizer(2.0),
        champions,
        CFG,
        "policy-2",
    )
    champion = champions.champions[0]
    assert champion.consecutive_wins == 0
    assert champion.policy_id == "policy-2"
    assert champion.best_coverage == 0.60
    assert champion.best_collision_rate == 0.00
    assert champion.best_return == 10.0
    np.testing.assert_array_equal(champion.param_snapshot.policy["w"], np.asarray([2.0], dtype=np.float32))
    np.testing.assert_array_equal(champion.normalizer_snapshot["mean"], np.asarray([2.0], dtype=np.float32))


def test_maybe_update_champions_regression_resets_consecutive_wins():
    champions = _promote_cluster(init_mosaic_champions(num_clusters=1))

    champions = maybe_update_champions(
        {0: _metrics(0.70, 0.00, 11.0)},
        _params(3.0),
        _normalizer(3.0),
        champions,
        CFG,
        "policy-3",
    )
    assert champions.champions[0].consecutive_wins == 1

    champions = maybe_update_champions(
        {0: _metrics(0.61, 0.00, 12.0)},
        _params(4.0),
        _normalizer(4.0),
        champions,
        CFG,
        "policy-4",
    )
    champion = champions.champions[0]
    assert champion.consecutive_wins == 0
    assert champion.policy_id == "policy-2"
    np.testing.assert_array_equal(champion.param_snapshot.policy["w"], np.asarray([2.0], dtype=np.float32))


def test_maybe_update_champions_clusters_are_independent():
    champions = init_mosaic_champions(num_clusters=3)

    champions = maybe_update_champions(
        {1: _metrics(0.60, 0.00, 10.0)},
        _params(1.0),
        _normalizer(1.0),
        champions,
        CFG,
        "policy-1",
    )
    champions = maybe_update_champions(
        {1: _metrics(0.60, 0.00, 10.0)},
        _params(2.0),
        _normalizer(2.0),
        champions,
        CFG,
        "policy-2",
    )

    assert champions.champions[1].policy_id == "policy-2"
    assert champions.champions[0].policy_id is None
    assert champions.champions[2].policy_id is None
    assert champions.champions[0].param_snapshot is None
    assert champions.champions[2].param_snapshot is None


def test_get_cluster_teacher_returns_champion_snapshot():
    champions = _promote_cluster(init_mosaic_champions(num_clusters=1))

    normalizer_snapshot, param_snapshot = get_cluster_teacher(champions, 0)

    np.testing.assert_array_equal(normalizer_snapshot["mean"], np.asarray([2.0], dtype=np.float32))
    np.testing.assert_array_equal(param_snapshot.policy["w"], np.asarray([2.0], dtype=np.float32))
    np.testing.assert_array_equal(param_snapshot.value["w"], np.asarray([102.0], dtype=np.float32))
