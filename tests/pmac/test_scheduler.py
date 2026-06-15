import numpy as np

from pmac.anchors import AnchorStore
from pmac.atlas import Atlas
from pmac.scheduler import Scheduler
from pmac.sentinels import SentinelStore


def _store():
    store = AnchorStore(capacity=1)
    store.add(
        np.zeros((1, 1), dtype=np.float32),
        np.zeros((1, 2), dtype=np.float32),
        np.zeros(1, dtype=np.float32),
        np.ones(1, dtype=np.float32),
        np.ones(1, dtype=np.float32),
    )
    return store


def _sentinels():
    return SentinelStore(np.zeros((1, 1), dtype=np.float32), np.zeros(1, dtype=np.int32))


def _atlas():
    atlas = Atlas()
    atlas.create_or_update_node("cur", 0, _store(), _sentinels())
    atlas.create_or_update_node("old-a", 1, _store(), _sentinels())
    atlas.create_or_update_node("old-b", 2, _store(), _sentinels())
    return atlas


def test_empirical_old_fraction_matches_config():
    atlas = _atlas()
    scheduler = Scheduler(old_skill_sample_fraction=0.3)
    draws = [scheduler.sample_source(atlas, "cur", key) for key in range(2000)]
    old_fraction = np.mean([draw != "cur" for draw in draws])

    assert abs(old_fraction - 0.3) <= 0.1


def test_boost_raises_regressed_node_share():
    atlas = _atlas()
    scheduler = Scheduler(old_skill_sample_fraction=1.0)
    before = [scheduler.sample_source(atlas, "cur", key) for key in range(1000)]
    before_share = np.mean([draw == "old-a" for draw in before])

    scheduler.boost(["old-a"])
    after = [scheduler.sample_source(atlas, "cur", key + 10_000) for key in range(1000)]
    after_share = np.mean([draw == "old-a" for draw in after])

    assert after_share > before_share
