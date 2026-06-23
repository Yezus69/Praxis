"""Five-game suite definition tests (spec 18)."""

from __future__ import annotations

from tfns.castm import suite


def test_canonical_atari57_has_57_unique_games():
    assert len(suite.ATARI_57) == 57
    assert len(set(suite.ATARI_57)) == 57
    names = suite.canonical_atari57()
    assert all(n.endswith("-v5") for n in names)
    # A few well-known anchors must be present.
    for g in ("Pong", "Breakout", "MontezumaRevenge", "Seaquest", "SpaceInvaders"):
        assert g in suite.ATARI_57


def test_sample_is_deterministic_and_frozen():
    a = suite.sample_suite(seed=57057, size=5)
    b = suite.sample_suite(seed=57057, size=5)
    assert a.games == b.games
    assert len(a.games) == 5
    assert len(set(a.games)) == 5  # without replacement
    assert all(g in suite.canonical_atari57() for g in a.games)


def test_persisted_suite_matches_sample():
    persisted = suite.load_suite()
    fresh = suite.sample_suite(seed=persisted.seed, size=len(persisted.games))
    assert persisted.games == fresh.games
    assert persisted.diagnostic_pair == (persisted.games[0], persisted.games[1])


def test_curriculum_orders_are_distinct_permutations():
    s = suite.sample_suite()
    orders = suite.curriculum_orders(s, n_orders=3)
    assert len(orders) == 3
    assert orders[0] == list(s.games)  # identity first
    # All orders are permutations of the same five games and are distinct.
    base = set(s.games)
    seen = set()
    for order in orders:
        assert set(order) == base
        seen.add(tuple(order))
    assert len(seen) == 3
