"""Fixed Atari five-game suite definition (spec section 18).

Sampling is deterministic and *frozen*: five games are drawn without replacement
from the canonical Atari-57 set using the fixed suite seed 57057. The ordered
list is persisted before training and reused for every method, seed, reference,
and ablation. A difficult game is never resampled after observing results
(spec 18.5, 26).

The experiment harness may know game names for environment creation and
evaluation. The agent and memory system may not (spec 18, 26): no game/task
identity ever enters policy inference, routing, memory addressing, optimization,
or replay selection.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass

import numpy as np


# Canonical Atari-57 benchmark set (Mnih et al. / Rainbow / Agent57), in the
# standard alphabetical canonical order. envpool exposes each as "<Name>-v5".
ATARI_57 = [
    "Alien", "Amidar", "Assault", "Asterix", "Asteroids", "Atlantis",
    "BankHeist", "BattleZone", "BeamRider", "Berzerk", "Bowling", "Boxing",
    "Breakout", "Centipede", "ChopperCommand", "CrazyClimber", "Defender",
    "DemonAttack", "DoubleDunk", "Enduro", "FishingDerby", "Freeway",
    "Frostbite", "Gopher", "Gravitar", "Hero", "IceHockey", "Jamesbond",
    "Kangaroo", "Krull", "KungFuMaster", "MontezumaRevenge", "MsPacman",
    "NameThisGame", "Phoenix", "Pitfall", "Pong", "PrivateEye", "Qbert",
    "Riverraid", "RoadRunner", "Robotank", "Seaquest", "Skiing", "Solaris",
    "SpaceInvaders", "StarGunner", "Surround", "Tennis", "TimePilot",
    "Tutankham", "UpNDown", "Venture", "VideoPinball", "WizardOfWor",
    "YarsRevenge", "Zaxxon",
]

SUITE_SEED = 57057
SUITE_SIZE = 5


def canonical_atari57() -> list[str]:
    """Return the canonical Atari-57 names with the envpool ``-v5`` suffix."""

    return [f"{name}-v5" for name in ATARI_57]


@dataclass(frozen=True)
class GameSuite:
    """A frozen ordered game suite."""

    seed: int
    games: tuple[str, ...]

    @property
    def diagnostic_pair(self) -> tuple[str, str]:
        """The first two sampled games (spec 18.6)."""

        return (self.games[0], self.games[1])

    def to_dict(self) -> dict:
        return {
            "seed": int(self.seed),
            "size": len(self.games),
            "games": list(self.games),
            "diagnostic_pair": list(self.diagnostic_pair),
            "source": "canonical_atari57",
        }


def sample_suite(seed: int = SUITE_SEED, size: int = SUITE_SIZE) -> GameSuite:
    """Sample ``size`` games without replacement from Atari-57 (deterministic)."""

    if len(ATARI_57) != 57:
        raise AssertionError(f"canonical set must have 57 games, has {len(ATARI_57)}")
    pool = canonical_atari57()
    rng = np.random.default_rng(int(seed))
    idx = rng.choice(len(pool), size=int(size), replace=False)
    games = tuple(pool[int(i)] for i in idx)
    return GameSuite(seed=int(seed), games=games)


def curriculum_orders(suite: GameSuite, n_orders: int = 3, seed: int = SUITE_SEED) -> list[list[str]]:
    """Return ``n_orders`` permutations of the suite (spec 18.7, 21.6).

    The identity order is always first; remaining orders are deterministic
    permutations. Different *orders* of the *same five games* are required for
    final experiments.
    """

    orders: list[list[str]] = [list(suite.games)]
    rng = np.random.default_rng(int(seed) + 1)
    seen = {tuple(suite.games)}
    attempts = 0
    while len(orders) < int(n_orders) and attempts < 1000:
        attempts += 1
        perm = tuple(rng.permutation(list(suite.games)).tolist())
        if perm not in seen:
            seen.add(perm)
            orders.append(list(perm))
    return orders


def default_suite_path() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    return os.path.join(here, "suites", f"five_game_suite_{SUITE_SEED}.json")


def persist_suite(path: str | None = None, seed: int = SUITE_SEED, size: int = SUITE_SIZE) -> str:
    """Sample and persist the frozen suite to JSON; return the path."""

    suite = sample_suite(seed=seed, size=size)
    out = path or default_suite_path()
    os.makedirs(os.path.dirname(out), exist_ok=True)
    payload = suite.to_dict()
    payload["curriculum_orders"] = curriculum_orders(suite)
    with open(out, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    return out


def load_suite(path: str | None = None) -> GameSuite:
    """Load a persisted suite (raises if absent — the suite must be frozen first)."""

    src = path or default_suite_path()
    with open(src, encoding="utf-8") as f:
        payload = json.load(f)
    return GameSuite(seed=int(payload["seed"]), games=tuple(payload["games"]))


__all__ = [
    "ATARI_57",
    "GameSuite",
    "SUITE_SEED",
    "SUITE_SIZE",
    "canonical_atari57",
    "curriculum_orders",
    "default_suite_path",
    "load_suite",
    "persist_suite",
    "sample_suite",
]


if __name__ == "__main__":
    path = persist_suite()
    suite = load_suite(path)
    print(f"persisted suite seed={suite.seed} -> {path}")
    print("games:", list(suite.games))
    print("diagnostic pair:", suite.diagnostic_pair)
