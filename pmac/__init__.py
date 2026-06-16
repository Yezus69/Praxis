"""PMA-C — continual PPO on full-ALE Atari with deployed certified-champion routing.

A single shared Nature-DQN CNN is trained sequentially across several real Atari games. Two
things are measured (see pmac/evaluation.py):
  * current_unified_retention — how much the mutable shared net forgets (the falsifiable metric;
    a behavior-conservation guard, pmac/conservation.py + the ppo_atari guard loss, reduces it);
  * deployed_retention — the deployed agent routes each protected skill to its frozen certified
    champion (pmac/deployment.py), so it does not lose protected skills by construction.

Entry point: pmac/experiments/continual_atari.py (single-game smoke: pmac/experiments/atari_smoke.py).
"""

__version__ = "0.1.0"
