## Multi-seed ablation sweep (5 games, 800k/game, stochastic eval, retention over LEARNED games only)

| ablation | seeds | mean retention (over learned games) | per-seed means |
|---|---|---|---|
| full | 3 | 0.754 ± 0.181 | [0.57, 1.0, 0.691] |
| no_memory_read | 2 | 0.841 ± 0.133 | [0.974, 0.708] |
| plain_ppo | 3 | 0.664 ± 0.101 | [0.535, 0.781, 0.676] |

**full** per-game retention (mean over seeds, learned games): BeamRider=0.60(n3), Qbert=1.00(n3), SpaceInvaders=0.50(n2)
**no_memory_read** per-game retention (mean over seeds, learned games): BeamRider=0.68(n2), Qbert=1.00(n2)
**plain_ppo** per-game retention (mean over seeds, learned games): BeamRider=0.33(n3), Qbert=1.00(n3)