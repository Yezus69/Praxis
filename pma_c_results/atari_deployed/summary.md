# Continual Full-ALE Atari — does PMA-C stop forgetting?

Games (sequential): Breakout-v5, SpaceInvaders-v5, BeamRider-v5, Asterix-v5, Qbert-v5

## 1. PRIMARY (falsifiable): shared mutable-net retention — does the conservation guard reduce forgetting?

Random-normalized `norm_retention` from the return matrix (same greedy eval protocol for every arm), per-game clipped to [0,1] then averaged. Higher = the *single shared net* forgot less. This is the apples-to-apples learning result. **'overwritten' column excludes the never-overwritten last game** (the clean forgetting measure). 'all' is mean+-SAMPLE-std across seeds.

> CAVEAT: n=2 seed(s). At n<=2 this is SUGGESTIVE (consistent direction), NOT a significance claim; treat the conservation effect as directional until n>=3 with a paired test / bootstrap CI. The DEPLOYED floor (Result 2) is structural and n-independent.

| arm | shared-net retention (all games) | shared-net retention (overwritten only) | worst game | mean final return |
|---|---|---|---|---|
| baseline | 0.342 ± 0.156 | 0.177 ± 0.196 | 0.006 ± 0.008 | 337.888 ± 98.907 |
| pmac_champions_only | 0.352 ± 0.025 | 0.190 ± 0.032 | 0.024 ± 0.034 | 354.375 ± 10.076 |
| pmac | 0.759 ± 0.004 | 0.699 ± 0.005 | 0.384 ± 0.127 | 602.288 ± 196.558 |

## 2. SECONDARY (structural): deployed champion-routing floor

With default safety routing the deployed agent serves each protected skill from its frozen certified champion, so **deployed_retention = 1.0 BY CONSTRUCTION** (deployed≡champion≡best). This is the architectural no-forgetting invariant (≡ certified per-task checkpointing + router), NOT a product of the conservation loss — the `champions_only` arm (conservation OFF) also reaches 1.0. The baseline has no champion store, so its deployed agent is the single mutable net and it forgets.

| arm | has champion store? | deployed floor mean | deployed floor worst |
|---|---|---|---|
| baseline | no (single net) | 0.332 ± 0.119 | 0.000 ± 0.000 |
| pmac_champions_only | yes | 1.000 ± 0.000 | 1.000 ± 0.000 |
| pmac | yes | 1.000 ± 0.000 | 1.000 ± 0.000 |

## 3. Plasticity — do new games still learn? (champions_only uses the IDENTICAL training procedure as baseline, conservation OFF)

| arm | per-game learned (peak) mean over seeds | ratio vs baseline |
|---|---|---|
| baseline | 575.4 | 1.000 |
| pmac_champions_only | 683.2 | 1.187 |
| pmac | 649.5 | 1.129 |

(Both champion arms learn new games at >= baseline level on average — ratio >= ~1.0 — so neither the champion/deployed guarantee nor the conservation guard sacrifices plasticity. champions_only uses the identical training procedure as baseline (conservation off); per-game scores differ only by GPU/envpool nondeterminism over 4M steps, not bit-identical. Per-game scores below.)

## Per-game (baseline) — mean over seeds

| game | learned | final(shared) | shared norm_ret | champion | deployed | deployed_ret(floor) | route |
|---|---|---|---|---|---|---|---|
| Breakout-v5 | 15.8 | 1.3 | 0.007 | 1.3 | 1.3 | 0.034 | current |
| SpaceInvaders-v5 | 430.6 | 133.8 | 0.118 | 203.5 | 203.5 | 0.038 | current |
| BeamRider-v5 | 690.0 | 645.0 | 0.730 | 625.2 | 625.2 | 0.500 | current |
| Asterix-v5 | 937.5 | 106.2 | 0.044 | 133.3 | 133.3 | 0.086 | current |
| Qbert-v5 | 803.1 | 803.1 | 1.000 | 1719.8 | 1719.8 | 1.000 | current |

## Per-game (pmac_champions_only) — mean over seeds

| game | learned | final(shared) | shared norm_ret | champion | deployed | deployed_ret(floor) | route |
|---|---|---|---|---|---|---|---|
| Breakout-v5 | 28.6 | 1.5 | 0.024 | 37.4 | 37.4 | 1.000 | champion |
| SpaceInvaders-v5 | 459.4 | 240.6 | 0.324 | 515.2 | 515.2 | 1.000 | champion |
| BeamRider-v5 | 972.0 | 561.0 | 0.309 | 959.7 | 959.7 | 1.000 | champion |
| Asterix-v5 | 1181.2 | 193.8 | 0.101 | 1164.6 | 1164.6 | 1.000 | champion |
| Qbert-v5 | 775.0 | 775.0 | 1.000 | 776.0 | 776.0 | 1.000 | champion |

## Per-game (pmac) — mean over seeds

| game | learned | final(shared) | shared norm_ret | champion | deployed | deployed_ret(floor) | route |
|---|---|---|---|---|---|---|---|
| Breakout-v5 | 18.7 | 14.9 | 0.594 | 22.4 | 22.4 | 1.000 | champion |
| SpaceInvaders-v5 | 316.2 | 227.5 | 0.481 | 370.4 | 370.4 | 1.000 | champion |
| BeamRider-v5 | 612.5 | 594.0 | 1.073 | 648.0 | 648.0 | 1.000 | champion |
| Asterix-v5 | 1168.8 | 1043.8 | 0.884 | 1116.7 | 1116.7 | 1.000 | champion |
| Qbert-v5 | 1131.2 | 1131.2 | 1.000 | 1102.1 | 1102.1 | 1.000 | champion |
