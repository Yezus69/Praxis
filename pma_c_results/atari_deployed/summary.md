# Continual Full-ALE Atari — does PMA-C stop forgetting?

Games (sequential): Breakout-v5, SpaceInvaders-v5, BeamRider-v5, Asterix-v5, Qbert-v5

## 1. PRIMARY (falsifiable): shared mutable-net retention — does the conservation guard reduce forgetting?

Random-normalized `norm_retention` from the return matrix (same greedy eval protocol for every arm), per-game clipped to [0,1] then averaged. Higher = the *single shared net* forgot less. This is the apples-to-apples learning result. **'overwritten' column excludes the never-overwritten last game** (the clean forgetting measure). 'all' is mean+-SAMPLE-std across seeds.

> CAVEAT: n=4 seeds (small sample). Read the paired-by-seed significance table below (per-seed sign count + 90% bootstrap CI), not a p-value. The cleanest conservation isolation is pmac vs champions_only (identical training procedure +/- the conservation guard). The DEPLOYED floor (Result 2) is structural and n-independent (1.0 by construction).

| arm | shared-net retention (all games) | shared-net retention (overwritten only) | worst game | mean final return |
|---|---|---|---|---|
| baseline | 0.375 ± 0.098 | 0.219 ± 0.123 | 0.003 ± 0.006 | 367.638 ± 80.035 |
| pmac_champions_only | 0.301 ± 0.063 | 0.126 ± 0.078 | 0.012 ± 0.024 | 322.475 ± 41.707 |
| pmac | 0.676 ± 0.176 | 0.595 ± 0.219 | 0.280 ± 0.201 | 537.856 ± 136.616 |

### Significance of the conservation effect (paired by seed, shared-net retention)

| contrast | metric | per-seed diffs | mean diff | 90% bootstrap CI | seeds with diff>0 |
|---|---|---|---|---|---|
| pmac − baseline | all 5 | +0.310, +0.525, +0.373, -0.003 | +0.301 | [+0.129, +0.449] | 3/4 |
| pmac − baseline | overwritten 4 | +0.387, +0.656, +0.466, -0.004 | +0.376 | [+0.161, +0.561] | 3/4 |
| pmac − pmac_champions_only | all 5 | +0.392, +0.422, +0.503, +0.183 | +0.375 | [+0.263, +0.463] | 4/4 |
| pmac − pmac_champions_only | overwritten 4 | +0.491, +0.528, +0.629, +0.229 | +0.469 | [+0.329, +0.579] | 4/4 |

(Positive = conservation retains more. At small n read the CI and the sign count, not a p-value; a consistent positive sign across all seeds + a CI excluding 0 is the bar.)

## 2. SECONDARY (structural): deployed champion-routing floor

With default safety routing the deployed agent serves each protected skill from its frozen certified champion, so **deployed_retention = 1.0 BY CONSTRUCTION** (deployed≡champion≡best). This is the architectural no-forgetting invariant (≡ certified per-task checkpointing + router), NOT a product of the conservation loss — the `champions_only` arm (conservation OFF) also reaches 1.0. The baseline has no champion store, so its deployed agent is the single mutable net and it forgets.

| arm | has champion store? | deployed floor mean | deployed floor worst |
|---|---|---|---|
| baseline | no (single net) | 0.365 ± 0.080 | 0.000 ± 0.000 |
| pmac_champions_only | yes | 1.000 ± 0.000 | 1.000 ± 0.000 |
| pmac | yes | 1.000 ± 0.000 | 1.000 ± 0.000 |

## 3. Plasticity — do new games still learn? (champions_only uses the IDENTICAL training procedure as baseline, conservation OFF)

| arm | per-game learned (peak) mean over seeds | ratio vs baseline |
|---|---|---|
| baseline | 578.6 | 1.000 |
| pmac_champions_only | 661.0 | 1.142 |
| pmac | 570.7 | 0.986 |

(Both champion arms learn new games at >= baseline level on average — ratio >= ~1.0 — so neither the champion/deployed guarantee nor the conservation guard sacrifices plasticity. champions_only uses the identical training procedure as baseline (conservation off); per-game scores differ only by GPU/envpool nondeterminism over 4M steps, not bit-identical. Per-game scores below.)

## Per-game (baseline) — mean over seeds

| game | learned | final(shared) | shared norm_ret | champion | deployed | deployed_ret(floor) | route |
|---|---|---|---|---|---|---|---|
| Breakout-v5 | 19.5 | 0.8 | 0.003 | 0.9 | 0.9 | 0.017 | current |
| SpaceInvaders-v5 | 432.8 | 146.6 | 0.059 | 180.6 | 180.6 | 0.019 | current |
| BeamRider-v5 | 673.5 | 670.5 | 0.876 | 653.6 | 653.6 | 0.580 | current |
| Asterix-v5 | 971.9 | 225.0 | 0.156 | 281.3 | 281.3 | 0.207 | current |
| Qbert-v5 | 795.3 | 795.3 | 1.000 | 1257.3 | 1257.3 | 1.000 | current |

## Per-game (pmac_champions_only) — mean over seeds

| game | learned | final(shared) | shared norm_ret | champion | deployed | deployed_ret(floor) | route |
|---|---|---|---|---|---|---|---|
| Breakout-v5 | 29.8 | 1.3 | 0.022 | 36.0 | 36.0 | 1.000 | champion |
| SpaceInvaders-v5 | 448.1 | 160.3 | 0.162 | 486.1 | 486.1 | 1.000 | champion |
| BeamRider-v5 | 902.0 | 407.0 | 0.154 | 1006.7 | 1006.7 | 1.000 | champion |
| Asterix-v5 | 1140.6 | 259.4 | 0.165 | 1282.3 | 1282.3 | 1.000 | champion |
| Qbert-v5 | 784.4 | 784.4 | 1.000 | 758.9 | 758.9 | 1.000 | champion |

## Per-game (pmac) — mean over seeds

| game | learned | final(shared) | shared norm_ret | champion | deployed | deployed_ret(floor) | route |
|---|---|---|---|---|---|---|---|
| Breakout-v5 | 23.8 | 15.5 | 0.561 | 27.8 | 27.8 | 1.000 | champion |
| SpaceInvaders-v5 | 296.2 | 181.2 | 0.329 | 352.7 | 352.7 | 1.000 | champion |
| BeamRider-v5 | 460.2 | 616.0 | 0.912 | 567.8 | 567.8 | 1.000 | champion |
| Asterix-v5 | 1181.2 | 984.4 | 0.822 | 1194.8 | 1194.8 | 1.000 | champion |
| Qbert-v5 | 892.2 | 892.2 | 1.000 | 875.0 | 875.0 | 1.000 | champion |
