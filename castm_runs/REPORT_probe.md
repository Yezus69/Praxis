# CASTM Atari Ladder Results

## Matched single-task references (stochastic eval)

| Game | reference best | reference final |
|---|---|---|
| Alien-v5 | 561.500 | 517.000 |
| Asterix-v5 | 975.000 | 882.500 |
| Defender-v5 | 7855.000 | 5450.000 |
| Phoenix-v5 | 3149.000 | 3149.000 |
| Tennis-v5 | -4.350 | -8.500 |

## Run: `castm_runs/oracle/probe_rank4`

| Game | S_rand | S_single | S_best | S_final | Progress | Retention | Forgetting |
|---|---|---|---|---|---|---|---|
| Alien-v5 | 173.000 | 561.500 | 480.500 | 480.500 | 0.792 | 1.000 | 0.000 |
| Asterix-v5 | 230.000 | 975.000 | 1215.000 | 1215.000 | 1.322 | 1.000 | 0.000 |
| Tennis-v5 | -23.850 | -4.350 | -24.000 | -24.000 | -0.008 | 1.000 | 0.000 |

- min progress = -0.008, min retention = 1.000, current progress = -0.008
- **Gate 21.4 (five-game):** min_P=-0.008 (>=0.90: False); or min_R=1.000 & P_cur=-0.008 (False)

### Oracle vs inferred address (spec 24)
| Game | oracle | inferred | route acc |
|---|---|---|---|
| Alien-v5 | 480.500 | 470.500 | 0.592 |
| Asterix-v5 | 1215.000 | 1295.000 | 1.000 |
| Tennis-v5 | -24.000 | -24.000 | 1.000 |
