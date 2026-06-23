# CASTM Atari Ladder Results

## Matched single-task references (stochastic eval)

| Game | reference best | reference final |
|---|---|---|
| Alien-v5 | 561.500 | 517.000 |
| Asterix-v5 | 975.000 | 882.500 |
| Defender-v5 | 7855.000 | 5450.000 |
| Phoenix-v5 | 3149.000 | 3149.000 |
| Tennis-v5 | -4.350 | -8.500 |

## Run: `castm_runs/oracle/five_seed1`

| Game | S_rand | S_single | S_best | S_final | Progress | Retention | Forgetting |
|---|---|---|---|---|---|---|---|
| Alien-v5 | 173.000 | 561.500 | 495.000 | 495.000 | 0.829 | 1.000 | 0.000 |
| Defender-v5 | 2927.500 | 7855.000 | 9742.500 | 9742.500 | 1.383 | 1.000 | 0.000 |
| Asterix-v5 | 230.000 | 975.000 | 497.500 | 497.500 | 0.359 | 1.000 | 0.000 |
| Tennis-v5 | -23.850 | -4.350 | -11.100 | -11.100 | 0.654 | 1.000 | 0.000 |
| Phoenix-v5 | 988.500 | 3149.000 | 3187.000 | 3187.000 | 1.018 | 1.000 | 0.000 |

- min progress = 0.359, min retention = 1.000, current progress = 1.018
- **Gate 21.4 (five-game):** min_P=0.359 (>=0.90: False); or min_R=1.000 & P_cur=1.018 (True)

### Oracle vs inferred address (spec 24)
| Game | oracle | inferred | route acc |
|---|---|---|---|
| Alien-v5 | 495.000 | 529.000 | 0.459 |
| Defender-v5 | 9742.500 | 7955.000 | 0.999 |
| Asterix-v5 | 497.500 | 322.500 | 1.000 |
| Tennis-v5 | -11.100 | -12.300 | 1.000 |
| Phoenix-v5 | 3187.000 | 1868.500 | 0.840 |
