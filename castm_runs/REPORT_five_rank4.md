# CASTM Atari Ladder Results

## Matched single-task references (stochastic eval)

| Game | reference best | reference final |
|---|---|---|
| Alien-v5 | 561.500 | 517.000 |
| Asterix-v5 | 975.000 | 882.500 |
| Defender-v5 | 7855.000 | 5450.000 |
| Phoenix-v5 | 3149.000 | 3149.000 |
| Tennis-v5 | -4.350 | -8.500 |

## Run: `castm_runs/oracle/five_rank4`

| Game | S_rand | S_single | S_best | S_final | Progress | Retention | Forgetting |
|---|---|---|---|---|---|---|---|
| Alien-v5 | 173.000 | 561.500 | 523.000 | 523.000 | 0.901 | 1.000 | 0.000 |
| Defender-v5 | 2927.500 | 7855.000 | 9877.500 | 9877.500 | 1.410 | 1.000 | 0.000 |
| Asterix-v5 | 230.000 | 975.000 | 1037.500 | 1037.500 | 1.084 | 1.000 | 0.000 |
| Tennis-v5 | -23.850 | -4.350 | -8.950 | -8.950 | 0.764 | 1.000 | 0.000 |
| Phoenix-v5 | 988.500 | 3149.000 | 2934.500 | 2934.500 | 0.901 | 1.000 | 0.000 |

- min progress = 0.764, min retention = 1.000, current progress = 0.901
- **Gate 21.4 (five-game):** min_P=0.764 (>=0.90: False); or min_R=1.000 & P_cur=0.901 (True)

### Oracle vs inferred address (spec 24)
| Game | oracle | inferred | route acc |
|---|---|---|---|
| Alien-v5 | 523.000 | 481.000 | 0.528 |
| Defender-v5 | 9877.500 | 9255.000 | 0.998 |
| Asterix-v5 | 1037.500 | 990.000 | 1.000 |
| Tennis-v5 | -8.950 | -8.850 | 1.000 |
| Phoenix-v5 | 2934.500 | 2472.000 | 0.870 |
