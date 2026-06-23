# CASTM Atari Ladder Results

## Matched single-task references (stochastic eval)

| Game | reference best | reference final |
|---|---|---|
| Alien-v5 | 561.500 | 517.000 |
| Asterix-v5 | 975.000 | 882.500 |
| Defender-v5 | 7855.000 | 5450.000 |
| Phoenix-v5 | 3149.000 | 3149.000 |
| Tennis-v5 | -4.350 | -8.500 |

## Run: `castm_runs/oracle/seed1`

| Game | S_rand | S_single | S_best | S_final | Progress | Retention | Forgetting |
|---|---|---|---|---|---|---|---|
| Alien-v5 | 173.000 | 561.500 | 564.500 | 564.500 | 1.008 | 1.000 | 0.000 |
| Defender-v5 | 2927.500 | 7855.000 | 7090.000 | 7090.000 | 0.845 | 1.000 | 0.000 |

- min progress = 0.845, min retention = 1.000, current progress = 0.845
- **Gate 21.2 (oracle 2-game):** P2=0.845 (>=0.90: False), R1=1.000 (>=0.90: True)
- **Gate 21.4 (five-game):** min_P=0.845 (>=0.90: False); or min_R=1.000 & P_cur=0.845 (False)

## Run: `castm_runs/oracle/seed3`

| Game | S_rand | S_single | S_best | S_final | Progress | Retention | Forgetting |
|---|---|---|---|---|---|---|---|
| Alien-v5 | 180.667 | 561.500 | 435.000 | 435.000 | 0.668 | 1.000 | 0.000 |
| Defender-v5 | 3090.000 | 7855.000 | 7186.667 | 7186.667 | 0.860 | 1.000 | 0.000 |

- min progress = 0.668, min retention = 1.000, current progress = 0.860
- **Gate 21.2 (oracle 2-game):** P2=0.860 (>=0.90: False), R1=1.000 (>=0.90: True)
- **Gate 21.3 (inferred 2-game):** routing_acc=1.000 (>=0.99: True), P2>=0.90: True
- **Gate 21.4 (five-game):** min_P=0.668 (>=0.90: False); or min_R=1.000 & P_cur=0.860 (False)

### Oracle vs inferred address (spec 24)
| Game | oracle | inferred | route acc |
|---|---|---|---|
| Alien-v5 | 435.000 | 360.667 | 0.889 |
| Defender-v5 | 7186.667 | 7946.667 | 1.000 |
