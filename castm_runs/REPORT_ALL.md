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

## Run: `castm_runs/oracle/five_order2`

| Game | S_rand | S_single | S_best | S_final | Progress | Retention | Forgetting |
|---|---|---|---|---|---|---|---|
| Asterix-v5 | 230.000 | 975.000 | 465.000 | 465.000 | 0.315 | 1.000 | 0.000 |
| Phoenix-v5 | 988.500 | 3149.000 | 2792.500 | 2792.500 | 0.835 | 1.000 | 0.000 |
| Defender-v5 | 2927.500 | 7855.000 | 5995.000 | 5995.000 | 0.623 | 1.000 | 0.000 |
| Alien-v5 | 173.000 | 561.500 | 610.500 | 610.500 | 1.126 | 1.000 | 0.000 |
| Tennis-v5 | -23.850 | -4.350 | -24.000 | -24.000 | -0.008 | 1.000 | 0.000 |

- min progress = -0.008, min retention = 1.000, current progress = -0.008
- **Gate 21.4 (five-game):** min_P=-0.008 (>=0.90: False); or min_R=1.000 & P_cur=-0.008 (False)

### Oracle vs inferred address (spec 24)
| Game | oracle | inferred | route acc |
|---|---|---|---|
| Asterix-v5 | 465.000 | 522.500 | 0.978 |
| Phoenix-v5 | 2792.500 | 1431.500 | 0.656 |
| Defender-v5 | 5995.000 | 5367.500 | 1.000 |
| Alien-v5 | 610.500 | 541.000 | 1.000 |
| Tennis-v5 | -24.000 | -24.000 | 1.000 |
