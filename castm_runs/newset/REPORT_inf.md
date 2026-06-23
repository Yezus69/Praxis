# CASTM Atari Ladder Results

## Matched single-task references (stochastic eval)

| Game | reference best | reference final |
|---|---|---|
| BeamRider-v5 | 728.600 | 596.400 |
| Breakout-v5 | 16.500 | 16.050 |
| Pong-v5 | -19.450 | -21.000 |
| Seaquest-v5 | 834.000 | 826.000 |
| SpaceInvaders-v5 | 456.250 | 456.250 |

## Run: `castm_runs/newset/plastic_inf_seed1`

| Game | S_rand | S_single | S_best | S_final | Progress | Retention | Forgetting |
|---|---|---|---|---|---|---|---|
| Breakout-v5 | 1.100 | 16.500 | 12.250 | 11.200 | 0.656 | 0.906 | 0.068 |
| Pong-v5 | -20.300 | -19.450 | -21.000 | -21.000 | -0.824 | 1.000 | 0.000 |
| SpaceInvaders-v5 | 110.750 | 456.250 | 205.750 | 201.750 | 0.263 | 0.958 | 0.012 |
| Seaquest-v5 | 58.000 | 834.000 | 787.000 | 763.000 | 0.909 | 0.967 | 0.031 |
| BeamRider-v5 | 481.800 | 728.600 | 582.600 | 582.600 | 0.408 | 1.000 | 0.000 |

- min progress = -0.824, min retention = 0.906, current progress = 0.408
- **Gate 21.4 (five-game):** min_P=-0.824 (>=0.90: False); or min_R=0.906 & P_cur=0.408 (False)

### Oracle vs inferred address (spec 24)
| Game | oracle | inferred | route acc |
|---|---|---|---|
| Breakout-v5 | 11.200 | 13.000 | 1.000 |
| Pong-v5 | -21.000 | -21.000 | 1.000 |
| SpaceInvaders-v5 | 201.750 | 234.250 | 0.983 |
| Seaquest-v5 | 763.000 | 808.000 | 1.000 |
| BeamRider-v5 | 582.600 | 591.800 | 0.993 |

## Run: `castm_runs/newset/plastic_inf_seed2`

| Game | S_rand | S_single | S_best | S_final | Progress | Retention | Forgetting |
|---|---|---|---|---|---|---|---|
| Breakout-v5 | 1.150 | 16.500 | 5.650 | 5.000 | 0.251 | 0.856 | 0.042 |
| Pong-v5 | -20.250 | -19.450 | -15.100 | -15.100 | 6.438 | 1.000 | 0.000 |
| SpaceInvaders-v5 | 107.250 | 456.250 | 345.250 | 320.000 | 0.610 | 0.894 | 0.072 |
| Seaquest-v5 | 66.000 | 834.000 | 804.000 | 795.000 | 0.949 | 0.988 | 0.012 |
| BeamRider-v5 | 396.400 | 728.600 | 572.400 | 572.400 | 0.530 | 1.000 | 0.000 |

- min progress = 0.251, min retention = 0.856, current progress = 0.530
- **Gate 21.4 (five-game):** min_P=0.251 (>=0.90: False); or min_R=0.856 & P_cur=0.530 (False)

### Oracle vs inferred address (spec 24)
| Game | oracle | inferred | route acc |
|---|---|---|---|
| Breakout-v5 | 5.000 | 5.650 | 1.000 |
| Pong-v5 | -15.100 | -16.400 | 1.000 |
| SpaceInvaders-v5 | 320.000 | 318.750 | 0.917 |
| Seaquest-v5 | 795.000 | 812.000 | 1.000 |
| BeamRider-v5 | 572.400 | 586.600 | 0.977 |
