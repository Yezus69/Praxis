# CASTM Atari Ladder Results

## Matched single-task references (stochastic eval)

| Game | reference best | reference final |
|---|---|---|
| BeamRider-v5 | 728.600 | 596.400 |
| Breakout-v5 | 16.500 | 16.050 |
| Pong-v5 | -19.450 | -21.000 |
| Seaquest-v5 | 834.000 | 826.000 |
| SpaceInvaders-v5 | 456.250 | 456.250 |

## Run: `castm_runs/newset/plastic_seed1`

| Game | S_rand | S_single | S_best | S_final | Progress | Retention | Forgetting |
|---|---|---|---|---|---|---|---|
| Breakout-v5 | 1.100 | 16.500 | 10.600 | 10.200 | 0.591 | 0.958 | 0.026 |
| Pong-v5 | -20.300 | -19.450 | -9.550 | -11.300 | 10.588 | 0.837 | 2.059 |
| SpaceInvaders-v5 | 110.750 | 456.250 | 352.000 | 350.500 | 0.694 | 0.994 | 0.004 |
| Seaquest-v5 | 58.000 | 834.000 | 822.000 | 822.000 | 0.985 | 1.000 | 0.000 |
| BeamRider-v5 | 481.800 | 728.600 | 756.600 | 756.600 | 1.113 | 1.000 | 0.000 |

- min progress = 0.591, min retention = 0.837, current progress = 1.113
- **Gate 21.4 (five-game):** min_P=0.591 (>=0.90: False); or min_R=0.837 & P_cur=1.113 (False)

## Run: `castm_runs/newset/naive_seed1`

| Game | S_rand | S_single | S_best | S_final | Progress | Retention | Forgetting |
|---|---|---|---|---|---|---|---|
| Breakout-v5 | 1.100 | 16.500 | 11.800 | 2.050 | 0.062 | 0.089 | 0.633 |
| Pong-v5 | -20.300 | -19.450 | -17.500 | -20.550 | -0.294 | -0.089 | 3.588 |
| SpaceInvaders-v5 | 110.750 | 456.250 | 439.000 | 155.750 | 0.130 | 0.137 | 0.820 |
| Seaquest-v5 | 58.000 | 834.000 | 818.000 | 26.000 | -0.041 | -0.042 | 1.021 |
| BeamRider-v5 | 481.800 | 728.600 | 970.200 | 970.200 | 1.979 | 1.000 | 0.000 |

- min progress = -0.294, min retention = -0.089, current progress = 1.979
- **Gate 21.4 (five-game):** min_P=-0.294 (>=0.90: False); or min_R=-0.089 & P_cur=1.979 (False)
