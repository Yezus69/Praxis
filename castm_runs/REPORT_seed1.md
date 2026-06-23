# CASTM Atari Ladder Results

## Matched single-task references (stochastic eval)

| Game | reference best | reference final |
|---|---|---|
| Alien-v5 | 561.500 | 517.000 |
| Asterix-v5 | 975.000 | 882.500 |
| Defender-v5 | 7855.000 | 5450.000 |

## Run: `castm_runs/oracle/seed1`

| Game | S_rand | S_single | S_best | S_final | Progress | Retention | Forgetting |
|---|---|---|---|---|---|---|---|
| Alien-v5 | 173.000 | 561.500 | 564.500 | 564.500 | 1.008 | 1.000 | 0.000 |
| Defender-v5 | 2927.500 | 7855.000 | 7090.000 | 7090.000 | 0.845 | 1.000 | 0.000 |

- min progress = 0.845, min retention = 1.000, current progress = 0.845
- **Gate 21.2 (oracle 2-game):** P2=0.845 (>=0.90: False), R1=1.000 (>=0.90: True)
- **Gate 21.4 (five-game):** min_P=0.845 (>=0.90: False); or min_R=1.000 & P_cur=0.845 (False)
