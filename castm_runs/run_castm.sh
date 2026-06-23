#!/usr/bin/env bash
set -e
source /opt/venv/bin/activate
cd /mnt/c/Users/Asav/source/repos/Praxis
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
echo "=== STAGE C: oracle 2-game seed1 ==="
python -m tfns.castm.train_castm --games Alien-v5 Defender-v5 \
  --steps-per-game 2000000 --num-envs 32 --eval-episodes 20 --seed 1 \
  --out-dir castm_runs/oracle/seed1 >> castm_runs/logs/oracle_seed1.log 2>&1
echo "=== STAGE E: oracle 5-game pilot seed1 ==="
python -m tfns.castm.train_castm --games Alien-v5 Defender-v5 Asterix-v5 Tennis-v5 Phoenix-v5 \
  --steps-per-game 1500000 --num-envs 32 --eval-episodes 20 --seed 1 \
  --out-dir castm_runs/oracle/five_seed1 >> castm_runs/logs/oracle_five_seed1.log 2>&1
echo "=== STAGE C replication: oracle 2-game seed2 ==="
python -m tfns.castm.train_castm --games Alien-v5 Defender-v5 \
  --steps-per-game 2000000 --num-envs 32 --eval-episodes 20 --seed 2 \
  --out-dir castm_runs/oracle/seed2 >> castm_runs/logs/oracle_seed2.log 2>&1
echo "ALL CASTM DONE"
