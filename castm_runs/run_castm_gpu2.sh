#!/usr/bin/env bash
set -e
source /opt/venv/bin/activate
cd /mnt/c/Users/Asav/source/repos/Praxis
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
echo "=== STAGE C/D: oracle 2-game seed3, 30-episode eval + inferred ==="
python -m tfns.castm.train_castm --games Alien-v5 Defender-v5 \
  --steps-per-game 2000000 --num-envs 32 --eval-episodes 30 --seed 3 \
  --out-dir castm_runs/oracle/seed3 >> castm_runs/logs/oracle_seed3.log 2>&1
echo "=== STAGE E (order 2): 5-game pilot, Asterix-first ==="
python -m tfns.castm.train_castm --games Asterix-v5 Phoenix-v5 Defender-v5 Alien-v5 Tennis-v5 \
  --steps-per-game 1500000 --num-envs 32 --eval-episodes 20 --seed 1 \
  --out-dir castm_runs/oracle/five_order2 >> castm_runs/logs/oracle_five_order2.log 2>&1
echo "GPU2 LADDER DONE"
