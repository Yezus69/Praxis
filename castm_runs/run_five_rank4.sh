#!/usr/bin/env bash
set -e
source /opt/venv/bin/activate
cd /mnt/c/Users/Asav/source/repos/Praxis
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
echo "=== 5-game scratch x4 (order1) ==="
python -m tfns.castm.train_castm --games Alien-v5 Defender-v5 Asterix-v5 Tennis-v5 Phoenix-v5 \
  --steps-per-game 1500000 --num-envs 32 --eval-episodes 20 --seed 1 --scratch-mult 4.0 \
  --out-dir castm_runs/oracle/five_rank4 >> castm_runs/logs/five_rank4.log 2>&1
echo "FIVE RANK4 DONE"
