#!/usr/bin/env bash
set -e
source /opt/venv/bin/activate
cd /mnt/c/Users/Asav/source/repos/Praxis
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
echo "=== PLASTICITY PROBE: scratch x4 on Alien->Asterix->Tennis ==="
python -m tfns.castm.train_castm --games Alien-v5 Asterix-v5 Tennis-v5 \
  --steps-per-game 2000000 --num-envs 32 --eval-episodes 20 --seed 1 --scratch-mult 4.0 \
  --out-dir castm_runs/oracle/probe_rank4 >> castm_runs/logs/probe_rank4.log 2>&1
echo "PROBE DONE"
