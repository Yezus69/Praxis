#!/usr/bin/env bash
set -e
source /opt/venv/bin/activate
cd /mnt/c/Users/Asav/source/repos/Praxis
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
echo "=== PLASTIC CASTM (no frozen weights), new set incl Breakout ==="
python -m tfns.castm.train_plastic \
  --games Breakout-v5 Pong-v5 SpaceInvaders-v5 Seaquest-v5 BeamRider-v5 \
  --steps-per-game 1500000 --num-envs 32 --eval-episodes 20 --seed 1 --mem-rank 64 \
  --out-dir castm_runs/newset/plastic_seed1 >> castm_runs/logs/newset_plastic.log 2>&1
echo "PLASTIC DONE"
