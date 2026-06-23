#!/usr/bin/env bash
set -e
source /opt/venv/bin/activate
cd /mnt/c/Users/Asav/source/repos/Praxis
GPU="$1"; SEED="$2"; OUT="$3"; RANK="${4:-64}"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
python -m tfns.castm.train_plastic \
  --games Breakout-v5 Pong-v5 SpaceInvaders-v5 Seaquest-v5 BeamRider-v5 \
  --steps-per-game 1500000 --num-envs 32 --eval-episodes 20 --seed "$SEED" --mem-rank "$RANK" \
  --out-dir "$OUT" >> "castm_runs/logs/$(basename $OUT).log" 2>&1
echo "DONE $OUT"
