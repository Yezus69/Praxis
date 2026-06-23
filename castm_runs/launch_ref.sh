#!/usr/bin/env bash
# usage: launch_ref.sh GAME GPU TOTAL_STEPS
set -e
source /opt/venv/bin/activate
cd /mnt/c/Users/Asav/source/repos/Praxis
GAME="$1"; GPU="$2"; STEPS="$3"
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES="$GPU"
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
python baseline_ppo.py --game "$GAME" --total-steps "$STEPS" --num-envs 32 --eval-episodes 20 --out-dir "castm_runs/refs/$GAME" > "castm_runs/logs/ref_${GAME}.log" 2>&1
