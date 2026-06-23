#!/usr/bin/env bash
set -e
source /opt/venv/bin/activate
cd /mnt/c/Users/Asav/source/repos/Praxis
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
for G in Alien-v5 Defender-v5 Asterix-v5 Tennis-v5 Phoenix-v5; do
  echo "=== REF $G ==="
  python baseline_ppo.py --game "$G" --total-steps 2000000 --num-envs 32 --eval-episodes 20 \
    --out-dir "castm_runs/refs/$G" >> "castm_runs/logs/ref_${G}.log" 2>&1
done
echo "ALL REFS DONE"
