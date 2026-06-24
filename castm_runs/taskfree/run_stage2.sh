#!/bin/bash
# Stage-2 (3-context alternating stream A->B->C->A->B, unannounced) + missing refs.
# Usage: bash run_stage2.sh <gpu_for_stage2> <gpu_for_refs>
set -e
source /opt/venv/bin/activate
cd /mnt/c/Users/Asav/source/repos/Praxis
export CUDA_DEVICE_ORDER=PCI_BUS_ID XLA_PYTHON_CLIENT_MEM_FRACTION=0.4
G2=${1:-1}; GR=${2:-2}

# Stage-2: SpaceInvaders -> Seaquest -> Breakout -> SpaceInvaders -> Seaquest (revisit SI, Seaquest)
CUDA_VISIBLE_DEVICES=$G2 python -m tfns.castm.train_taskfree \
  --games SpaceInvaders-v5 Seaquest-v5 Breakout-v5 SpaceInvaders-v5 Seaquest-v5 \
  --steps-per-game 300000 --num-envs 32 --eval-episodes 12 --mem-rank 64 --resolve-every 20 \
  --out-dir castm_runs/taskfree/stage2_3ctx > /tmp/stage2.out 2>/tmp/stage2.err &

# Matched 300k reference for Breakout (Stage-2 normalization)
CUDA_VISIBLE_DEVICES=$GR python baseline_ppo.py --game Breakout-v5 --total-steps 300000 \
  --num-envs 32 --eval-episodes 12 --out-dir castm_runs/taskfree/refs300/Breakout-v5 > /tmp/ref_bk300.out 2>&1
CUDA_VISIBLE_DEVICES=$GR python baseline_ppo.py --game SpaceInvaders-v5 --total-steps 300000 \
  --num-envs 32 --eval-episodes 12 --out-dir castm_runs/taskfree/refs300/SpaceInvaders-v5 > /tmp/ref_si300.out 2>&1
CUDA_VISIBLE_DEVICES=$GR python baseline_ppo.py --game Seaquest-v5 --total-steps 300000 \
  --num-envs 32 --eval-episodes 12 --out-dir castm_runs/taskfree/refs300/Seaquest-v5 > /tmp/ref_sq300.out 2>&1
wait
echo "STAGE2_DONE"
