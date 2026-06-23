#!/usr/bin/env bash
set -e
source /opt/venv/bin/activate
cd /mnt/c/Users/Asav/source/repos/Praxis
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=1
export XLA_PYTHON_CLIENT_MEM_FRACTION=0.9
for G in Breakout-v5 Pong-v5 SpaceInvaders-v5 Seaquest-v5 BeamRider-v5; do
  echo "=== REF $G ==="
  python baseline_ppo.py --game "$G" --total-steps 2000000 --num-envs 32 --eval-episodes 20 \
    --out-dir "castm_runs/newset/refs/$G" >> "castm_runs/logs/newset_ref_${G}.log" 2>&1
done
echo "=== NAIVE control (full finetune, no memory) ==="
python -m tfns.castm.train_castm \
  --games Breakout-v5 Pong-v5 SpaceInvaders-v5 Seaquest-v5 BeamRider-v5 \
  --steps-per-game 1500000 --num-envs 32 --eval-episodes 20 --seed 1 --naive --no-inferred-eval \
  --out-dir castm_runs/newset/naive_seed1 >> castm_runs/logs/newset_naive.log 2>&1
echo "REFS+NAIVE DONE"
