#!/bin/bash
# Multi-seed ablation sweep, GPU2 half. Runs jobs sequentially; each writes its own JSON.
set -u
cd /mnt/c/Users/Asav/source/repos/Praxis
source /opt/venv/bin/activate
mkdir -p /root/sweep
export CUDA_DEVICE_ORDER=PCI_BUS_ID
export CUDA_VISIBLE_DEVICES=2
export PYTHONPATH=.
export TF_CPP_MIN_LOG_LEVEL=3
export GAMES=SpaceInvaders-v5,Breakout-v5,BeamRider-v5,Asterix-v5,Qbert-v5
export PER_GAME=800000 NBLOCKS=1 NENVS=256 STOCH=1 EVAL_EP=48
JOBS=("plain_ppo 0" "plain_ppo 1" "plain_ppo 2" "no_memory_read 2")
for j in "${JOBS[@]}"; do
  set -- $j; ABL=$1; SD=$2
  OUT=/root/sweep/${ABL}_s${SD}.json
  echo "[gpu2] START $ABL seed=$SD -> $OUT"
  ABLATION=$ABL SEED=$SD RESULT_PATH=$OUT timeout 5400 python -u .codex/proof_run.py > /root/sweep/${ABL}_s${SD}.log 2>&1
  echo "[gpu2] DONE $ABL seed=$SD rc=$?"
done
echo "[gpu2] SWEEP COMPLETE" > /root/sweep/gpu2_done.txt
