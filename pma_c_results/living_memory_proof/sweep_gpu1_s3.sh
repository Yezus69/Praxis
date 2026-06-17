#!/bin/bash
set -u
cd /mnt/c/Users/Asav/source/repos/Praxis; source /opt/venv/bin/activate; mkdir -p /root/sweep
export CUDA_DEVICE_ORDER=PCI_BUS_ID CUDA_VISIBLE_DEVICES=1 PYTHONPATH=. TF_CPP_MIN_LOG_LEVEL=3
export GAMES=SpaceInvaders-v5,Breakout-v5,BeamRider-v5,Asterix-v5,Qbert-v5
export PER_GAME=800000 NBLOCKS=1 NENVS=256 STOCH=1 EVAL_EP=24
JOBS=("full 3" "no_memory_read 3" "plain_ppo 3")
for j in "${JOBS[@]}"; do
  set -- $j; ABL=$1; SD=$2; OUT=/root/sweep/${ABL}_s${SD}.json
  echo "[gpu1s3] START $ABL seed=$SD $(date +%H:%M)"
  ABLATION=$ABL SEED=$SD RESULT_PATH=$OUT timeout 7200 python -u .codex/proof_run.py > /root/sweep/${ABL}_s${SD}.log 2>&1
  echo "[gpu1s3] DONE $ABL seed=$SD rc=$? $(date +%H:%M)"
done
echo "COMPLETE $(date)" > /root/sweep/gpu1s3_done.txt
