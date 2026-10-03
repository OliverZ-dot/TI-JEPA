#!/bin/bash
# Official-scale (ViT-Tiny/14 + AdaLN transformer) training on REAL dm_control
# Reacher pixels, 3 arms run strictly sequentially (single GPU, ~20-24GB/proc).
# Same validated hyperparameter recipe as scripts/run_official_scale_cartpole.sh
# (AdamW, lr=5e-5, wd=1e-3, grad_clip=1.0, lambda_reg=5.0, lambda_v=1.0, steps=5000).
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PYTHON:-python3}"
CKPT=checkpoints_reacher_official
LOGS=logs_reacher_official
mkdir -p "$CKPT" "$LOGS" results/reacher_official

STEPS=5000
BS=64
LR=5e-5
LAMBDA_REG=5.0
LAMBDA_V=1.0
WD=1e-3
GC=1.0

echo "[$(date)] === 1/3 reacher baseline k=3 ==="
$PY -u reacher_train_official.py --model baseline --k 3 \
  --steps $STEPS --batch_size $BS --lr $LR --lambda_reg $LAMBDA_REG --weight_decay $WD --grad_clip $GC \
  --out $CKPT/reacher_baseline_k3.pt 2>&1 | tee "$LOGS/reacher_baseline_k3.log"

echo "[$(date)] === 2/3 reacher baseline k=1 (memoryless ablation) ==="
$PY -u reacher_train_official.py --model baseline --k 1 \
  --steps $STEPS --batch_size $BS --lr $LR --lambda_reg $LAMBDA_REG --weight_decay $WD --grad_clip $GC \
  --out $CKPT/reacher_baseline_k1.pt 2>&1 | tee "$LOGS/reacher_baseline_k1.log"

echo "[$(date)] === 3/3 reacher TI-JEPA ==="
$PY -u reacher_train_official.py --model tijepa --k 3 \
  --steps $STEPS --batch_size $BS --lr $LR --lambda_reg $LAMBDA_REG --lambda_v $LAMBDA_V \
  --weight_decay $WD --grad_clip $GC \
  --out $CKPT/reacher_tijepa.pt 2>&1 | tee "$LOGS/reacher_tijepa.log"

echo "[$(date)] === reacher training ALL DONE ==="
