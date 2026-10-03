#!/bin/bash
# 验证"真实Reacher v-probe在chance、branch-sep却很大"这个表征/行为分裂现象
# 是否是数据集本身i.i.d.随机力矩导致qvel快速去相关造成的(the results log r16的机制假说)。
# 用frameskip=1（不降采样，帧间dt最短，理论上qvel去相关最少）重建cache，
# 用完全相同的官方规模配方重训三臂，重跑Protocol A看v-probe是否明显回升。
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
export OMP_NUM_THREADS=4 MKL_NUM_THREADS=4
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PYTHON:-python3}"
CKPT=checkpoints_reacher_fs1
LOGS=logs_reacher_fs1
CACHE=cache/reacher_cache_fs1.npz

STEPS=5000
BS=64
LR=5e-5
LAMBDA_REG=5.0
LAMBDA_V=1.0
WD=1e-3
GC=1.0

echo "[$(date)] === 1/3 reacher(fs1) baseline k=3 ==="
$PY -u reacher_train_official.py --model baseline --k 3 --cache $CACHE \
  --steps $STEPS --batch_size $BS --lr $LR --lambda_reg $LAMBDA_REG --weight_decay $WD --grad_clip $GC \
  --out $CKPT/reacher_baseline_k3.pt 2>&1 | tee "$LOGS/reacher_baseline_k3.log"

echo "[$(date)] === 2/3 reacher(fs1) baseline k=1 ==="
$PY -u reacher_train_official.py --model baseline --k 1 --cache $CACHE \
  --steps $STEPS --batch_size $BS --lr $LR --lambda_reg $LAMBDA_REG --weight_decay $WD --grad_clip $GC \
  --out $CKPT/reacher_baseline_k1.pt 2>&1 | tee "$LOGS/reacher_baseline_k1.log"

echo "[$(date)] === 3/3 reacher(fs1) TI-JEPA ==="
$PY -u reacher_train_official.py --model tijepa --k 3 --cache $CACHE \
  --steps $STEPS --batch_size $BS --lr $LR --lambda_reg $LAMBDA_REG --lambda_v $LAMBDA_V \
  --weight_decay $WD --grad_clip $GC \
  --out $CKPT/reacher_tijepa.pt 2>&1 | tee "$LOGS/reacher_tijepa.log"

echo "[$(date)] === Protocol A (probe ladder) w/ fs1 cache ==="
$PY -u reacher_probe_ladder_official.py \
  --baseline_k3_ckpt $CKPT/reacher_baseline_k3.pt \
  --baseline_k1_ckpt $CKPT/reacher_baseline_k1.pt \
  --tijepa_ckpt $CKPT/reacher_tijepa.pt \
  --cache $CACHE \
  --out results/reacher_fs1/probe_ladder.json 2>&1 | tee "$LOGS/probe_ladder.log"

echo "[$(date)] === reacher(fs1) decorrelation-hypothesis test ALL DONE ==="
