#!/bin/bash
# 官方规模在 Pendulum 上的复现（同 run_official_scale_cartpole.sh，仅换 --env，
# 超参数已修复见该文件顶部注释：lr=5e-5, lambda_reg=5.0, lambda_v=1.0,
# weight_decay=1e-3, grad_clip=1.0, steps=5000, 外加 official_predictor.py 里
# 补上的 projector/pred_proj BatchNorm 抗坍缩机制）。
# 目的：CartPole 是"探针偏弱"那个环境，Pendulum 是"规划效果最强"(69%/55%closer,
# p<1e-9) 那个环境——两边都测才能说明官方规模的结论不是挑了个方便的环境。
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PYTHON:-python3}"
CKPT=checkpoints/official_scale
LOGS=logs
mkdir -p "$CKPT" "$LOGS" results/official_scale

STEPS=5000
BS=128
LR=5e-5
LAMBDA_REG=5.0
LAMBDA_V=1.0
WD=1e-3
GC=1.0
ENV=pendulum

echo "[$(date)] === 1/3 ${ENV} baseline k=3 ==="
$PY -m ti_jepa.train --env $ENV --model baseline \
  --backbone vit --predictor adaln --image_size 224 --z_dim 192 --feat_dim 192 --k 3 \
  --steps $STEPS --batch_size $BS --lr $LR --lambda_reg $LAMBDA_REG --weight_decay $WD --grad_clip $GC \
  --out $CKPT/${ENV}_baseline_k3.pt 2>&1 | tee "$LOGS/official_${ENV}_baseline_k3.log"

echo "[$(date)] === 2/3 ${ENV} baseline k=1 (memoryless ablation) ==="
$PY -m ti_jepa.train --env $ENV --model baseline \
  --backbone vit --predictor adaln --image_size 224 --z_dim 192 --feat_dim 192 --k 1 \
  --steps $STEPS --batch_size $BS --lr $LR --lambda_reg $LAMBDA_REG --weight_decay $WD --grad_clip $GC \
  --out $CKPT/${ENV}_baseline_k1.pt 2>&1 | tee "$LOGS/official_${ENV}_baseline_k1.log"

echo "[$(date)] === 3/3 ${ENV} TI-JEPA ==="
$PY -m ti_jepa.train --env $ENV --model tijepa \
  --backbone vit --predictor adaln --image_size 224 --q_dim 96 --v_dim 96 --feat_dim 192 --k 3 \
  --steps $STEPS --batch_size $BS --lr $LR --lambda_reg $LAMBDA_REG --lambda_v $LAMBDA_V \
  --weight_decay $WD --grad_clip $GC \
  --out $CKPT/${ENV}_tijepa.pt 2>&1 | tee "$LOGS/official_${ENV}_tijepa.log"

echo "[$(date)] === Protocol A: probe ladder (${ENV}) ==="
$PY -m ti_jepa.eval.probe_ladder --env $ENV \
  --baseline_ckpt $CKPT/${ENV}_baseline_k3.pt --tijepa_ckpt $CKPT/${ENV}_tijepa.pt \
  --out results/official_scale/probe_ladder_${ENV}.json 2>&1 | tee "$LOGS/official_probe_ladder_${ENV}.log"

echo "[$(date)] === Protocol B: kill experiment (${ENV}) ==="
$PY -m ti_jepa.eval.kill_experiment --env $ENV \
  --baseline_ckpt $CKPT/${ENV}_baseline_k3.pt \
  --baseline_k1_ckpt $CKPT/${ENV}_baseline_k1.pt \
  --tijepa_ckpt $CKPT/${ENV}_tijepa.pt \
  --n_pairs 150 --horizon 15 \
  --out_json results/official_scale/kill_experiment_${ENV}.json \
  --out_fig results/official_scale/kill_experiment_${ENV}.png 2>&1 | tee "$LOGS/official_kill_experiment_${ENV}.log"

echo "[$(date)] === ${ENV} ALL DONE ==="
