#!/bin/bash
# 第二个独立seed复现官方规模(ViT-Tiny/14 + AdaLN transformer) CartPole +
# Pendulum 全流程，回应"官方规模结果只有一个seed，统计稳健性存疑"这条审稿
# 风险。复用 run_official_scale_{cartpole,pendulum}.sh 验证过的完全相同配方
# (lr=5e-5, lambda_reg=5.0, lambda_v=1.0, weight_decay=1e-3, grad_clip=1.0,
# steps=5000)，只加 --seed 2，权重初始化/训练batch顺序不同，held-out eval
# 数据（get_or_make_data 内部固定 seed=0）和seed=0版本完全一致，可直接对比。
#
# 会先等待 GPU 上正在跑的 reacher_train_official.py（frameskip=1 decorrelation
# 实验）进程结束，避免两个 ViT-Tiny 规模的训练同时抢 46GB 显存导致 OOM。
set -e
cd "$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
PY="${PYTHON:-python3}"
CKPT=checkpoints/official_scale_seed2
LOGS=logs
mkdir -p "$CKPT" "$LOGS" results/official_scale_seed2

echo "[$(date)] waiting for GPU-heavy reacher_fs1 job to finish before starting seed2 official-scale runs..."
while pgrep -f "reacher_train_official.py" > /dev/null 2>&1; do
  sleep 30
done
echo "[$(date)] GPU free, starting official-scale seed2 runs."

STEPS=5000
BS=128
LR=5e-5
LAMBDA_REG=5.0
LAMBDA_V=1.0
WD=1e-3
GC=1.0
SEED=2

run_env () {
  local ENV=$1

  echo "[$(date)] === ${ENV} (seed=${SEED}) 1/3 baseline k=3 ==="
  $PY -m ti_jepa.train --env $ENV --model baseline --seed $SEED \
    --backbone vit --predictor adaln --image_size 224 --z_dim 192 --feat_dim 192 --k 3 \
    --steps $STEPS --batch_size $BS --lr $LR --lambda_reg $LAMBDA_REG --weight_decay $WD --grad_clip $GC \
    --out $CKPT/${ENV}_baseline_k3.pt 2>&1 | tee "$LOGS/official_seed2_${ENV}_baseline_k3.log"

  echo "[$(date)] === ${ENV} (seed=${SEED}) 2/3 baseline k=1 ==="
  $PY -m ti_jepa.train --env $ENV --model baseline --seed $SEED \
    --backbone vit --predictor adaln --image_size 224 --z_dim 192 --feat_dim 192 --k 1 \
    --steps $STEPS --batch_size $BS --lr $LR --lambda_reg $LAMBDA_REG --weight_decay $WD --grad_clip $GC \
    --out $CKPT/${ENV}_baseline_k1.pt 2>&1 | tee "$LOGS/official_seed2_${ENV}_baseline_k1.log"

  echo "[$(date)] === ${ENV} (seed=${SEED}) 3/3 TI-JEPA ==="
  $PY -m ti_jepa.train --env $ENV --model tijepa --seed $SEED \
    --backbone vit --predictor adaln --image_size 224 --q_dim 96 --v_dim 96 --feat_dim 192 --k 3 \
    --steps $STEPS --batch_size $BS --lr $LR --lambda_reg $LAMBDA_REG --lambda_v $LAMBDA_V \
    --weight_decay $WD --grad_clip $GC \
    --out $CKPT/${ENV}_tijepa.pt 2>&1 | tee "$LOGS/official_seed2_${ENV}_tijepa.log"

  echo "[$(date)] === ${ENV} (seed=${SEED}) Protocol A: probe ladder ==="
  $PY -m ti_jepa.eval.probe_ladder --env $ENV \
    --baseline_ckpt $CKPT/${ENV}_baseline_k3.pt --tijepa_ckpt $CKPT/${ENV}_tijepa.pt \
    --out results/official_scale_seed2/probe_ladder_${ENV}.json 2>&1 | tee "$LOGS/official_seed2_probe_ladder_${ENV}.log"

  echo "[$(date)] === ${ENV} (seed=${SEED}) Protocol B: kill experiment ==="
  $PY -m ti_jepa.eval.kill_experiment --env $ENV \
    --baseline_ckpt $CKPT/${ENV}_baseline_k3.pt \
    --baseline_k1_ckpt $CKPT/${ENV}_baseline_k1.pt \
    --tijepa_ckpt $CKPT/${ENV}_tijepa.pt \
    --n_pairs 150 --horizon 15 \
    --out_json results/official_scale_seed2/kill_experiment_${ENV}.json \
    --out_fig results/official_scale_seed2/kill_experiment_${ENV}.png 2>&1 | tee "$LOGS/official_seed2_kill_experiment_${ENV}.log"

  echo "[$(date)] === ${ENV} seed2 ALL DONE ==="
}

run_env cartpole
run_env pendulum

echo "[$(date)] === official-scale SEED2 (cartpole+pendulum) ALL DONE ==="
