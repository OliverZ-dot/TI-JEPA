#!/bin/bash
# 官方规模（ViT-Tiny/14 + AdaLN transformer depth6）CartPole 全流程：
# 3 个训练（严格顺序执行，不并发——每个进程 GPU 显存 ~20-24GB，46GB 的卡放不下两个；
# CPU 侧每个进程要把整份数据集(~17GB, train+test)读进内存，50GB cgroup 限额下也不能并发3个）
# 训练完之后自动跑 Protocol A（探针梯子）+ Protocol B（kill experiment）。
# 任何一步失败都会用 set -e 中止并在日志里看得出来。
#
# 重要修正记录（第一版跑出来后发现的bug，已修复，此为修复后重跑版本）：
# 第一版直接照抄了小CNN配方的 lr=1e-3/lambda_reg=1.0，那是专门为小CNN调过的值，
# 对官方 ViT-Tiny+AdaLN transformer 规模完全不对（官方config是 lr=5e-5，但官方
# 自己的 SIGReg 归一化方式和我们的实现不同，直接抄 lambda=0.09 数值一样坍缩）。
# 诊断：z_std/v_std 全部单调坍缩到~0，SIGReg loss 冻结在坍缩不动点（和已知的
# k=1/SIGReg-on-whole-z 坍缩消融症状一致）。修复：
#   1. lr=5e-5（抄官方数值，这个是对的）
#   2. 补上官方架构里漏掉的 projector/pred_proj（Linear->BatchNorm->GELU->Linear,
#      hidden=2048，见 official_predictor.py::MLPHead）——这是已知的自监督抗坍缩机制
#   3. AdamW + weight_decay=1e-3、grad_clip=1.0（抄官方 lewm.yaml）
#   4. lambda_reg 经短程(600步)扫描重新调到 5.0（不是官方数值0.09，因为我们的
#      SIGReg实现归一化方式不同；0.09/1.0 都坍缩，5.0 才稳定，z_std在600步内
#      稳定在 0.81-0.85，而不是单调滑向0——见 checkpoints/smoke_fix/ 的对比记录）
#   5. lambda_v 从 0.1 提到 1.0（同样的道理，给 v 更强的方差下限压力）
#   6. steps 从 3000 提到 5000（从零训练的ViT本来就需要更久收敛）
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
ENV=cartpole

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
