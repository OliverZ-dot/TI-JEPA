#!/bin/bash
# RSSM/Dreamer 风格递归聚合器对照实验（回应"为什么不直接用递归隐状态"这个
# 最直接的审稿人问题，见 the paper draft）。
#
# 用和小CNN基线完全相同的训练配方（lr=1e-3, lambda_reg=1.0, steps=8000,
# batch_size=256, k=3, z_dim=8, backbone=conv, image_size=64），只把
# predictor 从 HistoryPredictor（拼接k帧窗口）换成 RecurrentPredictor
# （GRUCell 逐步吸收k帧），在InertiaBall/Pendulum/CartPole三环境各训一个
# baseline_rnn，然后把它作为第4个臂接入 Protocol A/B/C，跟已经训好的
# baseline_k3 / baseline_k1 / tijepa checkpoint 一起对比。
#
# 关键：不重新训练 baseline_k3/baseline_k1/tijepa（沿用已有 checkpoint），
# 只新增 baseline_rnn 这一个臂，保证对比公平（同一批held-out数据/同一个
# env_cfg/同一套评测代码）。
set -e
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

PY="${PYTHON:-python3}"
LOGS=logs
OUT=results/rnn_ablation
mkdir -p "$OUT" "$LOGS"

STEPS=8000
BS=256
LR=1e-3
LAMBDA_REG=1.0
K=3
Z_DIM=8

train_rnn () {
  local ENV=$1
  local CKPT_DIR=$2
  echo "[$(date)] === training baseline_rnn on ${ENV} ==="
  $PY -m ti_jepa.train --env "$ENV" --model baseline --predictor rnn \
    --k $K --z_dim $Z_DIM --steps $STEPS --batch_size $BS --lr $LR --lambda_reg $LAMBDA_REG \
    --out "${CKPT_DIR}/baseline_rnn.pt" 2>&1 | tee "$LOGS/rnn_ablation_${ENV}_train.log"
}

eval_rnn () {
  local ENV=$1
  local CKPT_DIR=$2
  local PLAN_ARGS=$3   # n_tasks/horizon/a_max/success_thresh/mpc args, env-specific

  echo "[$(date)] === Protocol A (probe ladder) w/ RNN arm: ${ENV} ==="
  $PY -m ti_jepa.eval.probe_ladder --env "$ENV" \
    --baseline_ckpt "${CKPT_DIR}/baseline.pt" --tijepa_ckpt "${CKPT_DIR}/tijepa.pt" \
    --baseline_rnn_ckpt "${CKPT_DIR}/baseline_rnn.pt" \
    --out "${OUT}/probe_ladder_${ENV}.json" 2>&1 | tee "$LOGS/rnn_ablation_${ENV}_probe.log"

  echo "[$(date)] === Protocol B (kill experiment) w/ RNN arm: ${ENV} ==="
  $PY -m ti_jepa.eval.kill_experiment --env "$ENV" \
    --baseline_ckpt "${CKPT_DIR}/baseline.pt" --baseline_k1_ckpt "${CKPT_DIR}/baseline_k1.pt" \
    --tijepa_ckpt "${CKPT_DIR}/tijepa.pt" --baseline_rnn_ckpt "${CKPT_DIR}/baseline_rnn.pt" \
    --out_json "${OUT}/kill_experiment_${ENV}.json" --out_fig "${OUT}/kill_experiment_${ENV}.png" \
    2>&1 | tee "$LOGS/rnn_ablation_${ENV}_kill.log"

  echo "[$(date)] === Protocol C (planning, open+closed loop) w/ RNN arm: ${ENV} ==="
  $PY -m ti_jepa.eval.planning --env "$ENV" \
    --baseline_ckpt "${CKPT_DIR}/baseline.pt" --tijepa_ckpt "${CKPT_DIR}/tijepa.pt" \
    --baseline_rnn_ckpt "${CKPT_DIR}/baseline_rnn.pt" \
    $PLAN_ARGS --closed_loop \
    --out_json "${OUT}/planning_cem_${ENV}.json" 2>&1 | tee "$LOGS/rnn_ablation_${ENV}_plan.log"
}

# ---- InertiaBall ----
train_rnn inertia_ball checkpoints
eval_rnn inertia_ball checkpoints \
  "--n_tasks 200 --horizon 10 --success_thresh 0.05 --mpc_horizon 5 --mpc_total_steps 10"

# ---- Pendulum ----
train_rnn pendulum checkpoints/pendulum
eval_rnn pendulum checkpoints/pendulum \
  "--n_tasks 100 --horizon 8 --success_thresh 0.1 --a_max 0.08 --mpc_horizon 4 --mpc_total_steps 10"

# ---- CartPole ----
train_rnn cartpole checkpoints/cartpole
eval_rnn cartpole checkpoints/cartpole \
  "--n_tasks 100 --horizon 8 --success_thresh 0.1 --a_max 0.35 --mpc_horizon 4 --mpc_total_steps 10"

echo "[$(date)] === RNN baseline ablation ALL DONE ==="
