#!/bin/bash
# 第二个独立seed复现 recurrent-aggregator (RSSM/Dreamer风格) ablation，
# 回应"one seed per environment and arm 统计功效不足"这条审稿风险
# （见 the paper draft 和写作讨论）。
#
# 只新训 baseline_rnn（--seed 2，权重初始化和训练batch顺序不同，但
# get_or_make_data 内部固定 seed=0，held-out eval数据和seed=0版本完全一致，
# 可直接和已有 baseline_k3/baseline_k1/tijepa checkpoint 公平对比），
# 输出到独立的 checkpoints*_seed2 目录，不覆盖已有 seed=0 结果。
set -e
cd "$(dirname "$0")/.."

export OMP_NUM_THREADS=4
export MKL_NUM_THREADS=4

PY="${PYTHON:-python3}"
LOGS=logs
OUT=results/rnn_ablation_seed2
mkdir -p "$OUT" "$LOGS"
mkdir -p checkpoints_seed2 checkpoints_seed2/pendulum checkpoints_seed2/cartpole

STEPS=8000
BS=256
LR=1e-3
LAMBDA_REG=1.0
K=3
Z_DIM=8
SEED=2

train_rnn () {
  local ENV=$1
  local CKPT_DIR=$2
  echo "[$(date)] === training baseline_rnn (seed=${SEED}) on ${ENV} ==="
  $PY -m ti_jepa.train --env "$ENV" --model baseline --predictor rnn --seed $SEED \
    --k $K --z_dim $Z_DIM --steps $STEPS --batch_size $BS --lr $LR --lambda_reg $LAMBDA_REG \
    --out "${CKPT_DIR}/baseline_rnn_seed2.pt" 2>&1 | tee "$LOGS/rnn_ablation_seed2_${ENV}_train.log"
}

eval_rnn () {
  local ENV=$1
  local CKPT_DIR=$2
  local ORIG_CKPT_DIR=$3
  local PLAN_ARGS=$4

  echo "[$(date)] === Protocol A (probe ladder) seed2, ${ENV} ==="
  $PY -m ti_jepa.eval.probe_ladder --env "$ENV" \
    --baseline_ckpt "${ORIG_CKPT_DIR}/baseline.pt" --tijepa_ckpt "${ORIG_CKPT_DIR}/tijepa.pt" \
    --baseline_rnn_ckpt "${CKPT_DIR}/baseline_rnn_seed2.pt" \
    --out "${OUT}/probe_ladder_${ENV}.json" 2>&1 | tee "$LOGS/rnn_ablation_seed2_${ENV}_probe.log"

  echo "[$(date)] === Protocol B (kill experiment) seed2, ${ENV} ==="
  $PY -m ti_jepa.eval.kill_experiment --env "$ENV" \
    --baseline_ckpt "${ORIG_CKPT_DIR}/baseline.pt" --baseline_k1_ckpt "${ORIG_CKPT_DIR}/baseline_k1.pt" \
    --tijepa_ckpt "${ORIG_CKPT_DIR}/tijepa.pt" --baseline_rnn_ckpt "${CKPT_DIR}/baseline_rnn_seed2.pt" \
    --out_json "${OUT}/kill_experiment_${ENV}.json" --out_fig "${OUT}/kill_experiment_${ENV}.png" \
    2>&1 | tee "$LOGS/rnn_ablation_seed2_${ENV}_kill.log"

  echo "[$(date)] === Protocol C (planning) seed2, ${ENV} ==="
  $PY -m ti_jepa.eval.planning --env "$ENV" \
    --baseline_ckpt "${ORIG_CKPT_DIR}/baseline.pt" --tijepa_ckpt "${ORIG_CKPT_DIR}/tijepa.pt" \
    --baseline_rnn_ckpt "${CKPT_DIR}/baseline_rnn_seed2.pt" \
    $PLAN_ARGS --closed_loop \
    --out_json "${OUT}/planning_cem_${ENV}.json" 2>&1 | tee "$LOGS/rnn_ablation_seed2_${ENV}_plan.log"
}

# ---- InertiaBall ----
train_rnn inertia_ball checkpoints_seed2
eval_rnn inertia_ball checkpoints_seed2 checkpoints \
  "--n_tasks 200 --horizon 10 --success_thresh 0.05 --mpc_horizon 5 --mpc_total_steps 10"

# ---- Pendulum ----
train_rnn pendulum checkpoints_seed2/pendulum
eval_rnn pendulum checkpoints_seed2/pendulum checkpoints/pendulum \
  "--n_tasks 100 --horizon 8 --success_thresh 0.1 --a_max 0.08 --mpc_horizon 4 --mpc_total_steps 10"

# ---- CartPole ----
train_rnn cartpole checkpoints_seed2/cartpole
eval_rnn cartpole checkpoints_seed2/cartpole checkpoints/cartpole \
  "--n_tasks 100 --horizon 8 --success_thresh 0.1 --a_max 0.35 --mpc_horizon 4 --mpc_total_steps 10"

echo "[$(date)] === RNN baseline ablation SEED2 ALL DONE ==="
