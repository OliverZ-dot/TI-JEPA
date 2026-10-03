#!/bin/bash
# 等 cartpole 官方规模 pipeline (pid 传进来) 结束后，自动接着跑 pendulum 那一份，
# 保证 GPU 不空闲、不需要人工重新启动。
CARTPOLE_PID=$1
echo "[$(date)] watcher: waiting for cartpole pipeline pid=$CARTPOLE_PID to finish..."
while kill -0 "$CARTPOLE_PID" 2>/dev/null; do
  sleep 30
done
echo "[$(date)] watcher: cartpole pipeline finished, launching pendulum pipeline"
"$(dirname "${BASH_SOURCE[0]}")/run_official_scale_pendulum.sh" >> logs/official_scale_pipeline.log 2>&1
echo "[$(date)] watcher: pendulum pipeline finished too. all official-scale work done."
