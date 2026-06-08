#!/bin/bash
# Auto-launches B' (EMA-teacher + dual-KL anchor) on fs-mbz-gpu-970 GPUs 0-3
# once the base_teacher run has finished (i.e., its python process is gone).
# Intended to be invoked by ScheduleWakeup polling, not run manually.

set -uo pipefail

TARGET_HOST="fs-mbz-gpu-382"
EXP_NAME="jf_coder_7b_dapo_oci_4gpu_cons_ema_anchor"
JF_MODEL="/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Coder_7B_v1/snapshots/81815b050f535c622153b5f6df38efc71326f938"
LOG_FILE="/mnt/weka/home/hao.zhang/shao/verl/logs/cons_ema_anchor.log"

# Refuse to launch if any other verl training is still running on the target.
RUNNING=$(ssh "$TARGET_HOST" 'pgrep -af "main_ppo|run_dapo_jf" | grep -v _launch_cons_ema_pending | wc -l' 2>/dev/null || echo 0)
if [ "$RUNNING" -gt 0 ]; then
    echo "[launcher] target $TARGET_HOST still has $RUNNING training process(es); skip"
    exit 1
fi

echo "[launcher] launching B' on $TARGET_HOST GPUs 0-3"
ssh "$TARGET_HOST" "cd shao/verl && mkdir -p logs && \
  CUDA_VISIBLE_DEVICES=0,1,2,3 \
  EXP_NAME=${EXP_NAME} \
  JF_MODEL=${JF_MODEL} \
  TRAIN_FILE=/mnt/weka/home/hao.zhang/shao/verl/data/opencodeinstruct/train.parquet \
  CONSISTENCY_NOISE_SOURCE=uniform \
  CONSISTENCY_WEIGHT=0.001 \
  CONSISTENCY_TEACHER=ema \
  CONSISTENCY_TEACHER_PATH=${JF_MODEL} \
  CONSISTENCY_EMA_DECAY=0.999 \
  CONSISTENCY_ANCHOR_WEIGHT=0.001 \
  nohup bash scripts/run_dapo_jf_coder_4gpu_consistency.sh > ${LOG_FILE} 2>&1 < /dev/null & echo PID=\$!"
