#!/bin/bash
# Launch a "correct-only" cons-loss run on a remote node using GPUs 4-7
# (assumes GPUs 0-3 are occupied by a JacobiForcing CLLM training).
#
# Usage:
#   bash scripts/_launch_cons_correct_only.sh <target_host> <variant>
#     variant: "no_marker" or "sinusoidal"
#
# Refuses to launch if the target host already has a verl/main_ppo process.
# JacobiForcing CLLM processes (different python entry point) are unaffected
# because we pin CUDA_VISIBLE_DEVICES=4,5,6,7 and JF runs on 0-3.

set -uo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <target_host> <no_marker|sinusoidal>" >&2
    exit 2
fi

TARGET_HOST="$1"
VARIANT="$2"

JF_MODEL="/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Coder_7B_v1/snapshots/81815b050f535c622153b5f6df38efc71326f938"
TRAIN_FILE="/mnt/weka/home/hao.zhang/shao/verl/data/opencodeinstruct/train.parquet"

case "$VARIANT" in
    no_marker)
        EXP_NAME="jf_coder_7b_dapo_oci_4gpu_cons_correct_only"
        MARKER_ENV="CONSISTENCY_USE_DRAFT_MARKER=0"
        ;;
    sinusoidal)
        EXP_NAME="jf_coder_7b_dapo_oci_4gpu_cons_correct_only_sinus"
        MARKER_ENV="CONSISTENCY_USE_DRAFT_MARKER=1 CONSISTENCY_MARKER_TYPE=sinusoidal CONSISTENCY_MARKER_INIT_SCALE=1.0"
        ;;
    *)
        echo "Unknown variant: $VARIANT (expected no_marker|sinusoidal)" >&2
        exit 2
        ;;
esac

LOG_FILE="/mnt/weka/home/hao.zhang/shao/verl/logs/${EXP_NAME}.log"

# Refuse to launch if any verl training is already running on the target.
# (JacobiForcing CLLM uses train/soft_flexattn_train_cllm_multiblock.py, not
# main_ppo / run_dapo_jf, so the grep below won't false-positive on it.)
RUNNING=$(ssh "$TARGET_HOST" 'pgrep -af "main_ppo|run_dapo_jf" | grep -v _launch_cons | wc -l' 2>/dev/null || echo 0)
if [ "$RUNNING" -gt 0 ]; then
    echo "[launcher] target $TARGET_HOST already has $RUNNING verl process(es); skip" >&2
    exit 1
fi

echo "[launcher] launching $EXP_NAME on $TARGET_HOST GPUs 4-7"
ssh "$TARGET_HOST" "cd /mnt/weka/home/hao.zhang/shao/verl && \
  CUDA_VISIBLE_DEVICES=4,5,6,7 \
  EXP_NAME=${EXP_NAME} \
  JF_MODEL=${JF_MODEL} \
  TRAIN_FILE=${TRAIN_FILE} \
  CONSISTENCY_ENABLE=1 \
  CONSISTENCY_WEIGHT=0.001 \
  CONSISTENCY_FRACTION=0.10 \
  CONSISTENCY_TEACHER=self \
  CONSISTENCY_ANCHOR_WEIGHT=0.0 \
  CONSISTENCY_NOISE_SOURCE=uniform \
  CONSISTENCY_CORRECT_ONLY=1 \
  CONSISTENCY_CORRECT_THRESHOLD=1.0 \
  ${MARKER_ENV} \
  nohup bash scripts/run_dapo_jf_coder_4gpu_consistency.sh > ${LOG_FILE} 2>&1 < /dev/null & echo PID=\$!"
