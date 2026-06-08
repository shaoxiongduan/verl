#!/bin/bash
# Launch the "all traces + no marker + adaptive cons-weight scale" run.
# Variant of the cons001 baseline (no marker, all rollouts, λ=0.001) but with
# the new adaptive scaling that proved a +2.1pp win over baseline in the
# sinusoidal correct-only experiment (16:13 run, ended val=0.8125 vs
# baseline 0.7912 at step 300).
#
# Usage:
#   bash scripts/_launch_cons_all_no_marker_adaptive.sh <target_host> <cuda_devices>
#     cuda_devices: comma list, e.g. "0,1,2,3" or "0,1,2,3,4,5,6,7"
#
# Adaptive defaults match the sinusoidal_adapt run that worked:
#   ADAPTIVE_DECAY=0.99  WARMUP_STEPS=50  FLOOR=0.05  CEILING=1.0

set -uo pipefail

if [ "$#" -ne 2 ]; then
    echo "Usage: $0 <target_host> <cuda_devices>" >&2
    exit 2
fi

TARGET_HOST="$1"
CUDA_DEVS="$2"

JF_MODEL="/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Coder_7B_v1/snapshots/81815b050f535c622153b5f6df38efc71326f938"
TRAIN_FILE="/mnt/weka/home/hao.zhang/shao/verl/data/opencodeinstruct/train.parquet"

EXP_NAME="jf_coder_7b_dapo_oci_4gpu_cons_all_no_marker_adapt"
LOG_FILE="/mnt/weka/home/hao.zhang/shao/verl/logs/${EXP_NAME}.log"
SESSION="${EXP_NAME}"

EXISTS=$(ssh "$TARGET_HOST" "tmux has-session -t ${SESSION} 2>/dev/null && echo yes || echo no")
if [ "$EXISTS" = "yes" ]; then
    echo "[launcher] tmux session ${SESSION} already exists on ${TARGET_HOST}; refusing to re-launch" >&2
    exit 1
fi

echo "[launcher] launching ${EXP_NAME} on ${TARGET_HOST} GPUs ${CUDA_DEVS} (tmux session: ${SESSION})"

SKIP_RAY_STOP_PREFIX=""
if [ "${SKIP_RAY_STOP:-0}" = "1" ]; then
    SKIP_RAY_STOP_PREFIX="SKIP_RAY_STOP=1"
fi
RAY_TMPDIR_PREFIX=""
if [ -n "${RAY_TMPDIR:-}" ]; then
    RAY_TMPDIR_PREFIX="RAY_TMPDIR=${RAY_TMPDIR}"
fi

REMOTE_CMD="cd /mnt/weka/home/hao.zhang/shao/verl && \
  CUDA_VISIBLE_DEVICES=${CUDA_DEVS} \
  ${SKIP_RAY_STOP_PREFIX} \
  ${RAY_TMPDIR_PREFIX} \
  EXP_NAME=${EXP_NAME} \
  JF_MODEL=${JF_MODEL} \
  TRAIN_FILE=${TRAIN_FILE} \
  CONSISTENCY_ENABLE=1 \
  CONSISTENCY_WEIGHT=0.001 \
  CONSISTENCY_FRACTION=0.10 \
  CONSISTENCY_TEACHER=self \
  CONSISTENCY_ANCHOR_WEIGHT=0.0 \
  CONSISTENCY_NOISE_SOURCE=uniform \
  CONSISTENCY_CORRECT_ONLY=0 \
  CONSISTENCY_USE_DRAFT_MARKER=0 \
  CONSISTENCY_ADAPTIVE_SCALE=1 \
  CONSISTENCY_ADAPTIVE_DECAY=0.99 \
  CONSISTENCY_ADAPTIVE_WARMUP_STEPS=50 \
  CONSISTENCY_ADAPTIVE_FLOOR=0.05 \
  CONSISTENCY_ADAPTIVE_CEILING=1.0 \
  bash scripts/run_dapo_jf_coder_4gpu_consistency.sh 2>&1 | tee -a ${LOG_FILE}"

ssh "$TARGET_HOST" "tmux new-session -d -s ${SESSION} \"${REMOTE_CMD}\""
echo "[launcher] launched ${EXP_NAME}; attach with:  ssh ${TARGET_HOST} -t tmux attach -t ${SESSION}"
echo "[launcher] log file: ${LOG_FILE}"
