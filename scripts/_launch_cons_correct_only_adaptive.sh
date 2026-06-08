#!/bin/bash
# Launch a "correct-only + adaptive cons-weight scale" cons-loss run on a
# remote node in a tmux session. CUDA_VISIBLE_DEVICES is explicit so two of
# these can co-exist on one node (one on 0-3, one on 4-7).
#
# Usage:
#   bash scripts/_launch_cons_correct_only_adaptive.sh <target_host> <variant> <cuda_devices>
#     variant:       "no_marker" or "sinusoidal"
#     cuda_devices:  comma list, e.g. "0,1,2,3" or "4,5,6,7"
#
# Adds CONSISTENCY_ADAPTIVE_SCALE=1 + supporting knobs on top of the existing
# correct-only setup. Experiment name suffixed with _adapt so logs/runs don't
# collide with the non-adaptive baselines. No automatic process kill on the
# remote — caller is responsible for not double-launching on the same GPU set.

set -uo pipefail

if [ "$#" -ne 3 ]; then
    echo "Usage: $0 <target_host> <no_marker|sinusoidal> <cuda_devices>" >&2
    exit 2
fi

TARGET_HOST="$1"
VARIANT="$2"
CUDA_DEVS="$3"

JF_MODEL="/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Coder_7B_v1/snapshots/81815b050f535c622153b5f6df38efc71326f938"
TRAIN_FILE="/mnt/weka/home/hao.zhang/shao/verl/data/opencodeinstruct/train.parquet"

case "$VARIANT" in
    no_marker)
        EXP_BASE="jf_coder_7b_dapo_oci_4gpu_cons_correct_only_adapt"
        MARKER_ENV="CONSISTENCY_USE_DRAFT_MARKER=0"
        ;;
    sinusoidal)
        EXP_BASE="jf_coder_7b_dapo_oci_4gpu_cons_correct_only_sinus_adapt"
        MARKER_ENV="CONSISTENCY_USE_DRAFT_MARKER=1 CONSISTENCY_MARKER_TYPE=sinusoidal CONSISTENCY_MARKER_INIT_SCALE=1.0"
        ;;
    *)
        echo "Unknown variant: $VARIANT (expected no_marker|sinusoidal)" >&2
        exit 2
        ;;
esac

# CONSISTENCY_LOSS_TYPE override from env: "kl" (default, JF/CLLM-style) or
# "ce" (Nemotron-style hard cross-entropy on ground-truth tokens). When "ce"
# is set we suffix the exp name so logs/runs don't collide with the KL run.
LOSS_TYPE_ENV=""
EXP_NAME="${EXP_BASE}"
if [ "${CONSISTENCY_LOSS_TYPE:-kl}" = "ce" ]; then
    LOSS_TYPE_ENV="CONSISTENCY_LOSS_TYPE=ce"
    EXP_NAME="${EXP_BASE}_ce"
fi

LOG_FILE="/mnt/weka/home/hao.zhang/shao/verl/logs/${EXP_NAME}.log"
SESSION="${EXP_NAME}"

# Conservative collision guard: refuse if a tmux session of the same name
# already exists on the target. Different GPU sets get different exp names
# (and thus different sessions) by construction, so this only blocks true
# re-launches.
EXISTS=$(ssh "$TARGET_HOST" "tmux has-session -t ${SESSION} 2>/dev/null && echo yes || echo no")
if [ "$EXISTS" = "yes" ]; then
    echo "[launcher] tmux session ${SESSION} already exists on ${TARGET_HOST}; refusing to re-launch" >&2
    exit 1
fi

echo "[launcher] launching ${EXP_NAME} on ${TARGET_HOST} GPUs ${CUDA_DEVS} (tmux session: ${SESSION})"

# Optional knobs for parallel co-existence on one node:
#   SKIP_RAY_STOP=1   -> do not call `ray stop --force` at start (prevents
#                        killing a peer training's ray cluster on same host)
#   RAY_TMPDIR=/path  -> isolate ray state dir; required when two ray clusters
#                        run on the same host so they don't read each other's
#                        /tmp/ray session files.
SKIP_RAY_STOP_PREFIX=""
if [ "${SKIP_RAY_STOP:-0}" = "1" ]; then
    SKIP_RAY_STOP_PREFIX="SKIP_RAY_STOP=1"
fi
RAY_TMPDIR_PREFIX=""
if [ -n "${RAY_TMPDIR:-}" ]; then
    RAY_TMPDIR_PREFIX="RAY_TMPDIR=${RAY_TMPDIR}"
fi

# Build the env-prefixed command. tee captures to the log file AND keeps
# output in the tmux scrollback for interactive inspection.
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
  CONSISTENCY_CORRECT_ONLY=1 \
  CONSISTENCY_CORRECT_THRESHOLD=1.0 \
  CONSISTENCY_ADAPTIVE_SCALE=1 \
  CONSISTENCY_ADAPTIVE_DECAY=0.99 \
  CONSISTENCY_ADAPTIVE_WARMUP_STEPS=50 \
  CONSISTENCY_ADAPTIVE_FLOOR=0.05 \
  CONSISTENCY_ADAPTIVE_CEILING=1.0 \
  ${LOSS_TYPE_ENV} \
  ${MARKER_ENV} \
  bash scripts/run_dapo_jf_coder_4gpu_consistency.sh 2>&1 | tee -a ${LOG_FILE}"

ssh "$TARGET_HOST" "tmux new-session -d -s ${SESSION} \"${REMOTE_CMD}\""
echo "[launcher] launched ${EXP_NAME} on ${TARGET_HOST}; attach with:  ssh ${TARGET_HOST} -t tmux attach -t ${SESSION}"
echo "[launcher] log file: ${LOG_FILE}"
