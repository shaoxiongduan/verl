#!/usr/bin/env bash
# Standalone launcher for Run B (CE variant). Designed to run on a different
# node than Run A so the two can train in parallel.

set -x

cd /mnt/weka/home/hao.zhang/shao/verl

LOG_DIR=/mnt/weka/home/hao.zhang/shao/verl/logs
mkdir -p "${LOG_DIR}"

export EXP_NAME=jf_coder_7b_dapo_oci_4gpu_cons_correct_only_target03_no_marker_ce
export PROJECT_NAME=jacobi_forcing_dapo_opencodeinstruct
export CONSISTENCY_ENABLE=1
export CONSISTENCY_WEIGHT=0.001
export CONSISTENCY_CORRECT_ONLY=1
export CONSISTENCY_TARGET_RATIO=0.3
export CONSISTENCY_TARGET_LAMBDA_MIN=1e-6
export CONSISTENCY_TARGET_LAMBDA_MAX=1e-1
export CONSISTENCY_USE_DRAFT_MARKER=0
export CONSISTENCY_LOSS_TYPE=ce
unset CONSISTENCY_MARKER_TYPE
unset CONSISTENCY_LOG_GRAD_NORMS  # FSDP2 makes autograd.grad return 0 anyway
bash scripts/run_dapo_jf_coder_4gpu_consistency.sh \
    > "${LOG_DIR}/${EXP_NAME}.log" 2>&1
echo "[run_B_ce] exited with code $?"
