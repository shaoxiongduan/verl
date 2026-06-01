#!/usr/bin/env bash
# KL cons run combining three changes vs the previous target03 baseline:
#   1. target_ratio 0.3 -> 0.1                                    (CONSISTENCY_TARGET_RATIO)
#   2. EMA across step snapshots                                   (CONSISTENCY_TARGET_EMA_DECAY=0.9)
#   3. warmup_out schedule MULTIPLIER fades cons to 0 over second  (CONSISTENCY_SCHEDULE,
#      half of training, applied AFTER target_ratio computes lambda  RAMP_START_FRAC, RAMP_END_FRAC)

set -x

cd /mnt/weka/home/hao.zhang/shao/verl

LOG_DIR=/mnt/weka/home/hao.zhang/shao/verl/logs
mkdir -p "${LOG_DIR}"

export EXP_NAME=jf_coder_7b_dapo_oci_4gpu_cons_correct_only_target01_no_marker_emaschedule
export PROJECT_NAME=jacobi_forcing_dapo_opencodeinstruct
export CONSISTENCY_ENABLE=1
export CONSISTENCY_WEIGHT=0.001
export CONSISTENCY_CORRECT_ONLY=1
export CONSISTENCY_TARGET_RATIO=0.1
export CONSISTENCY_TARGET_LAMBDA_MIN=1e-6
export CONSISTENCY_TARGET_LAMBDA_MAX=1e-1
export CONSISTENCY_TARGET_EMA_DECAY=0.9
export CONSISTENCY_USE_DRAFT_MARKER=0
# Schedule MULTIPLIER applied on top of target_lambda. With actual_init=
# CONSISTENCY_WEIGHT (the base for the schedule shape) and WEIGHT_FINAL=0,
# warmup_out makes the multiplier go 1.0 -> 0.0 over [RAMP_START, RAMP_END]
# fraction of total_training_steps. Cons fully off in last 30% of training.
export CONSISTENCY_SCHEDULE=warmup_out
export CONSISTENCY_RAMP_START_FRAC=0.5
export CONSISTENCY_RAMP_END_FRAC=0.7
# CONSISTENCY_WEIGHT_FINAL inherited from warmup_out default (= 0).
unset CONSISTENCY_MARKER_TYPE
unset CONSISTENCY_LOSS_TYPE  # defaults to kl
unset CONSISTENCY_LOG_GRAD_NORMS
bash scripts/run_dapo_jf_coder_4gpu_consistency.sh \
    > "${LOG_DIR}/${EXP_NAME}.log" 2>&1
echo "[combined_run] exited with code $?"
