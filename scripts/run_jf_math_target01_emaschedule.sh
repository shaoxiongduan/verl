#!/usr/bin/env bash
# Math version of run_target01_no_marker_kl_combined.sh: best soft (KL) cons
# loss recipe so far — target_ratio=0.1 + EMA(0.9) + warmup_out schedule fading
# cons to 0 in the last 30% of training. KL divergence, no marker, no dflash.
# Coder-side EMA-adapt run hit val mean@8 0.813 (second-best after AR_V2 0.829).
set -x

cd /mnt/weka/home/hao.zhang/shao/verl

LOG_DIR=/mnt/weka/home/hao.zhang/shao/verl/logs
mkdir -p "${LOG_DIR}"

export EXP_NAME=jf_math_7b_dapo_ds_4gpu_cons_target01_emaschedule
export PROJECT_NAME=jacobi_forcing_dapo_deepscaler
export CONSISTENCY_ENABLE=1
export CONSISTENCY_WEIGHT=0.001
export CONSISTENCY_CORRECT_ONLY=1
export CONSISTENCY_TARGET_RATIO=0.1
export CONSISTENCY_TARGET_LAMBDA_MIN=1e-6
export CONSISTENCY_TARGET_LAMBDA_MAX=1e-1
export CONSISTENCY_TARGET_EMA_DECAY=0.9
export CONSISTENCY_USE_DRAFT_MARKER=0
export CONSISTENCY_SCHEDULE=warmup_out
export CONSISTENCY_RAMP_START_FRAC=0.5
export CONSISTENCY_RAMP_END_FRAC=0.7
# CONSISTENCY_WEIGHT_FINAL inherited from warmup_out default (= 0).
unset CONSISTENCY_MARKER_TYPE
unset CONSISTENCY_LOSS_TYPE  # defaults to kl
unset CONSISTENCY_LOG_GRAD_NORMS

bash scripts/run_dapo_jf_math_4gpu_consistency.sh \
    > "${LOG_DIR}/${EXP_NAME}.log" 2>&1
echo "[jf_math_target01_emaschedule] exited with code $?"
