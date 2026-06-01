#!/usr/bin/env bash
# Math version of run_ce_noisy_decay_k3.sh: K=3 independent uniform-noise tiles
# per clean block, dflash_ce loss (shift-by-1 + exp(-(k-1)/gamma) decay, γ=12),
# causal noisy-block attention, target_ratio=0.4 + EMA(0.9).
# Coder-side K=3 run: TPF 4.46 at val 0.785 (vs K=1 TPF 4.01 at same val).
set -x

cd /mnt/weka/home/hao.zhang/shao/verl

LOG_DIR=/mnt/weka/home/hao.zhang/shao/verl/logs
mkdir -p "${LOG_DIR}"

export EXP_NAME=jf_math_7b_dapo_ds_4gpu_cons_ce_noisy_decay_k3
export PROJECT_NAME=jacobi_forcing_dapo_deepscaler
export CONSISTENCY_ENABLE=1
export CONSISTENCY_WEIGHT=0.001
export CONSISTENCY_CORRECT_ONLY=1

export CONSISTENCY_NOISE_SOURCE=uniform
export CONSISTENCY_NUM_NOISY_TILES=3
unset CONSISTENCY_MASK_TOKEN_ID

export CONSISTENCY_LOSS_TYPE=dflash_ce
export CONSISTENCY_DFLASH_GAMMA=12.0

export CONSISTENCY_CAUSAL_REGION_SIZE=0

export CONSISTENCY_TARGET_RATIO=0.4
export CONSISTENCY_TARGET_EMA_DECAY=0.9
export CONSISTENCY_TARGET_LAMBDA_MIN=1e-6
export CONSISTENCY_TARGET_LAMBDA_MAX=1e-1

export CONSISTENCY_USE_DRAFT_MARKER=0
unset CONSISTENCY_MARKER_TYPE
unset CONSISTENCY_SCHEDULE
unset CONSISTENCY_LOG_GRAD_NORMS

bash scripts/run_dapo_jf_math_4gpu_consistency.sh \
    > "${LOG_DIR}/${EXP_NAME}.log" 2>&1
echo "[jf_math_ce_noisy_decay_k3] exited with code $?"
