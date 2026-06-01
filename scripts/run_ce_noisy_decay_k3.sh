#!/usr/bin/env bash
# Multi-noise ablation: K=3 independent noisy tiles per clean block.
# Same as run_ce_noisy_decay.sh but with CONSISTENCY_NUM_NOISY_TILES=3:
# diffusion-style Monte-Carlo averaging over noise samples per clean target.
# Sequence length roughly 2x; uses uniform noise + dflash_ce (shift + decay).
#
# A/B vs run_ce_noisy_decay.sh (K=1) — isolates whether multi-noise averaging
# improves draft quality / final val acc beyond the single-noise baseline.

set -x

cd /mnt/weka/home/hao.zhang/shao/verl

LOG_DIR=/mnt/weka/home/hao.zhang/shao/verl/logs
mkdir -p "${LOG_DIR}"

export EXP_NAME=${EXP_NAME:-jf_coder_7b_dapo_oci_4gpu_cons_ce_noisy_decay_k3}
export PROJECT_NAME=jacobi_forcing_dapo_opencodeinstruct
export CONSISTENCY_ENABLE=1
export CONSISTENCY_WEIGHT=0.001
export CONSISTENCY_CORRECT_ONLY=1

# Uniform noise (random vocab tokens), K=3 independent tiles per clean block
export CONSISTENCY_NOISE_SOURCE=uniform
export CONSISTENCY_NUM_NOISY_TILES=3
unset CONSISTENCY_MASK_TOKEN_ID

# Loss = dflash_ce (shift-by-1 + exp(-(k-1)/gamma) decay, γ=12)
export CONSISTENCY_LOSS_TYPE=dflash_ce
export CONSISTENCY_DFLASH_GAMMA=12.0

# Causal noisy attention (matches run_ce_noisy_decay.sh and dflash_causal_shiftfix)
export CONSISTENCY_CAUSAL_REGION_SIZE=0

# Adaptive weighting + EMA, same as the K=1 baseline
export CONSISTENCY_TARGET_RATIO=0.4
export CONSISTENCY_TARGET_EMA_DECAY=0.9
export CONSISTENCY_TARGET_LAMBDA_MIN=1e-6
export CONSISTENCY_TARGET_LAMBDA_MAX=1e-1

export CONSISTENCY_USE_DRAFT_MARKER=0
unset CONSISTENCY_MARKER_TYPE
unset CONSISTENCY_SCHEDULE
unset CONSISTENCY_LOG_GRAD_NORMS

bash scripts/run_dapo_jf_coder_4gpu_consistency.sh \
    > "${LOG_DIR}/${EXP_NAME}.log" 2>&1
echo "[ce_noisy_decay_k3] exited with code $?"
