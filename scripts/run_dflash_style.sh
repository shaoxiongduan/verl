#!/usr/bin/env bash
# dFlash-style cons-RL run:
#   - CONSISTENCY_NOISE_SOURCE=mask     (single mask token everywhere in noisy block)
#   - CONSISTENCY_LOSS_TYPE=dflash_ce    (per-position CE w/ exp(-(k-1)/gamma) weight)
#   - CONSISTENCY_CAUSAL_REGION_SIZE=-1 (fully bidirectional inside noisy block)
#   - CONSISTENCY_DFLASH_GAMMA=12        (extrapolated from paper: 4@8, 5@10, 7@16 → ~12@32)
# Does NOT preserve AR equivalence — user explicitly OK with that; the goal is
# to test whether dFlash-style draft training gives better first-iteration
# predictions than the current KL+random+causal recipe.

set -x

cd /mnt/weka/home/hao.zhang/shao/verl

LOG_DIR=/mnt/weka/home/hao.zhang/shao/verl/logs
mkdir -p "${LOG_DIR}"

export EXP_NAME=${EXP_NAME:-jf_coder_7b_dapo_oci_4gpu_cons_dflash_style_bidir}
export PROJECT_NAME=jacobi_forcing_dapo_opencodeinstruct
export CONSISTENCY_ENABLE=1
# target_ratio=0.4 + EMA keeps cons:|signed Σ pg| ≈ 0.4 at the per-step level
# (uses the per-step accumulator we built earlier; 1-step lag, 0.9 EMA decay).
# cons_loss starts at ~22 (model never trained on mask+bidir/causal input);
# adaptive weight handles the magnitude swing as the model converges to dFlash
# objective without us tuning a fixed weight.
export CONSISTENCY_WEIGHT=0.001                  # bootstrap before target_ratio kicks in
export CONSISTENCY_CORRECT_ONLY=1
export CONSISTENCY_NOISE_SOURCE=mask
export CONSISTENCY_MASK_TOKEN_ID=151643
export CONSISTENCY_LOSS_TYPE=dflash_ce
export CONSISTENCY_DFLASH_GAMMA=12.0
# CONSISTENCY_CAUSAL_REGION_SIZE overridable per launch: -1 = full bidir noisy
# block (matches dFlash); 0 = pure causal (original JF recipe). Both share all
# other cons knobs.
export CONSISTENCY_CAUSAL_REGION_SIZE=${CONSISTENCY_CAUSAL_REGION_SIZE:--1}
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
echo "[dflash_style] exited with code $?"
