#!/usr/bin/env bash
# Ablation: dflash_ce loss (shift-by-1 + exp(-(k-1)/gamma) position decay)
# applied to JF-style UNIFORM random noisy tokens (not mask). Causal attention.
#
# Direct A/B vs run_dflash_style.sh w/ CONSISTENCY_CAUSAL_REGION_SIZE=0
# (= jf_coder_7b_dapo_oci_4gpu_cons_dflash_causal_shiftfix). Only difference
# is noise source: uniform vs mask. Goal: isolate whether the dflash gap is
# driven by position decay + correct shift, or actually requires mask tokens.
#
# Hypothesis: most of the dflash win is from the decay weighting; mask vs
# noise should make a small difference once shift + decay are both in.

set -x

cd /mnt/weka/home/hao.zhang/shao/verl

LOG_DIR=/mnt/weka/home/hao.zhang/shao/verl/logs
mkdir -p "${LOG_DIR}"

export EXP_NAME=${EXP_NAME:-jf_coder_7b_dapo_oci_4gpu_cons_ce_noisy_decay}
export PROJECT_NAME=jacobi_forcing_dapo_opencodeinstruct
export CONSISTENCY_ENABLE=1
export CONSISTENCY_WEIGHT=0.001                  # bootstrap before target_ratio kicks in
export CONSISTENCY_CORRECT_ONLY=1

# JF-style noise: random vocab tokens (the historical default). Explicit for
# clarity; this is what `pack.py` falls back to when unset.
export CONSISTENCY_NOISE_SOURCE=uniform
unset CONSISTENCY_MASK_TOKEN_ID

# Loss = dflash_ce (per-position CE w/ shift-by-1 fix + exp(-(k-1)/gamma) decay)
export CONSISTENCY_LOSS_TYPE=dflash_ce
export CONSISTENCY_DFLASH_GAMMA=12.0

# Causal attention inside the noisy block (matches JF + dflash_causal_shiftfix)
export CONSISTENCY_CAUSAL_REGION_SIZE=0

# Same adaptive weighting + EMA as dflash_causal_shiftfix for a fair compare
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
echo "[ce_noisy_decay] exited with code $?"
