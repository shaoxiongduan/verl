#!/usr/bin/env bash
# Idea-A mask-tail, RUN A3: off-policy / AR with UNIFORM-RANDOM noise input
# (historical CLLM/dFlash setup) + mask-tail target. The "true synthetic"
# control — uniform input may be harder for the model to learn from than mask.
set -euo pipefail
cd /mnt/weka/home/hao.zhang/shao/verl

export PROJECT_NAME=jacobi_forcing_dapo_deepscaler_masktail
export EXP_NAME=jf_math_7b_masktail_A3_ar_uniform_m8

# --- cons loss: dflash_ce + mask-tail ---
export CONSISTENCY_LOSS_TYPE=dflash_ce
export CONSISTENCY_WEIGHT=0.001
export CONSISTENCY_MASK_TAIL=1
export CONSISTENCY_MASK_TOKEN_ID=151643
export CONSISTENCY_MASK_MARGIN=8

# --- variant: off-policy, uniform-random input init ---
export CONSISTENCY_ONPOLICY=0
export CONSISTENCY_NOISE_SOURCE=uniform

exec bash scripts/run_dapo_jf_math_4gpu_jacobi_onpolicy.sh "$@"
