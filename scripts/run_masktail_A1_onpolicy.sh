#!/usr/bin/env bash
# Idea-A mask-tail, RUN A1: ON-POLICY conditioning (real cascade-evolved tail
# input) + mask-tail target. Clean CE on [pml, pml+margin), mask-token target
# beyond. Flat-weighted (dflash decay dropped at the clean->mask boundary).
set -euo pipefail
cd /mnt/weka/home/hao.zhang/shao/verl

export PROJECT_NAME=jacobi_forcing_dapo_deepscaler_masktail
export EXP_NAME=jf_math_7b_masktail_A1_onpolicy_m8

# --- cons loss: dflash_ce + mask-tail ---
export CONSISTENCY_LOSS_TYPE=dflash_ce
export CONSISTENCY_WEIGHT=0.001          # user-specified lambda (launcher default was 0.01)
export CONSISTENCY_MASK_TAIL=1
export CONSISTENCY_MASK_TOKEN_ID=151643  # Qwen2.5 pad / dFlash mask token
export CONSISTENCY_MASK_MARGIN=8

# --- variant: on-policy real tail, no corruption ---
export CONSISTENCY_ONPOLICY=1
export CONSISTENCY_DRAFT_CORRUPT_PROB=0.0

exec bash scripts/run_dapo_jf_math_4gpu_jacobi_onpolicy.sh "$@"
