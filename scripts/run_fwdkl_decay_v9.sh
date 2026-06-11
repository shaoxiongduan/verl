#!/usr/bin/env bash
# v9: forward-KL-to-frozen-base cons loss (v8 recipe) + dFlash position decay
# on the KL (new CONSISTENCY_KL_DECAY path: pml-masked, w=exp(-(pos-pml)/gamma))
# + full-response cons coverage (MAX_PAIRS=64, random window subsample instead
# of first-N truncation).
set -euo pipefail
cd /mnt/weka/home/hao.zhang/shao/verl

export PROJECT_NAME=jacobi_forcing_dapo_deepscaler_onpolicy
export EXP_NAME=jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_kl_base_forward_decay_v9

# --- cons loss: forward KL to frozen base, decay-weighted ---
export CONSISTENCY_LOSS_TYPE=kl
export CONSISTENCY_DIVERGENCE=forward_kl
export CONSISTENCY_TEACHER=base
export CONSISTENCY_TEACHER_PATH=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1
export CONSISTENCY_KL_DECAY=1
export CONSISTENCY_DFLASH_GAMMA=12.0
export CONSISTENCY_WEIGHT=0.001
export CONSISTENCY_T_SOFT=1.0

# --- coverage: full response, uniform window subsample ---
export CONSISTENCY_MAX_PAIRS=64
export CONSISTENCY_PAIR_SAMPLE=random

# --- on-policy real cascade-tail inputs ---
export CONSISTENCY_ONPOLICY=1
export CONSISTENCY_DRAFT_CORRUPT_PROB=0.0

exec bash scripts/run_dapo_jf_math_4gpu_jacobi_onpolicy.sh "$@"
