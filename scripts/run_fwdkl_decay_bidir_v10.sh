#!/usr/bin/env bash
# v10 = v9 recipe (fwd-KL-to-base + KL decay + 64-pair random coverage,
# on-policy inputs) + TiDAR-style HYBRID attention inside the noisy block:
# first 8 intra-block positions stay causal (the commit/verify region),
# the remaining 24 attend bidirectionally within the block (the refinement
# scaffold). Trains a dual-mode drafter: causal frontier + bidir canvas.
set -euo pipefail
cd /mnt/weka/home/hao.zhang/shao/verl

export PROJECT_NAME=jacobi_forcing_dapo_deepscaler_onpolicy
export EXP_NAME=jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_kl_decay_bidir8_v10

export CONSISTENCY_LOSS_TYPE=kl
export CONSISTENCY_DIVERGENCE=forward_kl
export CONSISTENCY_TEACHER=base
export CONSISTENCY_TEACHER_PATH=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1
export CONSISTENCY_KL_DECAY=1
export CONSISTENCY_DFLASH_GAMMA=12.0
export CONSISTENCY_WEIGHT=0.001
export CONSISTENCY_T_SOFT=1.0

export CONSISTENCY_MAX_PAIRS=64
export CONSISTENCY_PAIR_SAMPLE=random

export CONSISTENCY_ONPOLICY=1
export CONSISTENCY_DRAFT_CORRUPT_PROB=0.0

# --- the v10 delta: hybrid causal(8) + bidir(24) mask in the noisy block ---
export CONSISTENCY_CAUSAL_REGION_SIZE=8

exec bash scripts/run_dapo_jf_math_4gpu_jacobi_onpolicy.sh "$@"
