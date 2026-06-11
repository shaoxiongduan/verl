#!/usr/bin/env bash
# v11.1 recipe (empirical canvas + CE-to-rollout + mult=10) initialized from
# the PRE-RL JacobiForcing Math 7B base — canvas+RL co-training from an
# unsharpened policy (user call 2026-06-12: don't retrofit onto v9_220).
set -euo pipefail
cd /mnt/weka/home/hao.zhang/shao/verl

export EXP_NAME=${EXP_NAME:-jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_v11p1_canvas_base}
export JF_MODEL=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1

exec bash scripts/run_v11p1_canvas.sh "$@"
