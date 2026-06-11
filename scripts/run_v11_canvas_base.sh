#!/usr/bin/env bash
# v11-base: identical canvas recipe to run_v11_canvas.sh but initialized from
# the PRE-RL JacobiForcing Math 7B base (not the v9_220 RL checkpoint).
# Question this answers: does canvas-mode training work better when RL and
# canvas learning happen together from scratch (policy not yet sharpened,
# entropy still high) vs. retrofitting the canvas onto a converged policy?
# Teacher = same base snapshot (at step 0, student == teacher, as in v9/v10).
set -euo pipefail
cd /mnt/weka/home/hao.zhang/shao/verl

export EXP_NAME=${EXP_NAME:-jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_v11_canvas_base}
export JF_MODEL=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1

exec bash scripts/run_v11_canvas.sh "$@"
