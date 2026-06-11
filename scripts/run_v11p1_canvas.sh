#!/usr/bin/env bash
# v11.1: canvas recipe revised from the 2026-06-12 decode-trace audit + matrix
# (TPF_RESULTS_LOG entries). Three deltas over run_v11_canvas.sh:
#   1. EMPIRICAL canvas construction — tiles match the measured hybrid-decode
#      state distribution (model-plausible tokens from the policy's own
#      cascade pools with position-decaying correctness P_NEAR→P_FAR, fresh-
#      noise tail of U{0..MAX_COMMIT} newly-entered positions) instead of the
#      f-mixture, which spent ~40% of budget on states decode never visits.
#      25% of canvas pairs keep the f-mixture (all-noise regime coverage).
#   2. CANVAS_LOSS=ce — canvas positions train CE to the clean ROLLOUT tokens
#      (decode-aligned: the AR zone verifies the policy's greedy text;
#      KL-to-base capped canvas accuracy at base↔policy agreement ~0.71).
#      Causal pairs keep the v9 fwd-KL+decay-to-base recipe untouched.
#   3. CANVAS_WEIGHT_MULT=10 — canvas term was ~1% of gradient budget
#      (cons_loss_effective 2.5e-4 vs pg 0.013); canvas argmax plateaued
#      0.32-0.38 from step 40. 10x puts it at ~10-20% of PG magnitude.
# Decode-side companion (eval): keep-everything update (argmax / tau>=4).
set -euo pipefail
cd /mnt/weka/home/hao.zhang/shao/verl

export EXP_NAME=${EXP_NAME:-jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_v11p1_canvas}

# --- v11.1 deltas ---
export CONSISTENCY_CANVAS_CONSTRUCTION=empirical
export CONSISTENCY_CANVAS_LEVELS_FRAC=0.25
export CONSISTENCY_CANVAS_P_NEAR=0.6
export CONSISTENCY_CANVAS_P_FAR=0.1
export CONSISTENCY_CANVAS_MAX_COMMIT=8
export CONSISTENCY_CANVAS_LOSS=ce
export CONSISTENCY_CANVAS_WEIGHT_MULT=10.0

exec bash scripts/run_v11_canvas.sh "$@"
