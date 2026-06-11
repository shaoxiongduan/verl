#!/usr/bin/env bash
# v11: dLLM-Jacobi hybrid ("assembly line") canvas training.
# Spec: v11_dllm_jacobi_hybrid.md (§3.4 run config). Init from the best causal
# checkpoint (v9 step 220: vanilla TPF 3.78 / reppen 4.10 / MATH val ~0.886).
#
# Per-pair mode mix (CONSISTENCY_CANVAS_FRAC):
#   causal pairs (β=0.5)  — v9 recipe exactly: on-policy cascade input, causal
#                            intra-tile attention, unmarked, fwd-KL + decay.
#   canvas pairs (1-β)    — §3.2: clean window with f·N positions renoised to
#                            fresh uniform randoms (f ~ U{levels}, far-weighted),
#                            ρ=0.1 plausible subs from cascade alt pools, FULL
#                            bidir intra-tile, constant sinusoidal marker ON,
#                            uniform-weight fwd-KL to clean-context logits.
# Canvas pairs are hardcoded full-bidir per-pair (attention.py canvas_pairs);
# CONSISTENCY_CAUSAL_REGION_SIZE stays 0 and governs causal pairs only.
#
# Watch: actor/cons_argmax_correct_canvas (canvas progress),
#        actor/cons_argmax_correct_causal (must hold v9 levels ~0.35→0.44),
#        MATH val (≥ ~0.88). Decision metric = offline TPF at ckpts
#        160/200/220/260 — merge promptly, max_ckpt_to_keep=5 deletes old ones.
set -euo pipefail
cd /mnt/weka/home/hao.zhang/shao/verl

export PROJECT_NAME=jacobi_forcing_dapo_deepscaler_onpolicy
export EXP_NAME=${EXP_NAME:-jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_v11_canvas}

# --- init from best causal ckpt (v9 s220) ---
export JF_MODEL=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/fwdkl_decay_v9_step_220

# --- cons loss (causal-pair half): forward KL to frozen base, decay-weighted ---
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

# --- on-policy real cascade-tail inputs (causal-pair half) ---
export CONSISTENCY_ONPOLICY=1
export CONSISTENCY_DRAFT_CORRUPT_PROB=0.0

# --- v11 canvas mode ---
export CONSISTENCY_CANVAS_FRAC=0.5
export CONSISTENCY_CANVAS_LEVELS=1.0,0.75,0.5,0.25,0.125
export CONSISTENCY_CANVAS_PLAUSIBLE_FRAC=0.1
export CONSISTENCY_USE_DRAFT_MARKER=1
export CONSISTENCY_MARKER_TYPE=constant      # fixed vector -> decode-matchable
export CONSISTENCY_MARKER_INIT_SCALE=1.0
export CONSISTENCY_CAUSAL_REGION_SIZE=0      # causal pairs stay pure causal

exec bash scripts/run_dapo_jf_math_4gpu_jacobi_onpolicy.sh "$@"
