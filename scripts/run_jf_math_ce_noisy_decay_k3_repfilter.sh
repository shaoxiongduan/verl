#!/usr/bin/env bash
# IDENTICAL to run_jf_math_ce_noisy_decay_k3.sh (K=3 noise tiles, dflash_ce
# loss, causal noisy attention, target_ratio=0.4 + EMA(0.9), uniform noise,
# correct-only filter) — PLUS a repetition filter on the cons-loss pool.
#
# CONSISTENCY_REPETITION_FILTER=1 drops trajectories whose response tokens
# contain a degenerate loop (n-gram repeating 3+ times in last 200 tokens)
# from the cons-loss POOL ONLY. DAPO reward and PG gradient are unchanged.
#
# Hypothesis (2026-06-07): the ce_noisy_decay_k3 model exhibits ~50%
# degenerate-loop rate on math (see v4 oracle-reinit analysis,
# eval_passk/tpf_results/_check_v4_repetition output). The cons loss may be
# REINFORCING those loops by training the model to predict the looped
# tokens cleanly even when given pure-noise drafts. Filtering loops out of
# the cons pool should remove that reinforcement without changing the
# policy gradient.
set -x

cd /mnt/weka/home/hao.zhang/shao/verl

LOG_DIR=/mnt/weka/home/hao.zhang/shao/verl/logs
mkdir -p "${LOG_DIR}"

export EXP_NAME=jf_math_7b_dapo_ds_4gpu_cons_ce_noisy_decay_k3_repfilter
export PROJECT_NAME=jacobi_forcing_dapo_deepscaler
export CONSISTENCY_ENABLE=1
export CONSISTENCY_WEIGHT=0.001
export CONSISTENCY_CORRECT_ONLY=1

# === Repetition filter (NEW vs run_jf_math_ce_noisy_decay_k3.sh) ============
export CONSISTENCY_REPETITION_FILTER=1
export CONSISTENCY_REPETITION_TAIL_N=200
export CONSISTENCY_REPETITION_MIN_REPEATS=3
# ============================================================================

export CONSISTENCY_NOISE_SOURCE=uniform
export CONSISTENCY_NUM_NOISY_TILES=3
unset CONSISTENCY_MASK_TOKEN_ID

export CONSISTENCY_LOSS_TYPE=dflash_ce
export CONSISTENCY_DFLASH_GAMMA=12.0

export CONSISTENCY_CAUSAL_REGION_SIZE=0

export CONSISTENCY_TARGET_RATIO=0.4
export CONSISTENCY_TARGET_EMA_DECAY=0.9
export CONSISTENCY_TARGET_LAMBDA_MIN=1e-6
export CONSISTENCY_TARGET_LAMBDA_MAX=1e-1

export CONSISTENCY_USE_DRAFT_MARKER=0
unset CONSISTENCY_MARKER_TYPE
unset CONSISTENCY_SCHEDULE
unset CONSISTENCY_LOG_GRAD_NORMS

bash scripts/run_dapo_jf_math_4gpu_consistency.sh \
    > "${LOG_DIR}/${EXP_NAME}.log" 2>&1
echo "[jf_math_ce_noisy_decay_k3_repfilter] exited with code $?"
