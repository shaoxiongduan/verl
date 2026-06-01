#!/usr/bin/env bash
# Wrapper: runs two cons-RL experiments sequentially on the same 4-GPU node.
#
# Both use the post-fix verl_hook normalization (cons gradient now scales like PG).
# CONSISTENCY_LOG_GRAD_NORMS=1 logs per-microbatch local cons/pg grad norms
# so we can confirm the normalization fix worked in wandb.
#
#   Run A: target_ratio=0.3 + no marker + KL  (mirror of sinus_target03 minus marker)
#   Run B: target_ratio=0.3 + no marker + CE  (same as A, with CE consistency loss)
#
# Note: launching back-to-back avoids Ray port + FSDP-offload RAM contention on
# a single H200 host (see feedback_no_parallel_fsdp_offload memory).

set -x

cd /mnt/weka/home/hao.zhang/shao/verl

LOG_DIR=/mnt/weka/home/hao.zhang/shao/verl/logs
mkdir -p "${LOG_DIR}"

# ---------------- Run A: KL, no marker, target_ratio=0.3 ----------------
# Clamp relaxed to [1e-6, 1e-1] (was [1e-5, 1e-2]). Pre-fix the cap at 1e-2
# masked target_ratio's per-MB enforcement during high-pg microbatches —
# weight got pinned to the ceiling instead of tracking 0.3 × pg/cons. Widening
# the band lets target_lambda actually reflect the ratio.
export EXP_NAME=jf_coder_7b_dapo_oci_4gpu_cons_correct_only_target03_no_marker
export PROJECT_NAME=jacobi_forcing_dapo_opencodeinstruct
export CONSISTENCY_ENABLE=1
export CONSISTENCY_WEIGHT=0.001
export CONSISTENCY_CORRECT_ONLY=1
export CONSISTENCY_TARGET_RATIO=0.3
export CONSISTENCY_TARGET_LAMBDA_MIN=1e-6
export CONSISTENCY_TARGET_LAMBDA_MAX=1e-1
export CONSISTENCY_USE_DRAFT_MARKER=0
unset CONSISTENCY_MARKER_TYPE
unset CONSISTENCY_LOSS_TYPE  # defaults to kl
unset CONSISTENCY_LOG_GRAD_NORMS  # FSDP2 makes autograd.grad return 0 anyway
bash scripts/run_dapo_jf_coder_4gpu_consistency.sh \
    > "${LOG_DIR}/${EXP_NAME}.log" 2>&1
echo "[pair] Run A exited with code $?"

# ---------------- Run B: CE, no marker, target_ratio=0.3 ----------------
# Sentinel: if /tmp/skip_run_b_in_wrapper exists, B has been (or is being)
# launched separately (e.g. on a different node). Skip to avoid double-launch.
if [ -f /tmp/skip_run_b_in_wrapper ]; then
    echo "[pair] /tmp/skip_run_b_in_wrapper present — Run B handled elsewhere, exiting."
    exit 0
fi

export EXP_NAME=jf_coder_7b_dapo_oci_4gpu_cons_correct_only_target03_no_marker_ce
export PROJECT_NAME=jacobi_forcing_dapo_opencodeinstruct
export CONSISTENCY_ENABLE=1
export CONSISTENCY_WEIGHT=0.001
export CONSISTENCY_CORRECT_ONLY=1
export CONSISTENCY_TARGET_RATIO=0.3
export CONSISTENCY_USE_DRAFT_MARKER=0
export CONSISTENCY_LOG_GRAD_NORMS=1
export CONSISTENCY_LOSS_TYPE=ce
unset CONSISTENCY_MARKER_TYPE
bash scripts/run_dapo_jf_coder_4gpu_consistency.sh \
    > "${LOG_DIR}/${EXP_NAME}.log" 2>&1
echo "[pair] Run B exited with code $?"
