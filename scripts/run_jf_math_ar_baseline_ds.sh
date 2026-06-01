#!/usr/bin/env bash
# Pure AR baseline on deepscaler (no consistency loss). Same base model
# + training data as run_jf_math_target01_emaschedule.sh / run_jf_math_ce_noisy_decay_k3.sh,
# so target01/k3 vs this is a clean cons-vs-no-cons comparison.
set -x

cd /mnt/weka/home/hao.zhang/shao/verl

LOG_DIR=/mnt/weka/home/hao.zhang/shao/verl/logs
mkdir -p "${LOG_DIR}"

export EXP_NAME=jf_math_7b_dapo_ds_4gpu_ar_baseline
export PROJECT_NAME=jacobi_forcing_dapo_deepscaler

# DISABLE cons hook — pure AR DAPO
export CONSISTENCY_ENABLE=0
unset CONSISTENCY_WEIGHT
unset CONSISTENCY_CORRECT_ONLY
unset CONSISTENCY_LOSS_TYPE
unset CONSISTENCY_NOISE_SOURCE
unset CONSISTENCY_NUM_NOISY_TILES
unset CONSISTENCY_MARKER_TYPE
unset CONSISTENCY_SCHEDULE
unset CONSISTENCY_TARGET_RATIO

# Use deepscaler train + multi-val (MATH + GSM8K), match cons ds runs
export TRAIN_FILE=/mnt/weka/home/hao.zhang/shao/verl/data/deepscaler/train.parquet
export VAL_FILES_LIST='[/mnt/weka/home/hao.zhang/shao/verl/data/openmathinstruct2/val.parquet,/mnt/weka/home/hao.zhang/shao/verl/data/gsm8k/test.parquet]'

# 300 steps to match the cons runs we're comparing against
export TOTAL_TRAINING_STEPS=300
export TOTAL_EPOCHS=1

bash scripts/run_dapo_jf_math_4gpu_consistency.sh \
    > "${LOG_DIR}/${EXP_NAME}.log" 2>&1
echo "[jf_math_ar_baseline_ds] exited with code $?"
