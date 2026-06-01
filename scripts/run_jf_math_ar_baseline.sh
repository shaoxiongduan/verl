#!/usr/bin/env bash
# Pure AR baseline on full OMI (no consistency loss). Trains until OMI epoch
# end (~104 steps with gen_oversample=64). Useful baseline against the cons
# variants trained on the same data, same model, same eval.
set -x

cd /mnt/weka/home/hao.zhang/shao/verl

LOG_DIR=/mnt/weka/home/hao.zhang/shao/verl/logs
mkdir -p "${LOG_DIR}"

export EXP_NAME=jf_math_7b_dapo_omi_4gpu_ar_baseline
export PROJECT_NAME=jacobi_forcing_dapo_openmathinstruct2

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

# Override base-script defaults to use OMI data (not deepscaler)
export TRAIN_FILE=/mnt/weka/home/hao.zhang/shao/verl/data/openmathinstruct2/train.parquet
# Keep multi-source val for cross-comparability with the cons runs
export VAL_FILES_LIST='[/mnt/weka/home/hao.zhang/shao/verl/data/openmathinstruct2/val.parquet,/mnt/weka/home/hao.zhang/shao/verl/data/gsm8k/test.parquet]'

# Total steps to roughly match OMI epoch end (~104 steps with gen_oversample=64)
export TOTAL_TRAINING_STEPS=120
export TOTAL_EPOCHS=1

bash scripts/run_dapo_jf_math_4gpu_consistency.sh \
    > "${LOG_DIR}/${EXP_NAME}.log" 2>&1
echo "[jf_math_ar_baseline] exited with code $?"
