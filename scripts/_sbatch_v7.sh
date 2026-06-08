#!/usr/bin/env bash
#SBATCH --job-name=shao_dllm
#SBATCH --nodes=1
#SBATCH --gpus=8
#SBATCH --time=120:00:00
#SBATCH --output=/mnt/weka/home/hao.zhang/shao/verl/logs/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_ce_pureclean_v7.log
#SBATCH --error=/mnt/weka/home/hao.zhang/shao/verl/logs/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_ce_pureclean_v7.log

set -eu
cd /mnt/weka/home/hao.zhang/shao/verl

export EXP_NAME=jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_ce_pureclean_v7
export CONSISTENCY_LOSS_TYPE=ce
export CONSISTENCY_FOCAL_GAMMA=0.0
export CONSISTENCY_NUM_NOISY_TILES=1
export CONSISTENCY_DRAFT_CORRUPT_PROB=0.0
export CONSISTENCY_WEIGHT=0.001
export CONSISTENCY_TARGET_RATIO=0.0

bash scripts/run_dapo_jf_math_4gpu_jacobi_onpolicy.sh
