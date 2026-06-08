#!/usr/bin/env bash
# Re-benchmark all major ckpts on DeepScaler-sampled prompts to match training
# distribution. T=1.0 trainparams, K=32, max_new=2048, BS=16, n=64.
set -eu
cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

PROMPTS=eval_passk/deepscaler_tpf_prompts.jsonl
OUT_DIR=eval_passk/tpf_results
mkdir -p "$OUT_DIR"

CUDA_DEV="${CUDA_DEV:-0}"

declare -A MODELS=(
  [base_jf_math_7b]=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1
  [math_ar_ds_step_300]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_ar_ds_step_300
  [dflashce_v2_step_20]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_v2_step_20
  [dflashce_fixed_step_20]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_fixed_step_20
  [dflashce_corrupt03_v4_step_20]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_corrupt03_v4_step_20
)

for name in "${!MODELS[@]}"; do
  model=${MODELS[$name]}
  out=$OUT_DIR/${name}__vllm_deepscaler_T1.jsonl
  traj=/tmp/vllm_tpf_traj_ds_${name}.jsonl
  rm -f "${traj}".*
  echo "=========================================="
  echo "=== DeepScaler T=1.0 max_new=2048 K=32: $name"
  echo "=========================================="
  CUDA_VISIBLE_DEVICES=$CUDA_DEV JACOBI_K=32 VLLM_TPF_TRAJ_PATH=$traj \
  python3 scripts/vllm_tpf_trajectories.py \
    --model "$model" \
    --prompts_jsonl "$PROMPTS" \
    --output_jsonl "$out" \
    --jacobi_block_len 32 --max_new_tokens 2048 \
    --batch_size 16 --temperature 1.0 2>&1
done
echo "=========================================="
echo "DONE"
