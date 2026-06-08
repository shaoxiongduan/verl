#!/usr/bin/env bash
# vLLM Jacobi greedy (T=0) at max_new=2048 — isolate temperature vs prior runs.
# Pairs with __vllm_trainparams.jsonl (same length cap, T=1.0) for delta-T analysis.
set -eu
cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

PROMPTS=eval_passk/eval_prompts_tpf.jsonl
K=32
T=0.0
MAXNEW=2048
BS=16
TAG=greedy2048
OUT_DIR=eval_passk/tpf_results
mkdir -p "$OUT_DIR"

declare -A MODELS=(
  [base_jf_math_7b]=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1
  [math_ar_ds_step_300]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_ar_ds_step_300
  [dflashce_v2_step_20]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_v2_step_20
  [dflashce_fixed_step_20]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_fixed_step_20
)

for name in base_jf_math_7b math_ar_ds_step_300 dflashce_v2_step_20 dflashce_fixed_step_20; do
  model=${MODELS[$name]}
  out=$OUT_DIR/${name}__vllm_${TAG}.jsonl
  traj=/tmp/vllm_tpf_traj_${TAG}_${name}.jsonl
  rm -f "${traj}".*
  echo "=========================================="
  echo "=== vLLM jacobi_patch ($TAG, T=$T, K=$K, max_new=$MAXNEW): $name"
  echo "=========================================="
  JACOBI_K=$K VLLM_TPF_TRAJ_PATH=$traj \
  python3 scripts/vllm_tpf_trajectories.py \
    --model "$model" \
    --prompts_jsonl "$PROMPTS" \
    --output_jsonl "$out" \
    --jacobi_block_len $K --max_new_tokens $MAXNEW \
    --batch_size $BS --temperature $T 2>&1
done
echo "DONE"
