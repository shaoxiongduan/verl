#!/usr/bin/env bash
# Greedy (T=0.0) TPF grid for 3 models × 2 codebases.
set -eu
cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

PROMPTS=eval_passk/eval_prompts_tpf.jsonl
K=32
T=0.0
MAXNEW=1024
BS=16
OUT_DIR=eval_passk/tpf_results
mkdir -p "$OUT_DIR"

declare -A MODELS=(
  [base_jf_math_7b]=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1
  [math_ar_ds_step_300]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_ar_ds_step_300
  [onpolicy_f30_v2_step_300]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/onpolicy_f30_v2_step_300
)

run_jf() {
  local name=$1; local model=$2
  local out=$OUT_DIR/${name}__jf_T0.jsonl
  echo "=========================================="
  echo "=== JF nanovllm (T=0): $name"
  echo "=========================================="
  python3 scripts/tpf_trajectories.py \
    --model "$model" \
    --prompts_jsonl "$PROMPTS" \
    --output_jsonl "$out" \
    --jacobi_block_len $K --max_new_tokens $MAXNEW \
    --batch_size $BS --temperature $T \
    --jacobi_on_policy 0
}

run_vllm() {
  local name=$1; local model=$2
  local out=$OUT_DIR/${name}__vllm_T0.jsonl
  local traj=/tmp/vllm_tpf_traj_T0_${name}.jsonl
  rm -f "${traj}".*
  echo "=========================================="
  echo "=== vLLM jacobi_patch (T=0): $name"
  echo "=========================================="
  JACOBI_K=$K VLLM_TPF_TRAJ_PATH=$traj \
  python3 scripts/vllm_tpf_trajectories.py \
    --model "$model" \
    --prompts_jsonl "$PROMPTS" \
    --output_jsonl "$out" \
    --jacobi_block_len $K --max_new_tokens $MAXNEW \
    --batch_size $BS --temperature $T
}

for name in base_jf_math_7b math_ar_ds_step_300 onpolicy_f30_v2_step_300; do
  model=${MODELS[$name]}
  run_jf "$name" "$model"
  run_vllm "$name" "$model"
done

echo "ALL DONE — greedy results in $OUT_DIR"
