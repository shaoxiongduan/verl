#!/usr/bin/env bash
# Run 3x2 TPF benchmark grid: {base, AR, KL-cons} × {JF nanovllm, vLLM jacobi_patch}
# Output one jsonl per (model, codebase) combo in eval_passk/tpf_results/
set -eu
cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

PROMPTS=eval_passk/eval_prompts_tpf.jsonl
K=32
T=0.6
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
  local out=$OUT_DIR/${name}__jf.jsonl
  echo "=========================================="
  echo "=== JF nanovllm: $name"
  echo "=========================================="
  python3 scripts/tpf_trajectories.py \
    --model "$model" \
    --prompts_jsonl "$PROMPTS" \
    --output_jsonl "$out" \
    --jacobi_block_len $K --max_new_tokens $MAXNEW \
    --batch_size $BS --temperature $T
}

run_vllm() {
  local name=$1; local model=$2
  local out=$OUT_DIR/${name}__vllm.jsonl
  local traj=/tmp/vllm_tpf_traj_${name}.jsonl
  rm -f "${traj}".*
  echo "=========================================="
  echo "=== vLLM jacobi_patch: $name"
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

echo "=========================================="
echo "ALL DONE — results in $OUT_DIR"
echo "=========================================="
ls -la "$OUT_DIR"
