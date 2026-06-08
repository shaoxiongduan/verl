#!/usr/bin/env bash
# Decode-Learning nanovllm Jacobi TPF for 4 models x {T=0 greedy, T=1.0}.
# Engine: scripts/tpf_trajectories.py -> Decode-Learning/nanovllm jacobi decoder.
# Same params as vLLM/JF-ref grids: K=32, max_new=2048, BS=16, n=64.
set -eu
cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

PROMPTS=eval_passk/eval_prompts_tpf.jsonl
K=32
MAXNEW=2048
BS=16
OUT_DIR=eval_passk/tpf_results
mkdir -p "$OUT_DIR"

declare -A MODELS=(
  [base_jf_math_7b]=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1
  [math_ar_ds_step_300]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_ar_ds_step_300
  [dflashce_v2_step_20]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_v2_step_20
  [dflashce_fixed_step_20]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_fixed_step_20
)

run_dl() {
  local name=$1; local model=$2; local T=$3; local tag=$4; local op=$5
  local out=$OUT_DIR/${name}__dl_${tag}.jsonl
  echo "=========================================="
  echo "=== DL nanovllm ($tag, T=$T, on_policy=$op): $name"
  echo "=========================================="
  python3 scripts/tpf_trajectories.py \
    --model "$model" \
    --prompts_jsonl "$PROMPTS" \
    --output_jsonl "$out" \
    --jacobi_block_len $K --max_new_tokens $MAXNEW \
    --batch_size $BS --temperature $T \
    --jacobi_on_policy $op 2>&1
}

for name in base_jf_math_7b math_ar_ds_step_300 dflashce_v2_step_20 dflashce_fixed_step_20; do
  model=${MODELS[$name]}
  run_dl "$name" "$model" 1.0 T1_2048 1
done
echo "DONE"
