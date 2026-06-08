#!/usr/bin/env bash
# vLLM Jacobi TPF for {base, AR-RL, cons v2 step20, cons fixed step20} x {T=0 greedy, T=0.6 nongreedy}
set -eu
cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

PROMPTS=eval_passk/eval_prompts_tpf.jsonl
K=32
MAXNEW=1024
BS=16
OUT_DIR=eval_passk/tpf_results
mkdir -p "$OUT_DIR"

declare -A MODELS=(
  [base_jf_math_7b]=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1
  [math_ar_ds_step_300]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_ar_ds_step_300
  [dflashce_v2_step_20]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_v2_step_20
  [dflashce_fixed_step_20]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_fixed_step_20
)

run_vllm() {
  local name=$1; local model=$2; local T=$3; local tag=$4
  local out=$OUT_DIR/${name}__vllm_${tag}.jsonl
  local traj=/tmp/vllm_tpf_traj_${name}_${tag}.jsonl
  rm -f "${traj}".*
  echo "=========================================="
  echo "=== vLLM jacobi_patch (T=${T}): $name"
  echo "=========================================="
  JACOBI_K=$K VLLM_TPF_TRAJ_PATH=$traj \
  python3 scripts/vllm_tpf_trajectories.py \
    --model "$model" \
    --prompts_jsonl "$PROMPTS" \
    --output_jsonl "$out" \
    --jacobi_block_len $K --max_new_tokens $MAXNEW \
    --batch_size $BS --temperature $T 2>&1
}

for name in base_jf_math_7b math_ar_ds_step_300 dflashce_v2_step_20 dflashce_fixed_step_20; do
  model=${MODELS[$name]}
  run_vllm "$name" "$model" 0.0 T0
  run_vllm "$name" "$model" 0.6 T06
done

echo "=========================================="
echo "DONE"
ls -la "$OUT_DIR"/*step_20__vllm_T*.jsonl 2>/dev/null || true
