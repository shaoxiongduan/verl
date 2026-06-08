#!/usr/bin/env bash
# TPF benchmark using TRAINING-MATCHED params: K=32, T=1.0, top_p=1.0, top_k=-1, max_new=2048.
# Two engines: vllm-jacobi (matches the rollout engine used during cons-RL),
# and jf-nanovllm (block decode, JacobiForcing reference).
# 4 models x 2 engines = 8 runs.
set -eu
cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

PROMPTS=eval_passk/eval_prompts_tpf.jsonl
K=32
T=1.0
MAXNEW=2048
BS=16
OUT_DIR=eval_passk/tpf_results
TAG=trainparams
mkdir -p "$OUT_DIR"

declare -A MODELS=(
  [base_jf_math_7b]=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1
  [math_ar_ds_step_300]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_ar_ds_step_300
  [dflashce_v2_step_20]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_v2_step_20
  [dflashce_fixed_step_20]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_fixed_step_20
)

run_vllm() {
  local name=$1; local model=$2
  local out=$OUT_DIR/${name}__vllm_${TAG}.jsonl
  local traj=/tmp/vllm_tpf_traj_${TAG}_${name}.jsonl
  rm -f "${traj}".*
  echo "=========================================="
  echo "=== vLLM jacobi_patch ($TAG, T=$T): $name"
  echo "=========================================="
  JACOBI_K=$K VLLM_TPF_TRAJ_PATH=$traj \
  python3 scripts/vllm_tpf_trajectories.py \
    --model "$model" \
    --prompts_jsonl "$PROMPTS" \
    --output_jsonl "$out" \
    --jacobi_block_len $K --max_new_tokens $MAXNEW \
    --batch_size $BS --temperature $T 2>&1
}

run_jf() {
  local name=$1; local model=$2
  local out=$OUT_DIR/${name}__jf_${TAG}.jsonl
  echo "=========================================="
  echo "=== JF nanovllm ($TAG, T=$T, block-decode): $name"
  echo "=========================================="
  python3 scripts/tpf_trajectories.py \
    --model "$model" \
    --prompts_jsonl "$PROMPTS" \
    --output_jsonl "$out" \
    --jacobi_block_len $K --max_new_tokens $MAXNEW \
    --batch_size $BS --temperature $T \
    --jacobi_on_policy 1 2>&1
}

for name in base_jf_math_7b math_ar_ds_step_300 dflashce_v2_step_20 dflashce_fixed_step_20; do
  model=${MODELS[$name]}
  run_vllm "$name" "$model"
  run_jf "$name" "$model"
done

echo "=========================================="
echo "DONE"
ls -la "$OUT_DIR"/*__{vllm,jf}_${TAG}.jsonl 2>/dev/null || true
