#!/usr/bin/env bash
# Postfix benchmark for trained ckpts on FIXED vLLM Jacobi:
#   BS=1 with gmu=0.6  (latency, where fix helps most)
#   BS=16 with gmu=0.85 (training-style batched throughput)
set -eu
cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

PROMPTS=eval_passk/eval_prompts_tpf.jsonl
K=32
MAXNEW=2048
OUT_DIR=eval_passk/tpf_results/postfix
mkdir -p "$OUT_DIR"

declare -A MODELS=(
  [base_jf_math_7b]=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1
  [math_ar_ds_step_300]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_ar_ds_step_300
  [dflashce_v2_step_60]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_v2_step_60
  [dflashce_corrupt03_step_60]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_corrupt03_step_60
)

run_one() {
  local name=$1; local model=$2; local T=$3; local BS=$4; local tag=$5; local GMU=$6
  local out=$OUT_DIR/${name}__${tag}.jsonl
  local traj=/tmp/vllm_pf_${tag}_${name}.jsonl
  rm -f "${traj}".*
  echo "=========================================="
  echo "=== POSTFIX: $name | T=$T BS_cap=$BS gmu=$GMU K=$K max_new=$MAXNEW ==="
  echo "=========================================="
  JACOBI_K=$K VLLM_TPF_TRAJ_PATH=$traj \
  python3 scripts/_bench_postfix_tpf_tps.py \
    --model "$model" --prompts_jsonl "$PROMPTS" --output_jsonl "$out" \
    --max_new_tokens $MAXNEW --max_num_seqs $BS --temperature $T --gpu_mem_util $GMU 2>&1 | tail -15
}

# Phase 1: BS=1 latency for all ckpts
for name in base_jf_math_7b math_ar_ds_step_300 dflashce_v2_step_60 dflashce_corrupt03_step_60; do
  model=${MODELS[$name]}
  run_one "$name" "$model" 0.0 1 greedy_bs1 0.6
  run_one "$name" "$model" 1.0 1 onpol_bs1  0.6
done

# Phase 2: BS=16 batched throughput
for name in base_jf_math_7b math_ar_ds_step_300 dflashce_v2_step_60 dflashce_corrupt03_step_60; do
  model=${MODELS[$name]}
  run_one "$name" "$model" 0.0 16 greedy_bs16 0.85
  run_one "$name" "$model" 1.0 16 onpol_bs16  0.85
done

echo "DONE"
