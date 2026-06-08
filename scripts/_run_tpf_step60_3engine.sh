#!/usr/bin/env bash
# Benchmark v4 step_60 ckpts (dflashce_v2 + dflashce_corrupt03) plus base + AR on
# THREE engines: vLLM Jacobi (T=0 + T=1.0), JF reference (greedy), DL nanovllm (T=1.0).
# K=32, max_new=2048, n=64, BS=16.
set -eu
cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

PROMPTS_JSONL=eval_passk/eval_prompts_tpf.jsonl
PROMPTS_PARQUET=eval_passk/eval_prompts_tpf.parquet
NUM=64
K=32
MAXNEW=2048
BS=16
TAG=step60
OUT_DIR=eval_passk/tpf_results
mkdir -p "$OUT_DIR"

JF_PY=/mnt/weka/home/hao.zhang/shao/JacobiForcing/.venv/bin/python
JF_SCRIPT=/mnt/weka/home/hao.zhang/shao/JacobiForcing/JacobiForcing/jf_inference_he_our_models.py

declare -A MODELS=(
  [base_jf_math_7b]=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1
  [math_ar_ds_step_300]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_ar_ds_step_300
  [dflashce_v2_step_60]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_v2_step_60
  [dflashce_corrupt03_step_60]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_corrupt03_step_60
)

run_vllm() {
  local name=$1; local model=$2; local T=$3; local tag=$4
  local out=$OUT_DIR/${name}__vllm_${tag}.jsonl
  local traj=/tmp/vllm_tpf_traj_${tag}_${name}.jsonl
  rm -f "${traj}".*
  echo "=========================================="
  echo "=== vLLM jacobi_patch ($tag, T=$T, K=$K, max_new=$MAXNEW): $name"
  echo "=========================================="
  JACOBI_K=$K VLLM_TPF_TRAJ_PATH=$traj \
  python3 scripts/vllm_tpf_trajectories.py \
    --model "$model" \
    --prompts_jsonl "$PROMPTS_JSONL" \
    --output_jsonl "$out" \
    --jacobi_block_len $K --max_new_tokens $MAXNEW \
    --batch_size $BS --temperature $T 2>&1
}

run_jfref() {
  local name=$1; local model=$2
  local out=$OUT_DIR/${name}__jfref_${TAG}.log
  echo "=========================================="
  echo "=== JF reference jacobi_forward_greedy ($TAG, N=$K, max_new=$MAXNEW, GREEDY): $name"
  echo "=========================================="
  TPF_MODEL_PATH=$model \
  TPF_DATA_PATH=$PROMPTS_PARQUET \
  NUM_PROMPTS=$NUM \
  N_TOKEN_SEQ_LEN=$K \
  MAX_NEW_TOKENS=$MAXNEW \
  DRAFT_INIT=prompt_sample \
  "$JF_PY" "$JF_SCRIPT" 2>&1 | tee "$out"
}

run_dl() {
  local name=$1; local model=$2; local T=$3; local tag=$4; local op=$5
  local out=$OUT_DIR/${name}__dl_${tag}.jsonl
  echo "=========================================="
  echo "=== DL nanovllm ($tag, T=$T, on_policy=$op): $name"
  echo "=========================================="
  python3 scripts/tpf_trajectories.py \
    --model "$model" \
    --prompts_jsonl "$PROMPTS_JSONL" \
    --output_jsonl "$out" \
    --jacobi_block_len $K --max_new_tokens $MAXNEW \
    --batch_size $BS --temperature $T \
    --jacobi_on_policy $op 2>&1
}

for name in dflashce_v2_step_60 dflashce_corrupt03_step_60; do
  model=${MODELS[$name]}
  run_vllm  "$name" "$model" 0.0 greedy2048
  run_vllm  "$name" "$model" 1.0 trainparams
  run_jfref "$name" "$model"
  run_dl    "$name" "$model" 1.0 T1_2048 1
done

echo "=========================================="
echo "DONE"
