#!/usr/bin/env bash
# TPF benchmark using TRAINING-MATCHED params:
#   - vllm-jacobi (the training rollout engine): K=32, T=1.0, top_p=1.0,
#     top_k=-1, max_new_tokens=2048 (matches max_response_length).
#     Generation method: shift-by-1 windowed Jacobi refresh (next draft =
#     target_argmax[n_acc+1:], 1 bonus committed/iter).
#   - JacobiForcing reference (TRUSTED — not nanovllm): jacobi_forward_greedy
#     from /mnt/weka/home/hao.zhang/shao/JacobiForcing. N_TOKEN_SEQ_LEN=32,
#     MAX_NEW_TOKENS=2048, DRAFT_INIT=prompt_sample (JF default).
#     Generation method: fill-one-block (run up to 128 Jacobi iters per block
#     until accepted prefix emerges, then commit prefix + bonus and start
#     next block). Greedy only — no temperature.
# 4 models x 2 engines = 8 runs.
set -eu
cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

PROMPTS_JSONL=eval_passk/eval_prompts_tpf.jsonl
PROMPTS_PARQUET=eval_passk/eval_prompts_tpf.parquet
NUM=64
K=32
T=1.0
MAXNEW=2048
BS=16
TAG=trainparams
OUT_DIR=eval_passk/tpf_results
mkdir -p "$OUT_DIR"

JF_PY=/mnt/weka/home/hao.zhang/shao/JacobiForcing/.venv/bin/python
JF_SCRIPT=/mnt/weka/home/hao.zhang/shao/JacobiForcing/JacobiForcing/jf_inference_he_our_models.py

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
  echo "=== vLLM jacobi_patch ($TAG, T=$T, K=$K, max_new=$MAXNEW): $name"
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
  echo "=== JF reference jacobi_forward_greedy ($TAG, N=$K, max_new=$MAXNEW): $name"
  echo "=========================================="
  TPF_MODEL_PATH=$model \
  TPF_DATA_PATH=$PROMPTS_PARQUET \
  NUM_PROMPTS=$NUM \
  N_TOKEN_SEQ_LEN=$K \
  MAX_NEW_TOKENS=$MAXNEW \
  DRAFT_INIT=prompt_sample \
  "$JF_PY" "$JF_SCRIPT" 2>&1 | tee "$out"
}

for name in base_jf_math_7b math_ar_ds_step_300 dflashce_v2_step_20 dflashce_fixed_step_20; do
  model=${MODELS[$name]}
  run_vllm  "$name" "$model"
  run_jfref "$name" "$model"
done

echo "=========================================="
echo "DONE"
ls -la "$OUT_DIR"/*__{vllm_trainparams.jsonl,jfref_trainparams.log} 2>/dev/null || true
