#!/usr/bin/env bash
# Single-prompt apples-to-apples comparison: JF reference vs vLLM patch
# on base pre-RL model, greedy, K=32, MAX_NEW=256.
set -eu
cd /mnt/weka/home/hao.zhang/shao/verl

MODEL=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1
PROMPT_PQ=/mnt/weka/home/hao.zhang/shao/verl/eval_passk/single_prompt.parquet
PROMPT_JL=/mnt/weka/home/hao.zhang/shao/verl/eval_passk/single_prompt.jsonl
K=32
MAXNEW=256
OUT_DIR=eval_passk/single_trace_out
mkdir -p "$OUT_DIR"

#### 1) JF reference (HF transformers + jacobi_forward_greedy) ####
echo "=========================================="
echo "=== JF REFERENCE: 1 prompt, base, greedy"
echo "=========================================="
JF_PY=/mnt/weka/home/hao.zhang/shao/JacobiForcing/.venv/bin/python
JF_SCRIPT=/mnt/weka/home/hao.zhang/shao/JacobiForcing/JacobiForcing/jf_inference_he_our_models.py
TPF_MODEL_PATH=$MODEL TPF_DATA_PATH=$PROMPT_PQ \
  NUM_PROMPTS=1 N_TOKEN_SEQ_LEN=$K MAX_NEW_TOKENS=$MAXNEW DRAFT_INIT=prompt_sample \
  "$JF_PY" "$JF_SCRIPT" 2>&1 | tee "$OUT_DIR/jf_ref.log"

#### 2) vLLM patch ####
echo "=========================================="
echo "=== vLLM PATCH: 1 prompt, base, greedy"
echo "=========================================="
. .venv/bin/activate
rm -f /tmp/vllm_single_trace.jsonl.*
JACOBI_K=$K VLLM_TPF_TRAJ_PATH=/tmp/vllm_single_trace.jsonl \
  python3 scripts/vllm_tpf_trajectories.py \
    --model $MODEL \
    --prompts_jsonl $PROMPT_JL \
    --output_jsonl $OUT_DIR/vllm.jsonl \
    --jacobi_block_len $K --max_new_tokens $MAXNEW \
    --batch_size 1 --temperature 0.0 2>&1 | tee "$OUT_DIR/vllm.log"

echo "=========================================="
echo "=== TRAJECTORY FILES"
echo "=========================================="
ls -la /tmp/vllm_single_trace.jsonl.*

echo "ALL DONE"
