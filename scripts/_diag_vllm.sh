#!/usr/bin/env bash
# Diagnose vLLM Jacobi slowness. Test all 4 modes x BS=1,4 with CUDA graphs.
set -eu
cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

MODEL=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1
PROMPTS=eval_passk/eval_prompts_tpf.jsonl
MAXNEW=256
N=8

for BS in 1 4; do
  for MODE in ar ngram jacobi; do
    echo "=========================================="
    echo "MODE=$MODE BS=$BS (cuda-graphs ON)"
    echo "=========================================="
    VLLM_DIAG_MODE=$MODE JACOBI_K=32 \
      python3 scripts/_diag_vllm_speed.py \
      --model "$MODEL" --prompts_jsonl "$PROMPTS" \
      --max_new_tokens $MAXNEW --max_num_seqs $BS --n_prompts $N 2>&1 | grep -E "^\[diag\]|enforce_eager|Capturing|Captured"
  done
done

echo "=========================================="
echo "RE-RUN with enforce_eager=True (no CUDA graphs) — only Jacobi BS=1, BS=4"
echo "=========================================="
for BS in 1 4; do
  VLLM_DIAG_MODE=jacobi JACOBI_K=32 \
    python3 scripts/_diag_vllm_speed.py \
    --model "$MODEL" --prompts_jsonl "$PROMPTS" \
    --max_new_tokens $MAXNEW --max_num_seqs $BS --n_prompts $N \
    --enforce_eager 2>&1 | grep -E "^\[diag\]"
done

echo "DONE"
