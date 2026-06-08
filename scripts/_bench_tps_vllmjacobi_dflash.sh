#!/usr/bin/env bash
# Proper BS sweep using max_num_seqs cap.
set -eu
cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

MODEL=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1
PROMPTS=eval_passk/eval_prompts_tpf.jsonl
PARQUET=eval_passk/eval_prompts_tpf.parquet
K=32
MAXNEW=512
OUT=eval_passk/tpf_results/tps_vllm_vs_dflash
mkdir -p "$OUT"

echo "==================== vLLM Jacobi proper BS sweep K=$K max_new=$MAXNEW ===================="
for BS in 1 4 16 32 64; do
  echo "--- max_num_seqs=$BS ---"
  JACOBI_K=$K python3 scripts/_bench_vllm_jacobi_bs.py \
    --model "$MODEL" --prompts_jsonl "$PROMPTS" --max_new_tokens $MAXNEW \
    --max_num_seqs $BS --temperature 0.0 2>&1 | tail -10
done

echo
echo "==================== dFlash 2-forward (single-prompt loop) ===================="
JF_PY=/mnt/weka/home/hao.zhang/shao/JacobiForcing/.venv/bin/python
JF_SCRIPT=/mnt/weka/home/hao.zhang/shao/JacobiForcing/JacobiForcing/jf_inference_dflash_2forward.py
TPF_MODEL_PATH=$MODEL TPF_DATA_PATH=$PARQUET NUM_PROMPTS=64 \
  N_TOKEN_SEQ_LEN=$K MAX_NEW_TOKENS=$MAXNEW \
  "$JF_PY" "$JF_SCRIPT" 2>&1 | tee "$OUT/dflash_2forward_bs1.log" | tail -50

echo "==================== DONE ===================="
