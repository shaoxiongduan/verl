#!/usr/bin/env bash
# vLLM Jacobi vs dFlash 2-forward TPS sweep on JF Math 7B (base, pre-RL).
# vLLM Jacobi: BS=1,4,16,32,64. K=32. max_new=512 (so all bs finish < 1 min).
# dFlash: single-prompt loop, K=32.
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

echo "==================== vLLM Jacobi TPS sweep (K=$K, max_new=$MAXNEW) ===================="
for BS in 1 4 16 32 64; do
  echo "--- vLLM Jacobi BS=$BS ---"
  traj=/tmp/_tpsbench_${BS}.jsonl; rm -f "${traj}".*
  JACOBI_K=$K VLLM_TPF_TRAJ_PATH=$traj \
  python3 scripts/vllm_tpf_trajectories.py \
    --model "$MODEL" \
    --prompts_jsonl "$PROMPTS" \
    --output_jsonl "$OUT/vllm_jacobi_BS${BS}.jsonl" \
    --jacobi_block_len $K --max_new_tokens $MAXNEW \
    --batch_size $BS --temperature 0.0 2>&1 | tail -25
done

echo
echo "==================== dFlash 2-forward TPS (single-prompt loop, K=$K, max_new=$MAXNEW) ===================="
JF_PY=/mnt/weka/home/hao.zhang/shao/JacobiForcing/.venv/bin/python
JF_SCRIPT=/mnt/weka/home/hao.zhang/shao/JacobiForcing/JacobiForcing/jf_inference_dflash_2forward.py
TPF_MODEL_PATH=$MODEL \
TPF_DATA_PATH=$PARQUET \
NUM_PROMPTS=64 \
N_TOKEN_SEQ_LEN=$K \
MAX_NEW_TOKENS=$MAXNEW \
"$JF_PY" "$JF_SCRIPT" 2>&1 | tee "$OUT/dflash_2forward_bs1.log" | tail -80

echo "==================== DONE ===================="
