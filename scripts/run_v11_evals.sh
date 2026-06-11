#!/usr/bin/env bash
# v11 eval battery (spec v11_dllm_jacobi_hybrid.md §3.6) for one merged HF ckpt.
# Usage: run_v11_evals.sh <HF_CKPT_DIR> <TAG> [GPU]
#   1. causal regression: vanilla jsim + reppen (compare v9_220: 3.78 / 4.10)
#   2. canvas quality: assembly decode bidir-vs-causal control, marker on/off
#      (untrained v9_220: causal 3.64 / bidir 3.41; success = trained bidir
#       BEATS its causal control)
# Outputs to eval_passk/tpf_results/v11/<TAG>/*.jsonl + summary lines on stdout.
set -euo pipefail
cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

CKPT="${1:?usage: $0 <HF_CKPT_DIR> <TAG> [GPU]}"
TAG="${2:?usage: $0 <HF_CKPT_DIR> <TAG> [GPU]}"
export CUDA_VISIBLE_DEVICES="${3:-0}"
PROMPTS=eval_passk/deepscaler_tpf_prompts_16.jsonl
OUT=eval_passk/tpf_results/v11/${TAG}
mkdir -p "${OUT}"

echo "=== [v11-eval ${TAG}] 1/5 vanilla streaming jsim (v9_220 ref: 3.78) ==="
python scripts/_sim_jacobi_predictor_refresh.py --model "${CKPT}" \
  --prompts_jsonl "${PROMPTS}" --out_jsonl "${OUT}/vanilla_jsim.jsonl" \
  --refresh none --refresh_to random --max_new 512 --max_iters 512

echo "=== [v11-eval ${TAG}] 2/5 reppen decode (v9_220 ref: 4.10) ==="
python scripts/_sim_reppen_decode.py --model "${CKPT}" \
  --prompts_jsonl "${PROMPTS}" --out_jsonl "${OUT}/reppen.jsonl" \
  --policy reppen --lookback 2

echo "=== [v11-eval ${TAG}] 3/5 assembly causal-canvas control (no marker) ==="
python scripts/_sim_assembly_decode.py --model "${CKPT}" \
  --prompts_jsonl "${PROMPTS}" --out_jsonl "${OUT}/asm_causal.jsonl" \
  --canvas_attn causal --canvas_update keepnoise

echo "=== [v11-eval ${TAG}] 4/5 assembly bidir + marker (the v11 target) ==="
python scripts/_sim_assembly_decode.py --model "${CKPT}" \
  --prompts_jsonl "${PROMPTS}" --out_jsonl "${OUT}/asm_bidir_marker.jsonl" \
  --canvas_attn bidir --canvas_update keepnoise --marker constant

echo "=== [v11-eval ${TAG}] 5/5 assembly bidir NO marker (marker ablation) ==="
python scripts/_sim_assembly_decode.py --model "${CKPT}" \
  --prompts_jsonl "${PROMPTS}" --out_jsonl "${OUT}/asm_bidir_nomarker.jsonl" \
  --canvas_attn bidir --canvas_update keepnoise

echo "=== [v11-eval ${TAG}] DONE — corpus TPF lines above; jsonl in ${OUT} ==="
