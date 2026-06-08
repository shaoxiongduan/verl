#!/usr/bin/env bash
# Bash loop wrapper — one Python process per (mode, BS) so the Jacobi patch
# state cannot leak across LLM instances.
set -uo pipefail

cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

TARGET=${TARGET:-/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1}
PROMPTS=${PROMPTS:-/mnt/weka/home/hao.zhang/shao/verl/eval_passk/eval_prompts_tpf.jsonl}
MAXNEW=${MAXNEW:-1024}
K=${K:-32}
GPU_MEM_UTIL=${GPU_MEM_UTIL:-0.85}
BS_LIST=${BS_LIST:-"1 2 4 8 16 32 64"}
MODES=${MODES:-"jacobi ar"}
OUT_CSV=${OUT_CSV:-eval_passk/tpf_results/dflash_qwen3_8b/bs_sweep_jacobi.csv}

mkdir -p "$(dirname "$OUT_CSV")"
echo "mode,bs,wall_s,output_tokens,tps" > "$OUT_CSV"

for MODE in $MODES; do
  for BS in $BS_LIST; do
    echo "[run] mode=$MODE  bs=$BS  K=$K  max_new=$MAXNEW" >&2
    line=$(python /mnt/weka/home/hao.zhang/shao/verl/scripts/_bench_vllm_jacobi_one.py \
      --target "$TARGET" --prompts_jsonl "$PROMPTS" \
      --max_new_tokens "$MAXNEW" --K "$K" --bs "$BS" --mode "$MODE" \
      --gpu_mem_util "$GPU_MEM_UTIL" 2>&1 | grep '^CSVROW,' | tail -1)
    if [ -n "$line" ]; then
      echo "${line#CSVROW,}" >> "$OUT_CSV"
      echo "[ok]  $line" >&2
    else
      echo "[fail] mode=$MODE bs=$BS — no CSVROW emitted" >&2
    fi
  done
done

echo "[done] CSV -> $OUT_CSV" >&2
