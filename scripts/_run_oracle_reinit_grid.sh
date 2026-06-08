#!/usr/bin/env bash
# Launch 5 parallel sim runs (one per mode) on different GPUs of the
# shao_dll node, then merge JSONL outputs into a single file for analysis.
#
# Usage:
#   srun --overlap --jobid=<JOBID> bash scripts/_run_oracle_reinit_grid.sh
set -euo pipefail

CKPT=${CKPT:-/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/ce_noisy_decay_k3_step_300}
PROMPTS=${PROMPTS:-/mnt/weka/home/hao.zhang/shao/verl/eval_passk/deepscaler_tpf_prompts.jsonl}
OUTDIR=${OUTDIR:-/mnt/weka/home/hao.zhang/shao/verl/eval_passk/tpf_results}
MAX_NEW=${MAX_NEW:-512}
K=${K:-32}
SEED=${SEED:-42}
TAG=${TAG:-oracle_reinit_v1}
NOISE_SOURCE=${NOISE_SOURCE:-uniform}
GPU_BASE=${GPU_BASE:-0}
PY=${PY:-/mnt/weka/home/hao.zhang/shao/verl/.venv/bin/python}

mkdir -p "$OUTDIR"

declare -a MODES=(natural oracle_shift0 oracle_shift1 oracle_shift2 oracle_shift3)

pids=()
for i in "${!MODES[@]}"; do
  mode="${MODES[$i]}"
  gpu=$((GPU_BASE + i))
  out_jsonl="$OUTDIR/${TAG}__${mode}.jsonl"
  log="$OUTDIR/${TAG}__${mode}.log"
  echo "[grid] launching mode=$mode on GPU $gpu -> $out_jsonl  (noise=$NOISE_SOURCE)"
  CUDA_VISIBLE_DEVICES=$gpu "$PY" \
    /mnt/weka/home/hao.zhang/shao/verl/scripts/_sim_jacobi_oracle_reinit.py \
    --model "$CKPT" \
    --prompts_jsonl "$PROMPTS" \
    --out_jsonl "$out_jsonl" \
    --modes "$mode" \
    --noise_source "$NOISE_SOURCE" \
    --K "$K" --max_new "$MAX_NEW" --seed "$SEED" \
    > "$log" 2>&1 &
  pids+=($!)
done

echo "[grid] launched ${#pids[@]} jobs; waiting"
for pid in "${pids[@]}"; do
  wait "$pid"
done

echo "[grid] all done; merging"
cat "$OUTDIR"/${TAG}__*.jsonl > "$OUTDIR/${TAG}__merged.jsonl"
wc -l "$OUTDIR"/${TAG}__*.jsonl
echo "[grid] merged -> $OUTDIR/${TAG}__merged.jsonl"
