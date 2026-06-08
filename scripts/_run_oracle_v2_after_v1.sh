#!/usr/bin/env bash
# Wait for v1 grid to finish (all 5 per-mode JSONLs reach 64 rows), then launch
# v2 with NOISE_SOURCE=uniform on the same GPUs. v1 used prompt_sample (the JF
# reference convention); v2 matches the cons_ce_noisy_decay_k3 training
# distribution (uniform random vocab).
set -euo pipefail

V1_DIR=/mnt/weka/home/hao.zhang/shao/verl/eval_passk/tpf_results
EXPECTED=64

echo "[v2-wait] waiting for v1 grid to finish (target ${EXPECTED} rows per mode)"
until [ "$(wc -l < $V1_DIR/oracle_reinit_v1__natural.jsonl 2>/dev/null || echo 0)" -ge $EXPECTED ] \
   && [ "$(wc -l < $V1_DIR/oracle_reinit_v1__oracle_shift0.jsonl 2>/dev/null || echo 0)" -ge $EXPECTED ] \
   && [ "$(wc -l < $V1_DIR/oracle_reinit_v1__oracle_shift1.jsonl 2>/dev/null || echo 0)" -ge $EXPECTED ] \
   && [ "$(wc -l < $V1_DIR/oracle_reinit_v1__oracle_shift2.jsonl 2>/dev/null || echo 0)" -ge $EXPECTED ] \
   && [ "$(wc -l < $V1_DIR/oracle_reinit_v1__oracle_shift3.jsonl 2>/dev/null || echo 0)" -ge $EXPECTED ]; do
  sleep 8
done
echo "[v2-wait] v1 done — running v2 (uniform noise)"

TAG=oracle_reinit_v2_uniform \
NOISE_SOURCE=uniform \
GPU_BASE=0 \
MAX_NEW=512 \
bash /mnt/weka/home/hao.zhang/shao/verl/scripts/_run_oracle_reinit_grid.sh
