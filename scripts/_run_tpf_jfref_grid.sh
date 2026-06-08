#!/usr/bin/env bash
# JF reference TPF (greedy, jacobi_forward_greedy from JacobiForcing repo)
# on 3 models: base, AR, KL-cons.
set -eu
cd /mnt/weka/home/hao.zhang/shao/verl

JF_PY=/mnt/weka/home/hao.zhang/shao/JacobiForcing/.venv/bin/python
JF_SCRIPT=/mnt/weka/home/hao.zhang/shao/JacobiForcing/JacobiForcing/jf_inference_he_our_models.py
PROMPTS_PARQUET=/mnt/weka/home/hao.zhang/shao/verl/eval_passk/eval_prompts_tpf.parquet

NUM=64
N=32   # block size; match our training
MAXNEW=1024
DRAFT_INIT=${DRAFT_INIT:-prompt_sample}

OUT_DIR=eval_passk/tpf_results
mkdir -p "$OUT_DIR"

declare -A MODELS=(
  [base_jf_math_7b]=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1
  [math_ar_ds_step_300]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_ar_ds_step_300
  [onpolicy_f30_v2_step_300]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/onpolicy_f30_v2_step_300
)

for name in base_jf_math_7b math_ar_ds_step_300 onpolicy_f30_v2_step_300; do
  model=${MODELS[$name]}
  out=$OUT_DIR/${name}__jfref.log
  echo "=========================================="
  echo "=== JF reference (greedy, N=$N, draft_init=$DRAFT_INIT): $name"
  echo "=========================================="
  TPF_MODEL_PATH=$model \
  TPF_DATA_PATH=$PROMPTS_PARQUET \
  NUM_PROMPTS=$NUM \
  N_TOKEN_SEQ_LEN=$N \
  MAX_NEW_TOKENS=$MAXNEW \
  DRAFT_INIT=$DRAFT_INIT \
  "$JF_PY" "$JF_SCRIPT" 2>&1 | tee "$out"
done

echo "ALL DONE — logs in $OUT_DIR/*__jfref.log"
