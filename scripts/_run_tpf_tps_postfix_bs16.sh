#!/usr/bin/env bash
# BS=16 phase only — original grid failed BS=16 due to 992 cap bug, now patched.
set -eu
cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

PROMPTS=eval_passk/eval_prompts_tpf.jsonl
K=32
MAXNEW=2048
OUT_DIR=eval_passk/tpf_results/postfix
mkdir -p "$OUT_DIR"

declare -A MODELS=(
  [base_jf_math_7b]=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1
  [math_ar_ds_step_300]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_ar_ds_step_300
  [dflashce_v2_step_60]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_v2_step_60
  [dflashce_corrupt03_step_60]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_corrupt03_step_60
)

for name in base_jf_math_7b math_ar_ds_step_300 dflashce_v2_step_60 dflashce_corrupt03_step_60; do
  model=${MODELS[$name]}
  for T in 0.0 1.0; do
    tag=$([ "$T" = "0.0" ] && echo greedy_bs16 || echo onpol_bs16)
    out=$OUT_DIR/${name}__${tag}.jsonl
    traj=/tmp/vllm_pf_${tag}_${name}.jsonl
    rm -f "${traj}".*
    echo "=========================================="
    echo "=== POSTFIX-BS16: $name | T=$T BS=16 gmu=0.85 K=$K max_new=$MAXNEW ==="
    echo "=========================================="
    JACOBI_K=$K VLLM_TPF_TRAJ_PATH=$traj \
    python3 scripts/_bench_postfix_tpf_tps.py \
      --model "$model" --prompts_jsonl "$PROMPTS" --output_jsonl "$out" \
      --max_new_tokens $MAXNEW --max_num_seqs 16 --temperature $T --gpu_mem_util 0.85 2>&1 | tail -15
  done
done
echo "DONE"
