#!/usr/bin/env bash
# BS=1 TPF+TPS for the final (step_300) ckpts of both runs.
set -eu
cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

PROMPTS=eval_passk/eval_prompts_tpf.jsonl
K=32
MAXNEW=2048
OUT_DIR=eval_passk/tpf_results/postfix
mkdir -p "$OUT_DIR"

declare -A MODELS=(
  [dflashce_v2_step_300]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_v2_step_300
  [dflashce_corrupt03_step_300]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_corrupt03_step_300
)

for name in dflashce_v2_step_300 dflashce_corrupt03_step_300; do
  model=${MODELS[$name]}
  for T in 0.0 1.0; do
    tag=$([ "$T" = "0.0" ] && echo greedy_bs1 || echo onpol_bs1)
    out=$OUT_DIR/${name}__${tag}.jsonl
    traj=/tmp/vllm_pf_${tag}_${name}.jsonl
    rm -f "${traj}".*
    echo "=========================================="
    echo "=== POSTFIX-S300: $name | T=$T BS=1 K=$K max_new=$MAXNEW ==="
    echo "=========================================="
    JACOBI_K=$K VLLM_TPF_TRAJ_PATH=$traj \
    python3 scripts/_bench_postfix_tpf_tps.py \
      --model "$model" --prompts_jsonl "$PROMPTS" --output_jsonl "$out" \
      --max_new_tokens $MAXNEW --max_num_seqs 1 --temperature $T --gpu_mem_util 0.6 2>&1 | tail -15
  done
done
echo "DONE"
