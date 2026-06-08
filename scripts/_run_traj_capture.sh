#!/usr/bin/env bash
set -eu
cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

PROMPTS=eval_passk/eval_prompts_tpf.jsonl
N=8
K=32
MAXNEW=2048
OUT_DIR=eval_passk/tpf_results/trajectory_viz

declare -A MODELS=(
  [base]=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1
  [consA_s300]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_v2_step_300
  [consB_s300]=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_corrupt03_step_300
)

mkdir -p "$OUT_DIR"
for label in base consA_s300 consB_s300; do
  model=${MODELS[$label]}
  echo "=========================================="
  echo "=== TRAJ CAPTURE: $label ==="
  echo "=========================================="
  JACOBI_K=$K TRAJ_LABEL=$label VLLM_TRAJ_CAPTURE_PATH=/tmp/_traj_capture_${label}.jsonl \
  python3 scripts/_bench_traj_capture.py \
    --model "$model" --prompts_jsonl "$PROMPTS" --out_dir "$OUT_DIR" \
    --max_new_tokens $MAXNEW --n_prompts $N --temperature 0.0 \
    --label "$label" 2>&1 | tail -15
done

echo "DONE — analyze + render"
python3 scripts/_traj_analyze_and_render.py --in_dir "$OUT_DIR" \
  --model_path /mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1 \
  --html_out "$OUT_DIR/viz.html" --stats_out "$OUT_DIR/stats.json" 2>&1 | tail -10
