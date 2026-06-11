#!/usr/bin/env bash
# Assembly-decode comparability matrix (2026-06-12): fills the arms missing
# from the step-80 battery so {v9_220 untrained, v11_s80} x {causal, bidir,
# bidir+marker} x {argmax, keepnoise} is complete. Spec baselines (3.64/3.41)
# were argmax-update; the step-80 battery ran keepnoise — never compare across
# update rules. 16 DS prompts, 512 tokens, greedy.
set -euo pipefail
cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

V9=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/fwdkl_decay_v9_step_220
S80=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/v11_canvas_step_80
P=eval_passk/deepscaler_tpf_prompts_16.jsonl
OUT=eval_passk/tpf_results/v11/matrix
mkdir -p "$OUT"

run() { # model tag attn update marker gpu
  local m=$1 tag=$2 attn=$3 upd=$4 mk=$5 gpu=$6
  CUDA_VISIBLE_DEVICES=$gpu python scripts/_sim_assembly_decode.py \
    --model "$m" --prompts_jsonl "$P" \
    --out_jsonl "$OUT/${tag}_${attn}_${upd}_${mk}.jsonl" \
    --canvas_attn "$attn" --canvas_update "$upd" --marker "$mk" \
    > "$OUT/${tag}_${attn}_${upd}_${mk}.log" 2>&1
  grep "CORPUS TPF" "$OUT/${tag}_${attn}_${upd}_${mk}.log" | sed "s/^/[${tag} ${attn} ${upd} ${mk}] /"
}

# Round 1 (4 parallel)
run "$V9"  v9220 causal argmax    none     4 &
run "$V9"  v9220 bidir  argmax    none     5 &
run "$V9"  v9220 bidir  argmax    constant 6 &
run "$V9"  v9220 causal keepnoise none     7 &
wait
# Round 2
run "$V9"  v9220 bidir  keepnoise none     4 &
run "$V9"  v9220 bidir  keepnoise constant 5 &
run "$S80" v11s80 causal argmax   none     6 &
run "$S80" v11s80 bidir  argmax   none     7 &
wait
# Round 3
run "$S80" v11s80 bidir  argmax   constant 4 &
wait
echo "=== MATRIX COMPLETE ==="
