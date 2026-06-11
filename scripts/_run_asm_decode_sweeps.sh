#!/usr/bin/env bash
# Hybrid-decode behavior sweeps on the latest merged canvas ckpt (v11 s100):
#   A. keep_tau sweep (keepnoise, bidir+marker)  — trace audit said tau=2.0
#      re-noises too aggressively now that kept tokens are 62% correct.
#   B. step-100 gap point (causal control vs bidir+marker, argmax update).
#   C. W_canvas residence curve (spec §3.6.4): W ∈ {24,32,56} = canvas {16,24,48}.
#   D. Gumbel canvas candidates (spec §2.4 option).
# 16 DS prompts, 512 tok, greedy commits. Node 422 GPUs 4-7.
set -euo pipefail
cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

M=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/v11_canvas_step_100
P=eval_passk/deepscaler_tpf_prompts_16.jsonl
OUT=eval_passk/tpf_results/v11/decode_sweeps
mkdir -p "$OUT"

run() { # tag gpu extra...
  local tag=$1 gpu=$2; shift 2
  CUDA_VISIBLE_DEVICES=$gpu python scripts/_sim_assembly_decode.py \
    --model "$M" --prompts_jsonl "$P" --out_jsonl "$OUT/${tag}.jsonl" "$@" \
    > "$OUT/${tag}.log" 2>&1
  grep "CORPUS TPF" "$OUT/${tag}.log" | sed "s/^/[$tag] /"
}

# Round 1: keep_tau sweep (keepnoise, bidir, marker)
run s100_keep_tau1 4 --canvas_attn bidir --canvas_update keepnoise --marker constant --keep_tau 1.0 &
run s100_keep_tau2 5 --canvas_attn bidir --canvas_update keepnoise --marker constant --keep_tau 2.0 &
run s100_keep_tau4 6 --canvas_attn bidir --canvas_update keepnoise --marker constant --keep_tau 4.0 &
run s100_keep_tau8 7 --canvas_attn bidir --canvas_update keepnoise --marker constant --keep_tau 8.0 &
wait
# Round 2: step-100 gap point (argmax) + W_canvas curve (argmax, marker)
run s100_causal_argmax 4 --canvas_attn causal --canvas_update argmax &
run s100_bidirmk_argmax 5 --canvas_attn bidir --canvas_update argmax --marker constant &
run s100_W24_argmax 6 --canvas_attn bidir --canvas_update argmax --marker constant --W 24 &
run s100_W56_argmax 7 --canvas_attn bidir --canvas_update argmax --marker constant --W 56 &
wait
# Round 3: gumbel candidates (keepnoise tau=2) + W extremes under keepnoise
run s100_gumbel_t07 4 --canvas_attn bidir --canvas_update keepnoise --marker constant --candidates gumbel --temp 0.7 &
run s100_W24_keep 5 --canvas_attn bidir --canvas_update keepnoise --marker constant --W 24 &
run s100_W56_keep 6 --canvas_attn bidir --canvas_update keepnoise --marker constant --W 56 &
wait
echo "=== DECODE SWEEPS COMPLETE ==="
