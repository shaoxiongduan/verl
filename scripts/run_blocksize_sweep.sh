#!/usr/bin/env bash
# Sweep block sizes 8, 16, 32, 64 for k3 and base on the same prompts.
# Use the per-iter probe to get per-iter accept breakdowns.
set -x
JF_PY=/mnt/weka/home/hao.zhang/shao/JacobiForcing/.venv/bin/python
PROBE=/mnt/weka/home/hao.zhang/shao/verl/scripts/jf_per_iter_probe.py
DATA=/mnt/weka/home/hao.zhang/shao/verl/data/openmathinstruct2/val.parquet
NUM=20
BASE_PATH=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1
K3_PATH=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_k3_ds_step_300

CUDA_VISIBLE_DEVICES=0 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=8 \
  TPF_MODEL_PATH=$BASE_PATH LABEL=base_b8 DRAFT_INIT=prompt_sample OUT_JSON=/tmp/bs_base_b8.json \
  $JF_PY $PROBE > /tmp/bs_base_b8.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=8 \
  TPF_MODEL_PATH=$K3_PATH LABEL=k3_b8 DRAFT_INIT=uniform OUT_JSON=/tmp/bs_k3_b8.json \
  $JF_PY $PROBE > /tmp/bs_k3_b8.log 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=2 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=16 \
  TPF_MODEL_PATH=$BASE_PATH LABEL=base_b16 DRAFT_INIT=prompt_sample OUT_JSON=/tmp/bs_base_b16.json \
  $JF_PY $PROBE > /tmp/bs_base_b16.log 2>&1 &
P3=$!
CUDA_VISIBLE_DEVICES=3 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=16 \
  TPF_MODEL_PATH=$K3_PATH LABEL=k3_b16 DRAFT_INIT=uniform OUT_JSON=/tmp/bs_k3_b16.json \
  $JF_PY $PROBE > /tmp/bs_k3_b16.log 2>&1 &
P4=$!
wait $P1 $P2 $P3 $P4
echo SWEEP_DONE

# Round 2: BS=64
CUDA_VISIBLE_DEVICES=0 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=64 \
  TPF_MODEL_PATH=$BASE_PATH LABEL=base_b64 DRAFT_INIT=prompt_sample OUT_JSON=/tmp/bs_base_b64.json \
  $JF_PY $PROBE > /tmp/bs_base_b64.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=64 \
  TPF_MODEL_PATH=$K3_PATH LABEL=k3_b64 DRAFT_INIT=uniform OUT_JSON=/tmp/bs_k3_b64.json \
  $JF_PY $PROBE > /tmp/bs_k3_b64.log 2>&1 &
P2=$!
wait $P1 $P2
echo ROUND2_DONE

cp /tmp/bs_*.json /mnt/weka/home/hao.zhang/shao/verl/research/
echo COPY_DONE
