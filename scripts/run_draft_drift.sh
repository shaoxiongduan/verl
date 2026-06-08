#!/usr/bin/env bash
set -x
JF_PY=/mnt/weka/home/hao.zhang/shao/JacobiForcing/.venv/bin/python
PROBE=/mnt/weka/home/hao.zhang/shao/verl/scripts/jf_draft_drift_probe.py
DATA=/mnt/weka/home/hao.zhang/shao/verl/data/openmathinstruct2/val.parquet
NUM=5
K3=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_k3_ds_step_300
BASE=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1

CUDA_VISIBLE_DEVICES=0 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=32 MAX_ITERS=12 \
  TPF_MODEL_PATH=$K3 MODE=chunked LABEL=k3_chunked OUT_JSON=/tmp/drift_k3_chunked.json \
  $JF_PY $PROBE > /tmp/drift_k3_chunked.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=32 MAX_ITERS=12 \
  TPF_MODEL_PATH=$K3 MODE=streaming LABEL=k3_stream OUT_JSON=/tmp/drift_k3_stream.json \
  $JF_PY $PROBE > /tmp/drift_k3_stream.log 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=2 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=32 MAX_ITERS=12 \
  TPF_MODEL_PATH=$BASE MODE=chunked LABEL=base_chunked OUT_JSON=/tmp/drift_base_chunked.json \
  $JF_PY $PROBE > /tmp/drift_base_chunked.log 2>&1 &
P3=$!
CUDA_VISIBLE_DEVICES=3 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=32 MAX_ITERS=12 \
  TPF_MODEL_PATH=$BASE MODE=streaming LABEL=base_stream OUT_JSON=/tmp/drift_base_stream.json \
  $JF_PY $PROBE > /tmp/drift_base_stream.log 2>&1 &
P4=$!
wait $P1 $P2 $P3 $P4
echo DRIFT_DONE
cp /tmp/drift_*.json /mnt/weka/home/hao.zhang/shao/verl/research/
echo COPY_DONE
