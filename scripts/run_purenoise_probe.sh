#!/usr/bin/env bash
set -x
JF_PY=/mnt/weka/home/hao.zhang/shao/JacobiForcing/.venv/bin/python
PROBE=/mnt/weka/home/hao.zhang/shao/verl/scripts/jf_purenoise_probe.py
DATA=/mnt/weka/home/hao.zhang/shao/verl/data/openmathinstruct2/val.parquet
NUM=20

CUDA_VISIBLE_DEVICES=0 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=32 \
  TPF_MODEL_PATH=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1 \
  LABEL=base OUT_JSON=/tmp/pn_base.json \
  $JF_PY $PROBE > /tmp/pn_base.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=32 \
  TPF_MODEL_PATH=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_ar_ds_step_300 \
  LABEL=ar OUT_JSON=/tmp/pn_ar.json \
  $JF_PY $PROBE > /tmp/pn_ar.log 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=2 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=32 \
  TPF_MODEL_PATH=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_t01_ds_step_300 \
  LABEL=t01 OUT_JSON=/tmp/pn_t01.json \
  $JF_PY $PROBE > /tmp/pn_t01.log 2>&1 &
P3=$!
CUDA_VISIBLE_DEVICES=3 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=32 \
  TPF_MODEL_PATH=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_k3_ds_step_300 \
  LABEL=k3 OUT_JSON=/tmp/pn_k3.json \
  $JF_PY $PROBE > /tmp/pn_k3.log 2>&1 &
P4=$!
wait $P1 $P2 $P3 $P4
echo PN_DONE
cp /tmp/pn_*.json /mnt/weka/home/hao.zhang/shao/verl/research/
echo COPY_DONE
