#!/usr/bin/env bash
set -x
JF_PY=/mnt/weka/home/hao.zhang/shao/JacobiForcing/.venv/bin/python
cd /mnt/weka/home/hao.zhang/shao/JacobiForcing/JacobiForcing

DATA=/mnt/weka/home/hao.zhang/shao/verl/data/openmathinstruct2/val.parquet
NUM=5
BS=32

# matched DRAFT_INIT per model: cons-trained -> uniform; AR/base -> prompt_sample
CUDA_VISIBLE_DEVICES=0 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=$BS \
  TPF_MODEL_PATH=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_t01_ds_step_300 \
  TRACE_OUT_JSON=/tmp/trace_t01_ds_s300_bs32.json TRACE_LABEL=target01_ds_s300_bs32 DRAFT_INIT=uniform \
  $JF_PY jf_inference_trace.py > /tmp/trace_t01_run_bs32.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=$BS \
  TPF_MODEL_PATH=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_k3_ds_step_300 \
  TRACE_OUT_JSON=/tmp/trace_k3_ds_s300_bs32.json TRACE_LABEL=k3_ds_s300_bs32 DRAFT_INIT=uniform \
  $JF_PY jf_inference_trace.py > /tmp/trace_k3_run_bs32.log 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=2 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=$BS \
  TPF_MODEL_PATH=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_ar_ds_step_300 \
  TRACE_OUT_JSON=/tmp/trace_ar_ds_s300_bs32.json TRACE_LABEL=ar_ds_s300_bs32 DRAFT_INIT=prompt_sample \
  $JF_PY jf_inference_trace.py > /tmp/trace_ar_run_bs32.log 2>&1 &
P3=$!
CUDA_VISIBLE_DEVICES=3 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=$BS \
  TPF_MODEL_PATH=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1 \
  TRACE_OUT_JSON=/tmp/trace_base_math_bs32.json TRACE_LABEL=jf_math_7b_base_bs32 DRAFT_INIT=prompt_sample \
  $JF_PY jf_inference_trace.py > /tmp/trace_base_run_bs32.log 2>&1 &
P4=$!
wait $P1 $P2 $P3 $P4
echo TRACE_DONE_BS32

$JF_PY jf_trace_to_html.py \
  /tmp/trace_base_math_bs32.json /tmp/trace_ar_ds_s300_bs32.json /tmp/trace_t01_ds_s300_bs32.json /tmp/trace_k3_ds_s300_bs32.json \
  > /tmp/math_4way_trace_bs32.html 2>> /tmp/html_render_bs32.log
echo HTML_DONE_BS32
cp /tmp/math_4way_trace_bs32.html /mnt/weka/home/hao.zhang/shao/verl/research/math_4way_trace_bs32.html
echo COPY_DONE_BS32
