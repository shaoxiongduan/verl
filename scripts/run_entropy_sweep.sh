#!/usr/bin/env bash
# Sweep entropy thresholds for k3. Compare to base.
set -x
JF_PY=/mnt/weka/home/hao.zhang/shao/JacobiForcing/.venv/bin/python
PROBE=/mnt/weka/home/hao.zhang/shao/verl/scripts/jf_entropy_refresh_probe.py
DATA=/mnt/weka/home/hao.zhang/shao/verl/data/openmathinstruct2/val.parquet
NUM=20
K3=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_k3_ds_step_300
BASE=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1

# Round 1: thresholds {inf (baseline), 4.0, 3.0, 2.0} for k3
CUDA_VISIBLE_DEVICES=0 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=32 \
  TPF_MODEL_PATH=$K3 DRAFT_INIT=uniform LABEL=k3_inf ENTROPY_THRESHOLD=inf \
  OUT_JSON=/tmp/ent_k3_inf.json \
  $JF_PY $PROBE > /tmp/ent_k3_inf.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=32 \
  TPF_MODEL_PATH=$K3 DRAFT_INIT=uniform LABEL=k3_th4 ENTROPY_THRESHOLD=4.0 \
  OUT_JSON=/tmp/ent_k3_th4.json \
  $JF_PY $PROBE > /tmp/ent_k3_th4.log 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=2 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=32 \
  TPF_MODEL_PATH=$K3 DRAFT_INIT=uniform LABEL=k3_th3 ENTROPY_THRESHOLD=3.0 \
  OUT_JSON=/tmp/ent_k3_th3.json \
  $JF_PY $PROBE > /tmp/ent_k3_th3.log 2>&1 &
P3=$!
CUDA_VISIBLE_DEVICES=3 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=32 \
  TPF_MODEL_PATH=$K3 DRAFT_INIT=uniform LABEL=k3_th2 ENTROPY_THRESHOLD=2.0 \
  OUT_JSON=/tmp/ent_k3_th2.json \
  $JF_PY $PROBE > /tmp/ent_k3_th2.log 2>&1 &
P4=$!
wait $P1 $P2 $P3 $P4
echo ROUND1_DONE

# Round 2: thresholds {1.0, 0.5, 0.0 (always replace)} for k3, plus baseline + best for base
CUDA_VISIBLE_DEVICES=0 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=32 \
  TPF_MODEL_PATH=$K3 DRAFT_INIT=uniform LABEL=k3_th1 ENTROPY_THRESHOLD=1.0 \
  OUT_JSON=/tmp/ent_k3_th1.json \
  $JF_PY $PROBE > /tmp/ent_k3_th1.log 2>&1 &
P1=$!
CUDA_VISIBLE_DEVICES=1 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=32 \
  TPF_MODEL_PATH=$K3 DRAFT_INIT=uniform LABEL=k3_th05 ENTROPY_THRESHOLD=0.5 \
  OUT_JSON=/tmp/ent_k3_th05.json \
  $JF_PY $PROBE > /tmp/ent_k3_th05.log 2>&1 &
P2=$!
CUDA_VISIBLE_DEVICES=2 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=32 \
  TPF_MODEL_PATH=$BASE DRAFT_INIT=prompt_sample LABEL=base_inf ENTROPY_THRESHOLD=inf \
  OUT_JSON=/tmp/ent_base_inf.json \
  $JF_PY $PROBE > /tmp/ent_base_inf.log 2>&1 &
P3=$!
CUDA_VISIBLE_DEVICES=3 TPF_DATA_PATH=$DATA NUM_PROMPTS=$NUM N_TOKEN_SEQ_LEN=32 \
  TPF_MODEL_PATH=$BASE DRAFT_INIT=prompt_sample LABEL=base_th3 ENTROPY_THRESHOLD=3.0 \
  OUT_JSON=/tmp/ent_base_th3.json \
  $JF_PY $PROBE > /tmp/ent_base_th3.log 2>&1 &
P4=$!
wait $P1 $P2 $P3 $P4
echo ROUND2_DONE

cp /tmp/ent_*.json /mnt/weka/home/hao.zhang/shao/verl/research/
echo COPY_DONE
