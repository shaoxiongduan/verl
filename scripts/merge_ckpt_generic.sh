#!/usr/bin/env bash
# Generic FSDP -> HF ckpt merger. Usage:
#   merge_ckpt_generic.sh <CKPT_ACTOR_DIR> <OUT_HF_DIR> [CUDA_DEVICES]
set -xeuo pipefail
cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

CKPT="${1:?usage: $0 <CKPT_ACTOR_DIR> <OUT_HF_DIR> [CUDA_DEVICES]}"
HF_OUT="${2:?usage: $0 <CKPT_ACTOR_DIR> <OUT_HF_DIR> [CUDA_DEVICES]}"
export CUDA_VISIBLE_DEVICES="${3:-0}"

python scripts/legacy_model_merger.py merge \
    --backend fsdp --local_dir "${CKPT}" --target_dir "${HF_OUT}"
echo "merged: ${CKPT} -> ${HF_OUT}"
