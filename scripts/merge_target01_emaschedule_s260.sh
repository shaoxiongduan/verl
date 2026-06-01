#!/usr/bin/env bash
# Merge FSDP ckpt of the combined run (target_ratio=0.1, EMA decay=0.9,
# warmup_out schedule) at peak val step 260 (mean@8 = 0.8194) to HF format.

set -xeuo pipefail

cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

CKPT=/mnt/weka/home/hao.zhang/shao/verl/ckpts/jacobi_forcing_dapo_opencodeinstruct/jf_coder_7b_dapo_oci_4gpu_cons_correct_only_target01_no_marker_emaschedule/global_step_260/actor
HF_OUT=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/target01_emaschedule_step_260

python scripts/legacy_model_merger.py merge \
    --backend fsdp \
    --local_dir "${CKPT}" \
    --target_dir "${HF_OUT}"

echo "merged -> ${HF_OUT}"
ls -la "${HF_OUT}"
