#!/usr/bin/env bash
# Merge CE run @ step 300 to HF format. CE was the loss-type=ce variant of
# the target_ratio=0.3 + per-step + relaxed-clamp KL run.
set -xeuo pipefail
cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate
CKPT=/mnt/weka/home/hao.zhang/shao/verl/ckpts/jacobi_forcing_dapo_opencodeinstruct/jf_coder_7b_dapo_oci_4gpu_cons_correct_only_target03_no_marker_ce/global_step_300/actor
HF_OUT=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/target03_ce_step_300
python scripts/legacy_model_merger.py merge \
    --backend fsdp --local_dir "${CKPT}" --target_dir "${HF_OUT}"
ls -la "${HF_OUT}" | head -8
