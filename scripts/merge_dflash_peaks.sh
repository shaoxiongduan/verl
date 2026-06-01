#!/usr/bin/env bash
# Merge dFlash bidir step 300 and causal step 240 (peak val) to HF.
set -xeuo pipefail
cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

NAME="${1:-bidir_300}"
case "${NAME}" in
    bidir_300)
        CKPT=/mnt/weka/home/hao.zhang/shao/verl/ckpts/jacobi_forcing_dapo_opencodeinstruct/jf_coder_7b_dapo_oci_4gpu_cons_dflash_bidir/global_step_300/actor
        HF_OUT=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/dflash_bidir_step_300
        ;;
    causal_240)
        CKPT=/mnt/weka/home/hao.zhang/shao/verl/ckpts/jacobi_forcing_dapo_opencodeinstruct/jf_coder_7b_dapo_oci_4gpu_cons_dflash_causal/global_step_240/actor
        HF_OUT=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/dflash_causal_step_240
        ;;
    causal_260)
        CKPT=/mnt/weka/home/hao.zhang/shao/verl/ckpts/jacobi_forcing_dapo_opencodeinstruct/jf_coder_7b_dapo_oci_4gpu_cons_dflash_causal/global_step_260/actor
        HF_OUT=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/dflash_causal_step_260
        ;;
    causal_300)
        CKPT=/mnt/weka/home/hao.zhang/shao/verl/ckpts/jacobi_forcing_dapo_opencodeinstruct/jf_coder_7b_dapo_oci_4gpu_cons_dflash_causal/global_step_300/actor
        HF_OUT=/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/dflash_causal_step_300
        ;;
    *)
        echo "unknown name: ${NAME}"; exit 1;;
esac

python scripts/legacy_model_merger.py merge \
    --backend fsdp --local_dir "${CKPT}" --target_dir "${HF_OUT}"
echo "merged -> ${HF_OUT}"
