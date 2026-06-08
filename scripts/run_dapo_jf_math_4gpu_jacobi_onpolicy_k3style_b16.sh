#!/usr/bin/env bash
# On-policy cons-RL with K3-style recipe: dflash_ce loss + position-decay weighting,
# at block_size=16 (matching JACOBI_K=16). Cascade-converged prefix (= n_acc tokens
# per trajectory) is MASKED OUT of cons loss, and decay starts at the first noisy
# position with weight 1.
set -xeuo pipefail

cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

project_name=${PROJECT_NAME:-jacobi_forcing_dapo_deepscaler_onpolicy_k3style}
exp_name=${EXP_NAME:-jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_k3style_b16}

# === Jacobi spec-decode (K=16 for the b=16 alignment) ===
export JACOBI_K=${JACOBI_K:-16}
CKPT_DIR=${CKPT_DIR:-/mnt/weka/home/hao.zhang/shao/verl/ckpts/${project_name}/${exp_name}}
mkdir -p "${CKPT_DIR}"
export JACOBI_TRAJ_PATH=${JACOBI_TRAJ_PATH:-/dev/shm/jacobi_traj_${project_name}_${exp_name}/traj}
mkdir -p "$(dirname "${JACOBI_TRAJ_PATH}")"
rm -f "${JACOBI_TRAJ_PATH}".req_*.jsonl 2>/dev/null || true

# === K3-style cons loss recipe ===
export CONSISTENCY_ENABLE=${CONSISTENCY_ENABLE:-1}
export CONSISTENCY_ONPOLICY=${CONSISTENCY_ONPOLICY:-1}
export CONSISTENCY_LOSS_TYPE=${CONSISTENCY_LOSS_TYPE:-dflash_ce}    # dflash CE w/ decay
export CONSISTENCY_DFLASH_GAMMA=${CONSISTENCY_DFLASH_GAMMA:-6.0}    # halved for b=16 (was 12 at b=32)
export CONSISTENCY_BLOCK_SIZE=${CONSISTENCY_BLOCK_SIZE:-16}
# K=1 noise tile (on-policy only provides one cascade draft per position; the
# multi-tile k=3 monte-carlo averaging doesn't apply here).
export CONSISTENCY_NUM_NOISY_TILES=${CONSISTENCY_NUM_NOISY_TILES:-1}
export CONSISTENCY_WEIGHT=${CONSISTENCY_WEIGHT:-0.01}
export CONSISTENCY_T_SOFT=${CONSISTENCY_T_SOFT:-1.0}
export CONSISTENCY_FRACTION=${CONSISTENCY_FRACTION:-0.30}
export CONSISTENCY_MAX_PAIRS=${CONSISTENCY_MAX_PAIRS:-16}
export CONSISTENCY_PAD_ID=${CONSISTENCY_PAD_ID:-151643}
export CONSISTENCY_DEBUG=${CONSISTENCY_DEBUG:-1}

export CONSISTENCY_CAUSAL_REGION_SIZE=${CONSISTENCY_CAUSAL_REGION_SIZE:-0}
export CONSISTENCY_SCHEDULE=${CONSISTENCY_SCHEDULE:-constant}
export CONSISTENCY_RAMP_START_FRAC=${CONSISTENCY_RAMP_START_FRAC:-0.0}
export CONSISTENCY_RAMP_END_FRAC=${CONSISTENCY_RAMP_END_FRAC:-1.0}

total_training_steps=${TOTAL_TRAINING_STEPS:-300}
export CONSISTENCY_TOTAL_STEPS=${CONSISTENCY_TOTAL_STEPS:-${total_training_steps}}

JF_MODEL=${JF_MODEL:-/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1}

TRAIN_FILE=${TRAIN_FILE:-/mnt/weka/home/hao.zhang/shao/verl/data/deepscaler/train.parquet}
VAL_FILES_LIST=${VAL_FILES_LIST:-'[/mnt/weka/home/hao.zhang/shao/verl/data/openmathinstruct2/val.parquet,/mnt/weka/home/hao.zhang/shao/verl/data/gsm8k/test.parquet]'}

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}

adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28

max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 2))
enable_overlong_buffer=True
overlong_buffer_len=512
overlong_penalty_factor=1.0

temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7
val_n=${VAL_N:-4}

train_prompt_bsz=32
n_resp_per_prompt=16
train_prompt_mini_bsz=32
gen_prompt_bsz_oversample=64

use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
gen_tp=1
sp_size=1
fsdp_size=4
actor_offload=True
ref_offload=True
loss_agg_mode="token-mean"
gpu_mem_util=0.55

export WANDB_API_KEY=${WANDB_API_KEY:-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}
export WANDB_ENTITY=${WANDB_ENTITY:-s4duan-uc-san-diego}
export WANDB_PROJECT=${WANDB_PROJECT:-${project_name}}
LOGGER=${LOGGER:-["console","wandb"]}

if [ "${SKIP_RAY_STOP:-0}" != "1" ]; then
    ray stop --force >/dev/null 2>&1 || true
fi

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files=${VAL_FILES_LIST} \
    data.prompt_key=prompt \
    data.truncation='left' \
    data.dataloader_num_workers=0 \
    data.max_prompt_length=${max_prompt_length} \
    data.max_response_length=${max_response_length} \
    data.train_batch_size=${train_prompt_bsz} \
    +data.gen_batch_size=${gen_prompt_bsz_oversample} \
    actor_rollout_ref.rollout.n=${n_resp_per_prompt} \
    algorithm.adv_estimator=${adv_estimator} \
    algorithm.use_kl_in_reward=${use_kl_in_reward} \
    algorithm.kl_ctrl.kl_coef=${kl_coef} \
    +algorithm.filter_groups.enable=True \
    +algorithm.filter_groups.metric=acc \
    +algorithm.filter_groups.max_num_gen_batches=6 \
    actor_rollout_ref.actor.fsdp_config.strategy=fsdp2 \
    critic.strategy=fsdp2 \
    actor_rollout_ref.actor.use_kl_loss=${use_kl_loss} \
    actor_rollout_ref.actor.kl_loss_coef=${kl_loss_coef} \
    actor_rollout_ref.actor.clip_ratio_low=${clip_ratio_low} \
    actor_rollout_ref.actor.clip_ratio_high=${clip_ratio_high} \
    actor_rollout_ref.actor.clip_ratio_c=10.0 \
    actor_rollout_ref.model.use_remove_padding=True \
    actor_rollout_ref.actor.use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.ref.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.rollout.log_prob_use_dynamic_bsz=${use_dynamic_bsz} \
    actor_rollout_ref.actor.ppo_max_token_len_per_gpu=${actor_ppo_max_token_len} \
    actor_rollout_ref.ref.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.rollout.log_prob_max_token_len_per_gpu=${infer_ppo_max_token_len} \
    actor_rollout_ref.model.path="${JF_MODEL}" \
    actor_rollout_ref.model.enable_gradient_checkpointing=True \
    actor_rollout_ref.model.mtp.enable=True \
    actor_rollout_ref.model.mtp.enable_rollout=True \
    actor_rollout_ref.model.mtp.method=ngram \
    actor_rollout_ref.model.mtp.num_speculative_tokens=${JACOBI_K} \
    actor_rollout_ref.actor.optim.lr=2e-6 \
    actor_rollout_ref.actor.optim.lr_warmup_steps=10 \
    actor_rollout_ref.actor.optim.weight_decay=0.1 \
    actor_rollout_ref.actor.ppo_mini_batch_size=${train_prompt_mini_bsz} \
    actor_rollout_ref.actor.fsdp_config.param_offload=${actor_offload} \
    actor_rollout_ref.actor.fsdp_config.optimizer_offload=${actor_offload} \
    actor_rollout_ref.actor.entropy_coeff=0 \
    actor_rollout_ref.actor.grad_clip=1.0 \
    actor_rollout_ref.actor.loss_agg_mode=${loss_agg_mode} \
    actor_rollout_ref.actor.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.rollout.gpu_memory_utilization=${gpu_mem_util} \
    actor_rollout_ref.rollout.tensor_model_parallel_size=${gen_tp} \
    actor_rollout_ref.rollout.enable_chunked_prefill=True \
    actor_rollout_ref.rollout.max_num_batched_tokens=$((max_prompt_length + max_response_length)) \
    actor_rollout_ref.rollout.temperature=${temperature} \
    actor_rollout_ref.rollout.top_p=${top_p} \
    actor_rollout_ref.rollout.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.temperature=${temperature} \
    actor_rollout_ref.rollout.val_kwargs.top_p=${val_top_p} \
    actor_rollout_ref.rollout.val_kwargs.top_k=${top_k} \
    actor_rollout_ref.rollout.val_kwargs.do_sample=True \
    actor_rollout_ref.rollout.val_kwargs.n=${val_n} \
    actor_rollout_ref.rollout.name=vllm \
    actor_rollout_ref.ref.fsdp_config.param_offload=${ref_offload} \
    actor_rollout_ref.ref.ulysses_sequence_parallel_size=${sp_size} \
    actor_rollout_ref.actor.fsdp_config.fsdp_size=${fsdp_size} \
    reward.reward_manager.name=dapo \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=False \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger="${LOGGER}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=20 \
    trainer.save_freq=20 \
    trainer.max_actor_ckpt_to_keep=${MAX_ACTOR_CKPT:-5} \
    trainer.total_epochs=${TOTAL_EPOCHS:-4} \
    trainer.total_training_steps=${total_training_steps} \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=0 \
    "$@"
