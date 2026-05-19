#!/usr/bin/env bash
# DAPO RL training of JacobiForcing_Math_7B_v1 on OpenMathInstruct-2 (math).
# 4 H200 GPUs, FSDP2 colocated rollout. Adapted from
# verl/experimental/one_step_off_policy/shell/dapo_7b_math_fsdp2_colocate.sh
# and the user's iterative_rl_training.py settings (Jacobi-specific knobs
# like rep penalty / TPF reward stripped — verl runs vanilla AR vLLM rollouts).
set -xeuo pipefail

cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

project_name=${PROJECT_NAME:-qwen_math_7b_dapo_omi}
exp_name=${EXP_NAME:-qwen_math_7b_dapo_omi_4gpu}

JF_MODEL=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--Qwen--Qwen2.5-Math-7B-Instruct/snapshots/ef9926d75ab1d54532f6a30dd5e760355eb9aa4d

TRAIN_FILE=${TRAIN_FILE:-/mnt/weka/home/hao.zhang/shao/verl/data/openmathinstruct2/train.parquet}
VAL_FILE=${VAL_FILE:-/mnt/weka/home/hao.zhang/shao/verl/data/openmathinstruct2/val.parquet}
CKPT_DIR=${CKPT_DIR:-/mnt/weka/home/hao.zhang/shao/verl/ckpts/${project_name}/${exp_name}}

# Number of GPUs (1 node, 4 of 8 H200s).
NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}

# DAPO algorithm
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28          # standard DAPO Clip-Higher (user's 10.0 disables it)

# Lengths. JF base supports rope to 32768; override max_position to enable
# 2k prompt + 4k response.
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 2))
enable_overlong_buffer=True
overlong_buffer_len=512
overlong_penalty_factor=1.0

# Sampling
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7

# Batch shape — user's request: 32 prompts/iter, 16 rollouts/prompt.
train_prompt_bsz=32
n_resp_per_prompt=16
train_prompt_mini_bsz=32      # one optimizer step per iter on 32 prompts
gen_prompt_bsz_oversample=64  # filter_groups dynamic sampling oversample budget

# Memory / sharding (4×H200 colocated)
use_dynamic_bsz=True
# Per-GPU token budget for actor & rollout log_prob micro-batches.
# H200 has 143GB. For 7B + activations w/ chunked checkpointing this is safe.
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
gen_tp=1                      # 7B fits on 1 GPU; 4-way DP rollout
sp_size=1                     # no Ulysses
fsdp_size=4                   # full FSDP across the 4 GPUs
actor_offload=True            # safer for 4-GPU full-batch step
ref_offload=True
loss_agg_mode="token-mean"
gpu_mem_util=0.55             # leave headroom for FSDP weights when colocated

# Logger
export WANDB_API_KEY=${WANDB_API_KEY:-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}
export WANDB_ENTITY=${WANDB_ENTITY:-s4duan-uc-san-diego}
export WANDB_PROJECT=${WANDB_PROJECT:-${project_name}}
LOGGER=${LOGGER:-["console","wandb"]}

# `ray stop` cleans any stale colocate cluster from a previous run.
ray stop --force >/dev/null 2>&1 || true

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
    data.prompt_key=prompt \
    data.truncation='left' \
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
    actor_rollout_ref.rollout.val_kwargs.n=1 \
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
    trainer.total_epochs=2 \
    trainer.total_training_steps=200 \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=10
