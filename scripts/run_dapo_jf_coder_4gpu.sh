#!/usr/bin/env bash
# DAPO RL training of JacobiForcing_Coder_7B_v1 on OpenCodeInstruct.
# 4 H200 GPUs, FSDP2 colocated rollout. Single-turn coder RL.
#
# Companion files (must exist):
#   scripts/data_preprocess/opencodeinstruct.py  -> data/opencodeinstruct/{train,val}.parquet
#   scripts/data_preprocess/humanevalplus.py     -> data/humanevalplus/val.parquet
#   scripts/reward_code_assert.py                -> custom compute_score_batch
#
# Differences from scripts/run_dapo_jf_4gpu.sh (math):
#   - model -> JacobiForcing_Coder_7B_v1
#   - dataset -> OpenCodeInstruct (in-distribution with JF paper) + HumanEval+ eval
#   - reward fn: math_dapo -> custom assert-style executor (scripts/reward_code_assert.py)
#     (DAPO reward manager kept; experimental loop calls compute_score via
#      asyncio.run_in_executor, so per-sample fn is already parallelized.)
#   - max_response_length: 2k -> 4k
#   - val_kwargs.n=8 for pass@k signal on HumanEval+
#   - gen_prompt_bsz oversample raised 64->96 (code prompts get filtered more)
#   - gpu_mem_util 0.55 -> 0.50 (longer KV at 4k resp)
set -xeuo pipefail

cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

project_name=${PROJECT_NAME:-jacobi_forcing_dapo_opencodeinstruct}
exp_name=${EXP_NAME:-jf_coder_7b_dapo_oci_4gpu}

JF_MODEL=/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Coder_7B_v1/snapshots/81815b050f535c622153b5f6df38efc71326f938

TRAIN_FILE=${TRAIN_FILE:-/mnt/weka/home/hao.zhang/shao/verl/data/opencodeinstruct/train.parquet}
VAL_FILE=${VAL_FILE:-/mnt/weka/home/hao.zhang/shao/verl/data/humanevalplus/val.parquet}
CKPT_DIR=${CKPT_DIR:-/mnt/weka/home/hao.zhang/shao/verl/ckpts/${project_name}/${exp_name}}

REWARD_FN=/mnt/weka/home/hao.zhang/shao/verl/scripts/reward_code_assert.py

NNODES=${NNODES:-1}
NGPUS_PER_NODE=${NGPUS_PER_NODE:-4}

# DAPO algorithm
adv_estimator=grpo
use_kl_in_reward=False
kl_coef=0.0
use_kl_loss=False
kl_loss_coef=0.0
clip_ratio_low=0.2
clip_ratio_high=0.28

# Lengths
max_prompt_length=$((1024 * 2))
max_response_length=$((1024 * 4))
overlong_buffer_len=1024
overlong_penalty_factor=1.0

# Sampling
temperature=1.0
top_p=1.0
top_k=-1
val_top_p=0.7
val_n=8                         # pass@k signal on HumanEval+

# Batch shape
train_prompt_bsz=32
n_resp_per_prompt=16
train_prompt_mini_bsz=32
gen_prompt_bsz_oversample=96    # more headroom for code's heavier all-pass/all-fail filter

# Memory / sharding
use_dynamic_bsz=True
actor_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 2))
infer_ppo_max_token_len=$(((max_prompt_length + max_response_length) * 3))
gen_tp=1
sp_size=1
fsdp_size=4
actor_offload=True
ref_offload=True
loss_agg_mode="token-mean"
gpu_mem_util=0.50

# Reward execution
reward_timeout_s=10
# memory_mb is intentionally left unset — RLIMIT_AS conflicts with numpy/torch
# shared libraries. Wallclock timeout is the real safety net. Set explicitly
# (e.g. reward_memory_mb=4096) only if subprocesses are OOMing the host.

# DAPO overlong-buffer reward shaping (same shape as math run).
enable_overlong_buffer=True
overlong_penalty_log=False

# Logger
export WANDB_API_KEY=${WANDB_API_KEY:-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}
export WANDB_ENTITY=${WANDB_ENTITY:-s4duan-uc-san-diego}
export WANDB_PROJECT=${WANDB_PROJECT:-${project_name}}
LOGGER=${LOGGER:-["console","wandb"]}

if [ "${SKIP_RAY_STOP:-0}" != "1" ]; then
    ray stop --force >/dev/null 2>&1 || true
fi

python3 -m verl.trainer.main_ppo \
    data.train_files="${TRAIN_FILE}" \
    data.val_files="${VAL_FILE}" \
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
    reward.custom_reward_function.path="${REWARD_FN}" \
    reward.custom_reward_function.name=compute_score \
    +reward.custom_reward_function.reward_kwargs.timeout_s=${reward_timeout_s} \
    +reward.reward_kwargs.overlong_buffer_cfg.enable=${enable_overlong_buffer} \
    +reward.reward_kwargs.overlong_buffer_cfg.len=${overlong_buffer_len} \
    +reward.reward_kwargs.overlong_buffer_cfg.penalty_factor=${overlong_penalty_factor} \
    +reward.reward_kwargs.overlong_buffer_cfg.log=${overlong_penalty_log} \
    +reward.reward_kwargs.max_resp_len=${max_response_length} \
    trainer.logger="${LOGGER}" \
    trainer.project_name="${project_name}" \
    trainer.experiment_name="${exp_name}" \
    trainer.n_gpus_per_node="${NGPUS_PER_NODE}" \
    trainer.nnodes="${NNODES}" \
    trainer.val_before_train=True \
    trainer.test_freq=20 \
    trainer.save_freq=20 \
    trainer.total_epochs=1 \
    trainer.total_training_steps=300 \
    trainer.default_local_dir="${CKPT_DIR}" \
    trainer.resume_mode=auto \
    trainer.log_val_generations=0
