#!/usr/bin/env bash
# Variant of run_dapo_jf_math_4gpu_consistency.sh with Jacobi spec-decoding rollout.
# Uses the jacobi_vllm_plugin (must be pip-installed in verl venv).
# Rollout produces Jacobi-style trajectories at TPF ~3.0-3.5 (vs AR's 1.0).
#
# Plugin auto-loads in every vLLM process via vllm.general_plugins entry point.
# We toggle on via the model.mtp config which verl reads at
# verl/workers/rollout/vllm_rollout/vllm_async_server.py:286-291.
set -xeuo pipefail

cd /mnt/weka/home/hao.zhang/shao/verl
source .venv/bin/activate

project_name=${PROJECT_NAME:-jacobi_forcing_dapo_deepscaler_jacobi}
exp_name=${EXP_NAME:-jf_math_7b_dapo_ds_4gpu_jacobi_k16}

# === Jacobi spec-decode config ===
# K=16 default chosen by audit agent: TPF ceiling + stable deep-position acceptance
# for cons-RL training data distribution.
export JACOBI_K=${JACOBI_K:-16}
# Optional: log per-iter trajectories to disk for analysis.
# export JACOBI_TRAJ_PATH=${CKPT_DIR:-/tmp}/jacobi_traj.jsonl

# === Cons loss config (same as baseline) ===
export CONSISTENCY_ENABLE=${CONSISTENCY_ENABLE:-1}
# Filter cons-loss pool to rollouts with acc==1.0. Matches the proven-good
# ce_noisy_decay_k3 reference; cons doesn't reinforce wrong drafts.
export CONSISTENCY_CORRECT_ONLY=${CONSISTENCY_CORRECT_ONLY:-1}
export CONSISTENCY_CORRECT_THRESHOLD=${CONSISTENCY_CORRECT_THRESHOLD:-1.0}
export CONSISTENCY_WEIGHT=${CONSISTENCY_WEIGHT:-0.01}
export CONSISTENCY_BLOCK_SIZE=${CONSISTENCY_BLOCK_SIZE:-32}
export CONSISTENCY_T_SOFT=${CONSISTENCY_T_SOFT:-1.0}
export CONSISTENCY_FRACTION=${CONSISTENCY_FRACTION:-0.10}
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
CKPT_DIR=${CKPT_DIR:-/mnt/weka/home/hao.zhang/shao/verl/ckpts/${project_name}/${exp_name}}

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
