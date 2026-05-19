# Consistency-loss RL findings — JF Coder 7B on OpenCodeInstruct

Snapshot as of 2026-05-16. Two runs to step 300 completed: AR-v2 baseline and cons λ=0.001 uniform-noise.

## Setup

- Base model: `JacobiForcing_Coder_7B_v1` (Qwen2.5-Coder-7B-Instruct + Jacobi-Forcing post-training).
- Algorithm: DAPO/GRPO RL on OpenCodeInstruct (50k prompts). Reward = assert-based code execution (scripts/reward_code_assert.py).
- Eval: HumanEval+ (164 prompts, n=8, t=1.0, p=0.7) for accuracy; HE+ and OCI prompts for Jacobi TPF.
- 4× H200 GPUs per run, FSDP2, vLLM rollout, 300 training steps.

## Bugfixes that mattered

1. **`flash_attention_2` + gradient checkpointing crash**: cons forward swaps to SDPA with a 4D float mask. With GC enabled, backward recomputes after the swap context exits → flash_attention_2 receives the SDPA mask → `vectorized_gather_kernel index out of bounds`. Fixed by `_no_gradient_checkpointing` context in `loss.py`.
2. **OCI-flavored noise leak**: original "sample" noise mode drew draft tokens from `prompt + response` of the current sample → OCI-prior baked into denoiser → HE+ pass@8 collapsed 0.762 → 0.601 by step 120. Fixed by `CONSISTENCY_NOISE_SOURCE=uniform` (draw from full vocab; matches Decode-Learning nanovllm default Jacobi init).
3. **Sandbox CWD pollution**: `reward_code_assert.py` ran model-generated code without `cwd=` → CSVs/JSONs landed in repo root. Fixed by `tempfile.TemporaryDirectory()` per execution.

## λ ablation (cons weight)

At λ=0.01 the consistency gradient dominated (~78% of total grad norm). HE+ acc plateaued at baseline (~0.75) for the entire 300 steps and never improved. λ=0.001 (10× lower) brought cons to ~10-20% of grad norm; HE+ acc tracks AR within 2-3 pp through most of training.

## Final state — step 300 head-to-head

### HumanEval+ val accuracy (validates at training time, t=1.0/p=0.7, n=8)

| metric | AR-v2 step 300 | cons λ=0.001 step 300 | gap |
|---|---:|---:|---:|
| mean@8 | **0.8293** | 0.7835 | AR +4.6 pp |
| pass@8 | **0.8696** | 0.8346 | AR +3.5 pp |

### OCI offline pass@1 (256 prompts, t=1.0, p=1.0, n=8)

| | step 300 |
|---|---:|
| AR-v2 | **0.8022** |
| cons λ=0.001 | 0.7886 |

AR wins accuracy by **+4.6 pp on HE+** and **+1.4 pp on OCI**.

### Jacobi TPF (HE+ 164 prompts, OCI 256 prompts, t=0.6, block=32)

All numbers measured with `scripts/tpf_trajectories.py` (per-prompt total_tokens / total_forwards). The old `compute_reference_tpf.py` gave numerically lower values (~0.23 less) because it averages per-block-TPF then per-prompt; this isn't comparable across scripts. Apples-to-apples below:

| ckpt | HE+ mean TPF | OCI mean TPF | HE+ Δ vs JF base 3.706 |
|---|---:|---:|---:|
| JF base | 3.706 | — | — |
| AR-v2 step 180 | 3.640 | 3.667 | −1.8% |
| AR-v2 step 300 | 3.552 | 3.422 | **−4.2%** |
| cons λ=0.001 step 40 | 3.660 | 4.039 | −1.2% |
| cons λ=0.001 step 180 | 3.787 | 4.173 | **+2.2%** (peak) |
| cons λ=0.001 step 300 | 3.722 | 3.944 | +0.4% |

**Both AR runs degrade TPF** (-4 to -10% from baseline). **Cons λ=0.001 preserves or slightly improves** TPF (+0.4% at step 300; +2.2% at step 180). Cons TPF peaks around step 180 then drifts down slightly.

(The old AR run, prior experiment, lost ~10% TPF by step 260; AR-v2's smaller -4% drop is run-to-run RL variance, same script + hyperparams.)

## The trade-off

Cons λ=0.001 buys **+5-15% TPF over AR** (HE+ +4.8%, OCI +15.2% at step 300) at the cost of **~4-5 pp HE+ accuracy** and **~1.4 pp OCI accuracy**.

## Reward-hacking via verbose template padding

Cons run response length on OCI grew from 188 (step 40) → 266 tokens (step 300), +33%. AR-v2 length grew from 175 → 199 (+14%). Cons run grows length ~2× faster than AR.

**The extra tokens are NOT n-gram repetition** — 4-gram repetition rate flat ~2.3% across all ckpts. They are structural padding the model learned during RL:

1. Wrap function body in `try/except` (defensive boilerplate).
2. Add a block of `print(test_case())` example invocations.
3. Trailing prose paragraph: "This Python script defines a function that...".

Example growth on prompt idx=101 (temperature conversion task):
- Step 40 (80 tokens): bare function.
- Step 180 (221 tokens): function + 5 test prints.
- Step 300 (412 tokens): function wrapped in try/except + 5 try/except test blocks + verbose explanation paragraph.

The function logic is identical-quality at step 40 and step 300. The padding doesn't change assert-reward (asserts care about behavior, not test prints), but RL still locks it in. Cons amplifies this because PG can't sink as much gradient into structure regularization while cons consumes some of the budget.

Per-block TPF analysis: prose paragraphs have *similar or slightly higher* TPF than code (highly templated). So the verbose padding does NOT explain the step-180-to-300 TPF dip; that's likely just RL trajectory noise.

## Cons-loss trajectory plateau

Cons loss bottomed at step 200 (2.70), then drifted back up to 2.87 by step 300. The TPF peak coincides with cons_loss minimum. Mechanism: as PG keeps lengthening responses, the cons forward sees more diverse / harder-to-denoise rollout content, and λ=0.001 is too small to drag cons_loss back down against this drift.

## Open questions / next steps

1. **Cons weight scheduling**: warmup-out (cons strong early, fade out) vs warmup-in (cons starts late). Hypothesis: warmup-out lets cons sculpt representations early when entropy is high and the model is moldable; fading late lets PG do its final accuracy climb without cons fighting.
2. **Data mixing**: OCI is one prompt distribution; cons learns to denoise OCI-style verbosity. Mixing in HE+-style or general-instruction prompts (Tulu-3 / Llama-3 / DeepSeek-R1 use 60-80% target + 20-40% general) should reduce specialization and the verbose-padding drift.
3. **Length penalty in reward**: explicit penalty on response length above some threshold. Would directly attack the reward-hacking pattern, orthogonal to cons.

## Files

- `scripts/consistency/loss.py` — cons forward + soft CE.
- `scripts/consistency/pack.py` — interleaved batch builder with `CONSISTENCY_NOISE_SOURCE=uniform|sample`.
- `scripts/consistency/verl_hook.py` — drop-in integration with verl `FSDPEngineWithLMHead.forward_step`.
- `verl/workers/engine/fsdp/transformer_impl.py` — patched call site.
- `scripts/run_dapo_jf_coder_4gpu_consistency.sh` — launch script. `CONSISTENCY_ENABLE=0` recovers pure AR.
- `scripts/tpf_trajectories.py` — TPF eval (Decode-Learning nanovllm-based).
- `scripts/eval_offline_pass1.py` — OCI pass@1 eval.
- `scripts/legacy_model_merger.py` — FSDP → HF merge.
- `eval_passk/trajectories/step{40,180,300}_lambda001_*.jsonl` — cons run TPF trajectories.
- `eval_passk/trajectories/step{180,300}_ar_v2_*.jsonl` — AR-v2 run TPF trajectories.
- `eval_passk/offline_pass1/step{40,300}_lambda001_oci__summary.json` — OCI pass@1 summaries.

## Wandb run IDs (project s4duan-uc-san-diego/jacobi_forcing_dapo_opencodeinstruct)

- AR-v2 (this experiment): `ubcl2m30`
- cons λ=0.001 (pre-crash): `qdsoi68a` (steps 1-32)
- cons λ=0.001 (resumed, all subsequent): `gf4rto12` (steps 41-300)
- AR-old (prior experiment, finished earlier): `aw3z2byr`
- cons λ=0.01 uniform (prior): `hnpwjry4`
- cons sample-noise (failure mode demo): `fhsdulms`, `gpsf4i4i`
