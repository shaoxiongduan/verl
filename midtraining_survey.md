# Mid-training survey: baking parallel-decoding capability outside RL

Date: 2026-06-12. Author: mid-training research agent (Claude).
Mission: survey how the field bakes dLLM/parallel-decoding capability in at the
SFT/mid-training stage, dissect their training code, and propose OUR pipeline.
Companion context: `v11_dllm_jacobi_hybrid.md` (spec), `eval_passk/tpf_results/
TPF_RESULTS_LOG.md` (2026-06-09..12: RL erodes TPF in every run; the v11.1
corruption→clean CE recipe at 40 RL steps beat 220 steps of the old recipe —
vanilla TPF 4.11, project best).

Local clones (all dissected at code level unless noted):
- CLLM: `/mnt/weka/home/hao.zhang/shao/Consistency_LLM`
- JacobiForcing: `/mnt/weka/home/hao.zhang/shao/JacobiForcing`
- Fast-dLLM (v2 under `v2/` + vendored LMFlow under `third_party/lmflow/`):
  `/mnt/weka/home/hao.zhang/shao/Fast-dLLM`
- Megatron-Bridge (Nemotron-Labs-Diffusion training code):
  `/mnt/weka/home/hao.zhang/shao/Megatron-Bridge`
- TiDAR: NO official code (paper arXiv 2511.08923 + community toy at
  `/mnt/weka/home/hao.zhang/shao/TiDAR-community` — toy deviates from paper,
  do not trust its masks)
- DiffusionGemma: no conversion code public, but the SAMPLER + SFT recipe are
  (google-deepmind/gemma `gemma/diffusion/`, google/hackable_diffusion)

---

## 0. Executive summary (cross-work consensus)

1. **Everyone trains x0-prediction**: CE from a corrupted input at a sampled
   noise level to the clean sequence. Nobody trains step-to-step transitions.
   Confirms spec §3.1 directly.
2. **Everyone co-trains an explicit AR loss in the same forward, and it is
   load-bearing**: CLLM AR×10; JacobiForcing AR×10; TiDAR α=1 (equal per-token);
   Nemotron AR=1.0 vs diff=0.3 (AR co-loss = their single biggest ablation gain,
   +7.48 avg pts); DiffusionGemma diffusion 1.0 + AR 1.0. The AR anchor is what
   prevents collapse and preserves base quality. Our RL currently plays this
   role; a dedicated mid-training stage must carry its own AR CE term.
3. **Dual-stream layout is the standard trick**: input `[x_corrupted | x_clean]`
   (Fast-dLLM v2, Nemotron, TiDAR doubled sequence; JacobiForcing interleaved
   (noisy_k, clean_last) block pairs) with SHARED RoPE positions between the two
   views, structured flex-attention mask (noisy: bidir-within-block +
   causal-to-clean-prefix; clean: strictly causal), clean-half logits give the
   AR loss "for free". Our verl cons pack is already exactly this family.
4. **Train-test corruption consistency is the most quantified lesson**
   (TiDAR Table 5: decode-matched corruption worth ~3 pp avg, +5.8 HumanEval).
   This is the published version of our v11.1 "empirical-state" finding.
5. **Mask-token works vs real-token corruption**: mask-token shops (FDv2,
   TiDAR, Nemotron) need no mode flag — the mask id is self-identifying.
   Real-token-corruption shops (DiffusionGemma uniform-random over vocab,
   CLLM/JF random-from-context) are the precedent for our canvas; DiffusionGemma
   additionally shows the **carry-loss** (supervise ALL canvas positions,
   including uncorrupted ones) is load-bearing when noise is real tokens — the
   model must learn to KEEP correct tokens. Our marker is justified: with no
   mask token, some input-level mode signal is needed (TiDAR shows nothing
   *beyond* such a signal is needed).
6. **Loss weighting: keep it flat.** No 1/t ELBO weighting (FDv2 drops it via
   complementary masks; DiffusionGemma uses none; TiDAR α-insensitive).
   Nemotron keeps 1/p_mask but adds global per-token batch averaging +
   per-DP-rank noise seeds as variance fixes. Simplest robust choice:
   unweighted CE, all positions, uniform noise-level sampling.
7. **Compute anchors**: capability conversion is cheap relative to pretraining.
   CLLM: 10M–200M tokens (hours–30h on 8×A100). Fast-dLLM v2: **1.3B tokens, 64
   A100 × 12 h** → fully functional 7B block-diffusion decoder. TiDAR: 50B
   (1.5B) / 150B (8B) for a *general* drafter at T/NFE 7.5–8.3. Nemotron: 300B
   joint (+1T AR warmup), but their 25B-token ablation already works.
   Domain-restricted (math) + already-Jacobi-trained init means our budget is
   the FDv2/CLLM end: **0.5–5B tokens, single-digit days on one 8×H200 node.**
8. **Framework reality check**: nobody hand-rolls. CLLM = HF Trainer + FSDP.
   JF = HF Trainer subclass + Accelerate + DeepSpeed ZeRO-3 + flex_attention.
   FDv2 = vanilla HF Trainer (LMFlow wrapper) + DS ZeRO-2, corruption injected
   inside `model.forward`. TiDAR = internal Megatron-LM+Torchtitan. Nemotron =
   Megatron-Bridge (released, recipes included). DiffusionGemma = JAX/Kauldron.
   Unsloth: only relevant as a *consumer* fine-tuning option for DiffusionGemma;
   not used by any of these training pipelines.

---

## 1. CLLM (Consistency LLMs, hao-ai-lab, arXiv 2403.00835)

Clone: `/mnt/weka/home/hao.zhang/shao/Consistency_LLM`. ~4 files matter.

**Objective.** Two losses, separate backwards, in a custom HF-Trainer
`training_step` (`cllm/cllm_trainer_global.py:19-108`):
- *Global consistency*: sample ONE random intermediate Jacobi-trajectory state
  per example; forward on it and on the fixed point (detached); soft CE
  H(softmax(fixed).detach(), log_softmax(intermediate)) — fwd-KL to the
  fixed-point distribution (`:71-97`). Same-weights self-distillation, no EMA.
  Already-converged prefix positions masked out of the loss (`:79-86`).
- *AR CE* on the model's own greedy output (`teacher_output_ids`, not GT),
  label-smoothed ε=0.1, **×10** (`:63-64`) — "to avoid pattern collapse".
- Paper: global (direct-to-fixed-point) beats local (adjacent-state) 3.0× vs
  2.4× speedup. Don't train transitions.

**Data.** Fully offline, one pass, generated by the pre-finetune model itself
(`data/generate_trajectory.py`): n-token blocks (16/32) initialized with
random tokens **sampled from the prompt** (`:201`), greedy Jacobi iteration to
fixed point, every iterate saved; per-block JSON. `--use_aug`: overwrite half
the still-wrong positions with fixed-point tokens — artificial corruption mixed
in because raw trajectories degenerate to "correct prefix + garbage tail".
Repetition-based quality filter. Sizes: GSM8K ≈10M tokens, ShareGPT ≈200M.

**Framework.** HF `Trainer` subclass; torchrun + HF-native FSDP
(`scripts/train_cllm.sh:34-35`); bf16, FA2, grad ckpt; bs=1/device.

**Scale.** 7B, 8×A100-40GB, LR 2e-5 cosine, 1 epoch; 2h (Spider) to 30h
(ShareGPT). 2.4–3.4× speedup, ~no accuracy loss.

**Transfers**: AR anchor ×10; corruption→fixed-point (not transitions);
masking converged positions out of the loss; in-distribution noise init
(prompt tokens, not uniform vocab); offline trajectories from the FROZEN
pre-stage model are fine because targets are defined by initial weights.
**Doesn't**: 100% causal, no bidir/marker anywhere; logit-level KL needs a
second forward; quality ceiling = own greedy output.

## 2. JacobiForcing (our base; arXiv 2512.14681)

Clone: `/mnt/weka/home/hao.zhang/shao/JacobiForcing`. This IS the mid-training
pipeline our base model came from — the most directly reusable asset found.

**Objective** (`train/soft_flexattn_cllm_trainer_multiblock.py`): packed layout
`[prompt][k_0][last_0]…[k_{T-1}][last_{T-1}]`, k_j = real noisy Jacobi iterate
of block j, last_j = fixed point (= AR greedy), **shared RoPE per pair**
(`:177-200`). Loss = AR CE ×10 on clean blocks (shift-by-1, EOS-truncated,
bridge pairs across blocks, `:420-543`) + consistency soft-CE: noisy-position
logits vs detached clean-position logits at matched offsets, single forward
(`:545-592`), divergence-gated (loss only from first k/last divergence onward,
`:127-168`). Strictly causal FlexAttention BlockMask everywhere — k_j sees
prompt + previous k blocks, last_j sees prompt + previous last blocks; streams
never cross (`:234-291`). Window variant bounds noisy-context exposure to w
blocks (across windows only clean visible).

**Noise schedule** lives in the DATA, not the trainer: cyclic linear schedule
`t_i = linspace(min,max,w)[itr mod w]`; the real trajectory iterate whose
measured suffix-divergence ratio is nearest the scheduled level is selected as
k_j (`generate_trajectory/data/2_prepare…progressive_noise_window.py:105-126`).
Decode-matched corruption, curriculum-in-data, static trainer.

**Two-flavor corruption** (their own bootstrap sequence): stage 0 model
("mask_1m_steps") was trained on ARTIFICIAL corruption — progressively replace
rightmost r tokens of each chunk with random tokens drawn from the preceding
≤128-token context (`1_progressive_masking_based_prepare_trajectory.py:64-87`);
then real Jacobi trajectories were generated FROM that model for the main run.
Artificial-corruption bootstrap → on-policy trajectory main course.

**Data.** OpenThoughts2-1M math split; packed datasets already sitting in the
repo: `data/OpenThoughts_Math_n16w16` (250k samples, ~430M tokens) and
`OpenThoughts_Math_n64w32` (230k samples). Coder: OpenCodeInstruct 450k.
bs=1 packed sample/device, no cross-sample packing.

**Framework.** Custom HF Trainer subclass + explicit Accelerate +
**DeepSpeed ZeRO-3** (CPU-offload config available), `flex_attention` attn
implementation, own AdamW β=(0.9,0.95) + HF cosine. torch 2.7.1, transformers
4.53, deepspeed 0.17.1.

**Scale.** Paper: 8×A100-80 + 8×H200; LR 1e-6; **two stages of 10k steps**
(stage 1 n=16/w=16 → stage 2 n=64/w=32, trajectories REGENERATED from the
stage-1 ckpt; ~20% extra speedup from stage 2). Our local repro: ~7.9 s/step
at 4 GPUs bs=1 → 10k steps ≈ 22 h.

**Transfers**: nearly everything — pair layout, shared RoPE, divergence
gating, schedule-in-data, two-stage regeneration, ZeRO-3 stack, and our local
marker retrofit (JF_USE_MARKER) already lives in this trainer. **Missing**:
bidirectional attention (every branch is causal) and the canvas corruption
family — exactly the v11/v11.1 deltas we'd add.
**Known sharp edges**: window trainer has a duplicated-method bug (second
`training_step` def wins → w=16 default, AR×1); long packed sequences NaN
(bf16+flex, see memory `feedback_jf_long_seq_nan`) — keep JF_FILTER_MAX_LEN.

## 3. Fast-dLLM v2 (NVIDIA, arXiv 2509.26328, ICLR 2026)

Clone: `/mnt/weka/home/hao.zhang/shao/Fast-dLLM` (`v2/` + `third_party/lmflow`).
**Key discovery**: the training objective is NOT in the repo's python — it
lives inside the HF trust_remote_code checkpoint's `modeling.py` (cached at
`~hao.zhang/.cache/huggingface/hub/models--Efficient-Large-Model--Fast_dLLM_v2_7B/
snapshots/0661ab…/modeling.py`). Corruption is applied inside `model.forward()`
when `self.training` — the data pipeline ships clean text + a stock collator,
noise is resampled GPU-side per step. Elegant pattern worth copying.

**Objective.** Per-32-token block: noise level t~U(ε,1) sampled independently
per block, i.i.d. Bernoulli(t) masking with a dedicated `|<MASK>|` token
(id 151665, added to vocab) on response tokens only (`modeling.py:580-598`).
Doubled sequence `[x_t ; x_0]` along seq dim, shared RoPE between halves
(`:197-207, 472-475`); clean half supplies cross-block KV context, its logits
discarded. **Complementary mask**: a second copy with the complement mask
concatenated along the BATCH dim → every token supervised exactly once →
1/t weighting dropped, plain unweighted CE on masked positions (`:600-613,
641`). Shift-by-1 retained (HF loss shift — logit at i predicts i+1).
Attention: M_BD (bidir within own noisy block) + M_OBC (noisy→clean strictly
earlier blocks) + M_BC (clean block-causal) via compiled flex_attention
(`:44-87`). **No AR CE term** — complementary coverage substitutes for it.
Pad-to-block-boundary with mask tokens during packing = +3.7 pts ablation
(`third_party/lmflow/pipeline/finetuner.py:175-200`).

**Framework.** Vendored LMFlow → vanilla HF `Trainer` + DeepSpeed **ZeRO-2**
no-offload; entry `v2/train_scripts/finetune.py` + `finetune_alpaca.sh`.

**Scale.** Qwen2.5-Instruct 1.5B/7B; seq 2048 packed; global batch 256
(524k tok/step); LR 1e-5 (7B); **2,500 steps ≈ 1.31B tokens; 64×A100 ~12 h**.
Note effective compute ≈ 4× nominal (2× seq doubling × 2× complementary copy).

**Transfers**: block 32 / decode sub-block 8 (mirrors our W=32/W_ar=8!);
per-block independent noise level; complementary-mask trick to equalize
supervision across noise levels without weights; in-forward corruption;
pad-to-block packing; the 1.3B-token/12h scale anchor. **Doesn't**: mask-token
corruption can never learn to REVISE wrong committed tokens (their decoder
never re-masks) — our plausible-wrong substitutions are strictly more general
for a refinement canvas; no marker needed because masks are self-identifying.

## 4. TiDAR (NVIDIA, arXiv 2511.08923) — paper only

Closest published cousin: causal "talk" section ≈ our AR/Jacobi zone, masked
"think" blocks ≈ our canvas, self-speculative rejection ≈ our prefix-match.

**Objective.** `L = 1/(1+α)·[α·mean(L_AR) + mean(L_diff)]`, α=1, both plain CE
in ONE forward over a doubled sequence `[x_1…x_S | m_1…m_S]`: clean copy
strictly causal with shift-by-1 NTP labels; mask copy partitioned into blocks
(4/8/16), each block bidir-within-itself + attends the clean causal prefix,
**labels UNshifted** (mask at i predicts x_i) — unshifted labels are what let
next-step drafts be computed before this step's verification resolves.
**Full-mask, single noise level** — justified explicitly by train-test
consistency with their ONE-shot drafting (Table 5: full-mask beats sampled-
ratio by ~3 pp avg / +5.8 HumanEval at equal T/NFE). No mode embedding of any
kind; mask id + attention topology carry the mode.

**Recipe.** Continual pretraining from AR BASE checkpoints (Qwen2.5-1.5B,
Qwen3-4B/8B); **50B tokens (1.5B) / 150B (8B)**; seq 4096 (8192 doubled);
batch 2M tokens; LR cosine 1e-5→3e-6, 1% warmup (tiny — knowledge-preserving
re-warm); bf16; modified Megatron-LM + Torchtitan, H100s; single stage, no SFT.

**Decode.** One forward = verify last step's block (causal logits, rejection
sampling) + pre-draft `block_len` candidate next-blocks conditioned on every
possible acceptance point (block_len² mask slots, sliced from one
pre-initialized FlexAttention mask). One-step diffusion drafting; rejected
drafts discarded, never re-denoised. Exact KV (rejected slots evicted).
T/NFE 7.45 (1.5B) / 8.25 (8B); 4.7–5.9× wall-clock vs AR; beats Block
Diffusion trained on the same recipe (44.0 vs 38.4 avg at 1.5B). β-mix
finding: verification can trust the DIFFUSION logit instead of the AR logit
("Trust Diff" slightly better on math at 8B) — quality is guaranteed by the
verification procedure, not by which head is smarter. Matches our "cons buys
TPF, not val".

**Transfers**: train-test corruption consistency as a PRINCIPLE (for our
persistent multi-pass canvas the consistent thing is multi-level empirical-
state corruption, not full-mask); α=1 equal per-token weighting is stable once
supervision is dense; tiny LR re-warm from base; train at the block size you
decode with; β logit-mixing at the verify boundary is a free ablation for our
acceptance rule. **Doesn't**: one-shot draft (no residence-time refinement —
they provide no evidence against ours, just don't do it); block_len² pre-draft
slots only economical at BS=1; 50–150B general-domain budget (we're math-only
and Jacobi-pre-trained; our reference point is FDv2's 1.3B, not this).

## 5. Nemotron-Labs-Diffusion (NVIDIA, tech report 2026-05; code RELEASED)

Code: `/mnt/weka/home/hao.zhang/shao/Megatron-Bridge`,
`src/megatron/bridge/diffusion/` (~700 lines of plug-in across 3 files on
stock Megatron infra). Tri-mode (AR / diffusion / self-speculation) 3B/8B/14B
from Ministral-3 bases.

**Objective.** `L = L_AR·1.0 + L_diff·0.3` (α=0.3 matched loss magnitudes;
both modes' accuracy peak at the SAME α — complementary, not zero-sum).
Dual-stream `[x_t | x_0]` (4096→8192 forward); one t~U(0,1) per sequence,
i.i.d. masking at p=(1-ε)t+ε with mask id, blocks (train 64 / inference 32)
exist only in the attention mask: noisy bidir-within-block + noisy→clean
strictly-earlier blocks + clean **strictly causal** (`diffusion/common/
dllm.py:181-213`) — strict causality of the clean stream (vs BD3-LM's block-
causal) is what makes the same-forward AR loss leakage-free (credited to Set
Block Decoding). Diffusion labels unshifted; 1/p_mask weighting kept but tamed
by **global per-token batch averaging** (+2.12 pts) and **per-DP-rank noise
seeds** (+0.71) (`dgpt_step.py:150-154, 321-326`). AR co-loss: +7.48 avg, the
biggest single ablation item. No mode token — mask id + attention pattern.
Untied diffusion output head (separate output layer init'd from embeddings).

**Self-speculation modes (Jacobi-adjacent; corrected 2026-06-12 after
primary-source re-read — do NOT equate with our assembly line):**
- *Linear SS (their DEFAULT shipping mode)*: **two forwards per cycle** —
  (1) append k [MASK] to the verified prefix, one diffusion forward denoises
  all k in one shot; (2) a second causal forward verifies, longest-prefix
  match + bonus token. Their own SOL section: "real TPF is the acceptance
  rate divided by two" — acceptance 6.82× but real per-forward TPF 3.41× on
  SPEED-Bench B=32. Structurally classic two-pass spec decode with shared
  weights, NOT our one-forward design.
- *Quadratic SS (Sec. 3.4 + App. C)*: single-forward draft+verify, but by
  their own words "following the same process as [20]" = **TiDAR's decode**.
  Input per cycle = previous k speculative tokens, EACH followed by k fresh
  masks (k² mask slots; k=16 → 273 query tokens): causal logits verify the
  carried drafts, mask blocks pre-draft the next block at every possible
  acceptance position. Fresh masks every iteration — no persistent draft
  state, no iterative refinement, unaccepted tokens discarded. TPF 6.38
  per-forward but at 1+k+k² query cost, and they ABANDON it in practice:
  FlexAttention kernel overhead makes it slower than linear SS on device.
  Our assembly line differs on every axis that matters: linear-cost (W-wide)
  persistent canvas vs k² branch materialization; iterative residence-time
  refinement vs one-shot mask prediction; real-token noise + marker (can
  learn to REVISE wrong tokens) vs mask-only corruption.
- TPF accounting caveat: their Table 5/6 linear-SS figures (4.52→5.99 w/
  LoRA) read as per-CYCLE acceptance, not per-forward (Sec. 4's definition
  halves them); don't compare raw against our per-forward jsim numbers.
- *Post-hoc frozen-backbone add-ons (these DO transfer)*: LoRA drafter
  (rank 128, o_proj only, ~0.4% params, top-K-truncated KL/TV + CE against
  the AR verifier on accepted+1 positions) → acceptance 4.52→5.99; trained
  commit-sampler classifier → +1.3× TPF over confidence thresholding;
  AR-diffusion ensemble verifier (TiDAR β-mix analog). Their stated future
  direction — commit "including at non-prefix positions" in a single forward
  — is the gap our canvas occupies; they motivate it but don't build it.

**Stages/scale.** 1T AR CPT (α=0) → 300B joint → 45B SFT (joint, answer-only
loss, seq 16k); 256×H100; GBS 512×4096; WSD LR 1e-5→3e-6. The released recipe
default reproduces the **25B-token ablation grade** (12.5k iters), which
already yields a working tri-mode 8B. Two-stage (strong AR first) beat
from-scratch-joint by +5.74.

**Framework usability for us.** Megatron-Bridge is the one mature released
framework with this objective built in: provider (~50-line dataclass swapping
core_attention) + DGPTStep (the whole training step incl. noising + dual
stream + both losses) + dllm.py mask utils; Qwen2 bridge EXISTS (qwen2_bridge
handles Qwen2.5 QKV bias); HF↔Megatron conversion both directions; torchrun
entry, Hydra-style overrides. Porting our objective = swap the mask predicate
(~15 lines), swap `forward_process_simple_masking` for our corruption, add a
custom embedding module for the marker (real Megatron surgery — the one part
with no hook). Sharp edges: no packed sequences (DGPTStep raises), no context
parallelism, compiled flex_attention not TE-fused, BlockMask fixed seq len,
MBS=1 at 4096 even for 3B (memory pressure from the 8192 forward).

## 6. DiffusionGemma (Google; sampler + SFT recipe public, conversion not)

Code ground truth: google-deepmind/gemma `gemma/diffusion/` +
google/hackable_diffusion (JAX/Kauldron). 26B-A4B MoE, adapted from the AR
checkpoint (token budget undisclosed).

**Objective (SFT recipe, code-confirmed).** Uniform-state discrete diffusion:
corruption = replace token with a draw from the FULL VOCAB uniformly (no mask
token; `CategoricalProcess.uniform_process`), linear schedule α(t)=1−t, one
t~U[1e-4,1−1e-4] per example. Pure x0-prediction. Loss = unweighted CE on
**ALL canvas positions including uncorrupted ones** (the carry-loss: model
must learn to KEEP correct tokens — load-bearing because with real-token noise
you can't tell noise from signal). **No time conditioning at all** (the time
arg is passed and ignored). Joint loss: diffusion CE 1.0 + **AR next-token CE
1.0** through the causal encoder path, same batch. Multi-canvas training:
corrupt everything, compute loss on ONE uniformly-sampled 256-token canvas
conditioned on prompt + CLEAN preceding canvases.

**Self-conditioning (their secret sauce we lack):** previous pass's logits
(stop-grad) → probability-weighted embedding average → RMSNorm+FFW+RMSNorm
block added onto canvas embeddings; applied with prob 0.5 during training
(two decoder passes); carried across refinement steps at inference. Carries
soft information that re-noising would otherwise destroy.

**Sampler (the rule our spec §2.4 adopted — exact semantics):** init canvas
fully uniform-random; ≤48 steps; temperature annealed 0.8→0.4 linear;
categorical-sample every position; accept by ENTROPY BUDGET (sort ascending,
accept while cumulative entropy ≤ 0.1 — not a per-token threshold); **every
non-selected position gets a fresh uniform-random token each step; nothing is
ever frozen** — acceptance is recomputed from scratch each step, so confident-
then-doubtful positions get re-noised (self-correction). Early stop: argmax
stable across steps AND mean entropy <0.005. Commit a finished 256-canvas via
causal prefill into KV; block-AR across canvases. ~15–20 committed tokens per
forward; 1,288 tok/s on H200 FP8 in vLLM (~6× AR). Quality admittedly below
AR Gemma 4 — speed-first recipe.

**Transfers**: the carry-loss (supervise uncorrupted positions — REQUIRED for
our real-token noise; check our loss does this); flat uniform t per
example/window; no time conditioning (argues for our minimal marker and
against any t-embedding our decode can't honestly populate); equal-weight AR
co-training; one-sampled-canvas-per-example conditioned on clean prefix =
exactly our window scheme; entropy-BUDGET acceptance + temperature anneal +
never-freeze are decode knobs to revisit once canvas quality rises (our
keep_tau sweep showed re-noising hurts NOW, at low canvas quality — their
never-freeze works because carry-loss training makes identity-on-clean
reliable). Self-conditioning is the one architectural addition worth
prototyping in v12+. **Caveat**: our incremental commits skew decode-time
noise levels low vs their all-or-nothing block commit — extra t-mass at low
noise is the one place to deviate from flat-uniform.

---

## 7. Framework comparison

| | objective home | trainer core | parallelism | our-objective fit | maturity/risk |
|---|---|---|---|---|---|
| CLLM | custom training_step | HF Trainer + torchrun FSDP | FSDP full-shard | causal-only, bs=1 | simple, dated |
| **JacobiForcing** | custom training_step | HF Trainer subclass + Accelerate | DeepSpeed ZeRO-3 (+CPU offload) | **pair layout, shared RoPE, flex BlockMask, marker retrofit ALREADY THERE**; needs bidir branch + canvas corruption | proven on OUR base model at OUR scale; known NaN + window-trainer bugs to patch |
| Fast-dLLM v2 | inside `model.forward` | vanilla HF Trainer (LMFlow) | DS ZeRO-2 | corruption-in-forward pattern is elegant; mask-token-centric | clean, small |
| TiDAR | n/a | Megatron-LM + Torchtitan (internal) | — | no code | n/a |
| Nemotron / Megatron-Bridge | DGPTStep + dllm.py | Megatron-Bridge recipes | TP/PP/DP, dist-opt, multi-node native | dual-stream + AR co-loss BUILT IN; Qwen2.5 bridge exists; marker = embedding surgery; no packing, no CP | most mature; heaviest port (~days), container/TE stack |
| DiffusionGemma | Kauldron configs | JAX/Flax | TPU-style | wrong ecosystem for us | n/a |

**Recommendation: build the mid-training stage on the JacobiForcing trainer
stack** (HF Trainer subclass + Accelerate + DeepSpeed ZeRO-3 + flex_attention).
Reasons: (1) it is literally the pipeline our base model came from — data
generators, packers, noise-schedule mapper, shared-RoPE pair layout, and our
sinusoidal-marker retrofit already exist and are debugged against this exact
model; (2) our verl `scripts/consistency/{pack,loss,attention}.py` modules are
the same design family and port almost line-for-line; (3) 7B on 1–2 nodes does
not need Megatron parallelism — ZeRO-3 (or a swap to FSDP2) carries it; (4) it
satisfies "mature framework, no hand-rolled loop" — the loop is HF Trainer.
**Fallback / scale-up path**: if we ever go multi-node ≥10B-token
conversion-grade, port to Megatron-Bridge (DGPTStep is a clean template; Qwen2
bridge exists; budget ~1 week including marker surgery).

---

## 8. Proposed pipeline: "v12 mid-train" (concrete)

### 8.1 Shape

Three phases, mirroring JF's own bootstrap and consistent with every survey
finding (artificial corruption → on-policy trajectories; strong-AR-first):

**Phase M0 — data.** Two sources, both cheap:
- (a) Reuse JF's packed `OpenThoughts_Math_n16w16` / `n64w32` (already on
  disk, ~430M/~340M tokens) for the causal-pair stream.
- (b) **Generate our own rollouts** (on-policy distillation flavor): greedy
  responses from the chosen init checkpoint on DeepScaler + OpenThoughts math
  prompts (vLLM batch on spare GPUs; ~100k prompts × ~700 tok ≈ 70M tokens of
  self-generated clean text, hours of GPU time). Per CLLM's lesson, targets
  defined by the FROZEN init model don't go stale offline. Canvas pairs are
  built from these by artificial empirical-state corruption (v11.1
  construction — fresh-random renoise at multi-level f + plausible-wrong
  substitutions calibrated to the decode-trace stats in
  `eval_passk/tpf_results/v11/traces/`), so no trajectory harvesting needed
  for phase 1.

**Phase M1 — mid-train (the main act).** JF trainer + our additions. Packed
layout per sample: `[prompt][pair_0]…[pair_{T-1}]`, each pair = (corrupted
tile | clean tile), shared RoPE, one MODE per pair:
- **Canvas pairs** (frac ~0.5): corrupted tile = empirical-state corruption of
  the clean window (multi-level f ∈ {1.0,…,0.125} uniform per pair, far-
  weighted positions, ρ≈0.1 plausible-wrong subs); attention bidir within
  tile, causal to prompt + prior CLEAN tiles; **marker ON**; loss = unweighted
  CE, shift-by-1, **ALL tile positions incl. uncorrupted (carry-loss —
  DiffusionGemma's load-bearing detail; VERIFIED already present: the v11.1
  canvas CE in `scripts/consistency/loss.py:794-806` selects by the pair-level
  canvas flag, so non-renoised positions are supervised too)**.
- **Causal Jacobi pairs** (frac ~0.5): v9-style — corrupted tile = JF noise-
  schedule-selected state (artificial in M1), causal mask, no marker,
  divergence-gated soft-CE/KL at matched offsets (JF `:545-592`) or our
  fwd-KL-decay; keeps the AR-zone drafting skill that v9/JF built.
- **AR CE anchor on the clean stream, weight 5–10** (CLLM/JF use 10; Nemotron
  shows it's the biggest win; we get it free from the clean tiles).
- Hyperparams (survey consensus): LR **1e-6–1e-5 cosine** (JF used 1e-6 on
  this exact model; TiDAR's preservation lesson says stay low), bf16, ZeRO-3,
  seq cap via JF_FILTER_MAX_LEN (NaN memory), 10k steps ≈ 1 epoch of the
  packed math set, save every 500, **checkpoint-select by TPF** (jsim-16
  vanilla + assembly battery — erosion/selection discipline from the RL runs
  applies here too).
- Init: **pre-RL JF base** (v11.1-base evidence: co-training from base beat
  retrofitting v9_220) — `models--JacobiForcing--JacobiForcing_Math_7B_v1/
  snapshots/e65283c…`.

**Phase M2 (optional, JF stage-2 pattern) — on-policy refresh.** Regenerate
trajectories/rollouts from the M1 checkpoint (real assembly-decode canvas
states for canvas pairs — harvest actual decode-time tiles via
`_sim_assembly_decode.py` tracing; real Jacobi iterates for causal pairs via
JF's greedy generator), re-pack with the schedule mapper, train 5–10k more
steps. This is the full OPD option: corruption distribution = the real thing,
teacher = clean-context logits of the frozen base (we already have the
fwd-KL-to-base plumbing).

**Phase M3 — RL, sparingly.** ≤40–80 DAPO steps with the existing cons loss as
guardrail (the v11.1 sweet spot), or skip entirely if M1/M2 TPF + val already
clear the bar. RL's job shrinks to reward polish, not capability.

### 8.2 What we adopt from whom (objective checklist)

- AR CE anchor in-forward, weight ≥5 [all five works]
- x0 corruption→clean CE, no transition targets [all]
- Multi-level uniform noise per pair; schedule lives in data [JF, FDv2, NLD]
- Decode-matched/empirical-state corruption [TiDAR Table 5; our v11.1]
- Carry-loss on uncorrupted positions [DiffusionGemma — REQUIRED for
  real-token noise]
- No 1/t weighting; flat per-token CE; consider per-DP-rank noise seeds +
  global token averaging if loss is noisy [FDv2, NLD]
- Divergence-gated masking for the CAUSAL pairs only [CLLM, JF]
- Shared-RoPE pair layout, single forward, detached clean branch [JF, FDv2,
  NLD, TiDAR]
- Marker on canvas tiles only (our substitute for the mask id) [TiDAR: some
  input-level mode signal suffices; nothing more needed]
- Pad/align tiles to block boundaries when packing [FDv2 +3.7 pts]
- v12+ candidates: self-conditioning block [DGemma], LoRA drafter +
  commit-sampler post-hoc [NLD], β logit-mix at verify [TiDAR]

### 8.3 Compute estimate (8×H200 node, GPUs 0-7 or 4-7)

- M0 rollout gen: ~100k prompts greedy on 4 GPUs vLLM ≈ 4–8 h.
- M1: JF measured ~8 s/step at bs=1×4 GPU on this model; 8 GPUs ⇒ ~4 s/step
  effective ⇒ 10k steps ≈ **11–22 h on one node** (≈0.4–0.8B tokens through
  the loss). Two epochs / 20k steps ≈ 1–2 days. Well inside FDv2's 1.3B-token
  anchor for a full conversion, and we start Jacobi-pre-trained.
- M2: regen ≈ 1 day (trajectory harvesting is the slow part), train ≈ 1 day.
- Total: **~3–5 node-days for the full M0–M2 program**; M1 alone (the minimum
  viable experiment) is ~1 day. Fits GPUs 4-7 of one shao_dllm node in
  half-speed mode if the RL boxes stay busy.

### 8.4 Success gates (same protocol as the RL line)

1. Causal regression: vanilla jsim ≥ 4.0 (v11.1-base s40 = 4.11 is the bar),
   MATH val ≥ 0.88 (watch: v11.1 currently pays ~2.5 pp val — the AR anchor
   weight is the knob, and mid-training LR 1e-6 should pay less than RL did).
2. Canvas: assembly bidir+marker BEATS causal control (gap > 0, vs best
   −0.048 so far).
3. Headline: assembly TPF ≥ 4.5 interesting, ≥ 6 compelling (spec §3.6).

### 8.5 Implementation work list

1. Port `scripts/consistency/{pack,loss,attention}.py` canvas-pair logic into
   the JF trainer as a third corruption flavor + bidir mask branch (the
   FlexAttention BlockMask builder at `trainer_multiblock.py:234-291` gets a
   per-pair mode flag, same design as our verl per-pair plumbing). ~1–2 days.
2. New packer stage: empirical-state corruption from rollout JSONs (reuse
   `2_prepare…progressive_noise_window.py` skeleton; swap iterate-selection
   for our corruption construction). ~1 day.
3. ~~Verify carry-loss~~ DONE: already all-position (`loss.py:794-806`,
   selection by pair-level canvas flag).
4. Patch known JF bugs: window-trainer duplicate `training_step`; keep
   JF_FILTER_MAX_LEN; NaN guard stays. ~2 h.
5. Rollout generation script (vLLM greedy over DeepScaler+OpenThoughts
   prompts). ~half day.
6. Eval loop unchanged (jsim/assembly batteries already exist in verl).

Open questions for the v11 owner (also sent via agent message):
- Init from pre-RL base vs v11p1_canvas_base_step_40? (Spec evidence favors
  base + co-train; but s40 already embodies M1-lite — a v11p1_s40-init M1 run
  is the cheapest A/B.)
- Canvas-pair fraction and AR-anchor weight priorities for the first sweep
  (proposal: frac 0.5, AR×5, LR 3e-6 as the center point).
- Do we want M2 (real harvested canvas states) before or after the first
  full TPF eval of M1? (Proposal: after — M1 is the cheap falsifier.)
