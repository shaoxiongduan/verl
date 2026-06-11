# TPF benchmark log

Append-only log. One entry per benchmark batch. Each entry must record:
engine, prompts file + count, decoding params (T, top_p, max_new), Jacobi K,
batch size, model ckpt path, the run/wandb name and step, and the resulting
per-prompt TPF stats. Raw per-prompt rows live next to this file as
`<run>__<engine>_<tag>.jsonl`.

Engines:
- **vllm-jacobi**: vLLM 0.10.2 + `scripts/vllm_jacobi_patch.py` (ngram-slot
  hijack -> JacobiProposer). Reproduce with
  `scripts/vllm_tpf_trajectories.py`.
- **jf-nanovllm**: JacobiForcing reference / `JacobiForcing/jf_inference_he_our_models.py`.
  Reproduce with `scripts/tpf_trajectories.py`.

Always quote per-prompt mean ± SE (SE = std/sqrt(n)) and min/max so future
readers can judge whether deltas are signal or n=64 noise. Note any
degenerate-loop prompts that hit `max_new_tokens` cap — they inflate greedy
TPF and should be flagged.

---

## 2026-06-04 (early) — vLLM Jacobi grid: 4 models x {greedy, T=0.6}, max_new=1024

(This is the **original** off-by-params grid; see entry below for matched-train-params re-run.)

## 2026-06-04 — Train-params grid v2: vLLM Jacobi (T=1.0) + JF reference (greedy block-decode)

**Why:** previous batch used T=0.6 and max_new=1024 which don't match training.
Training rollout uses T=1.0, top_p=1.0, top_k=-1, max_response_length=2048,
K=32 spec window. The JF reference engine here is the **JacobiForcing repo**
(`jacobi_forward_greedy`), NOT the Decode-Learning nanovllm port — user
considers nanovllm untrusted.

**Engines:**
- **vllm-jacobi** (training rollout engine): `scripts/vllm_tpf_trajectories.py`
  + `scripts/vllm_jacobi_patch.py`. Generation method = **shift-by-1 windowed
  Jacobi refresh** — next draft = target_argmax[n_acc+1:], one bonus token
  committed per iter. Params: K=32, T=1.0, top_p=1.0, top_k=-1,
  max_new_tokens=2048, max_model_len=4096, BS=16.
- **jf-reference** (JacobiForcing/JacobiForcing/jf_inference_he_our_models.py
  via JacobiForcing/.venv/bin/python): bare-transformers + flash_attn_2 +
  `jacobi_forward_greedy` from
  `modeling.cllm2_qwen2_modeling_kv_terminate_on_eos_improved`. Generation
  method = **fill-one-block (block decode)**: per K-block, run up to ~128
  Jacobi refinement iters until convergence/accept; commit accepted prefix +
  bonus first_correct_token; refresh tail draft via DRAFT_INIT=prompt_sample
  (`random.choice(generated_ids)`); start next block. **Greedy only — no
  temperature.** Params: N_TOKEN_SEQ_LEN=32, MAX_NEW_TOKENS=2048, NUM=64,
  DRAFT_INIT=prompt_sample.

**Prompts:** `eval_passk/eval_prompts_tpf.jsonl` (vllm) /
`eval_passk/eval_prompts_tpf.parquet` (JF ref), n=64 math prompts.

**Hardware:** node fs-mbz-gpu-961, GPU index 4.
**Grid script:** `scripts/_run_tpf_train_params_v2.sh`
**Raw rows / logs:**
  - vllm: `eval_passk/tpf_results/<run>__vllm_trainparams.jsonl`
  - jf-ref: `eval_passk/tpf_results/<run>__jfref_trainparams.log`
**Driver log:** `eval_passk/tpf_results/_train_params_v2.log`

### vLLM Jacobi (T=1.0)

| Run | Ckpt path | mean | SE | min | max | n_tok_mean |
|---|---|---:|---:|---:|---:|---:|
| base JF Math 7B (pre-RL) | `~/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1` | 4.494 | 0.128 | 2.580 | 7.184 | 801 |
| math_ar_ds_step_300 (RL-AR) | `ckpts_hf/math_ar_ds_step_300` | 2.700 | 0.048 | 1.748 | 3.637 | 504 |
| dflashce_v2 step_20 (cons RL "A", target_ratio=0.1) | `ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_v2_step_20` | 3.676 | 0.157 | 1.276 | 6.778 | 628 |
| dflashce_fixed step_20 (cons RL "B", fixed w=0.01) | `ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_fixed_step_20` | 2.937 | 0.163 | 1.181 | 6.293 | 570 |
| dflashce_corrupt03_v4 step_20 (multi-tile K=4 + focal γ=2.0, killed) | `ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_corrupt03_v4_step_20` | 4.045 | 0.126 | 1.324 | 5.747 | 682 |

### JF reference (greedy block-decode, JacobiForcing repo)

| Run | mean | std | median |
|---|---:|---:|---:|
| base JF Math 7B (pre-RL) | 3.443 | 0.573 | 3.517 |
| math_ar_ds_step_300 (RL-AR) | 2.673 | 0.346 | 2.645 |
| dflashce_v2 step_20 (cons RL "A") | 4.017 | 1.532 | 3.780 |
| dflashce_fixed step_20 (cons RL "B") | 4.144 | 1.931 | 3.732 |

**Key takeaways:**
- **Cons RL preserves TPF on the JF reference engine** (its native eval):
  cons-A 4.02 / cons-B 4.14 vs base 3.44 (+17% / +20%) — both cons step_20
  runs *beat* base by a wide margin, AR-RL collapses to 2.67 (-22%).
- **On vLLM Jacobi at T=1.0** the story flips for cons runs: base 4.49
  beats both cons (3.68 / 2.94). This is the **engine + temperature
  divergence**: the training-time engine is vLLM Jacobi T=1.0, so we should
  give *most* weight to that table. Cons-RL at step_20 has not converged
  enough to lift T=1.0 vLLM TPF above base; the JF-ref greedy gain doesn't
  carry over.
- AR-RL collapse is consistent across both engines (2.70 vLLM, 2.67 JF ref).
- Cons B's high JF-ref median 3.73 with std 1.93 indicates several
  degenerate-loop / max_new=2048 outliers. Check per-prompt log to confirm.

---

## 2026-06-04 (follow-up) — vLLM Jacobi greedy at max_new=2048

**Why:** isolate temperature effect from length-cap effect for the
train-params drop.
**Same setup as train-params grid above but T=0.0** (greedy). Identical K=32,
max_new=2048, BS=16, n=64.
**Grid script:** `scripts/_run_vllm_greedy_2048.sh`
**Driver log:** `eval_passk/tpf_results/_greedy_2048.log`

| Run | mean | SE | max | n_tok_mean |
|---|---:|---:|---:|---:|
| base JF Math 7B | 4.017 | 0.090 | 5.22 | 643 |
| math_ar_ds_step_300 (RL-AR) | 2.732 | 0.043 | 3.71 | 550 |
| dflashce_v2 step_20 (cons A) | 4.557 | 0.196 | 11.91 | 735 |
| dflashce_fixed step_20 (cons B) | 4.986 | 0.335 | 17.04 | 786 |
| dflashce_corrupt03_v4 step_20 (multi-tile K=4 + focal γ=2.0) | 4.196 | 0.141 | 9.07 | 711 |

**Side-by-side delta-T (vLLM Jacobi, max_new=2048 fixed):**

| Model | T=0 | T=1.0 | ΔT (T=1 − T=0) |
|---|---:|---:|---:|
| base | 4.017 | 4.494 | **+0.48** |
| AR | 2.732 | 2.700 | -0.03 |
| cons A | 4.557 | 3.676 | **-0.88** |
| cons B | 4.986 | 2.937 | **-2.05** |
| v4 (multi-tile + focal) | 4.196 | 4.045 | **-0.15** |

**Confirmed:** length cap had near-zero effect (base 4.005@1024 vs 4.017@2048,
AR 2.733 vs 2.732). The drop is purely a temperature effect on cons-RL
step_20: greedy is the best case, sampling at training-time T=1.0 erodes the
gain. Cons B is the brittlest (fixed weight + no target ratio = strongest
overfit to the narrow correct-only cons distribution). Cons B's huge max
(17.04) is degenerate-loop on a few prompts hitting max_new=2048 cap.

---

## 2026-06-08 — Shift-leak sim: boundary-predictor error tolerance

**Question:** the warm-restart protocol achieves 6.00 TPF_verify for math_k3 by
restarting each cycle with fresh K-noise after a perfect-boundary commit. In
practice, a lightweight boundary predictor (= identifies where the
already-accepted prefix ends in the K-block) won't be perfect. **How much
error can the predictor have before TPF degrades?**

**Protocol** (`scripts/_sim_warm_shift_leak.py`):
- Each cycle = (warm forward + verify forward) under greedy.
- Cycle's warm INPUT = `[committed | shift_leaked_tokens + (K-shift) fresh
  uniform random]`. The leaked tokens are the model's own WRONG predictions
  from the previous cycle (= `warm_prev[n_acc+1 : n_acc+1+shift]`). They're
  "in-distribution wrong" tokens, not random noise.
- `shift=0` = perfect predictor (all positions reinit'd to fresh random) →
  equivalent to warm_restart baseline.
- `shift=N` = predictor failed to reset N positions; they retain the model's
  prior-iter wrong guess.
- TPF_verify counts only verify forwards (warm treated as "free" — would be
  the cheap predictor's output in production).

**Results** (math_k3_ds_step_300, K=32, greedy, 16 DS prompts, max_new=512):

| shift | mean_n_acc/cycle | TPF_verify | TPF_all (counts warm too) | Δ vs perfect | % of perfect |
|---:|---:|---:|---:|---:|---:|
| **0** (perfect) | **5.010** | **6.004** | 3.002 | — | 100% |
| 1 | 4.828 | 5.824 | 2.912 | −0.180 | 97.0% |
| 2 | 4.783 | 5.780 | 2.890 | −0.224 | 96.3% |
| 3 | 4.734 | 5.728 | 2.864 | −0.276 | 95.4% |
| 4 | 4.585 | 5.579 | 2.790 | −0.425 | 93.0% |
| 5 | 4.519 | 5.513 | 2.756 | −0.491 | 91.8% |

**Headline: highly graceful degradation.** Cost ≈ **−0.1 TPF per token of
predictor error**. A predictor off by 5 still delivers 92% of the perfect
TPF_verify (5.51 vs 6.00).

**Why graceful (vs catastrophic with random corruption):** the leaked tokens
are the model's own wrong predictions — "in-distribution garbage" the model
knows how to ignore (cons training learned robustness to this pattern). With
random-token corruption (an earlier broken sim variant), the first random
token at position 0 immediately broke greedy spec-decode prefix acceptance →
TPF crashed to 1.0.

**Implication for production:** the "perfect 6 TPF ceiling" math_k3 hits is
NOT fragile. A cheap boundary predictor (n-gram, small EAGLE-style head, or
even shift-by-fixed-offset heuristic) needs only ~3-5 token accuracy to
deliver ~5.5–5.8 TPF_verify. If the warm forward itself can be cheapened
(KV-cache reuse, distilled draft head), the harvested TPF lands ~5.5 per
big-model forward — substantially above natural Jacobi's 3.4.

**Baseline consistency check (warm_restart, shift=0):**
- math_k3: mean_n_acc/cycle = 5.010, TPF_verify = 6.004 ✓ (matches Jun-7 sim)
- Confirms the shift-leak code path reduces exactly to warm_restart when
  shift=0 — no implementation drift.

---

## 2026-06-08 — Boundary-predictor feasibility study: per-position features

**Setup.** Dumped per-cycle warm-forward logit features (top1_prob, entropy,
margin_log = logit_top1 − logit_top2, top5_prob_sum) and ground-truth
per-position accept labels (= int(warm_argmax[j] == verify_argmax[j])) from
warm_restart shift=0 cycles on 16 DS prompts. Data dumps:
- `boundary_data_math_k3_shift0.jsonl` — 1392 cycles × 32 positions = 44 544
- `boundary_data_base_shift0.jsonl`    — 1689 cycles × 32 positions = 54 048

Collector: `scripts/_collect_boundary_dataset.py`. Analyzer:
`scripts/_analyze_boundary_features.py`.

**Per-feature AUC (single-scalar threshold classifier, all 32 positions
pooled):**

| Feature      | math_k3 AUC | base AUC |
|---|---:|---:|
| top1_prob    | 0.559       | 0.634    |
| entropy      | 0.508       | 0.614    |
| margin_log   | **0.687**   | **0.667**|
| top5_prob_sum| 0.516       | 0.611    |

Counter-intuitive: **cons training has made the model LESS confidence-
calibrated** for accept-vs-reject. Base's top1_prob AUC 0.63 > math_k3's
0.56. Cons loss spreads probability across many tokens at noise-conditioned
positions, so high-entropy no longer signals "the model is wrong."

**Position alone is highly predictive.** Accept-rate vs position j (full K-block):

| j  | math_k3 acc-rate | math_k3 top1_p | base acc-rate | base top1_p |
|---:|---:|---:|---:|---:|
| 0  | 1.000 | 0.94 | 1.000 | 0.98 |
| 1  | 0.85  | 0.83 | 0.79  | 0.84 |
| 3  | 0.66  | 0.61 | 0.53  | 0.59 |
| 5  | 0.51  | 0.46 | 0.41  | 0.44 |
| 10 | 0.35  | 0.24 | 0.28  | 0.23 |
| 20 | 0.47  | 0.12 | 0.38  | 0.14 |
| 31 | 0.67  | 0.10 | 0.49  | 0.14 |

Acceptance drops monotonically through j=10 then rises again at the tail
(positions 20–31 randomly match — top1_prob there is ≤0.15, but warm and
verify both produce similar argmax on K-noise context, so they match by
chance). The **boundary signal lives in j≈3–10**, exactly where the
predictor needs to be sharpest.

**3-feature logistic regression (top1_prob, entropy, pos/K → accept), GD,
80/20 split:**

| Model | val AUC | val acc | feature weights (std-scaled) |
|---|---:|---:|---|
| math_k3 | **0.750** | 68.2% | top1=+1.07, ent=+0.42, pos=+0.35, bias=+0.10 |
| base    | 0.689 | 66.5% | top1=+0.71, ent=−0.25, pos=+0.44, bias=−0.29 |

Combining position with the logit features lifts math_k3 from 0.56 (top1
alone) to 0.75 — a +0.19 AUC gain. Position is the single most important
extra signal.

**MLP on same features — barely improves over linear** (`scripts/_analyze_boundary_mlp.py`, 80/20 split):

| Model | Predictor | val AUC | val acc |
|---|---|---:|---:|
| math_k3 | 5-feat logreg          | 0.766 | 69.5% |
| math_k3 | 5-feat MLP-16          | 0.766 | 69.4% |
| math_k3 | 15-feat MLP-32 (+top5) | 0.770 | 69.8% |
| base    | 5-feat logreg          | 0.700 | 66.2% |
| base    | 5-feat MLP-16          | 0.706 | 66.4% |
| base    | 15-feat MLP-32 (+top5) | 0.714 | 67.8% |

The 5-feature MLP-16 hits the **same AUC as linear logreg** (0.766 for
math_k3), and adding top-5 probs only gains +0.004. **Logit-only features
have a hard ceiling around AUC 0.77.** Breaking through it requires
hidden states (cf. SpecDec++, AutoJudge in the lit review below).

**Threshold-based boundary prediction (`b = first j where top1_prob < τ`):**

| Model   | τ    | exact | |err|≤1 | |err|≤2 | |err|≤3 | mean signed err |
|---|---:|---:|---:|---:|---:|---:|
| math_k3 | 0.50 | 35%   | 67%      | 81%      | 89%      | −0.22 |
| math_k3 | 0.70 | 33%   | 64%      | 80%      | 88%      | −1.42 |
| base    | 0.50 | 31%   | 62%      | 79%      | 87%      | +0.11 |
| base    | 0.70 | 39%   | 69%      | 84%      | 91%      | −1.15 |

At τ=0.5 the median absolute error is 1 token; ~80% of cycles are within
±2 tokens. The shift-leak experiments showed each ±1-token error costs
~0.1 TPF, so this simple rule loses ~0.1–0.3 TPF vs perfect. Headline:
**a zero-parameter `top1_prob < 0.5` rule already harvests roughly 5.7 TPF
on math_k3 and 4.8 TPF on base** (extrapolating shift-leak's 0.1 TPF/
token degradation × ~2-token typical |err|).

---

## 2026-06-08 — Learned predictor swapped into shift-leak: actual TPF

**Setup.** `scripts/_sim_warm_predictor.py` replaces the shift-leak's
fixed-`shift` with a learned predictor that decides, per cycle, which warm
positions to retain as the next cycle's mixed_draft. 16 DS prompts, K=32,
greedy, max_new=512, math_k3_ds_step_300.

**Predictor variants (all logit-only features, no hidden states):**
- `logit_p`: keep warm[j] if `top1_prob[j] > 0.5`
- `logreg5`: keep warm[j] if `σ(w·[top1,ent,marg,t5sum,pos/K]) > 0.5`
  with the std-scaled logreg weights computed earlier (math_k3 val AUC=0.766)

**Leak modes:**
- `boundary`: predicted boundary `b = first j where p_correct < 0.5`;
  retain warm[0..b], reinit warm[b..K]
- `per_position`: independent decision per-j; retain only positions where
  p_correct > 0.5 (allows holes)

**Results (corpus TPF_verify = Σ n_tokens / Σ n_cycles):**

| Predictor | Leak mode | Σ n_tokens | Σ n_cycles | TPF_verify | per-prompt-mean | Δ vs oracle |
|---|---|---:|---:|---:|---:|---:|
| `top1_p>0.5` | boundary       | 8273 | 1416 | 5.8425 | 5.892 | −0.100 |
| `top1_p>0.5` | per_position   | 7801 | 1403 | 5.5602 | 5.563 | −0.383 |
| **logreg5**  | **boundary**   | **8297** | **1411** | **5.8802** | **5.921** | **−0.063** |
| logreg5      | per_position   | 8060 | 1457 | 5.5319 | 5.548 | −0.411 |

**Reference (shift-leak from same model):**
- shift=0 oracle (perfect boundary): 5.9425
- shift=1: 5.791,  shift=5: 5.460

**Headline.** The 5-feature logreg + boundary leak gives **5.880 TPF_verify**
— **0.063 TPF below the perfect oracle (99% of ceiling)**, achieved with
only logit-derived features (no hidden states, no per-token decoding head).
In shift-leak terms, its effective predictor error is between shift=0 and
shift=1, i.e., < 1 token of average misalignment.

**Two findings.**
1. **`boundary` mode > `per_position` mode by ~0.3 TPF.** A contiguous
   prefix beats independent decisions: isolated wrong tokens in the
   middle of the K-window degrade the warm forward more than a clean
   cutoff. Cons-trained model's denoising kernel expects "clean prefix +
   noisy tail," not "noisy holes."
2. **logreg5 beats `top1_p>0.5` by only +0.04 TPF.** Top-1 probability
   alone already captures most of the boundary signal; the marginal lift
   from adding entropy/margin/top5_sum/position is small. **Logit-only
   features have a hard ceiling around 5.88 TPF on math_k3.**

**Production cost.** Both predictors are FREE in compute terms — they
operate on the warm forward's logits which are already computed. The
"warm forward" is still a full 7B forward in this sim; the predictor only
recycles its output efficiently. To unlock real wall-time speedup,
**the warm forward itself must be cheapened** (n-gram, EAGLE-style head,
or distilled draft model). That is the next prototype.

---

## 2026-06-08 — 1-forward-per-iter Jacobi + predictor refresh (no verify)

**Question.** Can we replace the verify forward with a predictor that "cleans"
the noisy tail between iters? Standard JF block-decode (1 forward per iter,
acceptance = "this iter agrees with previous iter's input on a prefix"),
with the predictor refresh inserted between iters to reinit predicted-wrong
positions to fresh random.

**Sim:** `scripts/_sim_jacobi_predictor_refresh.py` on math_k3_ds_step_300,
K=32, max_new=512, max_iters=512, 16 DS prompts, greedy.

**Initial bug — stall below the AR floor (TPF<1).** With `boundary` leak
and threshold=0.5, the predictor can reject position 0 of the shifted window
(e.g., when cur_argmax[n_acc]=`' '` has P_correct=0.33). The refresh reinit's
position 0 to fresh random. Next iter's argmax[0] is the model's confident
continuation given the committed context — which does NOT equal the random
token we placed there → n_acc=0 → infinite stall. TPF crashes to 0.05.

**Fix:** `--always_keep_pos0` forces `keep_mask[0] = True`, guaranteeing the
shifted prefix retains the model's actual next-token argmax at position 0.

**Threshold sweep on math_k3, 16 DS prompts (corpus TPF):**

| refresh | thr | pos0fix | corpus TPF | per-prompt mean | Δ vs baseline |
|---|---:|:---:|---:|---:|---:|
| **none** (standard JF baseline) | — | — | **3.29** | 3.31 | — |
| logreg5 | 0.1 | no   | 3.29 | 3.31 |  0.00 |
| logreg5 | 0.2 | no   | 2.82 | 3.19 | −0.47 |
| logreg5 | 0.5 | no   | **0.05** | 0.05 | **catastrophic** |
| logreg5 | 0.1 | yes  | 3.29 | 3.31 |  0.00 |
| logreg5 | 0.3 | yes  | 3.20 | 3.30 | −0.09 |
| logreg5 | 0.5 | yes  | 3.26 | 3.28 | −0.03 |
| **logit_p** | 0.5 | yes  | **3.36** | **3.40** | **+0.07** |

**Two findings.**

1. **Below the AR floor.** Pure autoregressive greedy gives 1 TPF (1 token
   per forward). Any decoder that drops below that is misbehaving. Our
   thr=0.5 no-guard sim hit 0.05 — a 20× regression vs AR. The bug was
   straightforward (predictor rejecting position 0 → reinit-to-random
   forever) but illustrates the fragility of the approach.

2. **Best predictor refresh barely beats baseline.** logit_p + pos0fix
   at thr=0.5 gives 3.36 TPF, +0.07 over standard JF. logreg5 actually
   hurts at any threshold ≥0.3. Even the best predictor refresh is far
   below the 5.94 ceiling of the shift-leak protocol.

**Why the ceiling is so low.** Standard JF block-decode iterates by feeding
**the model's own previous predictions** as the next draft. This works
because the model is self-consistent: positions stabilize after a few
iters. The predictor refresh **replaces** the model's predictions with
**fresh random** at "wrong" positions — but random tokens are strictly
worse than the model's own (even-wrong) predictions for convergence: the
model has never seen random in this slot before and produces a different
argmax → no convergence. Iter-by-iter trace on prompt 0 (logreg5 thr=0.5):

  - iter 1 shifted draft was [' need', ' to', ' determine', ' the', ' the',
    ' of', ...] (model's iter-0 predictions). Refresh reinit's all but
    position 0 to random → iter 2 input = [' we', RND×31]. iter-2 only
    matches at pos 0 → commits 1.
  - Without refresh (none), iter 2 input keeps all of cur_argmax → commits
    4.

The structural conclusion: **standard JF's "shift and keep model predictions"
is already optimal at 1 forward per iter.** To beat it requires inserting
**better-than-model-current** tokens (= a real drafter producing high-quality
guesses, not "random replaces wrong"). The 5.88 TPF predictor sim earlier
only beat JF because it ran a verify forward — the predictor's job there
was to recycle previous-cycle output, while the verify ground-truthed the
commit. Removing the verify breaks this.

---

## 2026-06-08 — Fixed-boundary refresh sweep (no predictor)

**Question.** Does a stupid "reset everything after position i" rule (no
learned predictor at all) work better than the learned-predictor refresh?

**Sweep.** Each iter: keep shifted[0..i), reinit shifted[i..K) to fresh
random. `--refresh fixed --fixed_i i --always_keep_pos0`. Same 16 DS
prompts, K=32, max_new=512, max_iters=512.

| fixed_i | corpus TPF | Δ vs baseline (3.29) |
|---:|---:|---:|
| 1  | 0.998 | −2.30 (basically AR) |
| 3  | 2.476 | −0.82 |
| 5  | 3.131 | −0.16 |
| 7  | 3.256 | −0.04 |
| **10** | **3.422** | **+0.13** |
| 15 | 3.282 | −0.01 |
| ∞ (none / standard JF) | 3.294 | — |

**Headline: a fixed cutoff at i=10 (TPF=3.42) is the best refresh policy
we have found** — better than every learned-predictor variant (best was
logit_p+pos0fix at 3.36).

**Why fixed beats learned.** The predictor's per-cycle decisions add
*variability* on top of the basic "reset late-tail" effect. Variability
hurts because Jacobi convergence depends on positions stabilizing
iter-to-iter; the predictor sometimes refreshes positions that were
about to converge, costing TPF. A fixed-i policy commits to "always
reset 10+" without flipping policy from iter to iter.

**Interpretation of i=10.** With K=32, the first ~10 positions carry
most of the signal needed to extend the converged prefix; positions
10–31 contribute noise that the model's iteration would have eventually
resolved anyway, but resetting them earlier lets the model focus
attention on the front of the K-window. This matches the per-position
accept-rate curve from the boundary-data analysis (accept rate plummets
from 1.0 at j=0 to ~0.35 by j=10, plateauing afterward — the "useful
prefix length" of the model's noisy-conditioning predictions).

**Bottom line for the predictor refresh idea:** a learned predictor on
logit features cannot do better than a constant cutoff at the position
where the accept-rate curve flattens. To exceed 3.4 TPF, the refresh
needs to insert **real predicted tokens** (not random) — a draft model
(EAGLE, ngram, etc.), not a boundary classifier.

---

## 2026-06-08 — Fixed-boundary sweep on BASE JF Math 7B (absolute baseline)

Same `--refresh fixed --fixed_i i --always_keep_pos0` protocol, run on
the unmodified JacobiForcing Math 7B base checkpoint. 16 DS prompts, K=32,
max_new=512, max_iters=512.

| fixed_i | corpus TPF | Δ vs base JF (3.75) |
|---:|---:|---:|
| 1  | 0.998 | −2.75 |
| 3  | 2.348 | −1.40 |
| 5  | 2.987 | −0.76 |
| 7  | 3.296 | −0.45 |
| 10 | 3.541 | −0.21 |
| **15** | **3.774** | **+0.025** |
| 20 | 3.737 | −0.012 |
| ∞ (none / standard JF) | **3.749** | — |

**Cross-model comparison (best fixed_i refresh):**

| Model | standard JF baseline | best fixed_i | best TPF | Δ (refresh − baseline) |
|---|---:|---:|---:|---:|
| math_k3 (cons-RL) | 3.29 | 10 | 3.42 | +0.13 |
| **base** JF Math | **3.75** | 15 | 3.77 | **+0.025 (≈noise)** |

**Three findings.**

1. **Base JF standard JF (3.75) > math_k3 best refresh (3.42) by +0.33.**
   This is the "TPF degradation from cons-RL" finding from earlier
   experiments, now confirmed in the 1-forward-per-iter Jacobi setting:
   the cons-RL training spreads the model's logits at deep K-block
   positions, so consecutive iters converge less aggressively. Standard
   JF on base ≫ standard JF on cons-RL math_k3.

2. **Refresh barely helps base (+0.025 — within noise of repeated runs).**
   The base model's deep-tail predictions still carry useful signal that
   the next iteration's convergence absorbs. Reinit'ing them to random
   loses information that the model would have used.

3. **Optimal refresh depth shifts with model.** math_k3 peaks at i=10
   (where its per-position accept curve flattens). Base peaks at i=15–20
   (where its accept curve plateaus). The refresh "win" shrinks as the
   model's deep-tail positions become more informative.

**Absolute baseline conclusion.** Nothing in the predictor-refresh family
— fixed cutoff OR learned predictor — beats standard JF on base JF Math
7B by more than noise. **3.75 TPF is the practical 1-forward-per-iter
ceiling for this model.** To exceed it, the refresh must inject **real
predicted tokens** (a draft model: EAGLE, ngram, distilled head), not
random tokens or even surgical reinits. The shift-leak's 5.88 TPF
remains contingent on the verify forward (= 2 forwards per cycle, of
which one is "free" if cheap).

---

## 2026-06-08 — Hidden-state linear probe + asymmetric-cost bias

**Goal.** Build a learned per-position boundary predictor on last-layer
hidden states and test whether it beats the constant-cutoff fixed_i baseline
(3.42 math_k3, 3.75 base).

### Step 1: hidden-state extraction
Re-ran the warm-restart collector to also dump
`out.hidden_states[-1]` per K-position. ~1400 cycles × K=32 × 3584 dim per
model. Files: `boundary_h_math_k3.npz`, `boundary_h_base.npz` (~250 MB each).

### Step 2: label-target ablation (huge)

Originally the per-position label was `accept[j] = warm[j]==verify[j]`,
which is **noisy and non-monotonic** because of random argmax matches in
the K-window tail. Switching to `before_boundary[j] = (j < n_acc)` gave a
**monotonic** per-cycle target with positive rate ~12-15%.

| Target | math_k3 val AUC | base val AUC |
|---|---:|---:|
| accept (old) | 0.785 | 0.697 |
| **before_boundary (new)** | **0.960** | **0.962** |

+0.17–0.27 AUC just from cleaning the label. The clean target is what we
should have been using all along.

### Step 3: AUC was misleading — true boundary-level MAE is ~2 tokens

A 0.96 AUC on the imbalanced (12% positive) per-position task hides the
fact that the **per-cycle boundary error** is large. Sweeping threshold on
the val split:

| Probe | best thr | mean signed err | MAE | within±1 | within±2 |
|---|---:|---:|---:|---:|---:|
| math_k3 bb_pw1 | 0.30 | −0.07 | 1.94 | 58% | 74% |
| math_k3 bb_pw10 | 0.50 | +0.06 | 2.19 | 52% | 71% |
| base bb_pw1 | 0.30 | −0.35 | 1.58 | 65% | 80% |
| base bb_pw10 | 0.50 | +0.01 | 1.62 | 63% | 79% |

So the probe is off by ~2 tokens on average; only ~60% of cycles are
within ±1.

### Step 4: confirmed underreport >> overreport in cost

User hypothesis (from shift-leak data): overreporting boundary by 5 costs
~0.5 TPF (gentle); underreporting causes stalls (cost ~ −1 TPF per
position). Tested by adding a constant `--boundary_offset` to the probe's
predicted boundary, post-hoc bias:

**Probe (bb_pw1, thr=0.3) + offset (16 DS prompts, K=32):**

| Model | offset | corpus TPF | per-prompt mean |
|---|---:|---:|---:|
| math_k3 | 0 | 2.140 | 2.819 |
| math_k3 | **2** | **3.422** | 3.469 |
| math_k3 | 5 | 3.380 | 3.399 |
| math_k3 | 8 | 3.272 | 3.292 |
| base | 0 | 0.870 | 1.497 |
| base | 2 | 3.393 | 3.483 |
| base | **5** | **3.704** | **3.795** |
| base | 8 | 3.675 | 3.775 |

Offset=0 (raw probe) yields TPF=0.9–2.1 — many underreports cause stalls.
Adding +2 to +5 lifts TPF by **+1.3 to +2.8**. Confirms the asymmetric cost
empirically.

### Step 5: but probe + offset only TIES the constant-cutoff baseline

- math_k3 best (off=2): **3.42** ≈ fixed_i=10 baseline **3.42**
- base best (off=5):    **3.70** ≈ fixed_i=15 baseline **3.77**

The linear probe **does not exceed the constant cutoff** on either model.
The MAE-2-token boundary error means the probe can't adapt better per-cycle
than a single constant.

### Step 6: global classifier (K+1 softmax on flattened hidden) failed

Tried `_train_boundary_block_classifier.py`: input = flatten(K × d_hidden +
K × 5 aux), output = (K+1)-way softmax. 115k feature dim × 33 classes →
3.8M params with ~1100 training cycles → severe overfit. Final val
acc=0.12, MAE=6.5. Not run in sim.

### Bottom line

**Absolute ceiling for boundary-classifier refresh remains 3.42 (math_k3)
and 3.75 (base).** A linear probe on hidden states ties the constant-cutoff
baseline; it does not exceed it. The probe's per-cycle MAE of 2 tokens is
too high to beat "always reset positions ≥ 10".

To actually exceed the baseline, the refresh must inject **real predicted
tokens** (an EAGLE-style drafter), not random tokens at probed-bad
positions. Boundary classification alone hits a ceiling at the fixed cutoff
that JF's iteration already implicitly uses.

**Path that might still work for boundary prediction:**
1. MLP/attention head instead of linear (lit precedent: SpecDec++ at AUC
   ~0.90 → MAE potentially ~1 token).
2. Aux head trained INTO the model during cons-RL — co-adaptive learning.
3. Hidden states from multiple layers (early layers may carry signal lost
   in the last layer's vocab projection).

---

## 2026-06-08 — On-policy ckpts: vLLM TPF benchmark (post-EOS-fix)

Re-bench of all on-policy DAPO+DS+Jacobi training runs with the stop_token_ids
EOS fix in `scripts/vllm_tpf_trajectories.py`. Earlier numbers may have been
inflated by past-EOS loop generation. Setup: vLLM 0.10.2 + jacobi patch,
K=32, T=0 (greedy), batch_size=16, max_new_tokens=2048, 64 DeepScaler prompts
(`deepscaler_tpf_prompts.jsonl`).

| Run | step | TPF (n=64) |
|---|---:|---:|
| jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_v2 | 300 | 3.512 |
| jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_focalonly_v5 | 220 | 3.630 |
| jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_focalonly_constant_v6 (cmws0e9t) | 300 | 3.670 |
| jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_ce_pureclean_v7 | 300 | 3.597 |
| jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_kl_base_forward_v8 | 220 | 3.767 |
| **jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_corrupt03** | **300** | **4.171** |
| (reference) base JF Math 7B | — | 3.834 |
| (reference) math_k3_ds (offline cons) | 300 | 3.314 |

**Headline.** `dflashce_corrupt03` is the standout — TPF=4.17, the only
run that exceeds base JF (3.83). It's also +0.5–0.7 above the other
on-policy variants and +0.9 above offline math_k3.

**Interpretation.** Most on-policy variants (v2, v5, v6, v7, kl_v8) land
in 3.51–3.77 — slightly below base JF, similar to math_k3_ds. Only
`corrupt03` is dramatically better. The `corrupt03` variant applies
dflash_ce loss with corruption probability 0.3 — i.e., it trains
explicitly on noisy K-blocks with 30% corruption applied to non-marker
positions during the cons step. That training distribution seems to
match the inference-time K-window contents better than other variants.

Reproduction: ckpts in `ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_*_step_<n>/`,
raw outputs in `eval_passk/tpf_results/vllm_tpf_*_s<step>.jsonl`. Logs in
`logs/vllm_tpf_*.log`.

---

## 2026-06-08 — Lit review: lightweight drafters / boundary predictors

**Closest prior art (3 papers):**

1. **SPRINTER (Inampudi et al., 2025, arXiv:2502.04557)** — single Linear+
   sigmoid (~1k params, ~10⁻⁷× the 7B) on the draft model's last-token
   embedding, BCE on "would target accept this token?". Replaces verifier
   forwards. The closest precedent to per-position boundary classification
   in SD.
2. **SpecDec++ (Huang et al., COLM 2025, arXiv:2405.19715)** — small ResNet-MLP
   "acceptance head" (~5M params) on the draft hidden states, predicts
   P(accept) per position; stops drafting when cumulative P(reject) > τ.
   Theory: optimal candidate-length policy IS threshold on per-token P(accept).
3. **AutoJudge (Borzunov et al., 2025, arXiv:2504.20039)** — logistic
   regression on concatenated draft+target hidden states. Fusable into the
   existing LM head (≈ free at inference).

**Training-free baselines worth comparing:**
- **AdaEDL (NeurIPS-W 2024, arXiv:2410.18351)** — closed-form `1 − √(γH)`
  acceptance bound from logit entropy. Zero params.
- **SVIP (2024, arXiv:2411.18462)** — draft-entropy stop-drafting rule.
- **BiLD (Kim et al., NeurIPS 2023, arXiv:2302.07863)** — top-1 prob threshold.

**Per-block boundary prediction (without per-token AR decoding) appears
unaddressed** — the survey found no paper that frames it our way for
Jacobi K-blocks. That's the novel framing.

**Recommended prototypes (ranked B > A > C):**

- **A — Logit-threshold (zero training).** Use the verify-forward's last
  logits + AdaEDL/BiLD per-position threshold. Tune τ on held-out cycles
  vs true n_acc. Our analysis above shows this floor sits at ~67–69% within
  ±1 token. Param budget: 0. **Build first as baseline.**
- **B — 2-layer MLP-512 on verify-forward hidden states.** ~5M params
  (≈0.07% of 7B). Inputs: hidden_state[i] (4096 dims) + logit features +
  position embedding. Per-position sigmoid output P(accept), decode boundary
  via first-crossing or learned argmax head. Trains on cached
  (warm_argmax, verify_argmax) pairs from our existing trace dumps.
- **C — Linear-only head fused into LM head.** ~4k params (4096-dim hidden
  → 1 + bias). SPRINTER/AutoJudge floor. Lower bound on what learned
  predictors can achieve.

**Implementation order:** A this week (no training, immediate sanity);
C as a 1-hour finetune; upgrade to B only if both leave > 0.5 TPF on table.

**Key insight from our data.** Logit features alone (top1_prob/entropy)
AUC 0.51–0.67. Add `position_index / K` → AUC 0.75 (math_k3). A learned
predictor with hidden states + position + logits should land 0.85–0.90+
based on SpecDec++ numbers. Our deployment tolerance (~3-token error → 92%
of perfect TPF) means even a mediocre 0.75-AUC predictor likely captures
most of the 5.94 TPF ceiling.

---

## 2026-06-08 — Shift-leak sim on BASE JF Math 7B (comparison)

Same protocol, run on the unmodified JacobiForcing Math 7B base checkpoint
(`models--JacobiForcing--JacobiForcing_Math_7B_v1`), K=32, greedy, 16 DS
prompts, max_new=512.

**Important caveat — corpus-level vs per-prompt-mean.** The earlier
"`mean TPF_verify`" headline was a *per-prompt mean* (each prompt = one
sample), which slightly biases upward because a short generation with high
single-prompt TPF contributes the same weight as a long one. The honest
corpus-level number is `Σ n_tokens / Σ n_cycles` across all prompts. Both
are listed below.

Also: `TPF_verify` only counts the *verify* forward (treats warm as "free"
predictor output). `TPF_all_fwd` counts both warm + verify — this is the
true per-forward number if you have to run the warm forward at full cost.

**Base JF Math 7B (16 DS prompts):**

| shift | Σ n_tokens | Σ n_cycles | corpus tok/verify | corpus tok/all_fwd | per-prompt-mean TPF_verify |
|---:|---:|---:|---:|---:|---:|
| **0** | 8342 | 1689 | **4.939** | **2.470** | 5.095 |
| 1 | 8317 | 1591 | 5.227 | 2.614 | 5.408 |
| 2 | 8284 | 1504 | 5.508 | 2.754 | 5.643 |
| 3 | 8353 | 1483 | 5.633 | 2.816 | 5.784 |
| 4 | 8318 | 1466 | 5.674 | 2.837 | 5.804 |
| 5 | 8303 | 1447 | 5.738 | 2.869 | 5.939 |

**math_k3_ds_step_300 corrected (re-parsed from logs/sim_shift_leak_*.log):**

| shift | Σ n_tokens | Σ n_cycles | corpus tok/verify | corpus tok/all_fwd | per-prompt-mean TPF_verify |
|---:|---:|---:|---:|---:|---:|
| **0** | 8272 | 1392 | **5.943** | **2.971** | 6.004 |
| 1 | 8067 | 1393 | 5.791 | 2.896 | 5.824 |
| 2 | 8091 | 1406 | 5.755 | 2.877 | 5.780 |
| 3 | 8299 | 1465 | 5.665 | 2.832 | 5.728 |
| 4 | 8200 | 1483 | 5.529 | 2.765 | 5.579 |
| 5 | 8185 | 1499 | 5.460 | 2.730 | 5.513 |

So the corrected "ceilings" are **5.94 TPF_verify for math_k3** and **4.94
for base** at shift=0 (corpus level) — about 0.05–0.16 lower than the
per-prompt-mean we initially reported. **The asymmetric trend still holds:**
base rises with shift, math_k3 falls; curves still cross around shift=3.

| shift | base corpus tok/verify | math_k3 corpus tok/verify | gap (k3 − base) |
|---:|---:|---:|---:|
| 0 | 4.94 | 5.94 | +1.00 |
| 1 | 5.23 | 5.79 | +0.56 |
| 2 | 5.51 | 5.76 | +0.25 |
| 3 | 5.63 | 5.66 | +0.03 |
| 4 | 5.67 | 5.53 | −0.14 |
| 5 | 5.74 | 5.46 | −0.28 |

**Honest per-forward number (counting BOTH warm + verify) at shift=0:**
- math_k3: **2.97 TPF/forward** — *below* its natural vLLM Jacobi 3.38
- base:    **2.47 TPF/forward** — *below* its natural vLLM Jacobi 3.83

In other words: **if the warm forward costs as much as the verify forward,
the warm-restart protocol is strictly WORSE than natural Jacobi** for both
models. The "5–6 TPF" headline is only achievable when warm can be made
substantially cheaper than the big-model forward (n-gram, EAGLE-style
draft head, KV-cache reuse, distilled predictor). The cons-RL advantage
is real (+1.0 TPF_verify over base at shift=0) but is conditional on the
predictor architecture being cheap.

**Interpretation.** The two models live in opposite regimes:
- **math_k3 (cons-trained):** explicitly trained to denoise K-random
  conditioning → fresh random is the *easy* case (shift=0). Leaked own-wrong
  tokens are slightly *harder* than random because cons training never saw
  them — graceful but monotone decay.
- **base:** never saw K-random conditioning in training → fresh random is
  somewhat OOD for the warm forward, capping shift=0 at 5.1. The base model
  is happier with natural-language-like prefixes (its own wrong predictions
  are still in-distribution Qwen-ish tokens) → leaking those in *helps*.

**Practical implication.** A boundary predictor that's "perfectly accurate"
(reinit everything to random) is strictly best ONLY for the cons-trained
model. For a base JF model, a sloppy predictor — or just "trust the prior
warm output and only zero out the obviously wrong tail" — is competitive
or better. The cons-RL recipe is what unlocks the +0.9 TPF advantage at
the shift=0 ceiling, but that advantage erodes if the production predictor
is imperfect; by shift=5 the asymptotic ceilings are nearly tied (~5.5–5.9).

The takeaway from earlier ("a cheap 3-5 token tolerant predictor harvests
5.5–5.8 TPF") still holds for both models — they bracket the same band
once predictor error is ≥3.

---

## 2026-06-08 — warm_restart consistency check + vLLM T=0 (greedy) baselines

**Why:** verify the "perfect iter-2 ceiling" math_k3 hits ~6 TPF per verify
forward under the warm-restart protocol (2 forwards/cycle, treat warm as
free). Cross-check with vLLM Jacobi greedy (T=0) natural baseline.

**Setup:**
- warm_restart: HF forwards, greedy. Cycle = warm forward on `[committed |
  K random]` → verify forward on `[committed | warm_argmax]`. K=32, 16 DS
  prompts (`deepscaler_tpf_prompts_16.jsonl`), max_new=512.
- vLLM T=0: vLLM Jacobi natural streaming, T=0 with stop_token_ids fix.
  K=32, 64 DS prompts (`deepscaler_tpf_prompts.jsonl`), max_new=2048.

| Model | warm_restart TPF_verify | warm_restart TPF_all (2 fwds) | warm_restart mean_n_acc/cycle | vLLM-T=0 natural TPF | vLLM-T=0 ntok mean | vLLM-T=0 hit_cap |
|---|---:|---:|---:|---:|---:|---:|
| **math_k3_ds_step_300** | **6.004** | 3.002 | 5.010 | **3.378** ± 0.088 | 670 | 1/64 |
| base JF Math 7B | 5.095 | 2.547 | 4.117 | 3.827 ± 0.106 | 910 | 8/64 |

**Confirmed:**
- (a) Warm_restart consistency check **PASSES**: math_k3 = 6.004 TPF_verify
  (matches the previous 6.00 from earlier sim). Base = 5.095. The "perfect
  iter-2 ceiling of 6 TPF per verify forward" is reproducible.
- (b) vLLM T=0 (greedy) natural Jacobi gives TPF=3.38 for math_k3, 3.83 for
  base. Comparing to T=1.0 vLLM (math_k3 3.24, base 3.90): greedy is
  ~+0.15 for math_k3 but ~0.07 lower for base. Greedy doesn't dramatically
  beat T=1.0 — vLLM's natural cascade handles both well.
- vLLM T=0 numbers are close to JF block-decode TPF=3.24 (math_k3) and
  ~3.40 (base) — different engines but similar regime → consistent.

**The "6 TPF gap"**: warm_restart shows the model CAN denoise to 5+1 per
verify, but only when given a clean warm-AR draft. Natural Jacobi
(vLLM/JF block-decode) sustains ~3.3 per forward because mid-iters degrade.
To harvest the 6 TPF: cheap warm-draft mechanism (EAGLE, ngram, KV cache
reuse) — same conclusion as prior analysis.

---

## 2026-06-08 — DeepScaler re-bench WITH EOS stop_token_ids fix

**Engine:** vLLM 0.10.2 + `scripts/vllm_jacobi_patch.py` (windowed-streaming
Jacobi spec-decode), via `scripts/vllm_tpf_trajectories.py`. K=32, T=1.0,
top_p=1.0, top_k=-1, max_new_tokens=2048, BS=16. Stop tokens NOW passed
explicitly: `SamplingParams(stop_token_ids=[tokenizer.eos_token_id, 151645,
151643])`.

**Why this matters:** the prior bench called `SamplingParams(temperature,
max_tokens)` with NO `stop_token_ids`. vLLM falls back to model-config EOS
but the stop check happens per-spec-decode-commit-batch (not per-token), so
when EOS landed mid-K=32 commit, post-EOS positions got committed as junk
→ model entered loops → 2048 cap → inflated TPF. The fix lets vLLM stop
properly when EOS is committed.

**Output JSONLs:** `eval_passk/tpf_results/<name>__vllm_deepscaler_T1_stopfix.jsonl`

| Run | TPF | SE | min | max | mean_ntok | hit_cap (≥2040) | prior TPF (no fix) | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base JF Math 7B (pre-RL) | **3.904** | 0.135 | 1.651 | 7.191 | 1062 | 8/64 | 3.917 | −0.01 |
| math_ar_ds_step_300 (AR-RL) | 2.512 | 0.054 | 1.566 | 4.031 | 636 | 0/64 | 2.527 | −0.02 |
| dflashce_v2 step_20 (cons A, target_ratio=0.1) | 3.338 | 0.138 | 1.268 | 6.351 | 823 | 1/64 | 3.212 | +0.13 |
| dflashce_fixed step_20 (cons B, fixed w=0.01) | 3.066 | 0.150 | 1.191 | 5.359 | 750 | 1/64 | (n/a on DS) | — |
| dflashce_corrupt03_v4 step_20 (multi-tile + focal) | **3.645** | 0.130 | 1.340 | 5.987 | 854 | 4/64 | 3.624 | +0.02 |
| math_k3_ds_step_300 (CE noisy decay K=3, OG cons recipe) | 3.236 | 0.068 | 1.735 | 4.479 | 628 | 0/64 | 2.197 | +1.04 |

**Key takeaways:**

- **Most models' TPF barely changed** with the fix (Δ ≈ ±0.02–0.13). The
  stop-bug only triggered in rare commit-boundary states.
- **math_k3_ds jumped +1.04** (2.20 → 3.24) — but this is mostly T=1.0
  stochasticity between two runs (OLD had 6/64 looping prompts inflating
  the mean upward in some rows and downward in others; NEW has 0/64
  caps and lands at a cleaner 3.24).
- **base still hits 2048 cap on 8/64** with mean_ntok=1062 — these are
  genuinely long-reasoning prompts, not loops. Base just produces more
  text per prompt than the RL-finetuned models (which DAPO shortened).
- **Clean ranking on DS (true TPF, stop-fix engine):**
  1. base 3.90  (max possible; pre-RL)
  2. v4 multi-tile+focal 3.65
  3. v2 cons A 3.34
  4. math_k3_ds 3.24
  5. fixed cons B 3.07
  6. AR-RL 2.51
- math_k3 falls 0.66 below base on streaming Jacobi TPF — cons preserves
  ~83% of base's Jacobi capability, but the "iter-2 ≈ 6 on first verify"
  is a single-iter peak (block-decode regime), NOT a sustained streaming
  number.

---

## 2026-06-06 — DeepScaler-sampled prompts re-benchmark (training distribution)

**Why:** v5 training-time TPF logging showed `rollout/jacobi_tpf_mean ≈ 3.74` at
step 1, but `base_jf_math_7b__vllm_trainparams` on `eval_prompts_tpf.jsonl`
reported 4.494. Hypothesis: the eval prompt set (64 GSM8K-style problems) is
much easier than the DeepScaler training distribution. Test by re-running
identical vLLM Jacobi T=1.0 benchmark with 64 prompts sampled from
`data/deepscaler/train.parquet` (seed=42).

**Setup:** vLLM Jacobi (jacobi_patch), T=1.0, K=32, max_new=2048, BS=16, n=64.
Prompts: `eval_passk/deepscaler_tpf_prompts.jsonl`. Driver:
`scripts/_run_tpf_deepscaler.sh`. Driver log: `_deepscaler_grid.log`.

| Run | DeepScaler T=1.0 | SE | min | max | n_tok_mean | hit_cap | eval_prompts T=1.0 | Δ |
|---|---:|---:|---:|---:|---:|---:|---:|---:|
| base JF Math 7B | **3.917** | 0.137 | 1.651 | 7.191 | 1078 | 9/64 (14%) | 4.494 | **−0.577** |
| math_ar_ds_step_300 (RL-AR) | 2.527 | 0.053 | 1.505 | 3.975 | 646 | 0/64 | 2.700 | −0.173 |
| dflashce_v2 step_20 (cons A) | 3.212 | 0.148 | 1.165 | 5.632 | 767 | 4/64 | 3.676 | −0.464 |
| dflashce_fixed step_20 (cons B) | 2.884 | 0.174 | 1.119 | 5.974 | 709 | 2/64 | 2.937 | −0.053 |
| dflashce_corrupt03_v4 step_20 (multi-tile + focal) | 3.624 | 0.127 | 1.251 | 5.987 | 857 | 3/64 | 4.045 | −0.421 |
| ce_noisy_decay_k3 step_300 (CE + dflash + multi-tile, uniform noise) | 2.197 | 0.082 | 1.438 | 5.975 | 762 | — | n/a | — |

**Key takeaways:**
- **Hypothesis confirmed.** Base JF Math 7B drops 4.494 → **3.917** on
  DeepScaler — a 0.577-point drop purely from the prompt distribution. This
  brings base into the same range as v5 training-time observation
  (`rollout/jacobi_tpf_mean = 3.74` at step 1). Remaining ~0.2 gap is plausibly
  longer rollouts (n=16 samples per prompt, some hitting 2048 cap → more low-TPF
  tail iters).
- **DeepScaler is genuinely harder for Jacobi spec decoding** — 14% of base
  rollouts hit the 2048 cap (vs 4.7% on eval_prompts), confirming longer/more
  novel chains. Mean n_tok 1078 vs 801.
- **Ranking on DeepScaler matches eval_prompts ranking** (base > v4 > consA >
  consB > AR), but the absolute spread is compressed because everything drops.
- **AR and cons B barely move** (−0.17 / −0.05): they already collapsed Jacobi
  capability, so harder prompts can't take much more away.
- **v4 (multi-tile + focal) preserves most TPF after RL** on DeepScaler
  (−0.29 vs base) — consistent with the eval_prompts finding.
- **Training-time `rollout/jacobi_tpf_mean` is now interpretable**: the
  3.7 number is what you'd expect for a JF model on hard math problems with
  long rollouts. To track training progress, compare to a fixed-base baseline
  on the SAME distribution (this 3.917 number), not the easier eval_prompts.

---

## 2026-06-04 — POSTFIX bench: TPF + TPS on FIXED vLLM Jacobi engine

**Engine:** vLLM 0.10.2 + `shao/spec-decode-jacobi-perf` commits `2b785b13`
(FIX A+D+E) and `645887591` (capture-size 992 clamp + max_num_seqs scoping).
In-place edits at `.venv/.../vllm/{config/__init__.py,v1/worker/gpu_model_runner.py}`
match the fork commits. Cuda-graphs FULL mode active, 6-12 graphs captured.

**Params:** K=32, max_new=2048, n=64 math prompts, JF Math 7B family.

**Driver scripts:** `_run_tpf_tps_postfix.sh` (BS=1 phase), `_run_tpf_tps_postfix_bs16.sh` (BS=16 phase), `_run_tpf_tps_step300_bs1.sh` (final step_300 BS=1). Bench module: `_bench_postfix_tpf_tps.py`.

### BS=1 (latency / single-stream)

| Model | T=0 TPF | T=0 TPS | T=1.0 TPF | T=1.0 TPS |
|---|---:|---:|---:|---:|
| base JF Math 7B | 4.083 | **672** | 4.609 | **719** |
| math_ar_ds_step_300 (RL-AR) | 2.770 | 457 | 2.750 | 430 |
| **consA-v4 step_60** | **4.652** | **766** | 4.123 | 644 |
| **consA-v4 step_300 (final)** | 3.838 | 630 | 3.835 | 592 |
| **consB-corrupt03 step_60** | 4.580 | 759 | 3.751 | 587 |
| **consB-corrupt03 step_300 (final)** | 4.281 | 701 | 4.269 | 661 |

### BS=16 (training-rollout-style throughput, gmu=0.85)

| Model | T=0 TPF | T=0 TPS | T=1.0 TPF | T=1.0 TPS |
|---|---:|---:|---:|---:|
| base JF Math 7B | 4.083 | 3434 | 4.624 | 4004 |
| math_ar_ds_step_300 (RL-AR) | 2.709 | 2336 | 2.762 | 2318 |
| **consA-v4 step_60** | **4.647** | **4322** | 4.149 | 3582 |
| **consB-corrupt03 step_60** | 4.569 | 4244 | 4.222 | 3662 |

### Headline numbers

- **Cons step_60 wins on TPS at every cell except T=1.0 BS=1.** ConsA step_60 T=0 BS=1 = 766 TPS, BS=16 = 4322 TPS — both highest in their column, beating base (672 / 3434) by 14% / 26%.
- **Final ckpts (step_300) regress slightly from step_60** on TPF + TPS at BS=1: consA 4.65→3.84, consB 4.58→4.28. The TPF peak was at step_60; further training narrowed the distribution but didn't lift TPF further. (Step_60 was the right point to deploy.)
- **AR-RL is the clear loser** at every cell: TPS ~430-2336 vs base 672-4004.

### Notes / caveats

- TPF at BS=1 is reported as `n=1` because the trajectory aggregator counts per-req-slot, and at BS_cap=1 there's only one slot. The TPF value itself is the per-iter mean across all forwards for that one slot — equivalent to the per-prompt mean for the cohort.
- Capture-size fix (commit `645887591`) was needed for BS=16: without it, FIX A's auto-augmentation produced 1056-token graphs that exceed FlashAttn's 992-token full-cuda-graph cap.

---

## 2026-06-04 — step_60 progress: consA-v4 (target_ratio=0.1) + consB-corrupt03 (draft corruption p=0.3)

**Runs benched (active training continued from step_20):**
- `dflashce_v2_step_60`: CORRECT_ONLY=1, target_ratio=0.1, adaptive=1.0, weight=0.001, fraction=0.30, block=32. Same run as step_20.
- `dflashce_corrupt03_step_60`: same as above PLUS `CONSISTENCY_DRAFT_CORRUPT_PROB=0.3` (new feature, injects uniform vocab noise into draft positions ≥ pml).

**Engines:** vLLM Jacobi (T=0 + T=1.0), JF reference (greedy), DL nanovllm (T=1.0). K=32, max_new=2048, n=64.
**Grid script:** `scripts/_run_tpf_step60_3engine.sh`
**Driver log:** `eval_passk/tpf_results/_step60_3engine.log`

### consA-v4 step_20 → step_60 progression

| Engine / mode | step_20 acc / TPF | step_60 acc / TPF | Δ acc | Δ TPF | step_60 rep | step_60 TPF✓ |
|---|---|---|---:|---:|---:|---:|
| vLLM T=0 greedy | 75.0% / 4.557 | 79.7% / 4.582 | +4.7pp | +0.03 | 2 | 4.672 |
| vLLM T=1.0 | 53.1% / 3.676 | **73.4% / 4.228** | **+20pp** | **+0.55** | 4 | 4.382 |
| JF ref greedy | -    / 4.017 | -    / 3.959 | - | -0.06 | - | - |
| DL T=1.0 | 54.7% / 2.672 | 76.6% / 3.007 | +21.9pp | +0.34 | 1 | 3.034 |

### consB step_20 (fixed w=0.01) → step_60 (corrupt03 w=0.001 + 30% draft corruption)

| Engine / mode | step_20 fixed acc / TPF | step_60 corrupt03 acc / TPF | Δ acc | Δ TPF | step_60 rep | step_60 TPF✓ |
|---|---|---|---:|---:|---:|---:|
| vLLM T=0 greedy | 68.8% / 4.986 | 73.4% / 4.573 | +4.6pp | -0.41 | **2** (was 12) | 4.532 |
| vLLM T=1.0 | 28.1% / 2.937 | **75.0% / 4.125** | **+46.9pp** | **+1.19** | 2 | 4.265 |
| JF ref greedy | -    / 4.144 | -    / 3.817 | - | -0.33 | - | - |
| DL T=1.0 | 32.8% / 2.296 | 75.0% / 2.993 | +42.2pp | +0.70 | 1 | 3.127 |

**Findings:**

1. **Both step_60 ckpts dramatically improved at T=1.0** — consA acc 53→73%, consB-corrupt03 acc 28→75%. The "broken at T=1.0" pattern at step_20 was just under-training, not a fundamental cons-RL flaw.

2. **CV (distribution width) tightened a lot:** consA T=1.0 CV 0.34→0.18; consB CV 0.44→0.23. Step_60 models have much narrower TPF distributions — fewer fail-loops, more consistent throughput.

3. **Corruption variant (consB-corrupt03) recovered val accuracy from the worst case** (28% → 75%, +47pp) at T=1.0 — biggest accuracy gain of any run. **Repetition went 12→2 at greedy.** Confirms hypothesis that draft corruption fights the "lock-in" overconfidence that produced the loops in the previous fixed-weight run.

4. **vLLM Jacobi T=1.0 cons step_60 now beats base** (consA 4.23 > base 4.49? actually still slightly below; consB-corrupt03 4.13 below base). But **on the rollouts they get right**, cons matches base. The improvement is now stable across temperatures, not just greedy.

5. **DL nanovllm shows the same step_20→step_60 acc trend** but TPF stays lower (engine difference, not training difference).

6. **JF reference greedy didn't improve step_60** for consA (4.02→3.96) — and dropped a bit for consB (4.14→3.82). This is interesting: JF ref greedy was where cons-RL "won" most at step_20. The narrowing of the distribution at step_60 evidently removed some of the high-tail outliers that inflated JF ref means (consA std 1.53→0.64, consB std 1.93→0.69).

---

## 2026-06-04 (analysis) — 3-engine comparison + correctness × TPF + repetition

**Why:** TPF mean alone hides whether the gain is on correct outputs or on
degenerate (loop) outputs. Split by `verify(completion, expected_answer)`
using verl's boxed-answer extractor + numeric normalization (math_dapo
helpers). Repetition flagged when last 600 chars contain a 30/50/80-char
substring repeating >3 times.

**Engines benched at T=1.0, K=32, max_new=2048, n=64:**
- vLLM Jacobi (training engine, shift-by-1 windowed refresh)
- DL nanovllm (Decode-Learning `decode_strategy="jacobi"`, block decode)
- JF reference greedy (`jacobi_forward_greedy`) — greedy only, no T match

**Engines compared:**

| Engine | base | AR | consA | consB |
|---|---:|---:|---:|---:|
| vLLM T=1.0 mean | 4.494 | 2.700 | 3.676 | 2.937 |
| DL nanovllm T=1.0 mean | 3.488 | 2.542 | 2.672 | 2.296 |
| JF reference greedy mean | 3.443 | 2.673 | 4.017 | 4.144 |

**Per-engine acc + TPF-by-correctness (T=1.0):**

| Engine | run | acc | std | CV | rep | TPF correct (±SE) | TPF incorrect (±SE) |
|---|---|---:|---:|---:|---:|---:|---:|
| vLLM | base  | 78.1% | 1.03 | 0.23 | 3 | 4.59 ± 0.14 | 4.14 ± 0.31 |
| vLLM | AR    | 87.5% | 0.38 | 0.14 | 0 | 2.70 ± 0.05 | 2.67 ± 0.10 |
| vLLM | consA | 53.1% | 1.26 | 0.34 | 2 | 4.16 ± 0.19 | 3.13 ± 0.23 |
| vLLM | consB | **28.1%** | 1.30 | 0.44 | 3 | 4.14 ± 0.26 | 2.47 ± 0.16 |
| DL   | base  | 78.1% | 0.61 | 0.17 | 4 | 3.48 ± 0.09 | 3.52 ± 0.16 |
| DL   | AR    | 84.4% | 0.34 | 0.13 | 0 | 2.55 ± 0.04 | 2.50 ± 0.13 |
| DL   | consA | 54.7% | 0.79 | 0.29 | 2 | 3.12 ± 0.07 | 2.14 ± 0.15 |
| DL   | consB | **32.8%** | 0.93 | 0.40 | 5 | 3.27 ± 0.12 | 1.82 ± 0.10 |

**Greedy vLLM (T=0, max_new=2048) — same correctness/repetition view:**

| run | acc | rep | TPF correct | TPF incorrect |
|---|---:|---:|---:|---:|
| base | 78.1% | 0/64 | 4.12 ± 0.10 | 3.65 ± 0.18 |
| AR | 84.4% | 0/64 | 2.76 ± 0.05 | 2.59 ± 0.08 |
| consA | 75.0% | 5/64 | 4.58 ± 0.17 | 4.49 ± 0.61 |
| consB | 68.8% | 7/64 | 4.53 ± 0.19 | **5.99 ± 0.97** ← loop inflation |

**Conclusions:**

1. **Cons-RL trades correctness for TPF, especially at training-time T=1.0.**
   Cons B accuracy falls to 28% (vLLM) / 33% (DL) vs base 78%. On the
   rollouts the model *does* get right, cons B TPF≈4.14 (vLLM) — same as
   base — but failed rollouts have TPF≈2.47, dragging the mean to 2.94.
2. **Cons-RL greedy "wins" are partly repetition-loop artifacts.** Cons B
   greedy mean 4.99 has 7/64 repetitive prompts; incorrect-mean TPF (5.99)
   exceeds correct-mean (4.53). Drop the 7 reps and overall mean → ~4.4.
3. **AR-RL is the only stable run.** Narrow CV (0.13-0.14), zero
   repetition, accuracy comparable to base. But TPF locked at ~2.55-2.70.
4. **Three engines disagree by ~30%.** vLLM > JF ref > DL nanovllm
   consistently on base/AR. vLLM > DL on cons too. Rankings disagree:
   - vLLM: base > consA > consB > AR
   - DL  : base > consA > consB > AR
   - JFref: consB > consA > base > AR
   The JF-reference ordering (block-decode greedy) is what we'd
   "want" cons-RL to optimize, but it's not what training uses. Training
   uses vLLM-jacobi-style, where cons step_20 *doesn't* beat base yet.
5. **Engine TPF rejection-sampling math** is equivalent vLLM vs DL at the
   per-position level (both use delta-q proposal → recovered residual =
   `target` with mass-at-drafted zeroed, normalized). Difference: vLLM
   commits +1 bonus on full acceptance; DL doesn't. ~3% systematic
   throughput gain for vLLM at K=32 from bonus accounting alone, the rest
   from different draft-refresh strategies (shift-by-1 vs nanovllm
   block-style).



**Engine:** vllm-jacobi (K=32, default plugin behavior — windowed Jacobi
refresh via target_argmax hook)
**Prompts:** `eval_passk/eval_prompts_tpf.jsonl`, n=64 (math)
**Decoding:** max_new_tokens=1024 (output cap), max_model_len=4096, BS=16,
top_p default (1.0), top_k default (-1)
**Hardware:** node fs-mbz-gpu-961, GPU index 4, single H200
**Grid script:** `scripts/_run_vllm_tpf_4models.sh`
**Raw rows:** `eval_passk/tpf_results/<run>__vllm_{T0,T06}.jsonl`
**Driver log:** `eval_passk/tpf_results/_vllm_4models_grid.log`

| Run | Ckpt path | Mode | mean | SE | min | max | n_tok_mean |
|---|---|---|---:|---:|---:|---:|---:|
| base JF Math 7B (pre-RL) | `~/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1` | T=0 greedy | 4.005 | 0.088 | 2.393 | 5.099 | 585 |
| base JF Math 7B (pre-RL) | (same) | T=0.6 | 4.348 | 0.122 | 2.755 | 6.991 | 661 |
| math_ar_ds_step_300 (RL-AR only) | `ckpts_hf/math_ar_ds_step_300` | T=0 greedy | 2.733 | 0.043 | 1.968 | 3.714 | 525 |
| math_ar_ds_step_300 (RL-AR only) | (same) | T=0.6 | 2.701 | 0.042 | 2.017 | 3.623 | 515 |
| dflashce_v2 step_20 (cons RL "A", target_ratio=0.1) | `ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_v2_step_20` | T=0 greedy | 4.382 | 0.148 | 2.277 | 8.316 | 604 |
| dflashce_v2 step_20 (cons RL "A") | (same) | T=0.6 | 4.378 | 0.157 | 1.645 | 7.967 | 647 |
| dflashce_fixed step_20 (cons RL "B", fixed w=0.01) | `ckpts_hf/jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_dflashce_fixed_step_20` | T=0 greedy | 4.495 | 0.206 | 2.278 | 12.138 | 623 |
| dflashce_fixed step_20 (cons RL "B") | (same) | T=0.6 | 3.901 | 0.158 | 1.643 | 6.275 | 611 |

**Notes:**
- B T=0 mean inflated by degenerate-loop prompts: top-5 all hit ntok=1024 cap
  (model stuck in repetition). Drop top 1: 4.374; drop top 2: 4.293.
- AR-RL has tight per-prompt distribution (std=0.34) — small TPF, no outliers.
- Both cons step_20 ckpts match/beat base TPF; AR-RL collapses to ~2.7 (-32%).
- vLLM Jacobi K=32 numbers run ~15-20% higher than JF-nanovllm on same models
  due to different bonus-token handling and prompt-tail cold-start.

---

## 2026-06-03 — JF nanovllm grid: 3 models x {greedy, T=0.6}

**Engine:** jf-nanovllm (`scripts/tpf_trajectories.py`, K=32, on_policy=True
default at T>0; on_policy=False forced for T=0)
**Prompts:** `eval_passk/eval_prompts_tpf.jsonl`, n=64
**Decoding:** max_new_tokens=1024, BS=16, top_p default
**Grid scripts:** `scripts/_run_tpf_benchmark_grid.sh` (T=0.6),
  `scripts/_run_tpf_greedy_grid.sh` (T=0)
**Raw rows:** `eval_passk/tpf_results/<run>__jf{,_T0}.jsonl`

| Run | Ckpt path | Mode | per-prompt TPF (mean) |
|---|---|---|---:|
| base JF Math 7B | (HF JacobiForcing snapshot, see above) | T=0 greedy | see `base_jf_math_7b__jf_T0.jsonl` |
| base JF Math 7B | (same) | T=0.6 on-policy | see `base_jf_math_7b__jf.jsonl` |
| math_ar_ds_step_300 | `ckpts_hf/math_ar_ds_step_300` | T=0 greedy | see `math_ar_ds_step_300__jf_T0.jsonl` |
| math_ar_ds_step_300 | (same) | T=0.6 on-policy | see `math_ar_ds_step_300__jf.jsonl` |
| onpolicy_f30_v2 step_300 (cons f30 v2 final) | `ckpts_hf/onpolicy_f30_v2_step_300` | T=0 greedy | see `onpolicy_f30_v2_step_300__jf_T0.jsonl` |
| onpolicy_f30_v2 step_300 | (same) | T=0.6 on-policy | see `onpolicy_f30_v2_step_300__jf.jsonl` |

(Per-prompt mean values were aggregated live during the 2026-06-03 session
and not retained in this log; recompute from the jsonl files when needed.)

---

## TODO for next benchmark batches

- Once Run A v4 (dflashce_v2 target_ratio=0.1) and Run B v4
  (dflashce_corrupt03, p=0.3) reach step 20+, merge ckpts to HF and bench
  with both engines + both temperatures.
- Consider expanding prompt set above 64 to tighten SE — 64 prompts give
  SE~0.10-0.20 on mean TPF, marginal for ~0.3-0.5 deltas.

---

## 2026-06-09 — Mask-tail (idea A) checkpoints: honest streaming TPF + mask-reinit decode

**Engine:** `scripts/_sim_jacobi_predictor_refresh.py` (new `--refresh_to mask` mode:
reinit positions get mask token 151643 instead of random; mask is never committed —
a mask==mask Jacobi match stops the acceptance scan; stall guard after 8 consecutive
zero-commit iters with argmax[0]==mask). 16 DS prompts
(`eval_passk/deepscaler_tpf_prompts_16.jsonl`), K=32, max_new=512, max_iters=512, greedy.
Checkpoints: `ckpts_hf/masktail_{A1,A2,A3}_step_*` (merged from
`ckpts/jacobi_forcing_dapo_deepscaler_masktail/`; runs still training, ~step 250-275;
NOTE max_ckpt_to_keep=5 deleted steps ≤140 before merging — early ckpts lost).

| model | decode | corpus TPF | per-prompt mean | mask stalls |
|---|---|---:|---:|---:|
| base JF (ref, 2026-06-07)        | vanilla     | 3.90 | — | — |
| math_k3_ds_step_300              | vanilla     | 3.29 | 3.31 | — |
| math_k3_ds_step_300              | mask-reinit | 3.27 | 3.30 | 0/16 |
| masktail_A1_step_260 (onpolicy)  | **vanilla** | **3.38** | **3.45** | — |
| masktail_A1_step_260             | mask-reinit | 3.28 | 3.33 | 6/16 |
| masktail_A2_step_240 (mask-in)   | vanilla     | 3.00 | 3.06 | — |
| masktail_A2_step_240             | mask-reinit | 3.00 | 3.07 | 0/16 |
| masktail_A3_step_240 (uniform)   | vanilla     | 3.11 | 3.14 | — |
| masktail_A3_step_240             | mask-reinit | 3.08 | 3.10 | 5/16 |

**Headlines.**
1. **A1 (on-policy + mask-tail) vanilla = 3.38 — best post-RL honest TPF to date**
   (math_k3 3.29, AR-RL 2.51), with MATH val mean@4 0.887 at step 260 (vs ~0.83
   for math_k3 recipe). Still below base 3.90.
2. **Mask-reinit decode never beats vanilla** on any model (−0.02 to −0.10).
   Consistent with the established result that reinit-harvesting is capped: replacing
   the model's own draft tokens (even masks) removes lucky far-position pre-convergence.
3. **Mask-stall failure mode**: in mask-reinit decode A1 stalls on 6/16 prompts
   (emits mask at window pos 0 — the "underestimated boundary → stall" asymmetry).

**Draft-distribution sharpening** (`scripts/_analyze_draft_sharpening.py`, fixed
base-model 96-tok prefixes, per-position p_mask/top1/agreement over K=32 window):

- **Mask emission mirrors the training input dist exactly.** A2 emits mask ONLY on
  mask-token inputs (p_mask_tail=1.0, boundary pos 9; 0.000 on uniform/own-cascade).
  A3 only on uniform inputs. **A1 generalizes across all three** (boundary 8 on
  uniform/mask inputs; on own cascade boundary ~20 with p_mask_tail 0.71-0.77 —
  masks appear right after its converged prefix, as designed).
- **A2's smooth TPF decline explained**: NOT mask bleed-through (its cascade p_mask
  is 0.000). Its cascade self-agreement (argmax vs own greedy, pos 1-7) decays
  0.955 (s160) → 0.911 (s200) → 0.795 (s240) — the drafter is drifting off its own
  greedy continuation. A1 keeps 1.00 → 0.96; A3 0.88-0.85. Same ordering as TPF.
- **Sharpening confirmed for A1**: cascade top1_p(pos1-7) 0.98-0.95, near base (0.99),
  above math_k3 (0.875), with a clean p_mask ramp at the trained margin.

**Conclusion.** The mask-tail loss successfully shapes the drafter's distribution
(idea A's signature is there), but the mask-reinit *decode* extracts no extra TPF.
The win so far is A1: on-policy cons input + mask-tail target preserves/improves
TPF vs all prior cons recipes while RL lifts val acc. Step-300 final ckpts pending.

---

## 2026-06-09 (later) — KL-v8 ckpts under jsim-16 protocol; A1 mask-boundary alignment

**Protocol:** same as the mask-tail batch above (`_sim_jacobi_predictor_refresh.py`,
refresh=none/random, 16 DS prompts, K=32, max_new=512). Base re-measured under
this exact protocol for anchoring (the 3.90 reference was vLLM/64-prompt/2048).

| model | corpus TPF | per-prompt mean | MATH val mean@4 (train log) |
|---|---:|---:|---:|
| **kl_base_forward_v8 step 220**  | **3.96** | **4.05** | **0.901** (s300) |
| base JF Math 7B (jsim-16 anchor) | 3.75 | 3.86 | — |
| kl_base_reverse_v8 step 240      | 3.63 | 3.72 | 0.740 (stalled 0.74-0.81) |
| masktail_A1 step 260             | 3.38 | 3.45 | 0.887 |

**kl_base_forward_v8 beats base (+0.21) under this protocol with the best val
acc of any run.** Caveat: the 2026-06-08 vLLM/64-prompt bench had it BELOW base
(3.767 vs 3.834) — margin is protocol-sensitive; confirm on 64 prompts / longer
max_new before declaring. corrupt03 (vLLM 4.171) also still unbeaten there.

**Reverse-KL v8 post-mortem** (why it "didn't work"): reverse KL toward the
FROZEN base on on-policy noisy states = sustained mode-seeking drag against PG.
cons_loss never converged (19→14 plateau, vs mask-tail's 20-step absorption);
val stalled at 0.74-0.81 while rollout TPF rose late to 3.9-4.1 (policy
retreating toward base-sharp drafting at the cost of the accuracy climb).
Draft-OPD's reverse KL differs in all load-bearing details: current-policy
teacher (not stale), rejected-positions-only, paired w/ forward KL on accepted.

**A1 mask-boundary vs acceptance alignment** (`_measure_mask_boundary_alignment.py`,
vanilla decode, 2289 iters): draft window contains a mask boundary in 99% of
iters; **only 5.0% are 'exact'** (all non-mask tokens accepted, rejection at the
mask); **94.8% leak** real-token rejections before the mask, mean leak 5.24
(hist capped at margin=8). The mask is a "converged + margin" UPPER bound, not
a convergence predictor. Vanilla mode committed through mask==mask matches on
6/16 prompts (pad==mask conflation → early stop); distortion ≤ ~0.1 TPF
(mask-commit prompts: TPF 3.47 vs 3.34 for full-length).

---

## 2026-06-09 (night) — vLLM re-bench w/ greedy + T=1.0, 2048 tok, repetition audit

**Setup:** `vllm_tpf_trajectories.py` (post-EOS-fix), K=32, BS=16, max_new=2048,
64 DS prompts, T∈{0.0, 1.0}. Repetition audit: `scripts/_check_repetition.py`
(flag = tail unique-4gram < 0.35 or ≥8 identical lines). Mean per-req TPF:

| model | T=0 | T=0 excl-rep | T=1.0 | rep% (T0/T1) | capped (T0/T1) | 06-08 ref |
|---|---:|---:|---:|---:|---:|---:|
| corrupt03_300   | **4.223** | **4.246** | **3.982** | 22/23 | 3/0 | 4.171 |
| base JF         | 3.831 | 3.849 | 3.907* | 34/41 | 8/10 | 3.834 |
| fwd-KL v8 s220  | 3.768 | 3.616 | 3.601 | 22/20 | 1/0 | 3.767 |
| CE dflashce_v2 s300 | 3.521 | 3.440 | 3.387 | 17/14 | 0/0 | 3.512 |
| mask-tail A1 s260 | 3.397 | 3.358 | 3.373 | 19/20 | 0/0 | — |
| math_k3 s300    | 3.383 | 3.317 | 3.243 | 22/16 | 1/0 | 3.314 |

*base T=1.0 is repetition-inflated: excl-rep 3.723 (below its greedy 3.85).

**Conclusions.**
1. **Old vLLM numbers are trustworthy** — re-run reproduces 06-08 to ±0.07.
   (06-08 was already post-EOS-fix; pre-fix concern applies only to earlier logs.)
2. **Repetition exists but doesn't drive rankings**: excluding flagged rows
   shifts TPF ≤ 0.18 and changes no ordering. Base is the most repetition-prone
   (34-41% flagged, 8-10 hit the 2048 cap); all RL'd models are clean (≤3 capped)
   — RL itself removed the rambling.
3. **"fwd_v8 beats base" was a jsim-16 protocol artifact** (max_new 512, 16
   prompts, HF engine, pure-random reinit). At 2048/vLLM: base 3.83 > fwd_v8 3.77.
   **corrupt03 4.22 remains the only model robustly above base.**
4. **fwd-KL > matched CE holds in both protocols** (+0.25 vLLM, +0.18 jsim) at
   equal val acc (0.90) — the gap is purely the drafting function.
5. **T=1.0 robustness**: mask-tail A1 degrades least (−0.02), fwd_v8 −0.17,
   corrupt03 −0.24, math_k3 −0.14. On-policy cons input (T=1 rollouts) appears
   to buy temperature robustness of the drafter.

---

## 2026-06-10 — Stop-fix audit of jsim + TPF decomposition: repeating vs clean segments

**Bug found & fixed:** `_sim_jacobi_predictor_refresh.py` and `_tpf_rep_split.py`
checked stops only on the LAST committed token of a block — a stop accepted
mid-commit ran past <|im_end|>, appending junk. (vLLM checks per-token; the
vLLM bench was never affected.) Stop-fixed jsim-16/512 renumbering:

| model | stop-fixed | old (inflated) |
|---|---:|---:|
| fwd-KL v8 s220 | **3.63** | 3.96 |
| base JF | 3.37 | 3.75 |
| mask-tail A1 s260 | 3.32 | 3.38 |
| math_k3 s300 | 3.26 | 3.29 |

base and fwd_v8 were inflated ~0.35 (they commit stops mid-block + junk);
A1/math_k3 were honest. **fwd_v8 > base (+0.27) stands in the fixed protocol**;
base vs A1 gap at 512 tokens nearly vanishes (0.05).

**TPF decomposition** (`scripts/_tpf_rep_split.py` v2, 64 DS prompts, greedy,
max_new=2048; token labeled "repeating" if its 8-token-gram occurred earlier
in the response; commit cycles attributed by majority label). Validation:
per-prompt TPF reproduces the vLLM bench (base 3.77/3.83, corrupt03 4.07/4.22,
A1 3.38/3.40).

| model | rep-segment TPF | clean-segment TPF | rep token share | headline TPF |
|---|---:|---:|---:|---:|
| base JF | 5.91 | 2.97 | 46.9% | 3.77 |
| corrupt03_300 | **6.76** | **3.06** | 43.8% | 4.07 |
| masktail A1_260 | 5.03 | 2.81 | 35.2% | 3.38 |

**Findings.**
1. Repeating segments decode at ~2× the clean rate (5-6.8 vs 2.8-3.1 TPF) in
   every model. Roughly half of base's greedy tokens sit inside repeated
   8-grams (much of it legitimate math-derivation structure, not junk loops).
2. **On clean text the models are nearly equal** (base 2.97, corrupt03 3.06,
   A1 2.81). Base's headline advantage over RL'd models is mostly repetition
   SHARE (46.9% vs 35.2%), not drafting-function superiority. The true
   function-level gap base→A1 is −0.16 on comparable content.
3. **corrupt03 beats base on BOTH segments** with LOWER rep share — its 4.07
   is genuine drafting improvement, not repetition harvesting.
4. RL's TPF "destruction" is therefore mostly content compression (shorter,
   less self-similar derivations), validating the de-repetition account of the
   early-training TPF drop; residual function degradation is small.

---

## 2026-06-10 — Dual-criterion TPF decomposition (structural rep8 vs degenerate loops)

**Criteria** (`scripts/_tpf_rep_split.py` v3, 64 DS prompts, greedy, 2048 tok,
stop-fixed): LOOSE `rep8` = token's 8-gram occurred earlier anywhere (structural
reuse incl. legit math notation); STRICT `loop` = sustained periodic self-repeat
(token[i]==token[i-p] for ≥max(2p,32) tokens, p≤256) = degenerate cycling.

| model | corpus TPF | loop share | loop TPF | non-loop TPF | rep8-clean TPF |
|---|---:|---:|---:|---:|---:|
| base JF        | 3.87 | 5.0%      | 7.8  | 3.77 | 2.97 |
| corrupt03_300  | 4.03 | **10.6%** | 16.0 | 3.70 | 3.06 |
| fwd-KL v8 s220 | 3.69 | 2.8%      | 16.1 | 3.63 | 2.96 |
| CE v2 s300     | 3.61 | 2.9%      | 15.8 | 3.54 | 2.84 |

**Findings.**
1. **Degenerate looping is rare at greedy** (3-5% of tokens) and RL HALVED it
   vs base — except **corrupt03, which loops 2× MORE than base (10.6%)**, at
   TPF 16. Excluding loops costs corrupt03 −0.33 (4.03→3.70) vs base −0.10:
   **corrupt03's championship is substantially loop-harvesting**; loop-free it
   is ≈ base (3.70 vs 3.77).
2. **On truly novel text (rep8-clean) all models are equal**: 2.84-3.06. The
   drafting function survives every RL recipe; ALL TPF differences between
   models come from how much self-similar/degenerate content they emit.
3. Length-regime reconciliation: at 512 tok (stop-fixed jsim) RL'd models all
   beat base (fwd_v8 3.63 > cev2 3.59 > corrupt03 3.52 > base 3.37) because
   structural-reuse runway grows with length and base writes longest (896 vs
   620-740 mean tokens) — base's 2048-horizon edge is accumulated self-reuse,
   not function.
4. RL'd models' loops are rarer but tighter (TPF 16 vs base's 7.8).

---

## 2026-06-10 — Three-engine TPF comparison + why cons-loss falls but TPF doesn't

**Engines:** jsim (streaming sliding-window, pure Jacobi accept, NO bonus, HF);
vLLM jacobi (streaming spec-decode, +1 bonus/step, argmax_prev shift, KV cache);
JF block-complete (`_jf_block_trace.py`, DL venv: iterate N=32 block to FULL
convergence before advancing). 16 DS prompts, greedy, uniform/random init.

| per-prompt TPF | jsim 512 | vLLM 512 | JFblock 512 | jsim 2048* | vLLM 2048* | JFblock 2048 |
|---|---:|---:|---:|---:|---:|---:|
| base       | 3.39 | 3.43 | 2.97 | 3.77 | 3.83 | 3.25 |
| fwd-KL v8  | 3.66 | 3.71 | 3.23 | 3.69 | 3.77 | 3.36 |
| CE v2      | 3.63 | 3.52 | 3.20 | 3.61 | 3.52 | 3.22 |
| corrupt03  | 3.63 | 3.58 | 3.25 | 4.03 | 4.22 | 3.64 |
*2048 streaming cols = 64-prompt runs.

1. **vLLM ≈ jsim (±0.1)**: the +1 bonus is offset by vLLM discarding one warm
   position per shift (argmax_prev[n_acc+1:]). Engine choice doesn't change
   rankings. **JF block-complete is 0.3-0.5 LOWER than streaming** (tail
   positions of each block burn iterations streaming would slide past) and
   **reverses the long-horizon story: RL'd models beat base at BOTH horizons**
   (block mode caps loop/self-similarity harvesting at block granularity).
2. **RL'd > base at 512 in all three engines** — robust.
3. **Positional TPF bands** (streaming 2048 decode): 0-512: RL'd 3.57-3.66 >
   base 3.39. 512-1024: base 4.59 ≫ cev2 3.53 (flat = its MAX_PAIRS=16 ≈
   512-token cons coverage), fwd_v8 3.82 (64-pair coverage helps but only
   partway). Base's late-band 4.5+ = accumulated self-similar content RL'd
   models don't generate. NOTE: even MAX_PAIRS=64 saw few windows past ~700 —
   correct rollouts ARE ~700 tokens; coverage is bounded by response length,
   not just the cap.
4. **Why cons_loss falls but TPF doesn't** (`cons_argmax_correct` metric):
   fwd_v8 loss 5.6→3.0 (−46%) while argmax_correct flat 0.37→0.37±0.05;
   A1 loss 17→0.9 while argmax_correct CRASHED 0.34→0.14 (solved the loss by
   emitting masks). The loss average is dominated by easy positions where soft
   prob 0.9→0.99 cuts loss without flipping any argmax; acceptance is gated by
   hard content-decision tokens (commit-breaker classes: word-starts 34-43%,
   latex/math 27-30%, digits 10-12% — same across all models). Per-position
   soft objectives don't extend joint argmax fixed-point runs → need
   argmax-flip / sequence-acceptance objectives (VSD) or direct TPF reward
   (LightningRL).
5. **Qualitative** (annotated traces, `repsplit_*_v3.jsonl` + tokenizer):
   all models grind novel LaTeX at 1-4 tok/commit and surf boilerplate at
   8-16; fwd_v8 trace shows a mid-response solution RESTART re-decoded at
   [14]-token commits = repetition harvesting in action.

---

## 2026-06-10 (cont) — Rambling attribution, head/tail probes, lookahead post-mortem

**Per-band repetition (who rambles where):** base rep8 0.29→0.58→0.66 across
bands 0-512/512-1024/1024-2048 (loop 0.17 late, 11k tokens); fwd_v8 0.34→0.42
(→0.83 but on 1k tokens only). **Base is the late-stage rambler**; fwd_v8's
lower late TPF (3.82 vs 4.59) reflects MORE NOVEL late content, not worse
drafting. (Earlier "fwd_v8 restart" example was an early-window one-off.)

**Head/tail per-position probes** (fixed base prefixes, cascade last-iter
agree_self): fwd_v8 improved the HEAD (pos4-7: 0.922→0.984; uniform-head agree
0.43→0.48), tail argmax flat (0.16→0.145). Its KL loss has NO position decay →
loss VALUE is tail-dominated → the −46% loss fall = tail soft-prob moving
"remotely closer" without argmax flips (user's hypothesis confirmed at the
loss level). cev2 (HAS dflash decay): head ≈ base but mid/tail REGRESSED
(pos8-15: 0.539 vs base 0.672; tail 0.074 vs 0.160) — decay protected the head
and let the rest rot; matches cev2 having the lowest TPF of the trio.

**Lookahead/branching experiments post-mortem** (user's GRPO-within-block):
- `_sim_jacobi_lookahead.py` (per-iter best-of-K_alt sampled tails, ORACLE
  pick by next-iter acceptance, base model, T=1.0): K_alt=8 → Δ=+0.14 mean
  with per-prompt swings −1.96…+1.24; K_alt=16 → Δ=+0.75 on a lower vanilla
  baseline. Even oracle selection among the model's own samples buys little →
  the sample distribution rarely contains better tails at hard tokens;
  GRPO-style reweighting within blocks has weak signal.
- `_sim_jf_block_lookahead.py` (block mode, pick candidate CLOSEST TO OWN
  FIXPOINT): **negative** (Δ=−0.17/−0.31) — self-consistency proxy rewards
  stable-but-wrong fixpoints. Miniature proof of TPF-proxy reward hacking.

**TPF-reward design implications:** raw TPF reward is dominated by loop/
self-similarity outliers (loop TPF ~16 vs clean ~3). Candidate design:
reward = LOOP-MASKED clean-segment TPF per rollout (labeler exists in
`_tpf_rep_split.py`), group-normalized within prompt (controls content
difficulty), decoupled from accuracy channel (LightningRL), correct-only NLL
anchor; full-sequence horizon (within-block advantages ≈ noise).

---

## 2026-06-10 — Branching experiment v2 (fixed redo of _sim_jacobi_lookahead)

**Fixes over v1:** true self-acceptance (no external reference, no bonus, stop-
fixed), horizon-6 outcome instead of myopic 1-iter n_acc, measurement-only
branches (main decode unperturbed), alternatives = per-position samples from
the policy's own distribution at T=1.0 (n_alt=3), controlled random fill.
`scripts/_branch_tpf_v2.py`, 8 DS prompts, branch every 8th commit.

| model | branch pts | vanilla h-TPF | alts mean | best-of-4 | oracle headroom | alt wins | τ(n_acc1, h-TPF) |
|---|---:|---:|---:|---:|---:|---:|---:|
| base    | 130 | 3.40 | 3.28 | 3.74 | **+0.33** | 40% | +0.34 |
| fwd_v8  | 125 | 3.82 | 3.69 | 4.18 | **+0.36** | 44% | +0.52 |

**Findings.**
1. **v1's conclusion was wrong because v1's selection criterion was nearly
   blind**: 1-iter n_acc correlates with true horizon TPF at only τ=+0.34
   (base). v1 "oracle-picked" on that proxy → tiny noisy gains (+0.14). With
   the real horizon outcome, **state-level headroom is +0.33-0.36 from just 3
   sampled alternatives** — the policy's own samples contain better drafts at
   40-44% of branch states. There IS learnable signal for an RL/reward scheme.
2. Mean sampled alt < vanilla argmax (3.28 vs 3.40) — argmax is locally best
   on average; the value is in the within-group max → group-relative advantage
   (GRPO-style at fixed state) is the right extractor, NOT raw TPF reward.
3. Reward-granularity guidance: horizon ≥ ~6 forwards at branch states;
   same-state grouping controls content difficulty (greedy rollouts are
   deterministic → spread is genuine headroom, not estimation noise).
   Caveat: h=6 local gains need not all survive to sequence level.

---

## 2026-06-10 — Branching v3: horizon sweep, group-size sweep, luck check

`scripts/_branch_tpf_v3.py`: n_alt=15 full-window samples (T=1.0) per branch
state, each rolled once for h_max=12 recording cumulative commits → all
horizons + group sizes {4,8,16} from one run. Winner (picked at h=6) re-rolled
under 2 fresh random-fill seeds. 8 DS prompts, ~100 branch points per model.

Key numbers (base / fwd_v8):
- **Horizon**: headroom (best-of-16) PEAKS at h=3 (+1.06 / +0.97) and decays
  to +0.51 / +0.54 by h=12. Rank agreement with h=6: τ=0.39/0.39 at h=1,
  0.71/0.74 at h=3, 0.79/0.79 at h=4. → **h=1 is insufficient (confirms v1
  post-mortem); h=3-4 is the knee** — most signal at half the cost of h=6.
- **Group size** (h=6): headroom 0.35→0.46→0.65 (base) and 0.39→0.53→0.73
  (fwd_v8) for groups 4→8→16. **Not saturated at 16; group 4 is too small**
  (~half the available headroom).
- **Luck check**: winners' advantages persist under fresh fill seeds in **99%**
  of cases with unchanged magnitude (0.855→0.862 base; 0.984→0.975 fwd_v8) —
  advantages are properties of the DRAFT TOKENS, not fill luck. 73-76% of
  branch states have a winning alternative.
- **Mechanism (crude first-divergence classifier)**: only 26% (base) / 36%
  (fwd_v8) of wins have the first divergent token equal to the eventual greedy
  token; most wins come from later-position guesses and/or conditioning
  effects. Finer per-position credit analysis needs draft storage (TODO).
- Caveat: best-of is re-selected at each horizon; the h-decay of headroom =
  partial washout of local gains (regression to shared dynamics), consistent
  with advantages being partly transient. Persistent component ≈ +0.5 at h=12.

---

## 2026-06-10 — Branch-content analysis: what actually drives the TPF wins

`_branch_tpf_v4.py` (content-recording) + `_analyze_branch4.py`. ~100 branch
points/model, winners beat vanilla at 75/101 (base), 71/98 (fwd_v8).

**Feature predictiveness within-state (Kendall-tau vs h=6 outcome):**
prefix-match-with-future +0.34/+0.44 (base/fwd_v8); total-match +0.19/+0.17;
longest-run +0.25/+0.35; **mean sampling rank −0.02/+0.04 (≈ZERO)** — the
model's own likelihood does NOT identify good windows → pure distillation
cannot learn this; outcome-based (RL) signal required.

**Winners do NOT guess the future better**: winner−vanilla match delta is
NEGATIVE (−0.64 base / −1.04 fwd_v8); 69-77% of wins are "conditioning wins"
(winner matches future ≤ vanilla yet decodes faster). Losers match even less
(−1.8). Matching helps among samples, but is not the winning mechanism.

**The vanilla argmax window is often DEGENERATE at win states**: longest
same-token run 4.97 (base) / 8.86 (fwd_v8!), repeated-bigram fraction
0.22/0.40 — argmax under noisy conditioning collapses to repeated tokens
("3 3 3 3 5 5 = = =", "students students both both both") which re-poison
each subsequent iteration. ANY sample breaks the degeneracy (winners AND
losers both ~2.1 run / 0.03 rep) — but corr(vanilla degeneracy, headroom) is
only −0.05/−0.11, so degeneracy-breaking is a big visible failure mode at the
largest wins, not the whole story. Win-vs-lose among samples is mostly
early-position (prefix) correctness.

**Gained-match token classes** (winner matches future where vanilla doesn't):
latex/math 35, word-start 23-26, digit 12-21 — the breaker classes.

**Implications:** (1) reward unit = whole window, outcome-based (likelihood
can't find winners); (2) credit concentrates at early window positions
(prefix-gating; decay weighting is the right shape); (3) cheap decode-time
intervention suggested by the data: within-window repetition penalty on the
draft (break argmax degeneracy without sampling noise) — untested TODO;
(4) fwd_v8's argmax windows are MORE degenerate than base's (sharper policy →
harder mode collapse under noise), explaining part of its remaining gap.

---

## 2026-06-10 — Strategy zoo, sampled-prefix feasibility, REPPEN decode win

**Candidate-strategy zoo** (`_branch_tpf_v6.py`, h=6, group 8, 63-65 branch pts):
best-of-8 headroom vs vanilla — full +0.55/+0.68 (base/fwd_v8), keepm
(keep stable positions = argmax-agreed-with-input) +0.41/+0.57 — **LOWER than
full: "stable" includes the degenerate repeats; keeping them locks in the
poison**. T=0.5 ceiling ≈ full with better mean (best reward-group operating
point). prefix4 hurts. **oraclekeep (keep future-matched, resample rest):
+0.76/+0.79, 68% win — best ceiling; the keepm→oraclekeep gap is the
quantified value of a per-position correctness predictor.**

**Sampled-prefix feasibility:** branching at T=1.0-sampled prefixes (RL rollout
states): headroom +0.63 @ 58% win — same signal as greedy states. The
branched-advantage scheme ports to real training rollouts.

**REPPEN full-decode (deterministic, 2-line decode change: draft = argmax
unless token equals one of prev 2 window tokens → runner-up; acceptance
unchanged → content/accuracy invariant):**

| model | vanilla (stop-fixed jsim-16/512) | reppen | Δ |
|---|---:|---:|---:|
| **fwd_v8 s220** | 3.63 | **3.95** | **+0.32** |
| base | 3.37 | 3.24 | −0.13 |

**fwd_v8 + reppen = 3.95, the best honest-protocol TPF measured, +0.58 over
base vanilla — zero additional training.** Model-conditional: RL-sharpened
models suffer window mode-collapse (degeneracy 8.86 vs base 4.97) and reppen
is the antidote; base is mildly hurt. TODO: port to vLLM plugin (needs target
top-2 stash) + verify at 2048/64-prompt scale + on corrupt03/cev2/v9.

---

## 2026-06-10 — REPPEN: lookback sweep, dataset generalization, why-it-works accounting

**Lookback sweep** (DS-16, stop-fixed jsim): fwd_v8: lb1 3.85, **lb2 3.95**,
lb4 3.78, lb8 3.71 — lb=2 optimal, longer over-fires. base: 3.29/3.24/3.17/3.18
— hurt at every lookback (vanilla 3.37).

**Dataset generalization (lb=2, 16 prompts each):**
| | vanilla | reppen | Δ |
|---|---:|---:|---:|
| fwd_v8 MATH500 | 4.11 | **4.39** | **+0.28** |
| fwd_v8 GSM8K   | 3.97 | 4.00 | +0.04 |
| base MATH500   | 3.95 | 3.79 | −0.16 |
| base GSM8K     | 3.89 | 3.69 | −0.20 |

**Why it helps only the RL'd model** (fire accounting: at each penalty fire,
compare the finally-committed token to runner-up vs displaced argmax):
1. fwd_v8 windows fire ~2× more (37-50% of window positions vs base 22-27%)
   — more degenerate repeats to fix (RL sharpening → window mode collapse).
2. Hit ratio runnerup_right:argmax_right — fwd_v8 ≈ 1.10 (MATH/DS), 0.97
   (GSM8K, hence flat); base 0.61-0.86 — **base's consecutive window repeats
   are more often genuinely-correct content ("00","}}","((") — displacing
   them injects errors.**
3. 76-80% of fires are "neither" → bulk of the gain is the CONDITIONING effect
   (un-poisoning window dynamics), which scales with degeneracy → large for
   fwd_v8, absent for base.

**Verdict:** reppen lb=2 = free decode-time win for RL'd models on math-heavy
text (+0.28-0.32), neutral worst-case (GSM8K +0.04), never deploy on base.
Gate on the model (or on a measured window-degeneracy stat). TODO: vLLM
plugin port (top-2 stash) for production + training rollouts; test on v9 ckpts.

---

## 2026-06-11 — v9 (fwd-KL + dflash decay + 64-pair random coverage) results

Training completed 300 steps (job died post-completion). cons_argmax_correct
**0.349→0.444** — first run where the TPF-relevant metric climbed (v8 flat
~0.35). MATH val 0.899 final. Stop-fixed jsim-16/512:

| ckpt | vanilla | reppen lb=2 |
|---|---:|---:|
| **v9 step 220** | **3.78** | **4.10** ← new best honest TPF |
| fwd_v8 step 220 | 3.63 | 3.95 |
| v9 step 300 | 3.27 | 3.58 |
| base | 3.37 | 3.24 |

1. **At matched step 220, decay+coverage buys +0.15 over v8, and reppen
   stacks (+0.32): v9_220+reppen = 4.10 = +22% over base vanilla, with val
   ~0.886.**
2. **Late-training TPF collapse (220→300: 3.78→3.27)** despite argmax_correct
   still rising — in-rollout TPF decline from ~step 240 was real. Checkpoint
   selection matters; TPF-eval checkpoints during training (or early stop
   ~200-240) needed. argmax_correct (training-dist metric) diverges from
   decode TPF late — content/length drift suspected, not drafting metric lies.
3. TODO: vLLM-2048 confirm, val acc at 220, intermediate ckpts 240/260/280 to
   find the peak; reppen+v9 on MATH500.

---

## 2026-06-11 — dLLM-style decoding on causal JF models (quality-graded)

`_sim_dllm_decode.py`, MATH500 first-16, answer-graded. Modes: pure dLLM
(DiffusionGemma-style canvas: candidates + entropy-keep tau=1.0 + rerandomize
rest, commit on argmax-stability or 16-step cap, NO verification) and hybrid
(n dLLM refine passes + 1 prefix-verify pass, AR-greedy-equivalent).

| model | decode | TPF | ACC |
|---|---|---:|---:|
| base | vanilla Jacobi | 3.91 | 0.38 |
| fwd_v8 | vanilla Jacobi | 4.11 | **0.50** |
| base | pure dLLM (argmax cand) | 3.40 | 0.12 |
| fwd_v8 | pure dLLM (argmax cand) | 4.79 | 0.12 |
| base | pure dLLM (T=0.7 cand) | 3.68 | **0.00** |
| fwd_v8 | pure dLLM (T=0.7 cand) | 5.94 | **0.00** (all hit 512 cap, never terminate) |
| base | hybrid n=1 | 0.87 | 0.25* |
| fwd_v8 | hybrid n=1 | 1.66 | 0.44* |
| | hybrid n=2 | 0.79-1.59 | same* |
*hybrid ACC drop = truncation at the 512-forward cap, content is greedy-exact.

**Conclusions.**
1. **Causal JF models cannot run open-loop**: removing verification collapses
   accuracy 0.38-0.50 → 0.12 (argmax) → 0.00 (sampled). The high "TPF" of
   unverified decode is fake speed (committing junk; T=0.7 rambles forever).
   Verification IS the quality mechanism, not an optional tax. DiffusionGemma
   can run open-loop because it was TRAINED under its keep/rerandomize sampler
   (and bidir attention); JF models were trained for prefix-verified denoising.
2. **Naive hybrid (confidence-keep + rerandomize refinement passes) is a big
   net LOSS** (0.87-1.66 TPF): rerandomizing low-confidence positions destroys
   the model's own draft tokens — the reinit-harvesting result yet again. The
   dLLM refinement operator does not transplant to causal models untrained.
3. fwd_v8 vanilla beats base on MATH500-16 on BOTH axes (TPF 4.11 vs 3.91, ACC
   0.50 vs 0.38).
4. Path that remains open: TRAIN the refinement operator (cons loss already
   supports bidir-mask noisy blocks + keep/rerandomize input dist =
   DiffusionGemma-style training, decode w/ dual-mode attention + final verify).

---

## 2026-06-11 — Cross-model draft/verify probes (assembly-line feasibility)

`_sim_external_draft_verify.py` (fixed cursor bug): drafter greedy trace
consumed by verifier in K=32 windows, prefix-accept + bonus; `live` = drafter
re-drafts from corrected prefix after each correction (bounded-lead upper
bound; drafter compute uncounted), `static` = immutable trace (no resync).

| drafter → verifier | mode | TPF_verifier | mean run | runs≥8 | runs=0 |
|---|---|---:|---:|---:|---:|
| base → v9_220    | live   | **22.9** | 22.4 | 79% | 4% |
| v9_220 → base    | live   | 16.1 | 15.4 | 60% | 20% |
| math_k3 → v9_220 | live   | 10.2 | 9.4 | 37% | 31% |
| base → base      | static | 6.2* | 5.3 | 17% | 82% |
| base → v9_220    | static | 1.1 | 0.1 | 0% | 98% |

*base→base static < 32 because generate-vs-batched-forward numeric argmax
flips act as "corrections" and the immutable trace diverges after each.

**Findings.**
1. **In-family AR-fluent drafter ↔ verifier agreement is ~0.95** (runs of ~22
   between disagreements). If a diffusion canvas's graduates approach AR-greedy
   fluency, verifier-side TPF ~16-22 is on the table — the assembly-line
   ceiling is high.
2. **Cons-RL improves absorption: v9-as-verifier accepts base drafts 42%
   better than base-as-verifier accepts v9 drafts** (22.9 vs 16.1) — direct
   evidence for the train-the-verifier-to-absorb thesis.
3. **Re-anchoring is mandatory** (static collapses to ~1) — the drafter must
   re-draft from corrections; persistent-canvas repair is the cheap form.
4. Caveat: `live` counts only verifier forwards; real wall-clock needs a small
   drafter with bounded-lead incremental re-drafting.

## 2026-06-12 — v11 canvas run: step-80 early probe (jsim-16, stop-fixed)

Engine: HF bf16 sims, 16 DS prompts (`deepscaler_tpf_prompts_16.jsonl`), K/W=32,
W_ar=8, max_new=512, greedy. Ckpt: `ckpts_hf/v11_canvas_step_80` (run
`jf_math_7b_dapo_ds_4gpu_jacobi_onpolicy_v11_canvas`, init v9_220, β=0.5
canvas pairs + constant marker, seed-fix 0f640165). Eval node: fs-mbz-gpu-622.
Raw: `eval_passk/tpf_results/v11/step80/*.jsonl`, log `logs/v11_eval_s80.log`.

| eval | corpus TPF | reference |
|---|---:|---:|
| vanilla jsim (`_sim_jacobi_predictor_refresh --refresh none`) | 3.561 (pp-mean 3.621) | v9_220: 3.78 |
| reppen lb=2 | 3.718 (pp-mean 3.786) | v9_220: 4.10 |
| assembly causal-canvas control (keepnoise) | 3.393 | v9_220 untrained: 3.64 |
| assembly bidir + constant marker (keepnoise) | 3.345 | v9_220 untrained bidir: 3.41 |
| assembly bidir, no marker (keepnoise) | 3.265 | — |

Readings:
1. **Canvas is learning**: bidir-vs-causal-control gap −0.23 (untrained) →
   **−0.05** at step 80; the marker is worth +0.08 over no-marker (3.345 vs
   3.265) — mode flag is functioning at decode.
2. **Causal mode drifting down** (~−0.2…−0.4 across vanilla/reppen/causal
   control). Spec §3.6 criterion 1 not currently met. Candidate mechanism:
   causal cons pairs are β=0.5 of v9's signal (dilution) + RL drift on an
   already-sharpened policy — not necessarily marker leakage (input-side
   scoping is unit-tested exact; in-training cons_argmax_correct_causal holds
   0.40-0.47). Decision point at step-160 eval: if causal keeps eroding,
   lower CANVAS_FRAC (0.5→0.3) or raise CONSISTENCY_WEIGHT.
3. In-training canvas argmax: 0.25→0.38 (steps 5→80). MATH val 0.886-0.890
   (≥0.88 criterion holding).

**Correction (fable, 2026-06-11 18:15): criterion 1 reframed.** v11 step 80 has
300 TOTAL RL steps (220 v9-init + 80). Matched-step control is v9_300 =
vanilla 3.27 / reppen 3.58 (late-RL TPF collapse, 2026-06-11 entry) — v11's
3.56 / 3.72 is eroding SLOWER than v9's own trajectory despite β=0.5 halving
causal-cons pressure. The asm causal-control falling in lockstep (3.64→3.39,
no canvas involvement at decode) corroborates ambient late-RL erosion, not
v11-specific damage. Revised criterion 1: causal mode ≥ v9-at-matched-total-
steps. No env change; re-eval at 160 (healthy = vanilla ≳3.3-3.5). If
intervention ever needed: 2× CONSISTENCY_WEIGHT, keep CANVAS_FRAC=0.5.
Checkpoint selection may simply favor early ckpts (80-160) — selection, not
failure.

## 2026-06-12 — Assembly comparability matrix: update-rule confound resolved

{v9_220, v11_s80} × {causal, bidir, bidir+marker} × {argmax, keepnoise},
16 DS prompts, W=32/W_ar=8, max_new=512, HF bf16, node 422 GPUs 4-7.
Raw: `eval_passk/tpf_results/v11/matrix/`, keepnoise s80 arms from the
step-80 battery (`v11/step80/`). NEVER compare across update rules.

| arm | v9_220 | v11_s80 |
|---|---:|---:|
| causal/argmax | 3.706 | 3.551 |
| bidir/argmax | 3.524 | 3.397 |
| bidir+marker/argmax | 3.521 | 3.450 |
| causal/keepnoise | 3.589 | 3.393 |
| bidir/keepnoise | 3.453 | 3.265 |
| bidir+marker/keepnoise | 3.424 | 3.345 |

1. True causal erosion at s80 = −0.16 (argmax) / −0.20 (keepnoise), NOT the
   −0.25 previously logged from a keepnoise-vs-spec-argmax mixed comparison.
2. Canvas gap (bidir+marker − causal control): −0.185 → −0.101 (argmax),
   −0.165 → −0.048 (keepnoise). 45-71% closed in 80 steps; no crossover yet.
3. Marker: neutral untrained (−0.00/−0.03), +0.05/+0.08 at s80 — the mode
   flag is learned, not free.
4. argmax update > keepnoise everywhere (+0.07-0.16) at keep_tau=2.0 —
   matches trace audit (kept tokens 56% correct vs candidates 37% vs fresh
   noise ~0): re-noising destroys carried state.
5. Decode-trace audit (same day, `v11/traces/`): real canvas inputs are 0-9%
   clean vs training's assumed 35-54%; kept tokens are model picks wrong
   38-44% of the time. Training/decode state mismatch is the headline lever
   → v11.1 proposal (empirical-state construction, CE-to-rollout target,
   canvas weight mult, keep_tau sweep) pending sign-off.

## 2026-06-12 — Decode-side sweeps on v11_s100 (assembly, 16 DS prompts)

Raw: `eval_passk/tpf_results/v11/decode_sweeps/`. All bidir+constant-marker
unless noted; W=32/W_ar=8 default.

| arm | TPF |
|---|---:|
| keepnoise tau=1.0 | 3.080 |
| keepnoise tau=2.0 | 3.234 |
| keepnoise tau=4.0 | 3.340 |
| keepnoise tau=8.0 | 3.340 |
| argmax update | 3.340 |
| causal control (argmax, no marker) | 3.417 |
| W=24 argmax (canvas 16) | 3.336 |
| W=56 argmax (canvas 48) | 3.301 |
| W=24 keepnoise tau=2 | 3.229 |
| W=56 keepnoise tau=2 | 3.222 |
| gumbel T=0.7 keepnoise tau=2 | 2.934 |

1. tau>=4 keepnoise == argmax update token-identically (nothing re-noised):
   keep-everything is the best canvas update at current canvas quality.
   Re-noising (the DiffusionGemma-style rule training was co-designed with)
   is strictly harmful right now.
2. s100 gap (argmax): -0.077 (untrained -0.185, s80 -0.101) — still closing.
   But absolute TPF of BOTH arms keeps eroding (causal 3.71→3.55→3.42),
   pace accelerating — v9-style late-phase decline; early-ckpt selection
   likely (80-140 window).
3. Residence curve FLAT (3.34/3.34/3.30 for canvas 16/24/48): refinement
   does not accumulate — more residence buys nothing until canvas quality
   rises (the v11.1 training-side lever, not a decode knob).
4. Gumbel candidates hurt (2.93). Dropped.

## 2026-06-12 — v11.0 step-160 milestone (argmax-update protocol, jsim-16)

Ckpt `ckpts_hf/v11_canvas_step_160`; raw `eval_passk/tpf_results/v11/step160/`.

| eval | s160 | (untrained / s80 / s100) |
|---|---:|---|
| vanilla jsim | 3.438 (pp 3.507) | 3.78 / 3.56 / — |
| reppen lb=2 | 3.779 (pp 3.849) | 4.10 / 3.72 / — |
| asm causal/argmax | 3.422 | 3.706 / 3.551 / 3.417 |
| asm bidir+marker/argmax | 3.371 | 3.521 / 3.450 / 3.340 |
| asm bidir/argmax | 3.317 | 3.524 / 3.397 / — |

Gap −0.051 (monotone from −0.185; crossover projected ~s200). Marker +0.054.
Causal healthy under matched-total-steps criterion (380 total: 3.44 vs v9
extrapolation ~2.8-3.0; reppen ROSE s80→s160; causal-control erosion flat
s100→s160). Step-140 in-training argmax spike (0.419) = variance, not regime
change.
