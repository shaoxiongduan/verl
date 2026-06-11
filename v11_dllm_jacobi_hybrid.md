# v11: dLLM-Jacobi Hybrid ("Assembly Line") — Design & Implementation Spec

Date: 2026-06-11. Status: design final, implementation not started.
Companion context: `plan.md` (v11 section), `eval_passk/tpf_results/TPF_RESULTS_LOG.md`
(all baseline numbers cited below, entries 2026-06-09 … 06-11).

## 0. One-paragraph summary

One model, two modes, one forward per decode step. A sliding window after the
committed text holds two zones: a small **AR zone** (causal attention, plain
tokens) that verifies and commits via Jacobi prefix-match — output is exactly
the greedy AR text by construction — and a larger **canvas zone**
(bidirectional attention, tokens carry an additive sinusoidal **mode marker**)
that iteratively refines future tokens DiffusionGemma-style and never commits.
Tokens "graduate" from canvas to AR zone as the window slides (the marker is
simply not applied once they cross). The canvas gets one refinement pass per
forward for free ("residence-time refinement"); the system is self-stabilizing
(fast frontier → less-refined graduates → lower acceptance → slower frontier).
**v11 trains the canvas mode** — a multi-level corruption→clean CE in the dLLM
family (= our cons loss at sampled noise levels, bidir, marked) — initialized
from our best causal checkpoint. The AR/Jacobi mode needs no new training for
now (option: keep the current on-policy cons loss unchanged alongside).

## 1. Why we believe this works (measured evidence)

All numbers: stop-fixed jsim protocol, 16 DeepScaler prompts
(`eval_passk/deepscaler_tpf_prompts_16.jsonl`), K/W=32, max_new=512, greedy,
on H200 GPUs, HF bf16. Tools cited are in `scripts/`.

| fact | number | source tool |
|---|---|---|
| vanilla streaming Jacobi, v9_220 (best causal ckpt) | TPF 3.78 | `_sim_jacobi_predictor_refresh.py` |
| + reppen draft policy (decode-only) | 4.10 | `_sim_reppen_decode.py` |
| assembly decode, untrained, v9_220, causal-canvas control | 3.64 | `_sim_assembly_decode.py` |
| assembly decode, untrained, v9_220, **bidir canvas** | 3.41 | same |
| assembly decode, v10@step80 (hybrid-mask trained, wrong inputs) | causal 3.41 / bidir 3.26 (gap −0.15 vs v9's −0.23) | same |
| output == greedy AR text | verified token-exact (3/4 prompts 256/256; 1 bf16 near-tie flip) | inline check |
| agreement ceiling between two in-family AR models | runs of ~22 tokens (p≈0.95) → verifier-TPF 16–22 | `_sim_external_draft_verify.py` |

Readings: the zone/commit-cap **structure costs only −0.14**; **untrained**
bidir attention costs only −0.23 more (graceful: canvas only proposes, AR zone
filters); 80 steps of (mis-targeted) hybrid-mask training already shrank the
bidir gap 35%. The gap between current 3.4 and the 16–22 ceiling is canvas
quality — which is what v11 trains. Worst case degrades to current streaming
Jacobi (the AR zone alone is today's decoder).

## 2. Decode protocol (the environment training must match)

Window of W tokens appended to committed prefix; W_ar=8, W_canvas=24 (v1).

Per step (ONE forward):
1. Build input `[committed | window]`. Add the **marker vector** to the input
   embeddings of canvas positions only (positions ≥ W_ar of the window).
2. Attention mask (4D): causal everywhere, PLUS canvas positions attend
   bidirectionally within the canvas region. AR-zone rows are strictly causal
   → their logits are exact AR logits (this is the quality guarantee; never
   weaken it). Reference implementation: `scripts/_sim_assembly_decode.py`
   (`build_mask`), needs marker support added.
3. AR zone: standard Jacobi acceptance — n_acc = longest prefix where
   argmax == window token (positions 0..W_ar-1 only); commit
   `window[:n_acc] + [argmax[n_acc]]` (bonus), capped at W_ar. Truncate the
   commit at the first stop token (151645/151643) — mid-block stops MUST end
   generation (past bug, see log 2026-06-10).
4. Canvas update (the rule that makes training match decode — see §3):
   sample candidates from canvas logits (argmax for v1; Gumbel `logits/T+g`
   later), KEEP positions with entropy ≤ τ_keep, **re-noise all other
   positions with fresh uniform-random tokens** (NOT model leftovers).
5. Slide window by len(commit): AR-zone remainder keeps its Jacobi-corrected
   tokens; canvas tokens crossing the boundary graduate (lose the marker,
   become AR-zone draft input); fresh random tokens enter at the far end.

Dynamics (no scheduler needed): a token entering at distance d gets ~d/ν
refinement passes before graduation (ν = mean commit/forward). Stalls are free
refinement steps. Equilibrium ν* solves ν = q(W_canvas/ν)/(1−q(·)) where q(r)
= per-token agreement after r passes; unique & stable (monotone map).

## 3. Training: the canvas mode (v11 core)

### 3.1 The principle (why simple corruption→clean CE is correct)

Survey result (TiDAR / Fast-dLLM v2 / LLaDA / SDAR): every dLLM trains
**x0-prediction** — CE from a corrupted input at a sampled noise level to the
clean sequence. Nobody trains step-to-step transitions. This works because the
**samplers are designed so decode-time states stay inside the corruption
family**: masked models' states are subsets of masks; DiffusionGemma re-noises
rejected positions with fresh randoms each step so its canvas is always
≈ {kept tokens + fresh noise}. We adopt the same co-design: because our canvas
update re-noises non-kept positions with FRESH noise (§2.4), the decode-time
canvas states are, by construction, `{mostly-correct kept tokens + fresh noise
at some level}` — exactly what artificial corruption produces. So multi-level
corruption CE is on-policy-representative WITHOUT any rollout harvesting or
training-time sampler simulation. (TiDAR can use a single all-mask level
because its drafting is one-shot; our canvas is persistent/iterative and
visits the whole level range, hence multi-level.)

One residual gap + its patch: at decode, KEPT tokens are model-chosen and
occasionally confident-wrong; artificial corruption keeps ground truth.
Patch: a small fraction of "kept" positions get **plausible substitutions**
(tokens sampled from the model's top-k at that position) instead of truth.
Validated independently: corrupt03 (30% substitution-noise training) is our
best-drafting causal model (vLLM 4.22).

### 3.2 The canvas training pair (exact construction)

Work inside the existing cons pack (`scripts/consistency/pack.py`): pairs of
(noisy block | clean block) sharing RoPE positions, conditioned on the prompt
+ prior response. Per pair:

1. Take the clean response window `c[0..N)` (N = CONSISTENCY_BLOCK_SIZE = 32).
2. Sample noise level `τ ~ Uniform{levels}`; v1 levels: fraction renoised
   f ∈ {1.0, 0.75, 0.5, 0.25, 0.125} (uniform draw per pair). f=1.0 = fully
   fresh noise (a token's first canvas pass); f=0.125 = nearly graduated.
3. Choose the renoise set R: |R| = f·N positions. Position-weighted, not
   uniform: P(j ∈ R) increases with j (far positions noisier) — v1: sample R
   by weights w_j ∝ 0.5 + j/N. (Approximates the spatial age profile.)
4. Input tile: `x[j] = fresh_uniform_random if j ∈ R else c[j]`, then for a
   fraction ρ=0.1 of the NON-renoised positions, `x[j] = plausible_sub(c[j])`
   = a sample from the base model's top-8 at that position (precompute from
   rollout logits or approximate with a cached noise bank; v1 fallback:
   sample from the current policy's logits during the pack's existing
   alt-pool machinery, or skip ρ in v1.0 and add in v1.1).
5. Attention: **fully bidirectional within the tile** (`CONSISTENCY_CAUSAL_REGION_SIZE=-1`),
   causal to prompt/prior-clean (the mask machinery in
   `scripts/consistency/attention.py` already implements this).
6. **Marker ON** for the tile: additive sinusoidal vector on input embeddings
   at tile positions (existing machinery: `CONSISTENCY_USE_DRAFT_MARKER=1`;
   it is FSDP-safe — never index `embed.weight` outside forward, see memory).
7. Loss: CE (or fwd-KL to clean-context logits) from tile logits to clean
   tokens, **shift-by-1 convention kept** (logit at j → c[j+1]; same as
   dflash_ce path — do NOT reintroduce the in-place off-by-one bug, see
   feedback memory 2026-05-30). Position weighting: uniform in v1 (the decay
   weighting is an AR-zone concept; for the canvas all positions matter
   equally — TiDAR/LLaDA use uniform too). Mask out pad positions.

### 3.3 What happens to the AR/Jacobi mode

Phase A (this run): **no new AR-zone training.**
- Init from `ckpts_hf/fwdkl_decay_v9_step_220` (our best causal drafter:
  vanilla 3.78 / reppen 4.10, MATH val ~0.886). Its causal Jacobi skill is
  already trained.
- The RL loop (DAPO, unchanged) + the fwd-KL-to-base anchor continue to train
  the unmarked/causal mode every step.
- OPTION (recommended if cheap): keep the existing on-policy cons loss as a
  second pair type — a fraction β of cons pairs are standard v9-style pairs
  (on-policy cascade input, causal, unmarked, KL+decay), the rest (1−β) are
  canvas pairs (§3.2). v1: β=0.5. This is a PER-PAIR mode mix, NOT per-tile —
  avoids needing per-tile masks. Implementation: pack a batch with one mode
  per sequence-slot, gated by a per-pair flag carried through to the mask
  builder and loss.
- Phase B (later, only if graduate-verification turns out weak): train the AR
  zone on canvas graduates via one no-grad pass (input = canvas predictions,
  unmarked, causal, verify/correct loss). NOT in scope for v11.0.

### 3.4 Run configuration (v11.0)

Launcher: copy `scripts/run_fwdkl_decay_v9.sh` → `run_v11_canvas.sh` with:
```
JF_MODEL=ckpts_hf/fwdkl_decay_v9_step_220        # init from v9_220
CONSISTENCY_LOSS_TYPE=kl                          # fwd-KL (or ce; kl matches v9)
CONSISTENCY_DIVERGENCE=forward_kl
CONSISTENCY_TEACHER=base  CONSISTENCY_TEACHER_PATH=<base JF snapshot>
CONSISTENCY_KL_DECAY=1                            # applies to the CAUSAL pairs only
CONSISTENCY_MAX_PAIRS=64  CONSISTENCY_PAIR_SAMPLE=random
CONSISTENCY_ONPOLICY=1                            # for the causal-pair half
CONSISTENCY_WEIGHT=0.001
# --- new v11 knobs (to implement) ---
CONSISTENCY_CANVAS_FRAC=0.5            # fraction of pairs in canvas mode
CONSISTENCY_CANVAS_LEVELS=1.0,0.75,0.5,0.25,0.125
CONSISTENCY_CANVAS_PLAUSIBLE_FRAC=0.1  # 0 in v1.0 if alt-pool plumbing is slow
CONSISTENCY_USE_DRAFT_MARKER=1         # marker on canvas pairs ONLY
CONSISTENCY_CAUSAL_REGION_SIZE=-1      # full bidir — canvas pairs ONLY
```
Steps: 300, save every 20. **Checkpoint-select by TPF eval — v9 collapsed
3.78→3.27 from step 220→300; evaluate 160/200/220/260** (and merge promptly:
max_ckpt_to_keep=5 deletes old FSDP ckpts — we lost steps ≤140 of the
masktail runs this way).

### 3.5 Implementation checklist (file-level)

1. `pack.py`: per-pair mode flag; canvas-pair input construction (§3.2 steps
   1–4); marker application restricted to canvas pairs; plumb the flag into
   the mask builder and loss. (~half day)
2. `attention.py`: mask_mod takes per-pair causal_region (canvas pairs −1,
   causal pairs 0). The per-q-position machinery exists; add the per-pair
   lookup. (~2h)
3. `loss.py`: canvas-pair branch = uniform-weight shift-by-1 CE/KL to clean
   (mostly exists as the non-decay paths); route by pair flag. (~2h)
4. `verl_hook.py`: env knobs above; per-pair mode sampling. (~2h)
5. `_sim_assembly_decode.py`: add marker support (load marker vector, add to
   canvas-position embeddings via `inputs_embeds`), confidence-keep+renoise
   update (`--canvas_update keepnoise` exists), Gumbel candidates option.
   (~half day)
6. Unit test: CPU test that canvas pairs get bidir mask + marker and causal
   pairs stay bit-identical to v9 behavior (style of `test_mask_tail_loss.py`).

### 3.6 Evaluation plan & success criteria

All evals: stop-fixed, 16-prompt DS set first, MATH500-16 with answer grading
(`_sim_dllm_decode.py` has the grader) second.
1. **Causal-mode regression** (must-pass): vanilla jsim + reppen on v11 ckpts
   ≈ v9_220 (3.78/4.10), MATH val ≥ ~0.88. If the marked-canvas training
   degrades the unmarked mode, the mode separation failed → raise marker
   amplitude / lower CANVAS_FRAC.
2. **Canvas quality** (the target): assembly decode (`_sim_assembly_decode.py`
   + marker) — success = bidir canvas BEATS the causal-canvas control
   (untrained gap was −0.23; v11 should flip the sign), and graduate
   agreement-with-greedy at the boundary climbs toward the 0.95 ceiling.
   Diagnostic: per-position agreement vs distance-from-frontier
   (sharpening-probe style, `_analyze_draft_sharpening.py` adapted for the
   marker+bidir mask).
3. **Headline**: assembly TPF vs 4.10 (v9+reppen, the current champion) and
   vs 3.78 vanilla. The architecture is interesting at ≥4.5; compelling at ≥6.
4. Residence curve: TPF & graduate quality as functions of W_canvas
   ∈ {16, 24, 48} — checks the ν* = q(W/ν*) homeostasis prediction.

### 3.7 Known pitfalls (collected from this project's history)

- Stop tokens: truncate commits at the FIRST stop inside a block; check both
  151645 and 151643 at eval; never treat 151643 as stop when it's a mask/canvas
  filler context (here we use random noise, not 151643, partly for this reason).
- Numerics: `generate` vs batched-forward argmax flips at near-ties exist
  (~1/1000 tokens); don't chase exact-match failures below that rate.
- Shift-by-1: logit at position p predicts token p+1 under shared RoPE.
  The 2026-05-30 off-by-one bug cost a week of CE runs.
- FSDP: never index `embed.weight[i]` outside forward (0-storage views on
  non-owner ranks). The sinusoidal marker machinery exists because of this.
- Ray env: cons env vars must be set before `ray start` / in the launcher
  exports (wrapper-export pattern of `run_masktail_A1_onpolicy.sh` works).
- GPU etiquette: srun --overlap onto shared allocations; CHECK
  `nvidia-smi` immediately before launching (we OOM'd against another user's
  job); 4-GPU training per node; evals on the spare GPUs.
- In-rollout `jacobi_tpf_mean` is NOT the decision metric (argmax_prev,
  T=1.0); decide on stop-fixed jsim + (for finals) the vLLM 64-prompt bench.
- Watch `cons_argmax_correct`, not cons_loss (loss falls without TPF moving;
  v9's argmax_correct climb 0.349→0.444 was the first true positive signal).

## 4. Worked example (training pair, canvas mode)

Prompt: "Solve x²−5x+6=0." Rollout response (clean):
`... <prefix> | the roots are x = 2 and x = 3 . \boxed{2,3}` — take the
N=32-token window starting after the prefix; suppose tokens are
`[the][ roots][ are][ x][ =][ 2][ and][ x][ =][ 3][.][ \boxed][{2][,3][}] ...`

Draw f=0.5 → renoise 16 of 32 positions, far-weighted: say positions
{3,7,9,12,14,17,...} get fresh uniform-random vocab tokens; with ρ=0.1, ~2 of
the remaining positions get plausible substitutions (e.g., `[ roots]`→`[ solutions]`,
`[ 3]`→`[ -3]` sampled from top-8). Input tile (marked, fully bidir):
`[the][ solutions][ are][R74821][ =][ 2][ and][R3019][ =][ -3][.][R…]…`
Loss: CE/KL of tile logit at each position j against clean token j+1,
uniform weights, pads masked. The marker tells the model "dLLM mode"; bidir
attention lets `[ =][ 2]` on the left and `[.]` on the right jointly pin the
noised positions between them. The same window also appears as a clean block
at the same RoPE positions (existing pack layout) for the teacher/anchor.

At decode, this trained behavior runs in the canvas zone: fresh noise enters
at the far end, gets pinned down over ~3–6 residence passes by bidirectional
context, and graduates into the AR zone as a near-clean draft that the causal
Jacobi frontier verifies at high acceptance — committing, exactly, the AR text.
