# RL on Causal-Attention Jacobi Parallel Decoders: Research Plan

Date compiled: 2026-05-06
Status: Active research direction, awaiting key diagnostic experiments
Primary infrastructure: Decode-Learning repo, JacobiForcing_Math_7B_v1 base model, verl-compatible RL pipeline

---

## 0. Executive Summary

**Core observation**: Causal-attention Jacobi parallel decoding with rejection-sampling commit verification admits *exact AR log-likelihood factorization* via the chain rule (proof in `research/jacobi_rejection_ar_equivalence.tex`). This is a structural advantage no published parallel decoding paradigm exploits — dLLMs (LLaDA, MDLM) require ELBO surrogates, greedy-Jacobi (CLLM, JF) cannot claim AR equivalence. Plain AR-GRPO is therefore mathematically principled on Jacobi-with-rejection rollouts, with no need for trajectory-level surrogate losses.

**The actual research question** (after several pivots): does this structural cleanness translate to an empirical advantage? Specifically, does the model's training history (Jacobi forcing distillation) collapse the distribution in ways that *limit RL headroom* — even though the RL math is clean?

**Current best-guess paper shape**: Identify pass@k collapse from Jacobi distillation as a measurable problem, propose a diversity-preserving distillation method that retains TPF acceleration while preserving pass@k, validate that downstream RL gains are larger than vanilla JF + AR-GRPO. Three-pillar contribution: theorem + diagnostic finding + method.

**Key risk**: If pass@k turns out to be similar across paradigms, the contribution shrinks to a position paper / workshop note. Diagnostic experiment decides.

---

## 1. Background Context

### Setup
- **Base AR model**: Qwen2.5-Math-7B-Instruct
- **Distilled Jacobi model**: JacobiForcing_Math_7B_v1 (CLLM/JF-style trained on top of base)
- **Sampler**: Standard Jacobi decoding with **rejection-sampling commit verification** (not greedy match). Implementation at `nanovllm/engine/jacobi_decoding_nongreedy_on_policy.py`.
- **Block size**: 32 tokens. Causal attention throughout.
- **Existing RL pipeline**: GRPO with multiplicative reward (correctness × TPF), per-iteration training with iterative rollouts. Code in `train/iterative_rl_training.py`.

### Existing infrastructure
- `train/grpo_and_consistency_trainer.py:1761-1834` — CLLM-style consistency loss (block-paired soft CE), currently dormant
- `data/map_math_rollout_pipeline.py:1140-1239` — accept-chain-shift step-aware mechanism (`build_shifted_noisy_block`, `compute_accept_chain_shift_flags`)
- `nanovllm/engine/jacobi_decoding_nongreedy_on_policy.py` — RS-commit Jacobi sampler with `accepted_per_iter` tracking
- Iteration directories at `rl_iterations_jacobi_math_*` for various ablation conditions

### What was tried (and what we learned)
- **Multiplicative reward (correctness × TPF)** + plain AR-GRPO: produced repetition exploit. Documented in `research/jacobi_rl_consistency_brainstorm.md`.
- **Joint AR-GRPO + reward-weighted reverse-KL consistency** (labmate's experiment): collapsed. This is consistent with the 2025 literature: two mode-seeking signals (reverse-KL distillation + reverse-KL RL) reinforce each other into degeneracy.
- **Pivotality study** (`research/pivotality_test/`): per-token pivotality structure exists; H6 (block TPF correlated with avg flip iter, r=-0.856) confirmed strongly; H3 (mass-shift ≠ cascade pivotality) refuted.

---

## 2. Core Theoretical Insight (the proof)

### Statement
Under causal attention + per-token rejection-sampling commit (accept token `y_t` with probability `p_t(y_t)`, on first rejection commit a residual sample), the Jacobi rollout `y*` is distributed exactly as a standard left-to-right AR sample from `π_θ`:

```
y* ~ ∏_t π_θ(y*_t | y*_<t, x)
```

### Consequence
The policy gradient
```
∇𝔼[A(y*)] = 𝔼[A(y*) · Σ_t ∇log π_θ(y*_t | y*_<t, x)]
```
is unbiased and computable via a single teacher-forcing forward pass on `y*`. **No trajectory log-prob, no ELBO surrogate, no per-step PPO.**

### What the proof requires
- Causal attention (so `p_t^{(k_t)}` depends only on prefix, not on noisy suffix)
- RS commit rule (so committed token at position `t` is exactly a sample from `π_θ(· | y*_<t, x)`)
- Sweep order with no overwrite (committed positions stay committed)

### What the proof does NOT cover
- Greedy commit (filters toward mode, not exact AR)
- Bidirectional attention (suffix dependency breaks chain identity)
- Any commit rule that depends on values past position `t`

### Why this matters
The dLLM-RL literature (LLaDA 1.5 with VRPO, TraceRL, diffu-GRPO, AGRPO, d1) works hard to construct ELBO surrogates because they don't have AR factorization. ReFusion (Dec 2025) explicitly motivates its design as making "training objective tractable by simplifying combinatorial complexity." All of this machinery is unnecessary in the causal-Jacobi-with-RS regime.

---

## 3. Conceptual Framings

### View 1 vs View 2 (resolved)
- **View 1 (block snapshot)**: at iteration `j`, the entire block is at uniform noise level `j`. CLLM and JF use this framing.
- **View 2 (per-position commit-depth)**: each position `i` has its own commit-depth `d_i = k_i − j`. The user's `step_map` infrastructure encodes this.

Both are valid coordinate systems for the same underlying lattice. View 2 is strictly more expressive (View 1 is its block-marginalization). For samplers with sweep+commit structure, View 2 matches the geometry the sampler actually exploits.

**Subtle point**: The model's output is always position-aligned (standard NTP shift). The "diagonal direction" interpretation that initially confused things is just the inference trajectory's path through the per-position-noise-level state space — it's a path, not a direction the model denoises along.

### Chain-depth obstruction
Predicting `y*_{t+R}` from `y(0)` requires resolving `R-1` AR-conditional dependencies in a single forward pass. This is bounded by transformer depth `L`. For `R = 32` this is plausible; for `R ≥ 256` it's hopeless. **CLLM's empirical 2-3× speedup is hitting this information-theoretic ceiling, not a training quality issue.**

This is a structural reason the diffusion analogy for Jacobi is leaky: continuous diffusion's `z_t` always carries the same conditioning, just noised. Jacobi's "noise" includes *structural absence of conditioning* (early-iteration positions don't have committed prefix yet), which can't be undone by any training signal.

### Diffusion Forcing connection
**Jacobi parallel decoding = causal-attention discrete-token Diffusion Forcing with rejection-commit verification.** This single sentence places the work in the broader DF / Self-Forcing / CausVid literature.

| | DF | Jacobi |
|---|---|---|
| Per-element noise levels | continuous `t_i ∈ [0,T]` per frame | discrete commit-depth `d_i` per token |
| Frame at noise 0 | clean frame | committed token |
| Inference modes | full AR / full diffusion / frontier | sweep with rejection (specific frontier mode) |
| Attention | bidirectional | causal |
| Sample distribution guarantee | none | exact AR (per the proof) |

What Jacobi adds vs DF: causal attention + rejection commit (gives the AR proof). What it gives up: bidirectional flexibility.

---

## 4. Literature Landscape

### dLLM RL (the "wrong paradigm" comparison)
- **LLaDA 1.5** (arXiv 2505.19223) — Variance-Reduced Preference Optimization for masked diffusion. ELBO-DPO with bias/variance reduction.
- **TraceRL / dLLM-RL** (ICLR 2026, arXiv 2509.06949) — per-step trajectory PPO on dLLMs. SOTA on TraDo series.
- **diffu-GRPO / d1** (arXiv 2504.12216) — GRPO with Monte-Carlo log-likelihood estimators on dLLMs.
- **AGRPO** — claims +10% absolute on GSM8K, 3.4× over diffu-GRPO on Countdown.
- **ReFusion** (arXiv 2512.13586) — parallel autoregressive decoding for dLLMs, explicitly motivated by tractable training objective.

### Distillation for parallel decoders (the "starting point" comparison)
- **CLLM** (Kou et al., ICML 2024, arXiv 2403.00835) — Global Consistency loss + AR loss. View 1 framing. SFT only.
- **Jacobi Forcing** (Hu, Kou, ..., Zhang, Dec 2025, arXiv 2512.14681) — progressive consistency distillation, block-causal masks. SFT only. Doesn't reference DF.
- **CDLM** (Kim et al., arXiv 2511.19269) — block-causal diffusion + consistency.

### Diffusion forcing / few-step AR-diffusion (the "right framing" reference)
- **Diffusion Forcing** (Chen et al., NeurIPS 2024) — per-token noise levels, video. Single trained model handles all inference modes (AR/diffusion/hybrid).
- **Self Forcing** (Huang et al., NeurIPS 2025 Spotlight, arXiv 2506.08009) — train few-step student on its own AR rollouts; bridges train-test gap. Holistic video-level loss.
- **CausVid** (CVPR 2025, arXiv 2412.07772) — distill bidirectional teacher → causal AR student. DMD-based. 50→4 steps.
- **Causal Forcing** (Feb 2026, arXiv 2602.02214) — improves Self-Forcing with AR-teacher initialization. +19.3% Dynamic Degree.
- **rCM** (ICLR 2026) — JVP-based diffusion distillation.
- **Reward Forcing** (CVPR 2026, arXiv 2512.04678) — Re-DMD: reward-weighted distribution matching. Most directly relevant.
- **DOLLAR** (ICCV 2025) — few-step video gen via distillation + latent reward.

### V-GRPO and continuous-diffusion RL (related framing)
- **V-GRPO** (Tang et al., arXiv 2604.23380) — RL on continuous diffusion. Local clone at `research/v-grpo/`. Uses ELBO surrogate as log-prob substitute. Their "simple KL" regularizer is structurally equivalent to consistency loss.
- Key takeaway: V-GRPO arrives at the same `‖x_θ − x_θ_old‖²` regularizer for an independent reason (intractable trajectory KL). Corroborates that this regularizer shape is the right family for parallel-decoding RL.

### Mode collapse / RLVR limits (the "be careful" literature)
- **"KL-Regularized RL is Designed to Mode Collapse"** (arXiv 2510.20817) — argues standard RL+reverse-KL is structurally biased toward collapse.
- **Balanced Actor Initialization** (arXiv 2509.00309) — names "Sequence Length Collapse" and "Reward Hockey Stick Curve" as failure modes of RLHF on distillation-trained models.
- **DiverseGRPO** (arXiv 2512.21514) — diversity-aware GRPO.
- **VIDD / Iterative Distillation for Reward-Guided Diffusion** (arXiv 2507.00445) — switches to forward-KL to avoid mode collapse in diffusion RL.
- **"Does RL Really Incentivize Reasoning Capacity Beyond Base Model?"** (arXiv 2504.13837) — RL primarily compresses pass@k → pass@1.
- **"RLVR Implicitly Incentivizes Correct Reasoning"** (arXiv 2506.14245) — counter position.
- **Random rewards finding** — Qwen2.5-Math gains ~21% from random rewards vs ~29% from real rewards.

### dLLM exploration / diversity (directly relevant to Outcome B)
- **"Free Lunch for Pass@k? Low Cost Diverse Sampling for Diffusion Language Models"** (arXiv 2603.04893, March 2026) — *flagship paper*. Shows LLaDA-8B-Instruct pass@k collapses; training-free repulsion mechanism fixes it.
- **Time-Annealed Perturbation Sampling** (arXiv 2601.22629) — sibling diversity-enhancement method for dLLMs.
- **"Why Diffusion Language Models Struggle with Truly Parallel Decoding?"** (arXiv 2602.23225) — structural analysis of dLLM mode-coverage limits.

### Internal references
- `research/jacobi_rejection_ar_equivalence.tex` — the proof
- `research/jacobi_rl_consistency_brainstorm.md` — earlier brainstorm notes (Apr 2026)
- `research/pivotality_test/` — pivotality experiment results

---

## 5. Failed Attempts & Lessons Learned

### Joint AR-GRPO + reward-weighted reverse-KL consistency (collapsed)
**What was tried**: positive-advantage rollouts get consistency loss weight; reward-weighted reverse-KL between `π_θ(· | y^{(j)}_{<i})` and `π_{θ^-}(· | y*_{<i})`. Combined with multiplicative-reward AR-GRPO.

**What happened**: model collapsed.

**Diagnosis (post-hoc, grounded in literature)**:
- AR-GRPO with reverse-KL reference is mode-seeking.
- Reward-weighted reverse-KL consistency is also mode-seeking.
- Two mode-seeking signals reinforced each other onto narrow modes.
- Multiplicative reward + repetition exploit compounds the problem.

**Lesson**: Naive joint training of "RL + reward-weighted consistency" is collapse-prone by structure. Either:
1. Use mass-covering loss (forward-KL or symmetric-KL) for consistency
2. Add entropy regularization
3. Don't joint-train; do sequential CLLM→RL with strong reference-KL
4. Drop multiplicative reward, use accuracy-only reward

### Multiplicative reward → repetition exploit
**What was tried**: reward = correctness × TPF, plain AR-GRPO.

**What happened**: model learned to repeat large chunks of output. Repetition has high TPF (predictable for Jacobi), high correctness when easy, multiplicative reward exploits this.

**Lesson**: AR-GRPO has only one channel to influence TPF — sharpening `π^c`. The path of least resistance is repetition. Drop multiplicative reward in favor of accuracy-only for clean experiments; use consistency loss for the legitimate TPF channel only after the simple version is validated.

---

## 6. Experimental Plan (in priority order)

### Experiment 0 (THIS WEEK): pass@k diagnostic across paradigms

**Purpose**: Decide research direction. Compare diversity-preservation across decoding paradigms before any RL.

**Setup**:
- Models: Qwen2.5-Math-7B-Instruct, JacobiForcing_Math_7B_v1, LLaDA-Math (or closest reproducible dLLM)
- Eval sets: GSM8K test (1319 problems), MATH-500
- Sampling: temperature 1.0, 128 samples per problem
- Track: pass@k for k ∈ {1, 2, 4, 8, 16, 32, 64}

**Compute**: ~80M tokens per model, ~1 H100-day per model. Three models = ~3 H100-days total.

**Possible outcomes**:
| Outcome | pass@1 | pass@k | Implication |
|---|---|---|---|
| A | similar | similar | "Free lunch" — proceed with simple RL experiment, position paper |
| B | similar | Jacobi narrower | Distillation collapsed distribution → method paper on diversity-preserving JF |
| C | Jacobi worse | Jacobi narrower | Distillation hurt — need to rework the foundation |
| D | Jacobi better | Jacobi broader | Surprise — publishable on its own |

**Side experiment**: Run Jacobi-Forcing both with greedy commit and RS commit. If RS-commit pass@k is meaningfully higher, **the verification rule is doing diversity-preservation work** — clean structural finding.

### Experiment 1 (after Experiment 0): plain AR-GRPO on Jacobi-Forcing

**Purpose**: Validate the proof's empirical implication. Plain AR-GRPO on Jacobi-rollouts should improve accuracy without breaking TPF.

**Setup**:
- Use verl + Jacobi-Forcing checkpoint
- Reward: **accuracy-only** (NOT multiplicative — drops the repetition exploit)
- Track: GSM8K + MATH500 accuracy AND eval-time TPF, every N iterations
- Run length: ~50 iterations, similar budget to existing runs

**Comparison baselines**:
- Plain AR-GRPO on Qwen2.5-Math-7B-Instruct (AR baseline)
- TraceRL on LLaDA-Math (dLLM-RL baseline, if reproducible)
- Existing CLLM/JF SFT-only checkpoint (TPF baseline)

**What to report**:
- Accuracy curves over training iterations (all three paradigms on same axes)
- TPF dynamics during AR-GRPO on Jacobi-Forcing (does it degrade?)
- Compute used per percentage-point accuracy gain (efficiency comparison)

### Experiment 2 (conditional on Experiment 1): TPF preservation method

**Trigger**: Run if Experiment 1 shows TPF degradation during AR-GRPO.

**Candidates** (from §7 below):
1. RS-teacher trajectories for JF (free if confirmed by Exp 0 side experiment)
2. Soft-target / stochastic-target JF
3. Forward-KL consistency
4. Position-selective consistency

Implement and compare against vanilla JF + AR-GRPO.

### Experiment 3 (longer-term): block-scaling and chain-depth study

**Purpose**: Empirically validate the chain-depth ceiling on TPF.

**Setup**: Train Jacobi-Forcing variants with block sizes {16, 32, 64, 128}. Run RL post-training on each. Plot TPF as function of block size. Predict: superlinear cost on consistency loss past some block size, TPF saturates.

**Why it's interesting**: chain-depth bound is theoretical; empirical confirmation has not been published. Could ground a follow-up paper on architectural fixes (deeper transformer, multi-pass refinement).

---

## 7. Method Brainstorm (under Outcome B)

If pass@k diagnostic shows Jacobi-Forcing has narrower pass@k than AR base:

### Tier 1: high-leverage, low-effort
**(a) RS-teacher distillation**: use RS-commit Jacobi rollouts (which preserve full AR distribution per the proof) as JF training data instead of greedy fixed points. Smallest change, biggest potential win. Free given existing infrastructure.

**(b) Soft-target / stochastic-target JF**: sample K=4-8 rollouts per prompt at temperature 1.0; treat empirical distribution as target. Prevents mode collapse on greedy fixed point.

**(c) Forward-KL or symmetric-KL consistency**: switch divergence direction. Mass-covering instead of mode-seeking. Direct port of VIDD's mechanism for continuous diffusion.

**(d) Entropy-regularized JF**: explicit entropy term during JF training. ICLR 2025 entropy-controllable RL provides recipes.

### Tier 2: structurally novel
**(e) Position-selective distillation**: only apply consistency loss on positions where teacher is confident (low entropy). High-entropy positions stay flexible.
   - Recipe: weight consistency by `w_i = max(0, 1 − H_i / H_threshold)`
   - Connects to existing pivotality work (`research/pivotality_test/`)
   - Argument: rate-limiting positions for TPF are deterministic ones; distillation should only target those

**(f) Base-model KL anchoring**: explicit KL term to Qwen2.5-Math-7B-Instruct during JF training. Prevents drift from diverse base.

### Tier 3: ambitious / speculative
**(g) RL-aware distillation**: compute small RL gradient during distillation; reward distillation states with high RL gradient norm (= RL headroom). Computationally expensive but novel.

**(h) Repulsion-based GRPO sampling**: port the "Free Lunch for Pass@k" repulsion mechanism to GRPO rollouts. When generating G rollouts per prompt, repel them from each other in feature space. Increases pass@k headroom for GRPO directly.

**(i) Mixed teacher distillation**: distill from multiple AR teachers (different finetunes, different prompts, different temperatures). Match ensemble distribution.

### Recommended composition
For a paper-shaped contribution, combine 2-3 from Tier 1+2:
- **(a) RS-teacher** + **(c) forward-KL** + **(e) position-selective**
- This becomes "Diversity-Preserving Jacobi Forcing" with three principled components, each with literature precedent.

---

## 8. Paper Framing Options

### Option 1: Position paper (if Outcome A)
**"Causal-attention Jacobi parallel decoding admits exact AR log-likelihood, making plain AR-GRPO the principled RL post-training algorithm — without ELBO surrogates."**

- Three pillars: theorem + plain-AR-GRPO empirical results + comparative study with dLLM-RL methods
- Target: workshop or arXiv, build to main-track follow-up
- Risk: structurally simple, may need more experiments to land at top venue

### Option 2: Method paper (if Outcome B) — **MOST LIKELY**
**"Diversity-Preserving Jacobi Forcing for RL-Compatible Parallel Decoders"**

- Three pillars:
  1. Diagnostic finding: standard Jacobi distillation collapses pass@k, limiting RL headroom
  2. Mechanism analysis: greedy targets + reverse-KL drive collapse
  3. Method: RS-teacher / forward-KL / position-selective JF, validated on (preserved pass@k, retained TPF, larger downstream RL gains)
- Target: NeurIPS/ICLR main track
- Risk: depends on Outcome B holding empirically

### Option 3: TPF preservation paper (if Outcome from Experiment 1 shows TPF degradation)
**"Stable AR-GRPO on Parallel Decoders: A Lightweight TPF Preservation Method"**

- Identify TPF degradation during AR-GRPO; propose minimal stable consistency-style addition (likely sequential CLLM-then-RL with strong reference-KL)
- Cleaner methodological story than joint training
- Workshop / smaller venue

### Option 4: Comprehensive systems paper (if compute available)
**"RL-Friendliness Across Parallel Decoding Paradigms: An Empirical Characterization"**

- Matrix: paradigms × base models × benchmarks × compute
- Plot Pareto frontiers of accuracy vs TPF vs compute
- Argues for paradigm choice as a meta-level decision
- Target: position track or workshop, building toward systems venue (MLSys)

### Decision criteria
Run Experiment 0 first. Outcome determines paper.
- Outcome A → Option 1 (or wait for follow-ups for Option 4)
- Outcome B → Option 2 (preferred)
- Outcome from Experiment 1 (post-A) showing TPF degradation → Option 3

---

## 9. The Big-Picture Story

What started as "we have a novel method (joint RL + consistency)" has shifted to:

**"There is a structural choice in parallel-decoding paradigms (causal-attention + RS commit vs bidirectional + ELBO) that has dramatic consequences for RL post-training compatibility. The community has been working on the harder side (dLLM-RL with ELBO surrogates) without recognizing the easier side exists. Furthermore, even on the easier side, distillation methods designed for SFT (CLLM, JF) introduce a pre-collapse that limits RL headroom — a problem that maps onto the broader 2026 mode-collapse literature, and that admits paradigm-specific structural fixes (RS-teacher distillation, position-selective consistency)."**

This is a structural / characterization story, not a single-method story. The proof is one technical contribution; the diagnostic finding (if Outcome B) is another; the diversity-preserving JF method is the third.

---

## 10. Action Items

### Immediate (this week)
- [ ] Set up pass@k diagnostic on Qwen-AR, JF, LLaDA. ~1 day setup, ~3 days compute.
- [ ] Set up plain AR-GRPO on JF in verl. Accuracy-only reward. Track TPF as observation.
- [ ] Polish `research/jacobi_rejection_ar_equivalence.tex` for potential workshop submission.

### Short-term (after Experiment 0 results)
- [ ] If Outcome A: run Experiment 1; consider workshop submission with theorem + plain-AR-GRPO results.
- [ ] If Outcome B: implement RS-teacher JF distillation (Tier 1a). Start ablation grid for diversity-preserving methods.
- [ ] Run greedy-commit vs RS-commit pass@k side experiment on JF model.

### Medium-term (after method ablation)
- [ ] Block-scaling study (Experiment 3) for chain-depth empirical bound.
- [ ] Long-context evaluation (≥8k token derivations) to test TPF retention at scale.
- [ ] Comprehensive cross-paradigm comparison (Option 4 paper material).

### Things to NOT do (until simple version is validated)
- [ ] Joint reward-weighted reverse-KL consistency + AR-GRPO (collapsed; needs forward-KL or sequential).
- [ ] Multiplicative reward (repetition exploit; use accuracy-only).
- [ ] Fancy RL-aware distillation (Tier 3g) — expensive, save for follow-up.

---

## 11. Open Questions

1. **Does AR-GRPO preserve TPF on Jacobi-Forcing model with accuracy-only reward?** Empirical question, blocking on Experiment 1.

2. **Does RS-commit verification preserve more pass@k than greedy commit?** Tests whether the verification rule does diversity-preservation work. Blocking on side experiment within Experiment 0.

3. **At what block size does TPF saturate?** Tests chain-depth ceiling. Long-term Experiment 3.

4. **Can position-selective consistency (Tier 2e) preserve pass@k while still achieving TPF gains?** Tests whether the rate-limiting positions for TPF are also the entropy-poor positions. Conditional on Outcome B.

5. **Does RL on Jacobi extend pass@k beyond base, or just compress pass@k → pass@1?** The big RLVR debate, applied to this paradigm. Will require pass@k tracking throughout RL training.

6. **How does diversity-preserving JF + AR-GRPO compare to dLLM-RL methods (TraceRL, AGRPO) at matched compute?** Ultimate question for cross-paradigm story.

---

## 12. Proof Summary (for portability)

For convenience, the proof's structure (full LaTeX in `research/jacobi_rejection_ar_equivalence.tex`):

**Lemma (Commit rule is exact)**: If C is committed via "propose y ~ q; accept with probability p(y); on rejection commit bonus from p(· | · ≠ y)", then C ~ p regardless of q.

*Proof sketch*: For any z, Pr[C=z] = q(z)p(z) + Σ_{x≠z} q(x)(1-p(x)) · p(z)/(1-p(x)) = p(z)[q(z) + (1-q(z))] = p(z).

**Theorem (y* is exact AR sample)**: y* ~ ∏_t π_θ(y*_t | y*_<t, x).

*Proof sketch*: By induction on t. The sweep commits in causal order; committed positions never overwrite. At iteration k_t when position t commits, the prefix seen by the forward pass equals y*_<t (clean). By causal attention, the position-t logit is exactly π_θ(· | x, y*_<t). By the lemma, the committed token is drawn from this distribution.

**Corollary (AR GRPO is correct)**:
```
∇𝒥(θ) = 𝔼[A(y*) Σ_t ∇log π_θ(y*_t | y*_<t, x)]
```
unbiased when y* is sampled via the rejection-commit Jacobi sampler.

**Critical caveats**:
- Requires causal attention. Bidirectional attention breaks the chain identity.
- Requires RS commit specifically. Greedy commit produces a different (peakier) distribution; the proof doesn't apply.
- Trajectory log-prob decomposition is *not* needed. The proof bypasses it entirely.

---

## 13. Key Source URLs

### Proof and core references
- Jacobi rejection AR equivalence: `research/jacobi_rejection_ar_equivalence.tex` (internal)

### dLLM RL
- LLaDA: https://arxiv.org/abs/2502.09992
- LLaDA 1.5 (VRPO): https://arxiv.org/abs/2505.19223
- TraceRL / dLLM-RL: https://github.com/Gen-Verse/dLLM-RL
- d1 / diffu-GRPO: https://arxiv.org/abs/2504.12216
- ReFusion: https://arxiv.org/html/2512.13586v1

### Distillation
- CLLM: https://arxiv.org/abs/2403.00835
- Jacobi Forcing: https://arxiv.org/abs/2512.14681
- CDLM: https://arxiv.org/abs/2511.19269

### Diffusion forcing / AR diffusion
- Diffusion Forcing: https://www.diffusion-forcing.github.io/
- Self Forcing: https://arxiv.org/abs/2506.08009
- CausVid: https://arxiv.org/abs/2412.07772
- Causal Forcing: https://arxiv.org/abs/2602.02214
- Reward Forcing: https://arxiv.org/abs/2512.04678

### V-GRPO and continuous diffusion RL
- V-GRPO: https://arxiv.org/abs/2604.23380 (local clone at `research/v-grpo/`)

### Mode collapse / RLVR
- KL-Regularized RL is Designed to Mode Collapse: https://arxiv.org/abs/2510.20817
- Balanced Actor Initialization: https://arxiv.org/abs/2509.00309
- DiverseGRPO: https://arxiv.org/html/2512.21514
- VIDD: https://arxiv.org/html/2507.00445
- Does RL Really Incentivize Reasoning Beyond Base?: https://arxiv.org/abs/2504.13837
- RLVR Implicitly Incentivizes Correct Reasoning: https://arxiv.org/abs/2506.14245

### dLLM exploration (key for Outcome B)
- Free Lunch for Pass@k? (flagship): https://arxiv.org/abs/2603.04893
- Time-Annealed Perturbation Sampling: https://arxiv.org/html/2601.22629
- Why dLLMs Struggle with Truly Parallel Decoding: https://arxiv.org/html/2602.23225

### Surveys
- Survey on Parallel Text Generation: https://arxiv.org/html/2508.08712

---

## 14. Glossary

- **AR-GRPO**: Group Relative Policy Optimization with AR factorization of log-likelihood. The standard form when the proof applies.
- **Block snapshot (View 1)**: Trajectory representation where each iteration is a uniform-noise snapshot of the entire block.
- **Causal Forcing**: 2026 video method using AR-teacher for ODE initialization in causal-AR student distillation.
- **Chain depth ceiling**: Information-theoretic bound on Jacobi acceleration. Predicting position `t+R` from position `t`'s noisy state requires resolving R-1 sequential AR dependencies in one forward pass; bounded by transformer depth.
- **CLLM**: Consistency Large Language Models. SFT distillation method by Kou et al. 2024.
- **Commit-depth (View 2)**: Per-position noise level coordinate, `d_i = k_i − j` where `k_i` is position `i`'s commit iteration and `j` is current iteration.
- **DMD**: Distribution Matching Distillation. Distribution-level matching, not point-wise consistency.
- **JF**: Jacobi Forcing. Progressive consistency distillation by Hu, Kou et al. 2025.
- **Multiplicative reward**: correctness × TPF. Drives repetition exploit.
- **`π^c` / `π^n`**: Model's conditional distribution under clean / noisy prefix. Same parameters, different conditioning.
- **RS commit**: Rejection-sampling commit verification rule. Accept token with probability `p_t(y_t)`; on rejection, commit residual sample.
- **Sweep order**: Causal-order commit (position 1 before position 2, etc.).
- **TPF**: Tokens Per Forward (pass). Measure of Jacobi acceleration.
- **`y*`**: Final converged Jacobi rollout. Exact AR sample under causal+RS conditions.

---

# v11: Assembly-line architecture (designed 2026-06-11, with Claude)

**Decode** (validated untrained: structure cost −0.14, untrained-bidir −0.23,
v10@80 already −0.15; output = exact greedy AR, verified): sliding window
`[committed | AR zone W_ar=8 (causal, commits, plain tokens) | canvas W_d=24
(bidir, MARKED, never commits)]`, one forward/step via 4D mask
(`_sim_assembly_decode.py`). Canvas refines passively (1 pass/forward =
residence-time refinement); commits slide the window; canvas tokens graduate
into the AR zone BY DROPPING THE MARKER. Self-stabilizing equilibrium
ν* = q(W_d/ν*)/(1−q(...)); ceiling q∞≈0.95 (in-family probe → TPFv 16-22).

**Training (the v11 deltas over v10):**
1. **Mode separation, no AR-equivalence constraints in the canvas**:
   sinusoidal additive MARKER on canvas-zone positions (existing
   `CONSISTENCY_USE_DRAFT_MARKER` machinery, FSDP-safe) — explicit conditional
   mode switch. NOTE: TiDAR/dFlash get this implicitly because their draft
   regions contain mask tokens; a persistent canvas holds real-looking tokens,
   so the marker substitutes. Graduation = marker off; AR mode never sees it.
2. **Canvas objective freed from AR conventions**: in-place prediction
   (logit at j → clean token at j, NOT shift-by-1), bidir attention
   (CAUSAL_REGION_SIZE=W_ar), loss = CE/KL to clean, graded by position.
3. **Residence-time noise curriculum**: canvas inputs at graded refinement
   levels by distance-from-frontier (far = fresh noise/mask, near-boundary =
   lightly-corrupted/partially-refined) — train the marginal distribution of
   canvas states the decode visits.
4. **AR-zone inputs = canvas graduates** (two-pass): pass-1 no-grad forward of
   the marked canvas → its predictions at near positions become the
   (unmarked) AR-zone inputs in pass 2; AR-zone loss = verify/correct
   (decay-weighted CE/KL, causal). PG + base-anchor keep the AR mode pinned.
5. Later: self-conditioning (prev-step softmax-weighted embedding mix —
   SOFT_MIX revival), confidence-keep canvas update at decode.

**Why interference is solved**: the marker makes mode an input-side flag —
canvas training cannot drag the AR distribution because the AR mode is only
ever trained/evaluated unmarked. (DiffusionGemma's quality drop = one set of
weights, one law; we keep two laws, switched.)

**Implementation queue**: pack.py zone inputs + marker application (~half
day), loss.py in-place canvas branch (small), hook two-pass (~half day),
decode sim marker support (small). Then v11 run = v9 recipe + these.
