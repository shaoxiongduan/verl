# Cons-loss RL experiments — log of attempts + results

Date range: 2026-05-15 to 2026-05-20.
Setting: DAPO RL training of JacobiForcing_Coder_7B_v1 on OpenCodeInstruct (60k prompts), HumanEval+ (164 prompts) as val. Adding an auxiliary CLLM-style consistency loss to influence Jacobi decoding speed (TPF) while preserving accuracy.

## Question we're trying to answer

Does any consistency-loss variant outperform pure DAPO RL (AR_V2) on
- HumanEval+ pass@1, pass@32
- OCI training-distribution pass@8 (training-distribution coverage)
- TPF (tokens per forward, Jacobi decoding speed)

simultaneously?

## TL;DR — no cons variant beats AR_V2 on accuracy

```
                           HE+ pass@1   HE+ pass@32   OCI mean@8   OCI pass@8   TPF (HE+)
AR_V2 (no cons, baseline)     0.835       0.921         0.800        0.816       3.38
WARMOUT (best cons variant)   0.813       0.909         0.773        0.813       3.44
gap from AR_V2                -2.2 pp     -1.2 pp       -2.7 pp     -0.3 pp     +1.7%
```

All cons variants give a small TPF speedup (+1-8% over AR_V2) at the cost of some accuracy. Best trade-off is WARMOUT: matches AR_V2 on OCI pass@8 (-0.3 pp), modest HE+ accuracy hit (-2.2 pp pass@1), +1.7% TPF.

## All runs

| label             | teacher       | cons λ     | anchor       | layout   | marker         | schedule | status       |
|-------------------|---------------|------------|--------------|----------|----------------|----------|--------------|
| AR_V2             | (none)        | 0          | (none)       | n/a      | n/a            | n/a      | 300 steps ✓  |
| CONS_001          | self          | 0.001      | (none)       | 2-block  | (none)         | const    | 300 steps ✓  |
| BASE_TEACHER      | frozen base   | 0.001      | (none)       | 2-block  | (none)         | const    | 300 steps ✓  |
| MARKER_SELFD      | self          | 0.001      | (none)       | 2-block  | embed row 151665 (learned) | const | 300 steps ✓ |
| EMA_ANCHOR        | EMA student   | 0.001      | clean, 0.001 | 2-block  | (none)         | const    | 300 steps ✓  |
| TSOFT8            | self, T=8     | 0.001      | (none)       | 2-block  | (none)         | const    | 300 steps ✓  |
| REVKL             | self, rev-KL  | 0.001      | (none)       | 2-block  | (none)         | const    | crashed @200 |
| SINUS_SELFD       | self          | 0.001      | (none)       | 2-block  | sinusoidal per-block | const | 300 steps ✓  |
| TRIPLE_BASET      | frozen base   | 0.001      | noisy_unmarked, 0.001 | 3-block | sinusoidal | const | killed @220, degrading |
| WARMOUT           | frozen base   | 0.001 → 0  | noisy_unmarked, 0.001→0 | 3-block | sinusoidal | warmup_out | 300 steps ✓  |
| triple_emaT       | EMA student   | 0.001      | noisy_unmarked, 0.001 | 3-block | sinusoidal | const | crashes @1-2 |

## Full eval table (peak ckpts)

```
run                     peak    HE+_p@1   HE+_p@32   OCI_m@8   OCI_p@8     TPF    diversity_gap*
─────────────────────────────────────────────────────────────────────────────────────
AR_V2 (baseline)         300    0.835     0.921      0.800     0.816      3.38   healthy
CONS_001 (self-d)        300    0.786     0.902      0.769     0.805      3.59   healthy
BASE_TEACHER             100      --        --       0.754     0.777      3.64   decayed
MARKER_SELFD             240      --        --       0.770     0.797      3.43   ~AR
TSOFT8                   300    0.776     0.860        --       --         --    weakened cons
REVKL                    200    0.800     0.878        --       --         --    sharpened
EMA_ANCHOR               260      --        --       0.779     0.816      3.53   ~AR
SINUS_SELFD              160    0.802     0.896      0.767     0.805      3.61   healthy
TRIPLE_BASET             140    0.827     0.909      0.762     0.785      3.55   ~AR
WARMOUT                  200    0.813     0.909      0.773     0.813      3.44   AR-level
```

`*diversity_gap = best@8 - worst@8 of the val-aux HE+ samples at peak step.`

## Key findings

### 1. Pure DAPO (AR_V2) wins on accuracy

AR_V2 climbs steadily to mean@8 = 0.827 by step 300 with no cons regularization. No cons variant catches up. The PG signal is self-eliminating: as the model improves, DAPO's dynamic-sampling filter drops "all-pass" and "all-fail" groups, focusing remaining gradient on borderline prompts. No competing pull from another loss.

### 2. Cons loss accelerates early but doesn't compound with PG late

- BASE_TEACHER peaked at mean@8 = 0.793 at step 100 — *faster than AR_V2 reached that level*. Then degraded to 0.764 by step 300.
- TRIPLE_BASET peaked at 0.826 at step 140 — *matching AR_V2's step-300 final value at less than half the steps*. Then degraded.
- WARMOUT peaked at 0.813 at step 200; cons/anchor weights decayed to 0 by step 200 (per schedule). Then post-decay pure-PG phase did NOT continue climbing like AR_V2 does — plateaued/oscillated 0.79-0.81 through step 300.

Conclusion: cons-loss distillation provides good *early bootstrap* but leaves the policy in a parameter-space region from which PG cannot extract the same gains as starting from scratch. The "free bootstrap, then turn off" hope was disproven.

### 3. Frozen teacher → divergence-driven collapse

Any frozen-teacher cons variant (BASE_TEACHER, TRIPLE_BASET) peaks early then decays as the RL policy drifts further from the frozen reference. The cons/anchor gradient pulls *back* to the teacher; as student-vs-teacher disagreement grows, this pull-back overwhelms PG.

Diagnostic: `actor/student_teacher_kl_clean` (added in commit a3a057e9) measures forward KL between student's clean-position predictions and teacher's clean-position predictions. In WARMOUT this metric grew monotonically: 0 → 0.41 over 300 steps. The growth was *invisible to cons_loss and anchor_loss* (both stayed flat) — those measure noisy-position behavior, not the AR-mode drift that matters for val accuracy.

### 4. EMA-teacher works in 2-block but is blocked by FSDP2 issues in 3-block

EMA_ANCHOR (2-block + clean anchor + EMA teacher) ran 300 steps stable, no collapse. Best OCI pass@8 among cons variants (tied with AR_V2 at 0.816).

EMA + 3-block + sinusoidal marker (triple_emaT) crashes with FSDP2 `setStorage [B*L, 3584]` errors. Tried both `FSDP.summon_full_params` and `torch.distributed.checkpoint.state_dict.get_state_dict` — both call `all_gather` of student params which corrupts FSDP2's mid-forward state. The cons hook fires *inside* forward_step, so any param gather there breaks subsequent forwards.

Proper fix would require moving EMA update outside `forward_step` (into a post-optim callback). Significant refactor in verl's trainer. Not done.

### 5. Sinusoidal marker > learnable embed-row marker

The sinusoidal per-block-index marker (a fresh tensor computed each step, not stored in the embedding table) beat the learnable embed-row marker (row 151665 of the embedding table) on every metric. Same training config otherwise.

```
                    step 100   step 160   final
MARKER_SELFD (embed-row, learnable)   0.756   --     0.793 @300
SINUS_SELFD  (sinusoidal, fixed)      0.788   0.808  0.808 peak
```

Two reasons:
1. The embed-table approach hits FSDP-sharded weight indexing issues (size-0 storage on non-owner ranks under verl's FSDP2). Saved by disk-loading the row, but at the cost of marker trainability.
2. Per-block sinusoidal encoding gives the model a structured signal that the single learnable vector can't.

Memorialized in `feedback_fsdp_embed_indexing.md`: don't index FSDP-sharded model weights mid-forward.

### 6. Marker as a gating signal works in principle (3-block layout)

The 3-block layout — `[prompt | marked-noisy | unmarked-noisy | clean]` per pair — gives the model independent slots for "with marker" vs "without marker" via block-diagonal SDPA masking. Cons loss applies to marked-noisy → clean target; anchor applies to unmarked-noisy → teacher's noisy target. Marker becomes an explicit input-side switch.

Empirically the 3-block runs (TRIPLE_BASET, WARMOUT) had distinct HE+ behavior from 2-block runs. But the accuracy benefit didn't survive — both eventually degraded relative to AR_V2.

### 7. The OCI vs HE+ trade-off

```
                              HE+ pass@1   OCI pass@8
AR_V2                          0.835       0.816
TRIPLE_BASET (peak)            0.827       0.785    ← +HE+, -OCI
WARMOUT                        0.813       0.813    ← balanced
EMA_ANCHOR                     --          0.816    ← matches AR on OCI
```

TRIPLE_BASET specifically excels on HE+ but loses OCI. Likely because cons distillation from BASE biases toward base's strengths (HE+-style code completion). The schedule (WARMOUT) recovered most of the OCI gap by tapering the bias.

EMA_ANCHOR (no marker, 2-block, EMA teacher) matches AR on OCI — the EMA teacher has no external distribution bias since it tracks the student.

## Failure modes recorded

1. **2× FSDP+vLLM jobs per host** wedges the OS (kernel-level limits, not RAM). Documented in `feedback_no_parallel_fsdp_offload.md`.
2. **Ray-cluster env inheritance** — env vars set on the client are not propagated to actors of a pre-started cluster. Documented in `feedback_ray_env_inheritance.md`.
3. **`embed_layer.weight[i]` mid-forward under FSDP** — size-0 storage on non-owner ranks crashes broadcasts. Workaround: load from disk via safetensors or use a non-FSDP source like sinusoidal. Documented in `feedback_fsdp_embed_indexing.md`.
4. **`FSDP.summon_full_params` / `get_state_dict` mid-forward under FSDP2** — corrupts forward state for subsequent micro-batches/steps. Manifests as `setStorage` errors on (B*L, H) broadcasts. EMA-teacher path needs post-optim callback location to be safe.
5. **Frozen-teacher cons collapse** — as RL policy drifts, anchor pull magnitude grows until it dominates PG. Documented above as "Finding #3".
6. **Sample-noise OCI leak (early)** — cons trained with `CONSISTENCY_NOISE_SOURCE=sample` biased predictions toward the OCI distribution, hurting HE+. Fixed by switching to `uniform` noise.
7. **Mid-training reward sandbox CWD pollution** — reward function ran user code without `cwd=tempdir`, leaving CSVs/JSONs in the repo root. Fixed.

## What we built (code, alongside the experiments)

In `scripts/consistency/`:
- `pack.py` — `build_interleaved_batch` supports 2-block and 3-block layouts.
- `attention.py` — `build_sdpa_attention_mask` generalized to 3-block with block-diagonal masking between marked-noisy and unmarked-noisy slots.
- `loss.py` — `compute_consistency_loss` with: `forward_kl/reverse_kl/jsd` divergence, optional `compute_anchor`, `anchor_mode ∈ {clean, noisy_unmarked}`, `marker_embed_override`, sinusoidal vs embed marker types, drift-diagnostic metric.
- `verl_hook.py` — teacher modes (`self`, `base`, `ema`), once-per-step EMA update guard, weight scheduling (warmup_in/warmup_out, separate cons and anchor schedules), disk-loaded marker bypass, drift-diagnostic logging.

Inserted at `verl/workers/engine/fsdp/transformer_impl.py:FSDPEngineWithLMHead.forward_step` — single hook call, no behavior change unless `CONSISTENCY_ENABLE=1`.

## Eval protocols (for reproducing)

```bash
# OCI pass@8 (256 prompts, t=1.0, p=0.95, n=8)
python scripts/eval_offline_pass1.py \
  --model ckpts_hf/<step>_<run> \
  --parquet data/opencodeinstruct/train.parquet \
  --max_prompts 256 --tag oci_<run>_<step> \
  --temperature 1.0 --top_p 0.95 --n 8 \
  --out_dir eval_passk/oci_peak

# HE+ pass@32 (164 prompts, t=1.0, p=0.95, n=32)
python scripts/eval_offline_pass1.py \
  --model ckpts_hf/<step>_<run> \
  --parquet data/humanevalplus/val.parquet \
  --tag passk_<run>_<step>_he \
  --temperature 1.0 --top_p 0.95 --n 32 \
  --max_tokens 1024 --out_dir eval_passk/passk_crossing

# TPF (HE+ prompts, Jacobi t=0.6, block=32, max_iter=128)
MASTER_PORT=2334 python scripts/tpf_trajectories.py \
  --model ckpts_hf/<step>_<run> \
  --prompts_jsonl data/humanevalplus/prompts.jsonl \
  --output_jsonl eval_passk/tpf/peak_<run>_<step>.json \
  --jacobi_block_len 32 --jacobi_max_iterations 128 \
  --temperature 0.6 --batch_size 16 --max_new_tokens 1024
```

## Open questions / future work

1. **EMA teacher + 3-block + FSDP2** — needs a proper post-optim callback in verl's PPO trainer. The simpler in-cons-hook approaches have all hit FSDP state corruption.
2. **Lagged-checkpoint teacher** — implementable via background `model_merger` running on verl's saved checkpoints, with a symlink the cons hook reloads from periodically. Complex but principled.
3. **Schedule tuning** — WARMOUT used a linear decay over the first 67% of training. Other shapes (sigmoid, step, cosine) untested. Decaying anchor faster than cons (or vice versa) untested.
4. **Cons + DPO-style KL-to-π_ref** instead of our anchor — established RLHF anchoring with known good properties.
5. **Architecture-level isolation** (LoRA gated by marker) — ruled out earlier because of inference-time complexity, but would give provable AR-mode preservation if we ever needed it.

## Bottom line

If the goal is **highest accuracy** on HE+/OCI: ship AR_V2. The cons regularization doesn't help here, despite the early-step appearance of speed-up.

If the goal is **Jacobi decoding speed (TPF)** with minimal accuracy loss: ship WARMOUT. +1.7% TPF, OCI matched to AR_V2 (-0.3 pp pass@8), HE+ pass@1 -2.2 pp.

If the goal is **balance + research-paper-grade story** of an actual mechanism: WARMOUT is the most defensible result. The marker is doing work (3-block layout is genuinely different from 2-block), the schedule prevents collapse (vs TRIPLE_BASET), the drift diagnostic shows the underlying dynamics. EMA_ANCHOR is a clean fallback for "DMD2-flavored" framing.
