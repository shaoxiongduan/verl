"""Speed-of-light Jacobi-decoding experiment: per-iter ORACLE verification +
noise-reinit of unaccepted draft tokens.

Hypothesis (2026-06-07): training the cons model on PURE-noise drafts makes it
strong at refining noise into clean tokens, but the draft that vLLM-Jacobi
actually carries between iterations is `target_argmax[n_acc+1:]` (the "stale"
predictions made under confusing leftover context). These leftover tokens are
neither clean enough to accept nor noisy enough to be cleanly refined — they
trip up the cons model. If we had a perfect predictor telling us WHERE the
leftover-confusion zone starts in the next draft, we could replace exactly
those positions with pure noise and let the cons model do what it was trained
for. Tests how close to that upper bound we can get.

Decoding loop (shift-by-1 windowed refresh, mirrors vllm_jacobi_patch.py):

  Per real Jacobi iter:
    1. (verification, FREE)   forward(committed + draft) -> verify_argmax
                              B = greedy n_acc match between draft and verify_argmax
    2. modify draft           positions [B + shift, K) -> uniform-random vocab tokens
                              (shift ∈ {0,1,2,3}; shift=0 = perfect oracle,
                              shift>0 simulates a predictor that's late by `shift`)
    3. (real, COUNTED)        forward(committed + modified_draft) -> real_argmax
                              n_acc_real = greedy n_acc match
    4. commit                 modified_draft[:n_acc_real] + [real_argmax[n_acc_real]]
    5. build next draft       real_argmax[n_acc_real+1:] padded to K with prompt-sample

Modes:
  - natural          : step 1+2 skipped; standard shift-by-1 windowed refresh
                      (1 forward per iter). This is the baseline that matches
                      `scripts/vllm_jacobi_patch.py` semantics most closely.
  - oracle_shift{0,1,2,3}: full oracle loop above. TPF reported counts ONLY
                      the real forwards (step 3); verification forwards (step 1)
                      are the "speed-of-light free" oracle.

Output JSONL (one row per prompt per mode):
    { mode, batch_idx, n_tokens, n_iters_counted, tpf,
      mean_n_acc_real, mean_n_acc_verify (oracle modes only),
      completion_preview }

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/_sim_jacobi_oracle_reinit.py \\
        --model PATH --prompts_jsonl PATH \\
        --modes natural,oracle_shift0,oracle_shift1,oracle_shift2,oracle_shift3 \\
        --out_jsonl PATH --K 32 --max_new 1024 --seed 42
"""
from __future__ import annotations

import argparse
import json
import os
import random
import time

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


# -----------------------------------------------------------------------------
# Argument parsing
# -----------------------------------------------------------------------------
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--K", type=int, default=32, help="Jacobi block length")
    p.add_argument("--max_new", type=int, default=1024,
                   help="Max new tokens to commit per prompt")
    p.add_argument("--max_iters", type=int, default=1024,
                   help="Hard cap on Jacobi iters per prompt (anti-hang)")
    p.add_argument("--vocab_size", type=int, default=152064,
                   help="Qwen2.5 vocab (uniform-noise sampling range)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--modes", type=str,
                   default="natural,oracle_shift0,oracle_shift1,oracle_shift2,oracle_shift3",
                   help="Comma-separated list of modes to run.")
    p.add_argument("--noise_source", choices=["uniform", "prompt_sample"],
                   default="uniform",
                   help="How to sample noise tokens for cold-start, tail refill, "
                   "and oracle noise-fill. The ce_noisy_decay_k3 model was "
                   "trained with CONSISTENCY_NOISE_SOURCE=uniform — use uniform "
                   "to match that distribution. prompt_sample matches the JF "
                   "reference engine's DRAFT_INIT for an apples-to-apples vs "
                   "vLLM-Jacobi number.")
    p.add_argument("--limit_prompts", type=int, default=-1,
                   help="If >0, only run the first N prompts (debugging).")
    return p.parse_args()


def parse_mode(mode: str) -> tuple[bool, int]:
    """Returns (use_oracle, shift). natural -> (False, 0); oracle_shiftS -> (True, S)."""
    if mode == "natural":
        return False, 0
    if not mode.startswith("oracle_shift"):
        raise ValueError(f"unknown mode: {mode!r}")
    s = int(mode[len("oracle_shift"):])
    if s < 0:
        raise ValueError(f"shift must be >= 0, got {s}")
    return True, s


# -----------------------------------------------------------------------------
# Core helpers
# -----------------------------------------------------------------------------
def sample_noise_tokens(committed: list[int], n: int, vocab_size: int,
                         noise_source: str, rng: random.Random) -> list[int]:
    """Sample n noise tokens. uniform = uniform vocab (matches cons training
    when CONSISTENCY_NOISE_SOURCE=uniform). prompt_sample = pick from committed
    tokens (matches JF-reference DRAFT_INIT=prompt_sample)."""
    if noise_source == "uniform":
        return [rng.randrange(vocab_size) for _ in range(n)]
    if noise_source == "prompt_sample":
        if not committed:
            return [0] * n
        return [committed[rng.randrange(len(committed))] for _ in range(n)]
    raise ValueError(noise_source)


def greedy_n_acc(draft: list[int], target: list[int]) -> int:
    """Count consecutive matches from position 0. Used as the rejection-sampler
    acceptance count under greedy verification (T=0)."""
    n = 0
    for d, t in zip(draft, target):
        if d == t:
            n += 1
        else:
            break
    return n


@torch.no_grad()
def forward_K(model, committed: list[int], draft: list[int], K: int) -> list[int]:
    """One forward pass. Returns argmax at positions L-1..L-1+K-1, where L =
    len(committed). Logit at position L-1+j predicts the token AT L+j (= draft[j]).

    No KV cache reuse — matches vLLM-Jacobi semantics where each iter recomputes
    the full attention; simpler and the comparison stays apples-to-apples
    across modes."""
    device = model.device
    inp = torch.tensor([committed + draft], dtype=torch.long, device=device)
    L = len(committed)
    logits = model(input_ids=inp).logits[0, L - 1 : L - 1 + K, :]
    return logits.argmax(dim=-1).cpu().tolist()


# -----------------------------------------------------------------------------
# Simulation
# -----------------------------------------------------------------------------
@torch.no_grad()
def simulate(model, tok, prompt_ids: list[int], args, mode: str,
             rng: random.Random) -> dict:
    use_oracle, shift = parse_mode(mode)
    K = args.K
    # Stop tokens. Qwen2.5 chat uses 151643 (<|endoftext|>) and 151645 (<|im_end|>).
    # vLLM's spec-decode has a known stop-token-detection bug at commit
    # boundaries — it checks only the LAST committed token of each commit
    # step, so EOS landing mid-commit is missed and generation continues
    # past EOS into degenerate-loop territory. JF reference handles this
    # correctly (truncates the accepted prefix at first EOS); we mirror
    # that here so the simulator matches JF reference semantics.
    stop_ids = {int(tok.eos_token_id), 151645}
    if hasattr(tok, "pad_token_id") and tok.pad_token_id is not None:
        # don't add pad as a stop; just kept for reference
        pass

    committed = list(prompt_ids)
    prompt_len = len(committed)
    draft = sample_noise_tokens(committed, K, args.vocab_size, args.noise_source, rng)

    total_new = 0
    n_iters_counted = 0     # only "real" forwards
    n_iters_verify = 0      # bookkeeping for the free oracle forwards
    n_acc_real_history: list[int] = []
    n_acc_verify_history: list[int] = []

    while total_new < args.max_new and n_iters_counted < args.max_iters:
        if use_oracle:
            # STEP 1 (verification, FREE): forward on the current draft to see
            # exactly which positions would be accepted naturally. This tells us
            # the boundary B beyond which the draft is "confusing leftover".
            verify_argmax = forward_K(model, committed, draft, K)
            B = greedy_n_acc(draft, verify_argmax)
            n_acc_verify_history.append(B)
            n_iters_verify += 1

            # STEP 2: modify draft — keep first B + shift positions as-is,
            # noise-fill the rest using args.noise_source (default uniform,
            # matching CONSISTENCY_NOISE_SOURCE=uniform training-time noise;
            # see scripts/consistency/pack.py:316).
            cut = min(B + shift, K)
            if cut < K:
                modified_draft = list(draft[:cut]) + sample_noise_tokens(
                    committed, K - cut, args.vocab_size, args.noise_source, rng)
            else:
                modified_draft = list(draft)
        else:
            modified_draft = draft

        # STEP 3 (real, COUNTED): forward on the modified draft.
        real_argmax = forward_K(model, committed, modified_draft, K)
        n_acc_real = greedy_n_acc(modified_draft, real_argmax)
        n_acc_real_history.append(n_acc_real)
        n_iters_counted += 1

        # STEP 4: commit n_acc_real accepted draft tokens + 1 bonus, BUT cut
        # at the first EOS that appears anywhere in this commit batch (mirrors
        # JF reference's `out[0, :num_accepted] == eos_id` handling at
        # cllm2_qwen2_modeling_kv_terminate_on_eos_improved.py:208-211).
        accepted_part = [int(t) for t in modified_draft[:n_acc_real]]
        bonus_part: list[int] = []
        if n_acc_real < K:
            bonus_part = [int(real_argmax[n_acc_real])]
        commit_batch = accepted_part + bonus_part
        eos_pos = next((i for i, t in enumerate(commit_batch) if t in stop_ids), -1)
        if eos_pos >= 0:
            commit_batch = commit_batch[: eos_pos + 1]
            committed.extend(commit_batch)
            total_new += len(commit_batch)
            break
        committed.extend(commit_batch)
        if n_acc_real < K:
            total_new += n_acc_real + 1
        else:
            total_new += K

        # STEP 5: build next draft = shifted argmax tail + refill via
        # args.noise_source. With noise_source=uniform the n_acc_real+1 freed
        # tail slots become pure-random — matches the cons-noisy-decay-k3
        # training distribution.
        shifted_tail = list(real_argmax[n_acc_real + 1:])
        refill_n = K - len(shifted_tail)
        if refill_n > 0:
            shifted_tail = shifted_tail + sample_noise_tokens(
                committed, refill_n, args.vocab_size, args.noise_source, rng)
        draft = shifted_tail[:K]

    tpf = (total_new / n_iters_counted) if n_iters_counted else 0.0
    return {
        "mode": mode,
        "n_tokens": total_new,
        "n_iters_counted": n_iters_counted,
        "n_iters_verify": n_iters_verify,
        "tpf": tpf,
        "mean_n_acc_real": (sum(n_acc_real_history) / len(n_acc_real_history))
                            if n_acc_real_history else 0.0,
        "mean_n_acc_verify": (sum(n_acc_verify_history) / len(n_acc_verify_history))
                              if n_acc_verify_history else None,
        "n_acc_real_history": n_acc_real_history,
        "n_acc_verify_history": n_acc_verify_history if use_oracle else None,
        "completion_preview": tok.decode(committed[prompt_len:][:200],
                                         skip_special_tokens=False),
    }


# -----------------------------------------------------------------------------
# Main
# -----------------------------------------------------------------------------
def main() -> None:
    args = parse_args()
    modes = [m.strip() for m in args.modes.split(",") if m.strip()]
    for m in modes:
        parse_mode(m)  # validate early

    print(f"[oracle-sim] loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()

    prompts = [json.loads(l) for l in open(args.prompts_jsonl) if l.strip()]
    if args.limit_prompts > 0:
        prompts = prompts[: args.limit_prompts]
    print(f"[oracle-sim] {len(prompts)} prompts, K={args.K} max_new={args.max_new}, "
          f"modes={modes}, vocab={args.vocab_size}", flush=True)

    os.makedirs(os.path.dirname(args.out_jsonl) or ".", exist_ok=True)
    out_fp = open(args.out_jsonl, "w")

    per_mode_tpfs: dict[str, list[float]] = {m: [] for m in modes}
    per_mode_nacc_real: dict[str, list[float]] = {m: [] for m in modes}
    per_mode_nacc_verify: dict[str, list[float]] = {m: [] for m in modes}

    t_start = time.time()
    for i, pdata in enumerate(prompts):
        chat = [{"role": "user", "content": pdata["input"]}]
        prompt_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(prompt_text, return_tensors="pt").input_ids[0].tolist()

        # Per-prompt RNG seeded by (global_seed, prompt_idx). Each mode gets the
        # SAME stream of randomness for noise tokens / prompt-sample refills, so
        # mode-to-mode comparisons reflect mechanism not RNG luck.
        for mode in modes:
            rng = random.Random((args.seed * 1_000_003) ^ (i * 65537) ^ hash(mode))
            t_p = time.time()
            r = simulate(model, tok, prompt_ids, args, mode, rng)
            r["batch_idx"] = i
            r["task_id"] = pdata.get("id")
            r["wall_s"] = time.time() - t_p
            out_fp.write(json.dumps(r) + "\n")
            out_fp.flush()
            per_mode_tpfs[mode].append(r["tpf"])
            per_mode_nacc_real[mode].append(r["mean_n_acc_real"])
            if r["mean_n_acc_verify"] is not None:
                per_mode_nacc_verify[mode].append(r["mean_n_acc_verify"])
            print(f"[oracle-sim] [{i+1}/{len(prompts)}] mode={mode:14s} "
                  f"tok={r['n_tokens']:4d} iters={r['n_iters_counted']:4d} "
                  f"TPF={r['tpf']:.3f}  n_acc_real={r['mean_n_acc_real']:.2f}  "
                  + (f"n_acc_verify={r['mean_n_acc_verify']:.2f}  "
                     if r['mean_n_acc_verify'] is not None else "")
                  + f"({r['wall_s']:.1f}s)",
                  flush=True)

    out_fp.close()
    t_total = time.time() - t_start
    print(f"\n[oracle-sim] DONE in {t_total/60:.1f} min", flush=True)

    print(f"\n=== Summary (n={len(prompts)} prompts) ===")
    print(f"{'mode':18s} {'mean_TPF':>9s} {'SE':>7s} {'min':>6s} {'max':>6s} "
          f"{'n_acc_real':>10s} {'n_acc_verify':>13s}")
    for mode in modes:
        ts = per_mode_tpfs[mode]
        if not ts:
            continue
        mean_t = sum(ts) / len(ts)
        var = sum((x - mean_t) ** 2 for x in ts) / max(1, len(ts) - 1)
        se = (var / len(ts)) ** 0.5
        mean_acc_r = sum(per_mode_nacc_real[mode]) / len(per_mode_nacc_real[mode])
        mean_acc_v = (sum(per_mode_nacc_verify[mode]) / len(per_mode_nacc_verify[mode])
                      if per_mode_nacc_verify[mode] else None)
        acc_v_str = f"{mean_acc_v:13.3f}" if mean_acc_v is not None else f"{'—':>13s}"
        print(f"{mode:18s} {mean_t:9.3f} {se:7.3f} {min(ts):6.2f} {max(ts):6.2f} "
              f"{mean_acc_r:10.3f} {acc_v_str}")


if __name__ == "__main__":
    main()
