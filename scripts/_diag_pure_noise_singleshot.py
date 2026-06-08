"""Single-shot diagnostic: feed (prompt + K pure-noise tokens) ONCE and count
how many of the K argmax positions match the AR-greedy reference.

This is the "TPF~6 on pure noise" claim from training — independent of any
chained-iteration / Jacobi-convergence dynamics. If single-shot matches are
~6/K, the model HAS the noise-refinement capability and the issue is in our
multi-iter chaining. If single-shot matches are ~0/K, the model didn't learn
that capability and any chained scheme is bounded by what we see here.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/_diag_pure_noise_singleshot.py \\
        --model PATH --prompts_jsonl PATH --K 32 --n_prompts 16
"""
from __future__ import annotations

import argparse
import json
import random

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--vocab_size", type=int, default=152064)
    p.add_argument("--n_prompts", type=int, default=16)
    p.add_argument("--n_noise_trials", type=int, default=3,
                   help="Sample N independent noise tails per prompt; report "
                        "the mean across trials so we see noise-sample variance.")
    p.add_argument("--seed", type=int, default=42)
    return p.parse_args()


@torch.no_grad()
def greedy_ar_ref(model, prompt_ids: list[int], K: int, eos_id: int) -> list[int]:
    """AR-greedy K tokens. Returns the reference continuation."""
    device = model.device
    cur = list(prompt_ids)
    ref: list[int] = []
    for _ in range(K):
        inp = torch.tensor([cur], dtype=torch.long, device=device)
        log = model(input_ids=inp).logits[0, -1]
        t = int(log.argmax().item())
        ref.append(t)
        cur.append(t)
        if t == eos_id:
            break
    return ref


@torch.no_grad()
def forward_K(model, committed: list[int], draft: list[int]) -> list[int]:
    device = model.device
    inp = torch.tensor([committed + draft], dtype=torch.long, device=device)
    L = len(committed)
    K = len(draft)
    log = model(input_ids=inp).logits[0, L - 1 : L - 1 + K, :]
    return log.argmax(dim=-1).cpu().tolist()


def main() -> None:
    args = parse_args()
    print(f"[diag] loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0")
    model.eval()

    prompts = [json.loads(l) for l in open(args.prompts_jsonl) if l.strip()]
    prompts = prompts[: args.n_prompts]
    rng = random.Random(args.seed)
    eos_id = tok.eos_token_id

    # For each prompt: get AR-greedy ref of length K. Then for several noise
    # samples, forward (prompt + noise) and count how many argmax positions
    # match ref at the corresponding position.
    # Also report the prefix-match length (greedy AR acceptance count if we
    # used the argmax sequence as a "draft").
    per_prompt_match_counts: list[float] = []  # mean across noise trials
    per_prompt_prefix_lens: list[float] = []
    print(f"[diag] {len(prompts)} prompts, K={args.K}, "
          f"{args.n_noise_trials} noise trials each", flush=True)

    for i, pdata in enumerate(prompts):
        chat = [{"role": "user", "content": pdata["input"]}]
        prompt_text = tok.apply_chat_template(chat, tokenize=False,
                                              add_generation_prompt=True)
        prompt_ids = tok(prompt_text, return_tensors="pt").input_ids[0].tolist()
        ref = greedy_ar_ref(model, prompt_ids, args.K, eos_id)

        match_counts = []
        prefix_lens = []
        for trial in range(args.n_noise_trials):
            noise = [rng.randrange(args.vocab_size) for _ in range(args.K)]
            argmax_K = forward_K(model, prompt_ids, noise)
            # how many positions match ref
            m = sum(1 for a, r in zip(argmax_K, ref) if a == r)
            # prefix length: longest j such that argmax_K[:j] == ref[:j]
            p = 0
            for a, r in zip(argmax_K, ref):
                if a == r:
                    p += 1
                else:
                    break
            match_counts.append(m)
            prefix_lens.append(p)

        avg_match = sum(match_counts) / len(match_counts)
        avg_prefix = sum(prefix_lens) / len(prefix_lens)
        per_prompt_match_counts.append(avg_match)
        per_prompt_prefix_lens.append(avg_prefix)
        print(f"[diag] [{i+1}/{len(prompts)}] "
              f"matches={avg_match:.1f}/{args.K} (sample {match_counts}) | "
              f"AR-prefix={avg_prefix:.1f} (sample {prefix_lens}) | "
              f"ref_preview='{tok.decode(ref[:30])[:80]}...'",
              flush=True)

    n = len(per_prompt_match_counts)
    mean_match = sum(per_prompt_match_counts) / n
    mean_prefix = sum(per_prompt_prefix_lens) / n
    print(f"\n[diag] === Single-shot pure-noise predictions ===")
    print(f"[diag] mean matches / K   = {mean_match:.2f} / {args.K}  "
          f"(how many argmax positions match AR ref ANYWHERE)")
    print(f"[diag] mean AR-prefix len = {mean_prefix:.2f}  "
          f"(longest prefix of argmax that matches AR ref greedily)")
    print(f"[diag] If `matches/K` is ~6 then the model HAS the noise-refinement")
    print(f"[diag] capability. If `AR-prefix` is ~6 then Jacobi shift-by-1 with")
    print(f"[diag] pure-noise drafts could in principle reach TPF ~6.")


if __name__ == "__main__":
    main()
