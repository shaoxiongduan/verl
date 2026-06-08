"""Diagnostic: noisy-draft denoising accuracy per K-block position.

Same as _diag_logits_clean_draft.py but the K draft tokens are NOISE, not the
actual completion. This is the test the user actually wants: at each within-
block position j, given (prompt + completion[:start] + NOISE_K), does the
model predict completion[start + j + 1]?

This is the same context cons-loss reduces during training. If cons_loss
reduction from v4 actually learned anything useful, v4's per-position
log_p_correct should be HIGHER than base at noisy positions — especially at
small j (which determine TPF via the prefix-acceptance rule).

Noise types:
  uniform — uniform random over vocab (matches CONSISTENCY_NOISE_SOURCE=uniform)
  mask    — single mask token (matches CONSISTENCY_NOISE_SOURCE=mask)

Run with --noise_seed to keep noise identical across models.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/_diag_logits_noisy_draft.py \\
        --model PATH \\
        --completions_jsonl ... --prompts_jsonl ... \\
        --noise_type uniform --noise_seed 42 \\
        --out_jsonl ...
"""
from __future__ import annotations

import argparse
import json
import random

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--completions_jsonl", required=True)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--stride", type=int, default=32)
    p.add_argument("--max_blocks_per_req", type=int, default=20)
    p.add_argument("--noise_type", choices=["uniform", "mask"], default="uniform")
    p.add_argument("--noise_seed", type=int, default=42)
    p.add_argument("--mask_token_id", type=int, default=151643)
    p.add_argument("--vocab_size", type=int, default=152064)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    print(f"[diag] loading {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()

    prompts = [json.loads(l) for l in open(args.prompts_jsonl)]
    completions = [json.loads(l) for l in open(args.completions_jsonl)]
    n = min(len(prompts), len(completions))

    rng = random.Random(args.noise_seed)

    out_fp = open(args.out_jsonl, "w")
    n_rows = 0

    for batch_idx in range(n):
        pdata = prompts[batch_idx]
        cdata = completions[batch_idx]
        chat = [{"role": "user", "content": pdata["input"]}]
        prompt_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(prompt_text, return_tensors="pt").input_ids[0].tolist()

        comp_text = cdata.get("completion", "")
        comp_ids = tok(comp_text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()

        if len(comp_ids) < args.K + 2:
            continue

        pl = len(prompt_ids)
        n_blocks = 0
        for block_start in range(0, len(comp_ids) - args.K - 1, args.stride):
            if n_blocks >= args.max_blocks_per_req:
                break
            n_blocks += 1

            # Generate noise (deterministic given seed + block index).
            sub_rng = random.Random(args.noise_seed * 1000003 + batch_idx * 10007 + block_start)
            if args.noise_type == "uniform":
                noise = [sub_rng.randrange(args.vocab_size) for _ in range(args.K)]
            else:
                noise = [args.mask_token_id] * args.K

            # Input: prompt + completion[:block_start] + NOISE_K
            inp_ids = prompt_ids + comp_ids[:block_start] + noise
            with torch.no_grad():
                inp_t = torch.tensor([inp_ids], dtype=torch.long, device="cuda:0")
                out = model(input_ids=inp_t)
                logits = out.logits[0]

            for j in range(args.K):
                idx = pl + block_start + j
                if idx >= logits.shape[0]:
                    break
                target_idx = block_start + j + 1
                if target_idx >= len(comp_ids):
                    break
                target = int(comp_ids[target_idx])
                log_p = F.log_softmax(logits[idx].float(), dim=-1)
                log_p_correct = float(log_p[target].item())
                top1 = int(log_p.argmax().item())
                row = {
                    "batch_idx": batch_idx,
                    "block_start": block_start,
                    "j": j,
                    "target": target,
                    "noise_at_j": noise[j],
                    "argmax_correct": int(top1 == target),
                    "log_p_correct": log_p_correct,
                    "top1_token": top1,
                }
                out_fp.write(json.dumps(row) + "\n")
                n_rows += 1

        if (batch_idx + 1) % 4 == 0:
            print(f"[diag] {batch_idx + 1}/{n} prompts, {n_rows} rows", flush=True)

    out_fp.close()
    print(f"[diag] DONE: wrote {n_rows} rows to {args.out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
