"""Diagnose RL distribution-shift signatures (entropy collapse, JS divergence).

D1: per-token entropy curve as the model decodes a prompt greedily.
D3: per-token JS divergence between two checkpoints on the same prompts.

Outputs a JSONL with per-prompt per-position records {step, entropy, top1_id, top1_p}.
For D3 we re-run a second model on the SAME completion (force-decoding the first
model's outputs) so the per-position distributions are directly comparable.

Usage:
  # D1 only
  python scripts/diagnose_distribution.py --model PATH --prompts PROMPTS.jsonl \
    --output OUT.jsonl --n_prompts 50 --max_new_tokens 256

  # D1 + D3 (force-decoded second model)
  python scripts/diagnose_distribution.py --model PATH --model2 PATH2 \
    --prompts PROMPTS.jsonl --output OUT.jsonl --n_prompts 50

Output format (one row per prompt):
  {
    "task_id": ...,
    "prompt_len": int,
    "completion_ids": [int...],            # greedy decode from model
    "entropy_per_step": [float, ...],      # model's per-step entropy over completion
    "top1_p_per_step": [float, ...],
    "entropy2_per_step": [float, ...]?,     # model2's per-step entropy (force-decoded)
    "js_per_step": [float, ...]?,           # JS(model, model2) per step
  }
"""
from __future__ import annotations
import argparse
import json
import math
import os
import sys
import time

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def js_divergence(p_logits: torch.Tensor, q_logits: torch.Tensor) -> float:
    """Jensen-Shannon divergence between two distributions defined by logits.
    Uses log-space arithmetic to avoid numerical issues with sharp distributions.
    Returns a scalar (one value for the full vocab distribution at one position).
    """
    log_p = F.log_softmax(p_logits.float(), dim=-1)
    log_q = F.log_softmax(q_logits.float(), dim=-1)
    log_m = torch.logsumexp(torch.stack([log_p, log_q], dim=0), dim=0) - math.log(2.0)
    # KL(p || m) + KL(q || m), divided by 2
    p = log_p.exp()
    q = log_q.exp()
    kl_pm = (p * (log_p - log_m)).sum(-1)
    kl_qm = (q * (log_q - log_m)).sum(-1)
    return float(0.5 * (kl_pm + kl_qm))


def entropy_top1(logits: torch.Tensor) -> tuple[float, int, float]:
    """Return (entropy, argmax_id, top1_prob) for a single logit vector."""
    logp = F.log_softmax(logits.float(), dim=-1)
    p = logp.exp()
    H = float(-(p * logp).sum())
    top1_id = int(p.argmax())
    top1_p = float(p[top1_id])
    return H, top1_id, top1_p


def load_prompts(path: str, n: int, tok) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
            if len(rows) >= n:
                break
    return rows


def chat_input_ids(tok, prompt_text: str, device):
    msgs = [{"role": "user", "content": prompt_text}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(chat, return_tensors="pt").input_ids.to(device)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--model2", default=None, help="If set, force-decode this on model's output for JS.")
    p.add_argument("--prompts", required=True)
    p.add_argument("--output", required=True)
    p.add_argument("--n_prompts", type=int, default=50)
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--dtype", default="bfloat16")
    args = p.parse_args()

    dev = "cuda"
    dtype = getattr(torch, args.dtype)

    print(f"Loading tokenizer ({args.model}) ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token

    print(f"Loading model 1 ({args.model}) ...", flush=True)
    m1 = AutoModelForCausalLM.from_pretrained(
        args.model, device_map=dev, torch_dtype=dtype,
        attn_implementation="flash_attention_2",
    ).eval()
    m2 = None
    if args.model2:
        print(f"Loading model 2 ({args.model2}) ...", flush=True)
        m2 = AutoModelForCausalLM.from_pretrained(
            args.model2, device_map=dev, torch_dtype=dtype,
            attn_implementation="flash_attention_2",
        ).eval()

    prompts = load_prompts(args.prompts, args.n_prompts, tok)
    print(f"Diagnosing on {len(prompts)} prompts, max_new={args.max_new_tokens}", flush=True)

    eos_ids = {tok.eos_token_id, 151645, 151643}  # Qwen2.5 EOS/IM_END/EOT

    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    fout = open(args.output, "w")

    t0 = time.time()
    for pi, row in enumerate(prompts):
        prompt_text = row["input"] if "input" in row else row.get("prompt", row.get("question", ""))
        task_id = row.get("task_id", row.get("id", f"p{pi}"))
        ids = chat_input_ids(tok, prompt_text, dev)
        prompt_len = ids.size(1)

        entropy_steps: list[float] = []
        top1p_steps: list[float] = []
        completion_ids: list[int] = []
        entropy2_steps: list[float] = []
        js_steps: list[float] = []

        # Greedy decode with m1, recording per-step entropy + top1
        past = None
        cur = ids
        with torch.inference_mode():
            for step in range(args.max_new_tokens):
                if past is None:
                    out = m1(cur, use_cache=True)
                else:
                    out = m1(cur[:, -1:], past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits1 = out.logits[0, -1, :]
                H, top1, p1 = entropy_top1(logits1)
                entropy_steps.append(H)
                top1p_steps.append(p1)
                completion_ids.append(top1)
                if m2 is not None:
                    # Force-decode m2 on m1's greedy output. To do this efficiently
                    # without per-step KV cache rebuild on m2, we just run m2 on the
                    # full sequence at the end; see below.
                    pass
                cur = torch.cat([cur, torch.tensor([[top1]], device=dev)], dim=1)
                if top1 in eos_ids:
                    break

        # If m2 given, run it on the full (prompt + m1_completion) and pull
        # per-position logits over the completion span.
        if m2 is not None and completion_ids:
            full = cur  # already has prompt + completion
            with torch.inference_mode():
                out2 = m2(full, use_cache=False)
                logits2 = out2.logits[0]  # [seq, vocab]
                # Position i predicts token at position i+1. So logits at
                # position prompt_len-1 ... prompt_len-1+len(completion)-1
                # predict the completion tokens.
                start = prompt_len - 1
                # Re-walk and grab the same m1 logits we already have via a second
                # pass — simpler: just do an m1 pass on the full sequence to get
                # m1's "force-decoded" per-position logits, matched in indexing.
                out1_full = m1(full, use_cache=False)
                logits1_full = out1_full.logits[0]
                for j, _tok in enumerate(completion_ids):
                    pos = start + j
                    l1 = logits1_full[pos]
                    l2 = logits2[pos]
                    H2, _, _ = entropy_top1(l2)
                    entropy2_steps.append(H2)
                    js_steps.append(js_divergence(l1, l2))

        rec = {
            "task_id": str(task_id),
            "prompt_len": prompt_len,
            "completion_ids": completion_ids,
            "entropy_per_step": entropy_steps,
            "top1_p_per_step": top1p_steps,
        }
        if entropy2_steps:
            rec["entropy2_per_step"] = entropy2_steps
            rec["js_per_step"] = js_steps
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        if (pi + 1) % 5 == 0 or pi == 0:
            mean_H = sum(entropy_steps) / max(1, len(entropy_steps))
            mean_p1 = sum(top1p_steps) / max(1, len(top1p_steps))
            extra = ""
            if js_steps:
                extra = f"  JS_mean={sum(js_steps)/len(js_steps):.4f}"
            print(f"  [{pi+1}/{len(prompts)}] {task_id}: {len(entropy_steps)} tok  "
                  f"H_mean={mean_H:.3f}  top1_mean={mean_p1:.3f}{extra}  "
                  f"({time.time()-t0:.1f}s)", flush=True)

    fout.close()
    print(f"Wrote {len(prompts)} rows to {args.output}", flush=True)


if __name__ == "__main__":
    main()
