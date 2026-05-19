"""Diagnose where/why cons model's greedy decoding diverges from base.

For each prompt in a fixed set, runs greedy with full logit recording per
position. Saves a JSON of per-position (top_k_tokens, top_k_probs, entropy,
argmax_token) and the prompt's chat-templated input.

Usage:
  python scripts/diagnose_greedy_divergence.py \
    --model <hf-ckpt-path> --tag base \
    --prompt_indices 11,25,39,49,50,0,3,30,100,116 \
    --max_new_tokens 256 --top_k 5
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time

import numpy as np
import pyarrow.parquet as pq
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=None,
                   help="Defaults to model path (works for our HF-merged ckpts).")
    p.add_argument("--parquet", default="data/humanevalplus/val.parquet")
    p.add_argument("--tag", required=True)
    p.add_argument("--prompt_indices", required=True,
                   help="Comma-separated prompt indices to evaluate.")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--top_k", type=int, default=5)
    p.add_argument("--out_dir", default="eval_passk/diagnose_greedy")
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    indices = [int(x) for x in args.prompt_indices.split(",")]

    tok = AutoTokenizer.from_pretrained(args.tokenizer or args.model, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    eos_ids = {tok.eos_token_id, 151645, 151643}  # plus Qwen2.5 specials

    df = pq.read_table(args.parquet).to_pandas()

    out_rows = []
    for pi in indices:
        prompt_msg = df.iloc[pi]["prompt"]
        msgs = list(prompt_msg) if not isinstance(prompt_msg, str) \
            else [{"role": "user", "content": prompt_msg}]
        chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
        input_ids = tok(chat, return_tensors="pt").input_ids.cuda()

        pos_records = []
        cur = input_ids
        t0 = time.time()
        with torch.inference_mode():
            past = None
            for step in range(args.max_new_tokens):
                if past is None:
                    out = model(cur, use_cache=True)
                else:
                    out = model(cur[:, -1:], past_key_values=past, use_cache=True)
                past = out.past_key_values
                logits = out.logits[0, -1, :].float()
                probs = torch.softmax(logits, dim=-1)
                # Entropy of full distribution (nats)
                logp = torch.log_softmax(logits, dim=-1)
                entropy = float(-(probs * logp).sum().item())
                # Top-k
                topk = torch.topk(probs, k=args.top_k)
                topk_ids = topk.indices.cpu().tolist()
                topk_probs = topk.values.cpu().tolist()
                topk_tokens = [tok.decode([i]) for i in topk_ids]
                argmax_id = int(topk_ids[0])
                pos_records.append({
                    "step": step,
                    "argmax_id": argmax_id,
                    "argmax_token": topk_tokens[0],
                    "p_top1": float(topk_probs[0]),
                    "entropy": entropy,
                    "topk_ids": topk_ids,
                    "topk_probs": topk_probs,
                    "topk_tokens": topk_tokens,
                })
                cur = torch.cat([cur, torch.tensor([[argmax_id]], device=cur.device)], dim=1)
                if argmax_id in eos_ids:
                    break
        t1 = time.time()
        completion = tok.decode(cur[0, input_ids.shape[1]:].cpu().tolist(),
                                skip_special_tokens=False)
        print(f"  prompt {pi}: gen {len(pos_records)} tokens in {t1-t0:.1f}s", flush=True)
        out_rows.append({
            "prompt_idx": pi,
            "completion": completion,
            "num_tokens": len(pos_records),
            "positions": pos_records,
        })

    out_path = os.path.join(args.out_dir, f"{args.tag}.jsonl")
    with open(out_path, "w") as f:
        for row in out_rows:
            f.write(json.dumps(row) + "\n")
    print(f"\nWrote -> {out_path}", flush=True)


if __name__ == "__main__":
    main()
