"""Diagnostic: clean-draft verifier accuracy per K-block position.

For each (prompt, completion) pair, this slides a K-token window across the
completion. At each block start, it feeds `[prompt | completion[:start] |
completion[start:start+K]]` through the HF model with default causal attention
and records, for each within-block position j:

  - argmax_correct: does argmax(logit_at_pl+start+j) == completion[start+j+1]?
  - log_p_correct:  log_softmax(...)[completion[start+j+1]]
  - top1_token / top1_logp

The "draft" here IS the actual completion. So this measures, for THIS exact
completion: how often does the model's next-token prediction match the next
real token in the completion, given the prior K-1 tokens are already in
context? That is the exact criterion vLLM Jacobi spec-decode uses to ACCEPT
each draft token. Per-block aggregate "accepted prefix length" ≈ what TPF
measures.

Run twice (base vs RL'd ckpt) with the same `--completions_jsonl` so both
models are tested on identical drafts. The diff isolates where cons-RL changes
the verifier's prediction.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/_diag_logits_clean_draft.py \\
        --model PATH \\
        --completions_jsonl eval_passk/diag_traces/base_ds16/completions.jsonl \\
        --prompts_jsonl eval_passk/deepscaler_tpf_prompts_16.jsonl \\
        --out_jsonl eval_passk/diag_traces/base_ds16/clean__ckpt.jsonl
"""
from __future__ import annotations

import argparse
import json

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
    p.add_argument("--stride", type=int, default=32,
                   help="Block-start stride. K = non-overlapping windows.")
    p.add_argument("--max_blocks_per_req", type=int, default=20,
                   help="Cap blocks-per-request to bound runtime.")
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
            # Input: prompt + completion[:block_start + K]
            # Predicting positions [pl + block_start ... pl + block_start + K - 1] -> next token
            inp_ids = prompt_ids + comp_ids[: block_start + args.K]
            with torch.no_grad():
                inp_t = torch.tensor([inp_ids], dtype=torch.long, device="cuda:0")
                out = model(input_ids=inp_t)
                logits = out.logits[0]  # (L, V)

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
                top1_lp = float(log_p[top1].item())
                row = {
                    "batch_idx": batch_idx,
                    "block_start": block_start,
                    "j": j,
                    "target": target,
                    "argmax_correct": int(top1 == target),
                    "log_p_correct": log_p_correct,
                    "top1_token": top1,
                    "top1_logp": top1_lp,
                }
                out_fp.write(json.dumps(row) + "\n")
                n_rows += 1

        if (batch_idx + 1) % 4 == 0:
            print(f"[diag] {batch_idx + 1}/{n} prompts, {n_rows} rows", flush=True)

    out_fp.close()
    print(f"[diag] DONE: wrote {n_rows} rows to {args.out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
