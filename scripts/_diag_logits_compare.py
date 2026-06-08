"""Diagnostic: compare base vs RL'd model logits on identical Jacobi-rollout traces.

For each saved cascade trajectory (per-iter JSONL from vllm_jacobi_patch), this
script reconstructs the verifier input each Jacobi iteration sees — namely
`[prompt | response[:start] | draft_K_tokens]` with default causal attention —
forwards a single HF model, and records per-position logit diagnostics:

  - argmax_correct: argmax(logit_i) == response[start + i + 1]
  - log_p_correct:  log_softmax(logit_i)[response[start + i + 1]]
  - top1_token:     argmax(logit_i)
  - top1_logp:      max(log_softmax(logit_i))

Output JSONL has one row per (prompt, iter, position). Run twice (once per
ckpt) then diff. Position 0..K-1 in the noisy block; first n_acc positions are
ones the cascade already committed (model sees clean draft there), positions
[n_acc, K) are the "truly noisy" ones the cons loss targets.

Usage:
    CUDA_VISIBLE_DEVICES=0 python3 scripts/_diag_logits_compare.py \\
        --model /path/to/hf_ckpt \\
        --prompts_jsonl eval_passk/deepscaler_tpf_prompts_16.jsonl \\
        --completions_jsonl eval_passk/diag_traces/base_ds16/completions.jsonl \\
        --traj_glob 'eval_passk/diag_traces/base_ds16/traj.jsonl.*' \\
        --out_jsonl eval_passk/diag_traces/base_ds16/logits__base.jsonl
"""
from __future__ import annotations

import argparse
import glob
import json
import sys
from pathlib import Path

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer


def load_traj_files(pattern: str) -> dict[str, list[dict]]:
    """Per-PID JSONL → dict[req_id] -> sorted list of per-iter records."""
    per_req: dict[str, list[dict]] = {}
    for p in sorted(glob.glob(pattern)):
        with open(p) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                rid = str(rec.get("req_id", rec.get("request_id", "")))
                per_req.setdefault(rid, []).append(rec)
    for rid in per_req:
        per_req[rid].sort(key=lambda r: int(r.get("iter", 0)))
    return per_req


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--completions_jsonl", required=True)
    p.add_argument("--traj_glob", required=True)
    p.add_argument("--out_jsonl", required=True)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--max_iters_per_req", type=int, default=8,
                   help="Cap per-request iters sampled (across the trajectory) to control runtime.")
    return p.parse_args()


def main() -> None:
    args = parse_args()

    print(f"[diag] loading tokenizer + model from {args.model}", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(
        args.model, dtype=torch.bfloat16, device_map="cuda:0"
    )
    model.eval()

    prompts = [json.loads(l) for l in open(args.prompts_jsonl)]
    completions = [json.loads(l) for l in open(args.completions_jsonl)]
    if len(prompts) != len(completions):
        print(f"[diag] WARN prompts={len(prompts)} completions={len(completions)}", flush=True)

    # Group trajectories by req_id (the order vllm_jacobi_patch assigns is
    # batch-slot index, which is also prompt order).
    per_req = load_traj_files(args.traj_glob)
    rid_keys = sorted(per_req.keys(), key=lambda x: int(x) if x.isdigit() else x)
    print(f"[diag] loaded {len(per_req)} req trajectories from {args.traj_glob}", flush=True)

    out_fp = open(args.out_jsonl, "w")
    n_rows = 0

    for batch_idx, (pdata, cdata) in enumerate(zip(prompts, completions, strict=False)):
        if batch_idx >= len(rid_keys):
            break
        rid = rid_keys[batch_idx]
        recs = per_req[rid]
        chat = [{"role": "user", "content": pdata["input"]}]
        prompt_text = tok.apply_chat_template(chat, tokenize=False, add_generation_prompt=True)
        prompt_ids = tok(prompt_text, return_tensors="pt").input_ids[0].tolist()

        # Re-tokenize completion to get response_ids aligned to model vocab.
        comp_text = cdata.get("completion", "")
        resp_ids = tok(comp_text, return_tensors="pt", add_special_tokens=False).input_ids[0].tolist()

        # Reconstruct per-iter start position: pos advances by n_acc+1.
        pos = 0
        iter_idxs = list(range(len(recs)))
        # Subsample if too many.
        if len(iter_idxs) > args.max_iters_per_req:
            stride = len(iter_idxs) / args.max_iters_per_req
            iter_idxs = [int(i * stride) for i in range(args.max_iters_per_req)]
        # Need to walk all iters to track `pos`, but only forward sampled ones.
        for i_rec, rec in enumerate(recs):
            try:
                K = int(rec["num_draft"])
                n_acc = int(rec.get("n_acc", 0))
                draft = list(rec["draft"])
                if len(draft) != K:
                    pos += n_acc + 1
                    continue
            except (KeyError, TypeError, ValueError):
                pos += n_acc + 1 if 'n_acc' in dir() else 0
                continue

            if i_rec in iter_idxs and pos + K <= len(resp_ids):
                # Build verifier input: prompt + resp[:pos] + draft_K
                inp_ids = prompt_ids + resp_ids[:pos] + draft
                with torch.no_grad():
                    inp_t = torch.tensor([inp_ids], dtype=torch.long, device="cuda:0")
                    out = model(inp_ids=inp_t) if False else model(input_ids=inp_t)
                    logits = out.logits[0]  # (L, V)
                # Positions of the K draft tokens in the packed sequence:
                # prompt_len + pos + 0 ... prompt_len + pos + K-1
                # Logit at pos j predicts token at j+1.
                pl = len(prompt_ids)
                for j_local in range(K):
                    j_packed = pl + pos + j_local
                    if j_packed >= logits.shape[0] - 1:
                        break
                    log_p = F.log_softmax(logits[j_packed].float(), dim=-1)
                    # Target: next token in the response after position pos + j_local.
                    target_pos = pos + j_local
                    if target_pos >= len(resp_ids):
                        break
                    target_id = int(resp_ids[target_pos])
                    log_p_correct = float(log_p[target_id].item())
                    top1 = int(log_p.argmax().item())
                    top1_lp = float(log_p[top1].item())
                    row = {
                        "batch_idx": batch_idx,
                        "iter": int(rec.get("iter", i_rec)),
                        "pos_in_resp": int(pos + j_local),
                        "j_in_block": int(j_local),
                        "n_acc": n_acc,
                        "K": K,
                        "is_noisy": j_local >= n_acc,
                        "target": target_id,
                        "draft_at_j": int(draft[j_local]),
                        "argmax_correct": int(top1 == target_id),
                        "log_p_correct": log_p_correct,
                        "top1_token": top1,
                        "top1_logp": top1_lp,
                    }
                    out_fp.write(json.dumps(row) + "\n")
                    n_rows += 1
            pos += n_acc + 1
        if (batch_idx + 1) % 4 == 0:
            print(f"[diag] processed {batch_idx + 1}/{len(prompts)} reqs, {n_rows} rows", flush=True)

    out_fp.close()
    print(f"[diag] DONE: wrote {n_rows} rows to {args.out_jsonl}", flush=True)


if __name__ == "__main__":
    main()
