"""Branching correctness test: are high-entropy positions correctness-relevant?

Procedure (per prompt, per model):
  1. Greedy decode, record per-position entropy and top-2 tokens.
  2. Find top-K highest-entropy positions in the completion.
  3. For each such position P:
     a. Build the alt-prefix: prompt + greedy[:P] + top2[P]   (force the alt token at P).
     b. Greedy-continue from there with the same model.
     c. Score alt completion via the prompt's assertion tests.
     d. Score original greedy completion (once per prompt).
  4. Classify each fork as:
       - trivial  : original and alt have same correctness outcome (both pass or both fail)
       - decision : different outcomes (top-2 token matters)

Output: jsonl + summary stats.

Usage:
  python scripts/branching_correctness_test.py \
    --model PATH --tag TAG --parquet PATH \
    --n_prompts 50 --top_k_positions 5 --max_new_tokens 256
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from typing import List, Tuple

import torch
import torch.nn.functional as F
import pyarrow.parquet as pq
from transformers import AutoModelForCausalLM, AutoTokenizer

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward_code_assert import _run_one  # noqa: E402


def chat_input_ids(tok, prompt_text: str, device):
    msgs = [{"role": "user", "content": prompt_text}]
    chat = tok.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    return tok(chat, return_tensors="pt").input_ids.to(device)


@torch.inference_mode()
def greedy_decode_with_top2(model, ids, max_new_tokens, eos_ids):
    """Decode greedily, record per-step entropy and top-2 token ids.

    Returns:
      completion_ids: list[int] of greedily decoded tokens
      entropies: list[float] per step
      top2: list[(top1_id, top2_id)] per step
    """
    past = None
    cur = ids
    completion = []
    entropies = []
    top2_list = []
    for step in range(max_new_tokens):
        if past is None:
            out = model(cur, use_cache=True)
        else:
            out = model(cur[:, -1:], past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1, :].float()
        logp = F.log_softmax(logits, dim=-1)
        p = logp.exp()
        H = float(-(p * logp).sum())
        topk = torch.topk(p, k=2)
        top1 = int(topk.indices[0])
        top2 = int(topk.indices[1])
        entropies.append(H)
        top2_list.append((top1, top2))
        completion.append(top1)
        cur = torch.cat([cur, torch.tensor([[top1]], device=ids.device)], dim=1)
        if top1 in eos_ids:
            break
    return completion, entropies, top2_list


@torch.inference_mode()
def greedy_continue(model, ids, max_new_tokens, eos_ids):
    """Standard greedy continuation from given ids. Returns list of new tokens."""
    past = None
    cur = ids
    new_ids = []
    for step in range(max_new_tokens):
        if past is None:
            out = model(cur, use_cache=True)
        else:
            out = model(cur[:, -1:], past_key_values=past, use_cache=True)
        past = out.past_key_values
        logits = out.logits[0, -1, :]
        top1 = int(logits.argmax())
        new_ids.append(top1)
        cur = torch.cat([cur, torch.tensor([[top1]], device=ids.device)], dim=1)
        if top1 in eos_ids:
            break
    return new_ids


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--parquet", required=True,
                   help="verl-format parquet with prompt + reward_model.ground_truth")
    p.add_argument("--n_prompts", type=int, default=50)
    p.add_argument("--top_k_positions", type=int, default=5,
                   help="Branch at this many highest-entropy positions per prompt")
    p.add_argument("--max_new_tokens", type=int, default=256)
    p.add_argument("--out_dir", default="/mnt/weka/home/hao.zhang/shao/verl/eval_passk/branching_test")
    p.add_argument("--timeout_s", type=int, default=15)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.tag}.jsonl")
    summary_path = os.path.join(args.out_dir, f"{args.tag}__summary.json")

    print(f"Loading {args.model} ...", flush=True)
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)
    if tok.pad_token_id is None:
        tok.pad_token = tok.eos_token
    model = AutoModelForCausalLM.from_pretrained(
        args.model, device_map="cuda", torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    ).eval()
    eos_ids = {tok.eos_token_id, 151645, 151643}

    t = pq.read_table(args.parquet).to_pandas()
    n = min(args.n_prompts, len(t))
    print(f"Diagnosing {n} prompts, top_k_positions={args.top_k_positions}", flush=True)

    fout = open(out_path, "w")
    totals = {"trivial_pass_pass": 0, "trivial_fail_fail": 0,
              "decision_orig_pass_alt_fail": 0, "decision_orig_fail_alt_pass": 0,
              "total_forks": 0, "n_prompts": 0,
              "n_greedy_pass": 0}

    t0 = time.time()
    for pi in range(n):
        row = t.iloc[pi]
        prompt_msgs = row["prompt"]
        if hasattr(prompt_msgs, "tolist"):
            prompt_msgs = prompt_msgs.tolist()
        prompt_text = prompt_msgs[0]["content"] if isinstance(prompt_msgs, list) else str(prompt_msgs)
        gt = row["reward_model"]["ground_truth"]

        # Greedy decode + record top-2
        ids = chat_input_ids(tok, prompt_text, "cuda")
        prompt_len = ids.size(1)
        completion, entropies, top2 = greedy_decode_with_top2(model, ids, args.max_new_tokens, eos_ids)
        greedy_text = tok.decode(completion, skip_special_tokens=False)

        # Score greedy
        greedy_score = _run_one((greedy_text, gt, args.timeout_s, None))
        if greedy_score >= 1.0:
            totals["n_greedy_pass"] += 1

        # Find top-K highest entropy positions (within actual completion length)
        n_tokens = len(entropies)
        if n_tokens == 0:
            continue
        # Sort positions by entropy descending; only consider positions where top-2 differs from top-1
        pos_by_H = sorted(range(n_tokens), key=lambda i: -entropies[i])
        chosen = []
        for pos in pos_by_H:
            if entropies[pos] < 0.01:  # below this threshold, the fork is essentially a tie at top-1
                break
            if top2[pos][1] == top2[pos][0]:
                continue
            chosen.append(pos)
            if len(chosen) >= args.top_k_positions:
                break

        forks = []
        for pos in chosen:
            top1_id, top2_id = top2[pos]
            # Build alt prefix: prompt + completion[:pos] + top2_id
            alt_prefix_ids = ids.clone()
            if pos > 0:
                alt_prefix_ids = torch.cat([alt_prefix_ids,
                                             torch.tensor([completion[:pos]], device=ids.device)], dim=1)
            alt_prefix_ids = torch.cat([alt_prefix_ids,
                                         torch.tensor([[top2_id]], device=ids.device)], dim=1)
            # Continue greedy
            new_max = args.max_new_tokens - pos - 1
            new_max = max(1, new_max)
            alt_continue = greedy_continue(model, alt_prefix_ids, new_max, eos_ids)
            alt_completion = completion[:pos] + [top2_id] + alt_continue
            alt_text = tok.decode(alt_completion, skip_special_tokens=False)
            alt_score = _run_one((alt_text, gt, args.timeout_s, None))

            forks.append({
                "position": pos,
                "entropy": entropies[pos],
                "top1": top1_id,
                "top2": top2_id,
                "top1_text": tok.decode([top1_id]),
                "top2_text": tok.decode([top2_id]),
                "greedy_score": greedy_score,
                "alt_score": alt_score,
                "alt_len": len(alt_completion),
            })
            totals["total_forks"] += 1
            if greedy_score == alt_score:
                if greedy_score >= 1.0:
                    totals["trivial_pass_pass"] += 1
                else:
                    totals["trivial_fail_fail"] += 1
            else:
                if greedy_score >= 1.0:
                    totals["decision_orig_pass_alt_fail"] += 1
                else:
                    totals["decision_orig_fail_alt_pass"] += 1

        totals["n_prompts"] += 1
        rec = {
            "prompt_idx": pi,
            "task_id": str(row.get("extra_info", {}).get("task_id", pi)) if hasattr(row, "get") else str(pi),
            "greedy_completion_len": len(completion),
            "greedy_score": greedy_score,
            "forks": forks,
        }
        fout.write(json.dumps(rec) + "\n")
        fout.flush()
        if (pi + 1) % 5 == 0 or pi == 0:
            tot = totals["total_forks"]
            triv = totals["trivial_pass_pass"] + totals["trivial_fail_fail"]
            dec = totals["decision_orig_pass_alt_fail"] + totals["decision_orig_fail_alt_pass"]
            print(f"  [{pi+1}/{n}] forks_so_far={tot}  trivial={triv}/{tot} ({100*triv/max(1,tot):.1f}%)  "
                  f"decision={dec}/{tot} ({100*dec/max(1,tot):.1f}%)  ({time.time()-t0:.1f}s)", flush=True)

    fout.close()
    summary = {
        "tag": args.tag,
        "model": args.model,
        "n_prompts": totals["n_prompts"],
        "n_greedy_pass": totals["n_greedy_pass"],
        "greedy_pass_rate": totals["n_greedy_pass"] / max(1, totals["n_prompts"]),
        "total_forks": totals["total_forks"],
        "trivial_pass_pass": totals["trivial_pass_pass"],
        "trivial_fail_fail": totals["trivial_fail_fail"],
        "decision_orig_pass_alt_fail": totals["decision_orig_pass_alt_fail"],
        "decision_orig_fail_alt_pass": totals["decision_orig_fail_alt_pass"],
        "trivial_frac": (totals["trivial_pass_pass"] + totals["trivial_fail_fail"]) / max(1, totals["total_forks"]),
        "decision_frac": (totals["decision_orig_pass_alt_fail"] + totals["decision_orig_fail_alt_pass"]) / max(1, totals["total_forks"]),
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary -> {summary_path}", flush=True)
    print(json.dumps(summary, indent=2), flush=True)


if __name__ == "__main__":
    main()
