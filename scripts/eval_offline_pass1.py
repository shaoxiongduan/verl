"""Offline pass-rate eval: takes a verl-formatted parquet (prompt + reward_model.ground_truth),
samples n completions per prompt with given (temperature, top_p), and scores each
completion with reward_code_assert._run_one. Reports mean@n (== avg of per-prompt
average pass) so it matches verl's val-time 'val-core/.../acc/mean@n' number.

Usage:
  python scripts/eval_offline_pass1.py \
    --model ckpts_hf/step_40_consistency \
    --parquet data/humanevalplus/val.parquet \
    --tag step40_cons_val_protocol \
    --temperature 1.0 --top_p 0.7 --n 8 --max_tokens 1024

Multiple configs in one job: --configs "greedy:t=0,p=1,n=1;val:t=1,p=0.7,n=8;train:t=1,p=1,n=8"
"""
from __future__ import annotations
import argparse
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward_code_assert import _run_one  # noqa: E402


def parse_configs(spec: str) -> list[dict]:
    """Format: 'name:t=T,p=P,n=N;...' or single 'name'."""
    out = []
    for item in spec.split(";"):
        item = item.strip()
        if not item:
            continue
        name, rest = item.split(":", 1)
        kv = dict(s.split("=") for s in rest.split(","))
        out.append({
            "name": name.strip(),
            "temperature": float(kv["t"]),
            "top_p": float(kv["p"]),
            "n": int(kv["n"]),
        })
    return out


def score_batch(completions_and_truths, timeout_s=15):
    """Parallel scoring via ProcessPoolExecutor.
    Each work unit: (solution_str, ground_truth_str, timeout_s, memory_mb)"""
    args_list = [(c, gt, timeout_s, None) for c, gt in completions_and_truths]
    with ProcessPoolExecutor(max_workers=16) as pool:
        scores = list(pool.map(_run_one, args_list))
    return scores


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--parquet", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--configs", default="",
                   help="'name1:t=T,p=P,n=N;name2:...'. Overrides --temperature/--top_p/--n.")
    p.add_argument("--temperature", type=float, default=1.0)
    p.add_argument("--top_p", type=float, default=1.0)
    p.add_argument("--n", type=int, default=1)
    p.add_argument("--max_tokens", type=int, default=1024)
    p.add_argument("--max_prompts", type=int, default=None,
                   help="Cap on number of prompts (for quick OCI subset eval).")
    p.add_argument("--out_dir", default="eval_passk/offline_pass1")
    p.add_argument("--timeout_s", type=int, default=15)
    p.add_argument("--gpu_mem_util", type=float, default=0.85)
    args = p.parse_args()

    os.makedirs(args.out_dir, exist_ok=True)

    if args.configs:
        configs = parse_configs(args.configs)
    else:
        configs = [{"name": "single", "temperature": args.temperature,
                    "top_p": args.top_p, "n": args.n}]

    # Load + render chat templates ONCE.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(args.model, use_fast=True)

    t = pq.read_table(args.parquet).to_pandas()
    if args.max_prompts:
        t = t.head(args.max_prompts).reset_index(drop=True)
    prompts_raw = t["prompt"].tolist()
    ground_truths = [r["ground_truth"] for r in t["reward_model"].tolist()]

    chat_texts = []
    for p_raw in prompts_raw:
        # verl parquets store prompt as list[{"role","content"}] OR as a raw string.
        if isinstance(p_raw, str):
            messages = [{"role": "user", "content": p_raw}]
        else:
            messages = list(p_raw)
        chat_texts.append(tok.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True))

    print(f"# prompts: {len(chat_texts)}", flush=True)
    print(f"# configs: {[c['name'] for c in configs]}", flush=True)

    from vllm import LLM, SamplingParams
    llm = LLM(
        model=args.model,
        tensor_parallel_size=1,
        gpu_memory_utilization=args.gpu_mem_util,
        max_model_len=8192,
        dtype="bfloat16",
        enforce_eager=True,
    )

    summary = []
    for cfg in configs:
        name = cfg["name"]
        temp = cfg["temperature"]
        top_p = cfg["top_p"]
        n = cfg["n"]
        print(f"\n==== config {name}: temp={temp} top_p={top_p} n={n} ====", flush=True)

        sp = SamplingParams(
            n=n,
            temperature=temp,
            top_p=top_p,
            max_tokens=args.max_tokens,
            stop_token_ids=[151645, 151643],  # Qwen2.5 <|im_end|>, <|endoftext|>
        )
        t0 = time.time()
        outs = llm.generate(chat_texts, sp)
        t1 = time.time()
        print(f"  gen time: {t1-t0:.1f}s for {len(outs)} prompts, n={n}", flush=True)

        # Pair every (sample, gt) for parallel scoring
        pairs = []
        idx_map = []
        for pi, out in enumerate(outs):
            gt = ground_truths[pi]
            for k, c in enumerate(out.outputs):
                pairs.append((c.text, gt))
                idx_map.append((pi, k))

        t0 = time.time()
        scores = score_batch(pairs, timeout_s=args.timeout_s)
        t1 = time.time()
        print(f"  score time: {t1-t0:.1f}s for {len(scores)} samples", flush=True)

        # Per-prompt mean (== pass@1 averaged across n samples == acc@n metric)
        per_prompt = [0.0] * len(outs)
        for (pi, _k), s in zip(idx_map, scores, strict=False):
            per_prompt[pi] += s / n
        mean_an = sum(per_prompt) / len(per_prompt)
        # pass@1 over all (n*P) attempts
        flat_p1 = sum(scores) / len(scores)
        # pass@n (any-correct): fraction of prompts with at least one pass
        per_prompt_any = []
        for pi in range(len(outs)):
            row = [scores[i] for i in range(len(scores)) if idx_map[i][0] == pi]
            per_prompt_any.append(1.0 if any(s == 1.0 for s in row) else 0.0)
        pass_n = sum(per_prompt_any) / len(per_prompt_any)

        print(f"  acc/mean@{n} = {mean_an:.4f}  flat_pass@1 = {flat_p1:.4f}  pass@{n} = {pass_n:.4f}", flush=True)

        out_path = os.path.join(args.out_dir, f"{args.tag}__{name}.jsonl")
        with open(out_path, "w") as f:
            for pi, out in enumerate(outs):
                row = {
                    "prompt_idx": pi,
                    "completions": [c.text for c in out.outputs],
                    "per_sample_pass": [scores[i] for i in range(len(scores)) if idx_map[i][0] == pi],
                    "mean_pass": per_prompt[pi],
                }
                f.write(json.dumps(row) + "\n")
        print(f"  wrote {out_path}", flush=True)
        summary.append({"name": name, "temperature": temp, "top_p": top_p, "n": n,
                        "acc_mean_at_n": mean_an, "flat_pass_at_1": flat_p1, "pass_at_n": pass_n})

    summary_path = os.path.join(args.out_dir, f"{args.tag}__summary.json")
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)
    print(f"\nSummary -> {summary_path}", flush=True)
    for s in summary:
        print(f"  {s['name']:20s}  t={s['temperature']}/p={s['top_p']}/n={s['n']}  "
              f"acc/mean@n={s['acc_mean_at_n']:.4f}  pass@n={s['pass_at_n']:.4f}")


if __name__ == "__main__":
    main()
