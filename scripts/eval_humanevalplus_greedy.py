"""Greedy HumanEval+ eval for a single HF model directory using vLLM.

Usage:
  CUDA_VISIBLE_DEVICES=4,5,6,7 python scripts/eval_humanevalplus_greedy.py \
      --model <hf_dir_or_hub_id> --tag step_200 [--tp 1]

Outputs:
  eval_passk/greedy/<tag>.json   { "n": 164, "pass": int, "pass_rate": float }
"""

import argparse
import json
import os
import pathlib
import sys
import time

import pyarrow.parquet as pq

# Reuse the same code-execution reward used during RL.
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward_code_assert import _run_one  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tag", required=True)
    p.add_argument("--val_file", default="data/humanevalplus/val.parquet")
    p.add_argument("--out_dir", default="eval_passk/greedy")
    p.add_argument("--tp", type=int, default=1)
    p.add_argument("--max_response_length", type=int, default=4096)
    p.add_argument("--timeout_s", type=int, default=15)
    p.add_argument("--frequency_penalty", type=float, default=0.0)
    p.add_argument("--repetition_penalty", type=float, default=1.0)
    args = p.parse_args()

    from vllm import LLM, SamplingParams

    t = pq.read_table(args.val_file).to_pydict()
    n = len(t["prompt"])
    print(f"Loaded {n} HumanEval+ problems from {args.val_file}", flush=True)

    # vLLM expects pre-templated text. We have chat-format prompts; let vLLM
    # apply the chat template via `chat()` API.
    chats = [t["prompt"][i] for i in range(n)]  # already list[{role, content}]

    print(f"Loading vLLM model {args.model} (tp={args.tp}) ...", flush=True)
    t0 = time.time()
    llm = LLM(
        model=args.model,
        tensor_parallel_size=args.tp,
        gpu_memory_utilization=0.85,
        max_model_len=args.max_response_length + 2048,
        dtype="bfloat16",
        trust_remote_code=True,
    )
    print(f"Model loaded in {time.time() - t0:.1f}s", flush=True)

    sampling = SamplingParams(
        temperature=0.0,       # greedy
        top_p=1.0,
        max_tokens=args.max_response_length,
        frequency_penalty=args.frequency_penalty,
        repetition_penalty=args.repetition_penalty,
    )

    print("Generating ...", flush=True)
    t1 = time.time()
    outputs = llm.chat(chats, sampling, use_tqdm=True)
    print(f"Generated {len(outputs)} responses in {time.time() - t1:.1f}s", flush=True)

    print("Scoring ...", flush=True)
    t2 = time.time()
    results = []
    passes = 0
    for i, out in enumerate(outputs):
        sol = out.outputs[0].text
        gt = t["reward_model"][i]["ground_truth"]
        acc = _run_one((sol, gt, args.timeout_s, None))
        passes += int(acc)
        results.append({"task_id": t["extra_info"][i]["task_id"], "acc": acc})
    elapsed = time.time() - t2

    pathlib.Path(args.out_dir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.tag}.json")
    payload = {
        "model": args.model,
        "tag": args.tag,
        "n": n,
        "pass": passes,
        "pass_rate": passes / n,
        "scoring_seconds": elapsed,
        "per_task": results,
    }
    with open(out_path, "w") as f:
        json.dump(payload, f, indent=2)
    print(f"\n=== {args.tag} ===")
    print(f"pass@1 (greedy) = {passes}/{n} = {passes/n:.4f}")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
