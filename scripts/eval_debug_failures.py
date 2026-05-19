"""Dump raw greedy generations on specific HumanEval failures to diagnose
the gap vs the paper. Prints prompt + raw model output + decoded test result.
"""

import argparse
import json
import os
import sys

import pyarrow.parquet as pq

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward_code_assert import _run_one, extract_python  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--val_file", default="data/humaneval/val.parquet")
    p.add_argument("--task_ids", nargs="+", required=True,
                   help="e.g. HumanEval/1 HumanEval/10 ...")
    p.add_argument("--max_response_length", type=int, default=4096)
    p.add_argument("--stop_token_ids", nargs="+", type=int, default=None,
                   help="vLLM stop_token_ids. Try [151643, 151645].")
    args = p.parse_args()

    from vllm import LLM, SamplingParams

    t = pq.read_table(args.val_file).to_pydict()
    indices = []
    for tid in args.task_ids:
        for i in range(len(t["extra_info"])):
            if t["extra_info"][i]["task_id"] == tid:
                indices.append(i)
                break
    print(f"Found {len(indices)} matching task_ids", flush=True)
    chats = [t["prompt"][i] for i in indices]

    print(f"Loading {args.model} ...", flush=True)
    llm = LLM(model=args.model, tensor_parallel_size=1,
              gpu_memory_utilization=0.85,
              max_model_len=args.max_response_length + 2048,
              dtype="bfloat16", trust_remote_code=True)
    sp = SamplingParams(temperature=0.0, top_p=1.0,
                        max_tokens=args.max_response_length,
                        stop_token_ids=args.stop_token_ids)
    outs = llm.chat(chats, sp, use_tqdm=False)

    for i, out in zip(indices, outs):
        tid = t["extra_info"][i]["task_id"]
        gen = out.outputs[0].text
        finish_reason = out.outputs[0].finish_reason
        last_token = out.outputs[0].token_ids[-1] if out.outputs[0].token_ids else None
        gt = t["reward_model"][i]["ground_truth"]
        acc = _run_one((gen, gt, 15, None))
        code = extract_python(gen)

        print("=" * 80)
        print(f"== {tid}    finish={finish_reason}  last_token={last_token}  n_tokens={len(out.outputs[0].token_ids)}  acc={acc}")
        print("=" * 80)
        print(">>> CODE EXTRACTED (first 400 chars):")
        print(code[:400])
        print("---")
        print(">>> RAW GENERATION TAIL (last 400 chars):")
        print(gen[-400:])
        print()


if __name__ == "__main__":
    main()
