"""Greedy HumanEval eval using HF transformers `model.generate()` — direct
replication of research/JacobiForcing/JacobiForcing/ar_inference_baseline.py.
No vLLM. Lets us check if vLLM is the source of our JF baseline gap vs paper.
"""

import argparse
import json
import os
import sys
import time

import pyarrow.parquet as pq
import torch
from tqdm import tqdm

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from reward_code_assert import _run_one  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--tokenizer", default=None,
                   help="Defaults to Qwen2.5-Coder-7B-Instruct (paper choice).")
    p.add_argument("--tag", required=True)
    p.add_argument("--val_file", default="data/humaneval/val.parquet")
    p.add_argument("--out_dir", default="eval_passk/greedy_hf")
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--timeout_s", type=int, default=15)
    args = p.parse_args()

    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer_path = args.tokenizer or (
        "/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--Qwen--"
        "Qwen2.5-Coder-7B-Instruct/snapshots/c03e6d358207e414f1eca0bb1891e29f1db0e242"
    )
    print(f"tokenizer: {tokenizer_path}\nmodel: {args.model}", flush=True)
    tokenizer = AutoTokenizer.from_pretrained(tokenizer_path, use_fast=True)
    model = AutoModelForCausalLM.from_pretrained(
        args.model,
        device_map="cuda",
        torch_dtype=torch.bfloat16,
        attn_implementation="flash_attention_2",
    )
    model.eval()
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token

    eos_id = tokenizer.eos_token_id
    alt_eos_id = 151645
    print(f"eos_id={eos_id} alt_eos_id={alt_eos_id}", flush=True)

    t = pq.read_table(args.val_file).to_pydict()
    n = len(t["prompt"])

    passes = 0
    results = []
    t0 = time.time()
    with torch.inference_mode():
        for i in tqdm(range(n)):
            text = tokenizer.apply_chat_template(
                t["prompt"][i], tokenize=False, add_generation_prompt=True
            )
            inp = tokenizer([text], return_tensors="pt").to(model.device)
            out_ids = model.generate(
                **inp,
                max_new_tokens=args.max_new_tokens,
                do_sample=False,
                temperature=0.0,
                top_p=1.0,
                eos_token_id=[eos_id, alt_eos_id],
                pad_token_id=tokenizer.pad_token_id,
                use_cache=True,
            )
            new_ids = out_ids[0, inp["input_ids"].shape[1]:]
            gen = tokenizer.decode(new_ids, skip_special_tokens=False)
            acc = _run_one((gen, t["reward_model"][i]["ground_truth"], args.timeout_s, None))
            passes += int(acc)
            results.append({"task_id": t["extra_info"][i]["task_id"], "acc": acc})

    elapsed = time.time() - t0
    print(f"\n=== {args.tag} ===  pass@1 = {passes}/{n} = {passes/n:.4f}  ({elapsed:.1f}s)", flush=True)
    os.makedirs(args.out_dir, exist_ok=True)
    out_path = os.path.join(args.out_dir, f"{args.tag}.json")
    with open(out_path, "w") as f:
        json.dump({"model": args.model, "tag": args.tag, "n": n, "pass": passes,
                   "pass_rate": passes / n, "per_task": results}, f, indent=2)
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
