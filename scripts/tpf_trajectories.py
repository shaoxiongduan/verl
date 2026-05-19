"""Run Decode-Learning's nanovllm Jacobi inference on HumanEval+ and save
per-prompt trajectories: completion text + per-block TPF series + aggregate
TPF.  Used downstream for repetition analysis and TPF distribution plots.

Output JSONL, one row per prompt:
  { task_id, prompt, completion, num_tokens, num_forwards, tpf,
    tpf_per_block: [t0, t1, ...] }
"""

import argparse
import json
import os
import sys

# Ensure we import Decode-Learning's nanovllm (not any verl-side install).
DL_ROOT = "/mnt/weka/home/hao.zhang/shao/Decode-Learning"
sys.path.insert(0, DL_ROOT)
from nanovllm import LLM, SamplingParams  # noqa: E402


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--output_jsonl", required=True)
    p.add_argument("--max_new_tokens", type=int, default=1024)
    p.add_argument("--jacobi_block_len", type=int, default=32)
    p.add_argument("--jacobi_max_iterations", type=int, default=128)
    p.add_argument("--temperature", type=float, default=0.6)
    p.add_argument("--batch_size", type=int, default=16)
    args = p.parse_args()

    prompts = []
    with open(args.prompts_jsonl, "r") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            prompts.append(json.loads(line))
    print(f"Loaded {len(prompts)} prompts", flush=True)

    print(f"Loading model: {args.model}", flush=True)
    llm = LLM(model=args.model, tensor_parallel_size=1, max_model_len=4096)
    tokenizer = llm.tokenizer

    os.makedirs(os.path.dirname(args.output_jsonl) or ".", exist_ok=True)
    out_fp = open(args.output_jsonl, "w")

    sp = SamplingParams(
        temperature=args.temperature,
        max_tokens=args.max_new_tokens,
        decode_strategy="jacobi",
        jacobi_block_len=args.jacobi_block_len,
        jacobi_max_iterations=args.jacobi_max_iterations,
        jacobi_on_policy=True,
    )

    total = 0
    for start in range(0, len(prompts), args.batch_size):
        batch = prompts[start : start + args.batch_size]
        chat_texts = []
        for pdata in batch:
            chat_texts.append(
                tokenizer.apply_chat_template(
                    [{"role": "user", "content": pdata["input"]}],
                    tokenize=False,
                    add_generation_prompt=True,
                )
            )

        records = llm.generate(chat_texts, sp)

        for pdata, per_prompt in zip(batch, records, strict=False):
            # Each per_prompt is a dict of block_idx -> {tokens_per_forward, teacher_output_ids, ...}.
            # Block keys look like "itr_0", "itr_1", ..., "itr_31"; previous
            # implementation sorted by str (lex order: "itr_0", "itr_1",
            # "itr_10", "itr_2", ...) which caused the LAST visited block to
            # be itr_9 (or similar non-final block) for prompts with 11+
            # blocks. Since nanovllm stores `num_forwards` as a CUMULATIVE
            # counter, picking up the lex-last block underestimates the true
            # total num_forwards and thus OVERESTIMATES the per-prompt TPF.
            # Fix: sort numerically by extracting the integer block index, and
            # take max(num_forwards) (monotone so robust to ordering).
            def _block_idx(k):
                s = str(k)
                # "itr_5" -> 5; tolerate raw ints / odd keys.
                if "_" in s:
                    tail = s.rsplit("_", 1)[1]
                    if tail.isdigit():
                        return int(tail)
                try:
                    return int(s)
                except (TypeError, ValueError):
                    return -1

            tpf_series = []
            completion_ids = []
            num_forwards = 0
            num_tokens = 0
            for k in sorted(per_prompt.keys(), key=_block_idx):
                blk = per_prompt[k]
                if not isinstance(blk, dict):
                    continue
                if "tokens_per_forward" in blk:
                    tpf_series.append(float(blk["tokens_per_forward"]))
                if "num_forwards" in blk:
                    # Cumulative — max == final block's value; robust to order.
                    num_forwards = max(num_forwards, int(blk["num_forwards"]))
                if "teacher_output_ids" in blk and blk["teacher_output_ids"]:
                    # All blocks end up holding the final completion ids;
                    # take whichever non-empty entry.
                    completion_ids = blk["teacher_output_ids"]

            # `teacher_output_ids` is prompt + completion. Strip the prompt prefix.
            prompt_tok_ids = tokenizer(chat_texts[batch.index(pdata)]).input_ids
            if completion_ids and len(completion_ids) > len(prompt_tok_ids):
                comp_only = completion_ids[len(prompt_tok_ids):]
            else:
                comp_only = completion_ids
            completion_text = tokenizer.decode(comp_only, skip_special_tokens=False)
            num_tokens = len(comp_only)

            tpf_overall = (num_tokens / num_forwards) if num_forwards > 0 else 0.0

            row = {
                "task_id": pdata.get("id"),
                "prompt": pdata["input"],
                "completion": completion_text,
                "num_tokens": num_tokens,
                "num_forwards": num_forwards,
                "tpf": tpf_overall,
                "tpf_per_block": tpf_series,
            }
            out_fp.write(json.dumps(row) + "\n")
            total += 1
            print(f"  [{total}/{len(prompts)}] {row['task_id']}: tok={num_tokens} fw={num_forwards} tpf={tpf_overall:.2f}", flush=True)

        out_fp.flush()

    out_fp.close()
    print(f"\nWrote {total} rows -> {args.output_jsonl}", flush=True)
    llm.exit()


if __name__ == "__main__":
    main()
