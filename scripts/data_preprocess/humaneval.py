"""Preprocess openai/openai_humaneval (plain HumanEval) to verl parquet.

Same schema as humanevalplus.py — only difference is the source dataset has
fewer/simpler tests per task. Lets us reproduce the JF paper's 83.5 number
on the JF base and compare to our RL'd checkpoint.
"""

import argparse
import json
import os

from datasets import Dataset, load_dataset

# Verbatim prompt from the JacobiForcing paper's HumanEval eval script
# (research/JacobiForcing/JacobiForcing/jacobi_forcing_inference_humaneval.py).
# Using the paper's prompt closes the JF baseline gap we saw with a generic
# instruction; relative gains over JF baseline don't depend on this choice
# but absolute numbers do.
PROMPT_TEMPLATE = (
    "Please continue to complete the function. You are not allowed to modify "
    "the given code and do the completion only. Please return all completed "
    "function in a codeblock. Here is the given code to do completion:\n"
    "```python\n{prompt}\n```"
)


def build(example, idx):
    prompt = example["prompt"].strip()
    test = example["test"]
    entry_point = example["entry_point"]
    test_code = test + f"\n\ncheck({entry_point})\n"
    return {
        "data_source": "humaneval",
        "prompt": [{"role": "user", "content": PROMPT_TEMPLATE.format(prompt=prompt)}],
        "ability": "code",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps(
                {"test_code": test_code, "entry_point": entry_point, "language": "python"}
            ),
        },
        "extra_info": {"split": "test", "index": idx, "task_id": example["task_id"]},
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/mnt/weka/home/hao.zhang/shao/verl/data/humaneval")
    parser.add_argument("--hf_name", default="openai/openai_humaneval")
    parser.add_argument("--hf_split", default="test")
    args = parser.parse_args()

    ds = load_dataset(args.hf_name, split=args.hf_split)
    print(f"Loaded {len(ds)} rows; columns: {ds.column_names}")

    rows = [build(ds[i], i) for i in range(len(ds))]
    os.makedirs(args.local_dir, exist_ok=True)
    out = os.path.join(args.local_dir, "val.parquet")
    Dataset.from_list(rows).to_parquet(out)
    print(f"Wrote {len(rows)} rows -> {out}")


if __name__ == "__main__":
    main()
