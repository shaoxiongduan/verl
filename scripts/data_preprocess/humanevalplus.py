"""Preprocess evalplus/humanevalplus to verl parquet for coder RL evaluation.

HumanEval+ is the dataset Jacobi Forcing reports its coder numbers on
(alongside MBPP). Each row produces a single eval example:

    {
      "data_source": "humanevalplus",
      "prompt":       [{"role": "user", "content": <function-signature + docstring>}],
      "ability":      "code",
      "reward_model": {"style": "rule",
                       "ground_truth": json.dumps({"test_code": "<harness>",
                                                   "entry_point": "<fn name>",
                                                   "language": "python"})},
      "extra_info":   {"split": "test", "task_id": ...},
    }

Routes through the same reward_code_assert.compute_score_batch as training,
so the eval signal matches what the policy is being optimized for.
"""

import argparse
import json
import os

from datasets import Dataset, load_dataset

# Same paper-style prompt template as humaneval.py — verbatim from
# research/JacobiForcing/JacobiForcing/jacobi_forcing_inference_humaneval.py
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
    # HumanEval+'s `test` defines a `check(candidate)` function. After running
    # the model code, we invoke `check(<entry_point>)`.
    test_code = test + f"\n\ncheck({entry_point})\n"

    return {
        "data_source": "humanevalplus",
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
    parser.add_argument("--local_dir", default="/mnt/weka/home/hao.zhang/shao/verl/data/humanevalplus")
    parser.add_argument("--hf_name", default="evalplus/humanevalplus")
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
