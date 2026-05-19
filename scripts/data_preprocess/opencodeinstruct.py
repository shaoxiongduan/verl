"""Preprocess nvidia/OpenCodeInstruct to verl parquet for single-turn coder RL.

OpenCodeInstruct ships with assert-style unit tests (the same format
JacobiForcing_Coder_7B_v1 was distilled on). Each row becomes:

    {
      "data_source": "opencodeinstruct",          # matches reward_code_assert.py
      "prompt":       [{"role": "user", "content": <instruction>}],
      "ability":      "code",
      "reward_model": {"style": "rule",
                       "ground_truth": json.dumps({"test_code": "...",
                                                   "entry_point": "..." or None,
                                                   "language": "python"})},
      "extra_info":   {"split": ..., "index": ..., "task_id": ...},
    }

Defensive about column names — OpenCodeInstruct has gone through a few
revisions and field names vary across mirrors.
"""

import argparse
import json
import os
import random

from datasets import load_dataset

CANDIDATE_PROMPT_KEYS = ["input", "instruction", "prompt", "question", "task"]
CANDIDATE_TEST_KEYS = ["unit_tests", "tests", "test", "test_code", "test_cases"]
CANDIDATE_SOLUTION_KEYS = ["output", "solution", "response", "code", "canonical_solution"]
CANDIDATE_ENTRY_KEYS = ["entry_point", "function_name", "fn_name"]


def pick_first(example, candidates):
    for k in candidates:
        if k in example and example[k]:
            return example[k]
    return None


PROMPT_TEMPLATE = (
    "{instruction}\n\n"
    "Write a complete Python solution. Put your final answer inside a single "
    "```python ... ``` code block.\n"
)


def build_example(example, idx, split):
    instruction = pick_first(example, CANDIDATE_PROMPT_KEYS)
    tests = pick_first(example, CANDIDATE_TEST_KEYS)
    entry = pick_first(example, CANDIDATE_ENTRY_KEYS)
    if not instruction or not tests:
        return None

    # Normalize: OCI stores `unit_tests` as a JSON-encoded string of a list
    # of assert strings. Decode it if possible.
    if isinstance(tests, str):
        try:
            parsed = json.loads(tests)
            if isinstance(parsed, list):
                tests = parsed
        except (json.JSONDecodeError, ValueError):
            pass
    if isinstance(tests, list):
        test_code = "\n".join(str(t) for t in tests)
    else:
        test_code = str(tests)

    ground_truth = json.dumps(
        {
            "test_code": test_code,
            "entry_point": entry,
            "language": "python",
        }
    )

    return {
        "data_source": "opencodeinstruct",
        "prompt": [{"role": "user", "content": PROMPT_TEMPLATE.format(instruction=instruction)}],
        "ability": "code",
        "reward_model": {"style": "rule", "ground_truth": ground_truth},
        "extra_info": {
            "split": split,
            "index": idx,
            "task_id": str(example.get("id", example.get("task_id", idx))),
        },
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--local_dir", default="/mnt/weka/home/hao.zhang/shao/verl/data/opencodeinstruct")
    parser.add_argument("--hf_name", default="nvidia/OpenCodeInstruct")
    parser.add_argument("--hf_split", default="train")
    parser.add_argument("--n_train", type=int, default=50000, help="random subset for first run")
    parser.add_argument("--n_val", type=int, default=512)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    print(f"Loading {args.hf_name} split={args.hf_split} ...")
    ds = load_dataset(args.hf_name, split=args.hf_split)
    print(f"Columns: {ds.column_names}")
    print(f"Rows: {len(ds)}")
    print("First row keys/values (truncated):")
    sample = ds[0]
    for k, v in sample.items():
        print(f"  {k}: {str(v)[:160]}")

    # Filter to rows that have both an instruction and tests, and (mirroring
    # the JacobiForcing paper) keep only entries whose canonical solution
    # passes all unit tests. OCI stores `average_test_score` as a string.
    def passes_all(example):
        score = example.get("average_test_score")
        try:
            return float(score) >= 1.0
        except (TypeError, ValueError):
            return False

    def has_required(example):
        return (
            bool(pick_first(example, CANDIDATE_PROMPT_KEYS))
            and bool(pick_first(example, CANDIDATE_TEST_KEYS))
            and passes_all(example)
        )

    ds = ds.filter(has_required, num_proc=8)
    print(f"After required-fields + passes-all filter: {len(ds)} rows")

    # Materialize the filtered view into a contiguous Arrow table — without
    # this, every downstream random access pays O(N) for the index map.
    print("Flattening filtered view ...", flush=True)
    ds = ds.flatten_indices(num_proc=8)

    # NOTE: a global shuffle on 1.6M filtered rows triggers a multi-GB random
    # write that deadlocks on shared NFS-style storage. Instead we:
    #   (1) pick a contiguous block of (n_train+n_val) rows from the start
    #       of the flattened table — OCI's row order isn't curriculum-sorted,
    #       so this is fine,
    #   (2) shuffle just that small block in-memory and split train/val.
    n_total = min(args.n_train + args.n_val, len(ds))
    print(f"Selecting first {n_total} flattened rows ...", flush=True)
    ds = ds.select(range(n_total)).flatten_indices(num_proc=4)

    os.makedirs(args.local_dir, exist_ok=True)
    from datasets import Dataset

    # Read all selected rows into memory (n_total is small, ~50k).
    rng = random.Random(args.seed)
    indices = list(range(n_total))
    rng.shuffle(indices)
    val_pos = set(indices[: args.n_val])

    print("Building rows ...", flush=True)
    val_rows, train_rows = [], []
    for src_idx in range(n_total):
        split = "val" if src_idx in val_pos else "train"
        new_idx = len(val_rows) if split == "val" else len(train_rows)
        row = build_example(ds[src_idx], new_idx, split)
        if row is None:
            continue
        (val_rows if split == "val" else train_rows).append(row)
        total_built = len(val_rows) + len(train_rows)
        if total_built % 10000 == 0:
            print(f"  built {total_built}/{n_total}", flush=True)

    Dataset.from_list(train_rows).to_parquet(os.path.join(args.local_dir, "train.parquet"))
    print(f"Wrote {len(train_rows)} train rows", flush=True)
    Dataset.from_list(val_rows).to_parquet(os.path.join(args.local_dir, "val.parquet"))
    print(f"Wrote {len(val_rows)} val rows", flush=True)


if __name__ == "__main__":
    main()
