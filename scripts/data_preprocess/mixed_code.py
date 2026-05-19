"""Build a mixed-distribution code-RL parquet from OpenCodeInstruct + AceCode-87K
+ MBPP-sanitized, in the verl format that reward_code_assert.py expects.

Motivation: pure-OCI RL specialized the model to OCI's verbose problem-statement
output format and hurt HumanEval+ generalization. Mixing in AceCode and MBPP
(both inline-assert) adds format / problem-style diversity while staying
compatible with the existing assert-based reward.

All three sources emit `test_code` as concatenated inline asserts — the reward
function runs `candidate + "\n\n" + test_code` so no harness changes are needed.

Output schema (per row, matches opencodeinstruct.py):
    {
      "data_source": "opencodeinstruct" | "acecode" | "mbpp",
      "prompt":       [{"role": "user", "content": <instruction>}],
      "ability":      "code",
      "reward_model": {"style": "rule",
                       "ground_truth": json.dumps({"test_code": "...",
                                                   "entry_point": None,
                                                   "language": "python"})},
      "extra_info":   {"split": ..., "index": ..., "task_id": ..., "source": ...},
    }
"""
from __future__ import annotations

import argparse
import json
import os
import random
from typing import Optional

from datasets import load_dataset


PROMPT_TEMPLATE = (
    "{instruction}\n\n"
    "Write a complete Python solution. Put your final answer inside a single "
    "```python ... ``` code block.\n"
)


def _row(data_source: str, instruction: str, test_code: str, split: str,
         index: int, task_id: str, entry_point: Optional[str] = None) -> dict:
    return {
        "data_source": data_source,
        "prompt": [{"role": "user", "content": PROMPT_TEMPLATE.format(instruction=instruction)}],
        "ability": "code",
        "reward_model": {
            "style": "rule",
            "ground_truth": json.dumps({
                "test_code": test_code,
                "entry_point": entry_point,
                "language": "python",
            }),
        },
        "extra_info": {
            "split": split,
            "index": index,
            "task_id": str(task_id),
            "source": data_source,
        },
    }


def load_opencodeinstruct(n_train: int, n_val: int, seed: int) -> tuple[list[dict], list[dict]]:
    """Sequentially read the first n_train+n_val rows from OCI in streaming mode.

    OCI is ~5M rows. Full random sampling (either `select(idxs).to_list()` for
    scattered indexes, or streaming-reservoir over all 5M rows) was prohibitive
    (5-30+ min). OCI is not curated by category — sequential prefix from the
    parquet shards is a representative-enough sample for RL, and finishes in
    under a minute.
    """
    pick_n = n_train + n_val
    ds = load_dataset("nvidia/OpenCodeInstruct", split="train", streaming=True)
    print(f"  OCI: sequential-prefix sample of first {pick_n} rows ...", flush=True)
    reservoir: list[dict] = []
    seen = 0
    for ex in ds:
        seen += 1
        reservoir.append(ex)
        if seen % 5000 == 0:
            print(f"    OCI streamed {seen}/{pick_n} rows...", flush=True)
        if seen >= pick_n:
            break
    print(f"  OCI: took {len(reservoir)} rows", flush=True)

    # Local shuffle to mix train+val without picking them in order.
    random.Random(seed).shuffle(reservoir)
    rows_all: list[dict] = []
    for new_idx, ex in enumerate(reservoir):
        instr = ex.get("input") or ex.get("instruction") or ex.get("question")
        tests = ex.get("unit_tests") or ex.get("tests") or ex.get("test_list")
        entry = ex.get("entry_point") or ex.get("function_name")
        if not instr or not tests:
            continue
        if isinstance(tests, str):
            try:
                parsed = json.loads(tests)
                if isinstance(parsed, list):
                    tests = parsed
            except (json.JSONDecodeError, ValueError):
                pass
        test_code = "\n".join(map(str, tests)) if isinstance(tests, list) else str(tests)
        rows_all.append(_row(
            "opencodeinstruct", instr, test_code, "tmp", new_idx,
            ex.get("id", ex.get("task_id", new_idx)), entry_point=entry,
        ))
    val_rows = rows_all[:n_val]
    for r in val_rows:
        r["extra_info"]["split"] = "val"
    train_rows = rows_all[n_val:n_val + n_train]
    for r in train_rows:
        r["extra_info"]["split"] = "train"
    print(f"  OCI emitted: train={len(train_rows)}  val={len(val_rows)}", flush=True)
    return train_rows, val_rows


def load_acecode(n: int, seed: int, max_prompt_chars: int = 6000) -> list[dict]:
    """AceCode-87K: 'question' (free-text), 'test_cases' (list of inline asserts).

    Stream the dataset and inline-filter by prompt length, collecting up to `n`
    rows. The HF `select(scattered_idxs).to_list()` path was unusably slow even
    on this 87k-row dataset (>2 min and still running). Sequential iteration is
    fine since AceCode isn't sorted by anything we care about.

    Filter by raw char length as a cheap proxy for token length (~4 chars/token
    → 6000 chars ≈ 1500 tokens, within max_prompt_length=2048).
    """
    ds = load_dataset("TIGER-Lab/AceCode-87K", split="train", streaming=True)
    print(f"  AceCode: streaming up to {n} rows (filtering by {max_prompt_chars}-char prompt) ...", flush=True)
    rows: list[dict] = []
    seen = skipped = 0
    for ex in ds:
        seen += 1
        q = ex.get("question") or ""
        tc = ex.get("test_cases")
        if not q or not tc or len(q) > max_prompt_chars:
            skipped += 1
            continue
        test_code = "\n".join(map(str, tc))
        rows.append(_row("acecode", q, test_code, "train", len(rows),
                         ex.get("id", len(rows))))
        if len(rows) >= n:
            break
        if len(rows) % 2000 == 0 and len(rows) > 0:
            print(f"    AceCode kept {len(rows)}/{n} (seen={seen}, skipped={skipped})", flush=True)
    print(f"  AceCode emitted: {len(rows)} (seen={seen}, skipped={skipped})", flush=True)
    return rows


def load_mbpp_sanitized() -> list[dict]:
    """MBPP sanitized (train + validation), 120 + 43 = 163 problems.
    test split (257) is intentionally excluded since it's used by MBPP+ eval."""
    ds = load_dataset("google-research-datasets/mbpp", "sanitized")
    rows = []
    for split_name in ("train", "validation"):
        sp = ds[split_name]
        for i in range(len(sp)):
            ex = sp[i]
            instr = ex.get("prompt") or ex.get("text")
            tests = ex.get("test_list")
            if not instr or not tests:
                continue
            test_code = "\n".join(map(str, tests))
            rows.append(_row("mbpp", instr, test_code, "train",
                             len(rows), ex.get("task_id", i)))
    print(f"  MBPP sanitized emitted: {len(rows)} (train+validation, test excluded)")
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="/mnt/weka/home/hao.zhang/shao/verl/data/mixed_code")
    p.add_argument("--n_oci_train", type=int, default=50000)
    p.add_argument("--n_oci_val", type=int, default=512)
    p.add_argument("--n_acecode", type=int, default=10000)
    p.add_argument("--max_prompt_chars", type=int, default=6000,
                   help="Drop AceCode prompts longer than this in raw chars (≈4 chars/token).")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--no_mbpp", action="store_true", help="Skip MBPP sanitized.")
    args = p.parse_args()

    random.seed(args.seed)
    os.makedirs(args.out_dir, exist_ok=True)

    print("Loading OpenCodeInstruct ...")
    oci_train, oci_val = load_opencodeinstruct(args.n_oci_train, args.n_oci_val, args.seed)
    print("\nLoading AceCode-87K ...")
    ace_train = load_acecode(args.n_acecode, args.seed, args.max_prompt_chars)
    mbpp_train: list[dict] = []
    if not args.no_mbpp:
        print("\nLoading MBPP sanitized ...")
        mbpp_train = load_mbpp_sanitized()

    train_rows = oci_train + ace_train + mbpp_train
    random.Random(args.seed + 100).shuffle(train_rows)
    # Renumber indices after shuffle, preserving original task_id in extra_info.
    for i, r in enumerate(train_rows):
        r["extra_info"]["index"] = i
    val_rows = oci_val

    import pyarrow as pa
    import pyarrow.parquet as pq
    train_path = os.path.join(args.out_dir, "train.parquet")
    val_path = os.path.join(args.out_dir, "val.parquet")
    pq.write_table(pa.Table.from_pylist(train_rows), train_path)
    pq.write_table(pa.Table.from_pylist(val_rows), val_path)

    print(f"\nWrote train: {len(train_rows)} rows -> {train_path}")
    print(f"Wrote val:   {len(val_rows)} rows -> {val_path}")

    # Source breakdown
    from collections import Counter
    c = Counter(r["data_source"] for r in train_rows)
    print("\nTrain source breakdown:")
    for src, n in c.most_common():
        print(f"  {src:>20}: {n:>6}  ({100*n/len(train_rows):.1f}%)")


if __name__ == "__main__":
    main()
