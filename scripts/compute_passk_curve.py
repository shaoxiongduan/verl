"""Compute unbiased pass@k curve from eval_offline_pass1's per-prompt JSONL.

Pass@k estimator from Codex / Chen et al. 2021 ("Evaluating Large Language Models
Trained on Code", arXiv:2107.03374, Eq. 1):

    pass@k = E_problem[ 1 - C(n-c, k) / C(n, k) ]

where n = total samples per problem, c = correct samples out of n. Defined only
when k <= n. Estimator is unbiased over the random sample of n attempts.

Usage:
  python scripts/compute_passk_curve.py \
    --jsonls passk_base_he__single.jsonl passk_ar300_he__single.jsonl ... \
    --tags base ar_v2_300 cons_const_300 \
    --ks 1 2 4 8 16 32
"""
from __future__ import annotations
import argparse
import json
import math
from typing import List


def pass_at_k_unbiased(n: int, c: int, k: int) -> float:
    """Codex unbiased estimator. Returns 0.0 if k > n."""
    if k > n:
        return float("nan")
    if n - c < k:
        return 1.0
    # 1 - C(n-c, k) / C(n, k)
    # Compute in log space to avoid overflow for large n
    log_num = sum(math.log(n - c - i) for i in range(k))
    log_den = sum(math.log(n - i) for i in range(k))
    return 1.0 - math.exp(log_num - log_den)


def load_rows(path: str) -> list[dict]:
    rows = []
    with open(path) as f:
        for line in f:
            rows.append(json.loads(line))
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonls", nargs="+", required=True)
    p.add_argument("--tags", nargs="+", required=True,
                   help="Display labels parallel to --jsonls")
    p.add_argument("--ks", nargs="+", type=int, default=[1, 2, 4, 8, 16, 32])
    args = p.parse_args()
    assert len(args.jsonls) == len(args.tags)

    results = {}
    for tag, path in zip(args.tags, args.jsonls):
        rows = load_rows(path)
        # Each row has "per_sample_pass": [0.0 or 1.0, ...] of length n.
        ns, cs = [], []
        for r in rows:
            psp = r.get("per_sample_pass", [])
            if not psp:
                continue
            ns.append(len(psp))
            cs.append(int(sum(s == 1.0 for s in psp)))
        if not ns:
            results[tag] = {}
            continue
        n_max = max(ns)
        n_min = min(ns)
        per_k = {}
        for k in args.ks:
            if k > n_min:
                # Skip k > min n to keep estimates unbiased.
                continue
            vals = [pass_at_k_unbiased(n, c, k) for n, c in zip(ns, cs)]
            per_k[k] = sum(vals) / len(vals)
        results[tag] = {
            "n_prompts": len(ns),
            "n_samples_per_prompt": (n_min, n_max),
            "pass_at_k": per_k,
        }

    # Print as table
    print("=" * 100)
    print(f"Pass@k crossing analysis ({results[args.tags[0]].get('n_prompts','?')} prompts)")
    print("=" * 100)
    header = f"{'k':>4} | " + " | ".join(f"{t:>20}" for t in args.tags)
    print(header)
    print("-" * len(header))
    all_ks = sorted({k for r in results.values() for k in r.get("pass_at_k", {})})
    for k in all_ks:
        row = f"{k:>4} | "
        row += " | ".join(
            f"{results[t]['pass_at_k'][k]:>20.4f}" if k in results[t].get("pass_at_k", {})
            else f"{'-':>20}"
            for t in args.tags
        )
        print(row)

    print()
    # Compute the Yue et al. 2025 "crossing" diagnostic: where does base catch up?
    # If base pass@k > rl pass@k for some k > 1, mode collapse is documented.
    print("Crossing analysis (where pass@k_base > pass@k_RL implies collapse):")
    if "base" in args.tags:
        base_pk = results["base"]["pass_at_k"]
        for t in args.tags:
            if t == "base":
                continue
            tk = results[t]["pass_at_k"]
            crossings = [k for k in sorted(tk) if k in base_pk and base_pk[k] > tk[k]]
            if crossings:
                print(f"  base > {t} at k = {crossings}")
            else:
                print(f"  base never overtakes {t} in measured ks")


if __name__ == "__main__":
    main()
