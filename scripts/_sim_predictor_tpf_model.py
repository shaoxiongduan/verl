"""TPF model using (pred_b, true_b) pairs + shift-leak lookup.

For each cycle i (chronological order within each prompt):
  Cap commits at probe boundary; carry over the over-pred amount as a "shift"
  to the next iter. The next iter's TPF is read from the shift-leak table.

  Strategy:
    incoming_shift = max(0, pred_{i-1} - true_{i-1} - 1) capped at TABLE_MAX
    if pred_i < true_i:
        commits_i = pred_i            # under-pred → capped at probe
    else:
        commits_i = SHIFT_TABLE[incoming_shift]  # degraded by inherited shift

    next_shift = max(0, pred_i - true_i - 1)

  This uses the empirical shift_table (math_k3 and base) from prior sims.

  Reports total tokens, total cycles, and TPF_verify under this model.
"""
from __future__ import annotations
import argparse
import json
import numpy as np


# Empirical TPF_verify from shift-leak sims (per verify forward)
SHIFT_TABLE = {
    "math_k3": {0: 5.94, 1: 5.79, 2: 5.76, 3: 5.67, 4: 5.53, 5: 5.46,
                6: 5.40, 7: 5.34, 8: 5.28, 9: 5.22, 10: 5.16},  # extrapolated linearly after 5
    "base":    {0: 4.94, 1: 5.23, 2: 5.51, 3: 5.63, 4: 5.67, 5: 5.74,
                6: 5.74, 7: 5.74, 8: 5.74, 9: 5.74, 10: 5.74},  # base PLATEAUS / improves
}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--csv", required=True, help="pred_vs_true csv: prompt_idx,cycle_idx,pred_b,true_b,err")
    p.add_argument("--model", required=True, choices=["math_k3", "base"])
    p.add_argument("--max_shift", type=int, default=10)
    p.add_argument("--add_offset", type=int, default=0, help="add to pred_b before applying caps (sim retraining)")
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--label", default="")
    return p.parse_args()


def main():
    args = parse_args()
    table = SHIFT_TABLE[args.model]
    rows = []  # list of (prompt_idx, cycle_idx, pred, true)
    with open(args.csv) as f:
        next(f)  # header
        for line in f:
            pi, ci, pb, tb, _ = line.strip().split(",")
            rows.append((int(pi), int(ci), int(pb) + args.add_offset, int(tb)))

    # Group by prompt to process cycles in order
    from collections import defaultdict
    by_prompt = defaultdict(list)
    for pi, ci, pb, tb in rows:
        by_prompt[pi].append((ci, pb, tb))
    for pi in by_prompt:
        by_prompt[pi].sort()

    total_tok = 0.0
    n_cycles = 0
    n_under = 0; n_match = 0; n_over = 0
    shifts_inherited = []
    K = args.K

    for pi, cycles in by_prompt.items():
        last_shift = 0  # at start of prompt, no shift
        for ci, pb, tb in cycles:
            pb = min(pb, K); pb = max(pb, 0)
            shifts_inherited.append(last_shift)
            if pb < tb:
                # under-pred: commit pb tokens, no shift carries forward
                commits = float(pb)
                n_under += 1
                this_shift = 0
            elif pb == tb:
                # exact match: full commit, no shift
                commits = float(tb + 1)  # bonus
                n_match += 1
                this_shift = 0
            else:
                # over-pred: full commit, shift = pb - tb - 1
                # but base TPF for THIS iter is degraded by INHERITED shift
                # Use the shift_table lookup
                s = min(last_shift, args.max_shift)
                # Normalize: if shift=0 the table value IS the oracle TPF; for our (pred,true)
                # data the oracle commit count for this cycle is tb+1. We need to map.
                # Approach: use ratio table_TPF[s] / table_TPF[0] as multiplicative degradation
                degradation = table[s] / table[0]
                commits = (tb + 1) * degradation
                n_over += 1
                this_shift = min(pb - tb - 1, args.max_shift)
            total_tok += commits
            n_cycles += 1
            last_shift = this_shift

    tpf_model = total_tok / max(1, n_cycles)
    shifts = np.array(shifts_inherited)
    print(f"=== {args.label or args.csv}  offset={args.add_offset} ===")
    print(f"  n_cycles={n_cycles}  total_tok={total_tok:.1f}  TPF_model={tpf_model:.3f}")
    print(f"  P(under)={n_under/n_cycles:.3f}  P(match)={n_match/n_cycles:.3f}  P(over)={n_over/n_cycles:.3f}")
    print(f"  mean inherited_shift = {shifts.mean():.2f}")
    print(f"  inherited shift histogram: ", end="")
    for s in range(8):
        print(f"s={s}: {(shifts==s).sum()}", end="  ")
    print()


if __name__ == "__main__":
    main()
