"""Analyze per-cycle (pred_b, true_b) pairs from wmp sims.

Reports:
  - Full per-cycle list (or sample if too long)
  - Distribution of pred_b vs true_b (joint frequency)
  - Under-prediction analysis: how often pred_b < true_b, by how much, and
    how much TPF would be lost IF commits were capped at pred_b.
"""
from __future__ import annotations
import argparse, json
from collections import Counter
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonl", required=True)
    p.add_argument("--max_print", type=int, default=50, help="max cycles to print verbatim")
    return p.parse_args()


def main():
    args = parse_args()
    pred = []; true_ = []
    for line in open(args.jsonl):
        r = json.loads(line)
        for pb, tb in zip(r["pred_b_history"], r["true_b_history"]):
            pred.append(int(pb)); true_.append(int(tb))
    pred = np.array(pred); true_ = np.array(true_)
    n = len(pred)
    err = pred - true_
    print(f"n_cycles={n}")
    print(f"true_b: mean={true_.mean():.3f}  median={np.median(true_):.0f}  std={true_.std():.3f}")
    print(f"pred_b: mean={pred.mean():.3f}  median={np.median(pred):.0f}  std={pred.std():.3f}")
    print(f"err   : mean={err.mean():+.3f}  MAE={np.abs(err).mean():.3f}  std={err.std():.3f}")

    print(f"\n  P(pred < true) = {(pred < true_).mean():.3f}  (under-predicts; would cost TPF if commits capped at pred)")
    print(f"  P(pred = true) = {(pred == true_).mean():.3f}")
    print(f"  P(pred > true) = {(pred > true_).mean():.3f}  (over-predicts; would commit wrong tokens if no verify)")

    # Hypothetical TPF if commits capped at min(pred_b, true_b+1) (i.e., probe is the commit gate)
    # In current 2-fwd sim, we commit min(true_n_acc+1, K). With probe-cap: min(pred_b, true_n_acc+1, K)
    # Loss per cycle if probe under-predicts: (true_n_acc + 1) - pred_b
    K = 32
    commits_oracle = np.minimum(true_ + 1, K)
    commits_probe  = np.minimum(np.minimum(pred, true_ + 1), K)
    loss_per_cycle = commits_oracle - commits_probe
    print(f"\nHYPOTHETICAL: if pred_b CAPS commits (commits = min(pred, true+1)):")
    print(f"  Σ oracle commits = {commits_oracle.sum()}  →  TPF_per_verify_oracle = {commits_oracle.mean():.3f}")
    print(f"  Σ probe-capped commits = {commits_probe.sum()}  →  TPF_per_verify_probe = {commits_probe.mean():.3f}")
    print(f"  total tokens LOST = {loss_per_cycle.sum()}  ({100*loss_per_cycle.sum()/commits_oracle.sum():.2f}% of oracle)")
    print(f"  avg loss per cycle = {loss_per_cycle.mean():.3f}")
    print(f"  P(any loss) = {(loss_per_cycle > 0).mean():.3f}")

    # Distribution of loss size
    print(f"\n  loss-per-cycle histogram (only cycles where loss>0):")
    only_loss = loss_per_cycle[loss_per_cycle > 0]
    if len(only_loss) > 0:
        for v in sorted(set(only_loss))[:15]:
            c = (only_loss == v).sum()
            print(f"    loss={int(v):>3}: {c:>5}  ({c/len(only_loss)*100:.1f}%)")

    # Joint distribution (pred, true) compact
    print(f"\nJoint frequency (pred_b vs true_b, top 30 cells):")
    joint = Counter(zip(pred.tolist(), true_.tolist()))
    for (p, t), c in joint.most_common(30):
        marker = "  match" if p == t else ("  UNDER" if p < t else "  OVER")
        print(f"  (pred={p:>2}, true={t:>2}): {c:>5} cycles{marker}")

    # Per-cycle list (or sample)
    print(f"\nPer-cycle (pred, true) list (first {args.max_print}):")
    for i in range(min(args.max_print, n)):
        print(f"  cyc {i:>4d}: pred={pred[i]:>2}  true={true_[i]:>2}  err={pred[i]-true_[i]:+d}")

    # Distribution by true_b
    print(f"\nFor each true_b value, what does the probe predict?")
    print(f"{'true_b':>7} {'n_cycles':>8} {'pred_mean':>10} {'pred_median':>11} {'P(under)':>9} {'P(match)':>9} {'P(over)':>8}")
    for tb in sorted(set(true_.tolist())):
        mask = true_ == tb
        n_t = mask.sum()
        if n_t < 5: continue
        p_t = pred[mask]
        under = (p_t < tb).mean(); match = (p_t == tb).mean(); over = (p_t > tb).mean()
        print(f"{tb:>7} {n_t:>8} {p_t.mean():>10.2f} {int(np.median(p_t)):>11d} {under:>9.3f} {match:>9.3f} {over:>8.3f}")


if __name__ == "__main__":
    main()
