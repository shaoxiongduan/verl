"""Check whether vLLM windowed-Jacobi reaches a steady-state distribution
(Mode B) or has iter-dependent shifts (Mode A).

Mode B (desired): per-iter draft input distribution converges after warmup.
                  Useful for on-policy cons-RL — model only needs to learn 1
                  input distribution.
Mode A (undesired): distribution keeps shifting across iters within a "call"
                    and resets between calls. ~10 distinct distributions.

Method:
  Bucket all spec-decode iters by global-iter range (early/mid/late).
  For each bucket, compute:
    - mean n_acc per request
    - per-position entropy of target_argmax
    - n_acc histogram

  If buckets look identical (within sampling noise) → steady state (Mode B).
  If they drift systematically → shifting (Mode A).

Reference (from agent's empirical probes on k3 model, BS=32):
  Windowed steady-state H ≈ 2.7 bits (math content soup)
  Windowed per-iter accept: iter 0 = 1.00, iter 1 = 4.25, iter 2 = 2.70,
                            iter 3-4 ≈ 3, then ~2.7 steady state
"""
from __future__ import annotations
import argparse
import glob
import json
import math
from collections import Counter, defaultdict


def H(counts: Counter) -> float:
    tot = sum(counts.values())
    if tot == 0:
        return 0.0
    return -sum((c/tot) * math.log2(c/tot) for c in counts.values() if c > 0)


def analyze_buckets(traj_glob: str, bucket_edges=(1, 5, 20, 100, 1_000_000)):
    """Bucket records by iter and compute distribution stats per bucket."""
    files = sorted(glob.glob(traj_glob))
    if not files:
        return None

    # Buckets: (lo, hi] global-iter ranges
    buckets = [(lo, hi) for lo, hi in zip(bucket_edges[:-1], bucket_edges[1:])]

    # Per-bucket aggregations
    n_acc_hist_b = {b: Counter() for b in buckets}
    tok_counts_b = {b: defaultdict(Counter) for b in buckets}  # pos -> Counter
    seen_b = {b: defaultdict(int) for b in buckets}
    n_reqs_b = {b: 0 for b in buckets}

    K_max = 0

    for fp in files:
        with open(fp) as fh:
            for line in fh:
                rec = json.loads(line)
                nd = rec["num_draft"]
                # Skip warmup-phase records
                if len(nd) > 256 and set(nd) == {1}:
                    continue
                it = rec["iter"]
                # Pick bucket
                b = None
                for lo, hi in buckets:
                    if lo <= it < hi:
                        b = (lo, hi); break
                if b is None:
                    continue

                for i, n in enumerate(nd):
                    if n <= 0:
                        continue
                    K_max = max(K_max, n)
                    n_acc = rec["n_acc"][i]
                    tgt = rec["target_argmax"][i]
                    n_acc_hist_b[b][n_acc] += 1
                    n_reqs_b[b] += 1
                    for p in range(min(n, len(tgt))):
                        seen_b[b][p] += 1
                        tok_counts_b[b][p][tgt[p]] += 1

    # Per-bucket summary
    summary = []
    for b in buckets:
        if n_reqs_b[b] == 0:
            continue
        hist = n_acc_hist_b[b]
        total_acc = sum(k * v for k, v in hist.items())
        mean_acc = total_acc / max(1, n_reqs_b[b])
        # per-position entropies
        pos_H = {}
        for p in range(K_max):
            if p in tok_counts_b[b] and seen_b[b][p] > 0:
                pos_H[p] = H(tok_counts_b[b][p])
        mean_pos_H = sum(pos_H.values()) / max(1, len(pos_H))
        summary.append({
            "bucket": b, "n_reqs": n_reqs_b[b],
            "mean_n_acc": mean_acc, "mean_pos_H_bits": mean_pos_H,
            "pos_H": pos_H, "n_acc_hist": dict(sorted(hist.items())),
        })
    return {"K": K_max, "buckets": summary, "n_files": len(files)}


def print_report(label, rep):
    print(f"\n{'=' * 70}")
    print(f"{label}  (K={rep['K']}, files={rep['n_files']})")
    print("=" * 70)
    print(f"{'iter range':>14}  {'reqs':>7}  {'mean_n_acc':>10}  {'mean H/pos':>11}  H(pos 0):.1f  H(pos K-1):.1f")
    for b in rep["buckets"]:
        lo, hi = b["bucket"]
        h0 = b["pos_H"].get(0, float("nan"))
        hL = b["pos_H"].get(rep["K"] - 1, float("nan"))
        print(f"  [{lo:>5},{hi:>5})  {b['n_reqs']:>7}  {b['mean_n_acc']:>10.3f}  {b['mean_pos_H_bits']:>11.3f}  {h0:>5.2f}  {hL:>5.2f}")
    # Position-wise drift across buckets (last vs first bucket)
    if len(rep["buckets"]) >= 2:
        first, last = rep["buckets"][0], rep["buckets"][-1]
        print(f"\n  drift first→last bucket:  Δmean_n_acc={last['mean_n_acc']-first['mean_n_acc']:+.3f}  "
              f"Δmean_H={last['mean_pos_H_bits']-first['mean_pos_H_bits']:+.3f}")
        print(f"  per-pos H drift (last - first):")
        for p in range(rep["K"]):
            h_first = first["pos_H"].get(p, float("nan"))
            h_last = last["pos_H"].get(p, float("nan"))
            if not math.isnan(h_first) and not math.isnan(h_last):
                print(f"    pos {p:>2}:  {h_first:.2f} → {h_last:.2f}  (Δ={h_last-h_first:+.2f})")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="+")
    args = ap.parse_args()
    for g in args.globs:
        rep = analyze_buckets(g)
        if rep is None:
            print(f"no files for {g}")
            continue
        label = g.split("/")[-1].split(".")[0]
        print_report(label, rep)


if __name__ == "__main__":
    main()
