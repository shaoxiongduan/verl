"""Per-block-position analysis of Jacobi rollout trajectories.

Reads JSONL trajectories written by vllm_jacobi_patch and computes, for each
draft position p ∈ [0, K-1]:

  - acceptance rate:    P(draft[p] == target_argmax[p])
  - survival rate:      P(n_acc >= p+1)  — chain reached this position
  - argmax-marginal entropy: Shannon entropy of target_argmax[p] over all
                             (req, iter) samples (proxy for "how diverse
                             is the model's prediction at this slot")
  - top-10 token concentration: fraction of mass on top-10 most-frequent tokens

Compare across K to assess training stability.

Usage:
    python analyze_jacobi_trajectories.py TRAJ_GLOB...
e.g.:
    python analyze_jacobi_trajectories.py \\
        /mnt/weka/home/hao.zhang/shao/verl/scripts/traj_bs128_k8.jsonl.* \\
        /mnt/weka/home/hao.zhang/shao/verl/scripts/traj_bs128_k16.jsonl.* \\
        /mnt/weka/home/hao.zhang/shao/verl/scripts/traj_bs128_k32.jsonl.*
"""
from __future__ import annotations
import argparse
import glob
import json
import math
import sys
from collections import Counter, defaultdict


def shannon_entropy(counts: Counter) -> float:
    total = sum(counts.values())
    if total == 0:
        return 0.0
    h = 0.0
    for c in counts.values():
        p = c / total
        if p > 0:
            h -= p * math.log2(p)
    return h


def analyze(traj_glob: str) -> dict:
    files = sorted(glob.glob(traj_glob))
    if not files:
        return {"glob": traj_glob, "error": "no files"}

    # Per-position aggregations:
    #   acc_match[p] = # of (req, iter) where draft[p] == target_argmax[p]
    #   acc_seen[p]  = # of (req, iter) where the draft had position p at all
    #   reached[p]   = # of (req, iter) where chain reached position p (n_acc >= p+1)
    #   tok_counts[p]= Counter of target_argmax[p] across all (req, iter)
    acc_match = defaultdict(int)
    acc_seen = defaultdict(int)
    reached = defaultdict(int)
    tok_counts: dict[int, Counter] = defaultdict(Counter)

    total_iters = 0
    total_active_reqs = 0
    total_committed = 0
    K_seen = 0
    n_acc_hist: Counter = Counter()
    chain_extension_to: defaultdict = defaultdict(int)

    for fp in files:
        with open(fp) as f:
            for line in f:
                rec = json.loads(line)
                # Skip vLLM warmup phase (huge num_draft list of 1s)
                if len(rec["num_draft"]) > 256 and set(rec["num_draft"]) == {1}:
                    continue
                total_iters += 1
                # FIX alignment: _PENDING_DRAFT_PER_REQ in the patch was
                # indexed by propose()-call sequence, not by batch index. To
                # align, walk nd in order and consume drafts from a queue:
                # the i-th non-zero nd entry takes draft[i] from the queue.
                draft_queue = [d for d in rec["draft"] if d]  # non-empty
                draft_iter = iter(draft_queue)
                for i, nd in enumerate(rec["num_draft"]):
                    if nd <= 0:
                        continue
                    K_seen = max(K_seen, nd)
                    total_active_reqs += 1
                    n_acc = rec["n_acc"][i]
                    n_acc_hist[n_acc] += 1
                    total_committed += n_acc + 1  # accepted spec + bonus

                    # Aligned draft (may be empty if queue exhausted = bug fallback)
                    draft = next(draft_iter, [])
                    tgt = rec["target_argmax"][i]
                    for p in range(min(nd, len(tgt))):
                        acc_seen[p] += 1
                        tok_counts[p][tgt[p]] += 1
                        if p < len(draft) and draft[p] == tgt[p]:
                            acc_match[p] += 1
                        if p < n_acc:
                            reached[p] += 1
                        if p == n_acc:
                            chain_extension_to[p] += 1

    K = K_seen
    rows = []
    for p in range(K):
        seen = acc_seen[p]
        if seen == 0:
            continue
        ent = shannon_entropy(tok_counts[p])
        # Top-10 concentration
        topN = tok_counts[p].most_common(10)
        top_mass = sum(c for _, c in topN) / seen
        rows.append({
            "pos": p,
            "seen": seen,
            "acc_rate": acc_match[p] / seen,
            "survival": reached[p] / seen,
            "entropy_bits": ent,
            "top10_mass": top_mass,
            "unique_tokens": len(tok_counts[p]),
        })

    return {
        "glob": traj_glob,
        "n_files": len(files),
        "K": K,
        "total_spec_iters": total_iters,
        "total_active_reqs": total_active_reqs,
        "total_committed": total_committed,
        "avg_tpf": total_committed / total_active_reqs if total_active_reqs else 0,
        "n_acc_hist": dict(sorted(n_acc_hist.items())),
        "per_pos": rows,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("globs", nargs="+", help="trajectory file globs")
    args = ap.parse_args()

    reports = []
    for g in args.globs:
        r = analyze(g)
        reports.append(r)

    # ---- per-config summary ----
    print("\n" + "=" * 78)
    print(f"{'config':>14}  {'K':>3}  {'iters':>6}  {'reqs':>7}  {'TPF':>6}  {'n_acc histogram':>40}")
    print("=" * 78)
    for r in reports:
        if "error" in r:
            print(f"  {r['glob']}: {r['error']}")
            continue
        label = r["glob"].split("/")[-1].split(".")[0]
        hist = r["n_acc_hist"]
        # Show top n_acc buckets
        hist_str = " ".join(f"{k}:{v}" for k, v in list(hist.items())[:10])
        print(f"{label:>14}  {r['K']:>3}  {r['total_spec_iters']:>6}  {r['total_active_reqs']:>7}  {r['avg_tpf']:>6.2f}  {hist_str}")

    # ---- per-position breakdown ----
    print("\n" + "=" * 90)
    print("PER-POSITION ANALYSIS")
    print("=" * 90)
    print(f"{'pos':>4}  ", end="")
    for r in reports:
        label = r["glob"].split("/")[-1].split(".")[0]
        # truncated label
        label = label.replace("traj_", "")
        print(f"{label[:14]:>14}  ", end="")
    print()
    print(f"{'    ':>4}  ", end="")
    for _ in reports:
        print(f"{'acc/surv/H(bits)':>14}  ", end="")
    print()
    print("-" * 90)

    # Find max K across reports
    max_K = max(r.get("K", 0) for r in reports)
    for p in range(max_K):
        print(f"{p:>4}  ", end="")
        for r in reports:
            row = next((x for x in r.get("per_pos", []) if x["pos"] == p), None)
            if row is None:
                print(f"{'  -  ':>14}  ", end="")
            else:
                cell = f"{row['acc_rate']:.2f}/{row['survival']:.2f}/{row['entropy_bits']:.1f}"
                print(f"{cell:>14}  ", end="")
        print()

    # ---- diversity summary ----
    print("\n" + "=" * 80)
    print("DIVERSITY (mean over positions): entropy bits, top10-mass, unique tokens")
    print("=" * 80)
    for r in reports:
        if "per_pos" not in r:
            continue
        label = r["glob"].split("/")[-1].split(".")[0]
        if not r["per_pos"]:
            continue
        mean_ent = sum(x["entropy_bits"] for x in r["per_pos"]) / len(r["per_pos"])
        mean_top10 = sum(x["top10_mass"] for x in r["per_pos"]) / len(r["per_pos"])
        mean_uniq = sum(x["unique_tokens"] for x in r["per_pos"]) / len(r["per_pos"])
        print(f"  {label:>14}: H={mean_ent:.2f} bits  top10_mass={mean_top10:.3f}  uniq={mean_uniq:.0f}")


if __name__ == "__main__":
    main()
