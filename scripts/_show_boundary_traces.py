"""Pretty-print sample cycles from the boundary dataset.

Pick cycles with varied n_acc (0, 3, 6, 10, 15, K) and dump:
  - committed context tail
  - per-position table: j, warm_tok, verify_tok, match?, top1_p, entropy, margin
  - clear boundary marker
"""
from __future__ import annotations
import argparse, json, math
from collections import defaultdict
from transformers import AutoTokenizer


def load_cycles(fn: str):
    out = []
    for line in open(fn):
        out.append(json.loads(line))
    return out


def pick_diverse(cycles, targets=(0, 1, 3, 6, 10, 15, 20, 25, 32)):
    """For each target n_acc, return one cycle whose n_acc matches (or closest)."""
    by_nacc = defaultdict(list)
    for c in cycles:
        by_nacc[c["n_acc"]].append(c)
    picked = {}
    for t in targets:
        if t in by_nacc and len(by_nacc[t]) > 0:
            # Prefer a cycle from prompt_idx 0 if available for reproducibility
            cand = by_nacc[t]
            cand.sort(key=lambda c: (c["prompt_idx"], c["cycle_idx"]))
            picked[t] = cand[0]
        else:
            # Find closest
            nearest = min(by_nacc.keys(), key=lambda k: abs(k - t)) if by_nacc else None
            if nearest is not None and nearest not in picked.values():
                picked[t] = by_nacc[nearest][0]
    # dedupe by (prompt_idx, cycle_idx)
    seen = set(); out = []
    for t, c in picked.items():
        key = (c["prompt_idx"], c["cycle_idx"])
        if key in seen: continue
        seen.add(key); out.append((t, c))
    return out


def render_cycle(target_n, c, tok, prompt_text_cache):
    K = len(c["accept"])
    n_acc = c["n_acc"]
    # Decode tokens (warm + verify) compactly
    warm = c["warm_argmax"]; verify = c["verify_argmax"]
    print(f"\n{'='*78}")
    print(f"target_n_acc={target_n}  prompt_idx={c['prompt_idx']}  cycle_idx={c['cycle_idx']}  L_committed={c['L_committed']}  **TRUE n_acc={n_acc}**")
    print(f"{'='*78}")
    print(f"{'j':>3} {'warm_tok':>14} {'verify_tok':>14} {'match':>5} {'top1_p':>7} {'entropy':>7} {'margin':>7}  status")
    print("-" * 78)
    for j in range(K):
        wt = repr(tok.decode([warm[j]]))[:14]
        vt = repr(tok.decode([verify[j]]))[:14]
        match = "✓" if c["accept"][j] else "✗"
        p1 = c["top1_prob"][j]; ent = c["entropy"][j]; mg = c["margin_log"][j]
        status = ""
        if j == n_acc and n_acc < K:
            status = " ← BOUNDARY"
        elif j < n_acc:
            status = "  accepted"
        elif j == n_acc + 1 and n_acc + 1 == K:
            status = "  (eob)"
        elif j > n_acc:
            status = "  rejected"
        print(f"{j:>3} {wt:>14} {vt:>14} {match:>5} {p1:>7.3f} {ent:>7.3f} {mg:>7.3f}{status}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--targets", type=str, default="0,1,3,6,10,15,20,32")
    args = p.parse_args()

    print(f"[trace] loading data: {args.data}")
    cycles = load_cycles(args.data)
    print(f"[trace] {len(cycles)} cycles total")
    # Distribution
    nacc_dist = defaultdict(int)
    for c in cycles:
        nacc_dist[c["n_acc"]] += 1
    print(f"[trace] n_acc distribution (top 15):")
    for nacc in sorted(nacc_dist.keys()):
        bar = "█" * min(50, nacc_dist[nacc] // 5)
        print(f"  n_acc={nacc:>3}: {nacc_dist[nacc]:>5}  {bar}")

    print(f"\n[trace] loading tokenizer: {args.tokenizer}")
    tok = AutoTokenizer.from_pretrained(args.tokenizer)
    targets = [int(x) for x in args.targets.split(",")]

    picks = pick_diverse(cycles, targets=tuple(targets))
    for tgt, c in picks:
        render_cycle(tgt, c, tok, {})


if __name__ == "__main__":
    main()
