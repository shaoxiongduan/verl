"""Decode and print a few example Jacobi iterations for qualitative eyeball.

Reads a trajectory JSONL and pretty-prints, per sample iter:
  - the K drafts we sent (decoded as text)
  - the K target argmaxes (decoded as text)
  - n_acc (chain length)
  - which positions matched

Usage:
    python decode_jacobi_traces.py TRAJ_FILE [--n_examples 5] [--model ...]
"""
from __future__ import annotations
import argparse
import json
import sys
from transformers import AutoTokenizer


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("traj", help="trajectory JSONL file")
    ap.add_argument("--n_examples", type=int, default=5)
    ap.add_argument("--model", default="/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_k3_ds_step_300")
    ap.add_argument("--min_acc", type=int, default=2,
                    help="only show iters where at least one req accepted >= min_acc")
    args = ap.parse_args()

    tok = AutoTokenizer.from_pretrained(args.model)

    shown = 0
    with open(args.traj) as f:
        for line in f:
            if shown >= args.n_examples:
                break
            r = json.loads(line)
            nd = r["num_draft"]
            # Skip warmup
            if len(nd) > 256 and set(nd) == {1}:
                continue
            # Find a req with non-trivial acceptance
            for i, (n, na) in enumerate(zip(nd, r["n_acc"])):
                if n <= 0 or na < args.min_acc:
                    continue
                d = r["draft"][i]
                t = r["target_argmax"][i]
                if not d:  # draft logging broken in this file
                    continue
                print("=" * 90)
                print(f"iter={r['iter']}  batch_pos={i}  K={n}  n_acc={na}  bonus_tok={r['bonus'][i]}")
                print(f"  draft tokens:   {d}")
                print(f"  target argmax:  {t}")
                # Per-position match
                matches = ["✓" if (j < len(d) and j < len(t) and d[j] == t[j]) else "✗"
                           for j in range(n)]
                print(f"  per-pos match:  {' '.join(matches)}")
                # Decoded
                draft_str = tok.decode(d[:n]) if d else "(empty)"
                target_str = tok.decode(t[:n])
                bonus_str = tok.decode([r["bonus"][i]])
                print(f"  draft  decoded: {draft_str!r}")
                print(f"  target decoded: {target_str!r}")
                print(f"  bonus  decoded: {bonus_str!r}")
                # Accepted prefix
                acc_prefix = t[:na]
                acc_str = tok.decode(acc_prefix) if na > 0 else "(none)"
                committed = acc_prefix + [r["bonus"][i]]
                print(f"  committed:      {tok.decode(committed)!r}  ({na+1} tokens)")
                shown += 1
                break

    if shown == 0:
        print(f"No iters with draft logging found in {args.traj}", file=sys.stderr)


if __name__ == "__main__":
    main()
