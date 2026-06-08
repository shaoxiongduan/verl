"""Repetition check on v4 oracle-reinit grid output.

For each per-prompt row:
  - Detect substring repetition in completion_preview (200 chars). Flag if any
    20/40-char substring repeats >= 2 times.
  - Detect tail-of-history repetition in n_acc_real_history: if the last 40
    iters have mean n_acc >= 15, that's a loop signature (the model is
    accepting its own repeated draft each iter).
  - Combine: a prompt is "looped" if EITHER heuristic fires.

Output per-mode:
  - n_total, n_looped, % looped
  - mean_TPF overall vs mean_TPF only for non-looped vs only for looped
"""
from __future__ import annotations

import argparse
import glob
import json
import math
import os
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--indir", required=True)
    return p.parse_args()


def has_substring_repetition(text: str, min_len: int = 40, max_len: int = 80,
                              min_repeats: int = 3) -> bool:
    """True iff some substring of length [min_len, max_len] appears >= min_repeats
    times in `text`. Cheap O(n^2 / step) check."""
    n = len(text)
    if n < min_len * min_repeats:
        return False
    for L in (min_len, min_len + 10, min_len + 20, max_len):
        if L > n // 2:
            continue
        # slide
        seen: dict[str, int] = {}
        for i in range(0, n - L + 1, 4):
            s = text[i : i + L]
            seen[s] = seen.get(s, 0) + 1
            if seen[s] >= min_repeats:
                return True
    return False


def history_looped(hist: list[int], K: int = 32, tail_n: int = 40,
                    mean_thresh: float = 15.0) -> bool:
    if hist is None or len(hist) < tail_n:
        return False
    tail = hist[-tail_n:]
    return (sum(tail) / len(tail)) >= mean_thresh


def main() -> None:
    args = parse_args()
    pattern = os.path.join(args.indir, f"{args.tag}__*.jsonl")
    files = sorted([f for f in glob.glob(pattern) if not f.endswith("merged.jsonl")])
    if not files:
        raise SystemExit(f"no files: {pattern}")

    by_mode: dict[str, list[dict]] = defaultdict(list)
    for fp in files:
        with open(fp) as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)
                by_mode[r["mode"]].append(r)

    modes_order = [m for m in
                   ["natural", "oracle_shift0", "oracle_shift1",
                    "oracle_shift2", "oracle_shift3"]
                   if m in by_mode]

    print(f"\n=== {args.tag}: repetition split ===")
    print(f"{'mode':18s} {'n':>4s} {'n_loop':>7s} {'%loop':>6s} "
          f"{'TPF_all':>9s} {'TPF_noloop':>11s} {'TPF_loop':>9s} "
          f"{'n_tok_noloop':>13s} {'n_tok_loop':>11s}")
    for mode in modes_order:
        rows = by_mode[mode]
        tpfs = [r["tpf"] for r in rows]
        looped = []
        for r in rows:
            text_rep = has_substring_repetition(r.get("completion_preview", ""))
            hist_rep = history_looped(r.get("n_acc_real_history"))
            looped.append(text_rep or hist_rep)
        n_loop = sum(looped)
        n_noloop = len(rows) - n_loop
        mean_all = sum(tpfs) / len(tpfs) if tpfs else 0
        noloop_tpfs = [t for t, L in zip(tpfs, looped) if not L]
        loop_tpfs = [t for t, L in zip(tpfs, looped) if L]
        mean_noloop = sum(noloop_tpfs) / len(noloop_tpfs) if noloop_tpfs else float("nan")
        mean_loop = sum(loop_tpfs) / len(loop_tpfs) if loop_tpfs else float("nan")
        ntok_noloop = (sum(r["n_tokens"] for r, L in zip(rows, looped) if not L) /
                       max(1, n_noloop))
        ntok_loop = (sum(r["n_tokens"] for r, L in zip(rows, looped) if L) /
                     max(1, n_loop))
        print(f"{mode:18s} {len(rows):4d} {n_loop:7d} {100*n_loop/len(rows):5.1f}% "
              f"{mean_all:9.3f} {mean_noloop:11.3f} {mean_loop:9.3f} "
              f"{ntok_noloop:13.0f} {ntok_loop:11.0f}")


if __name__ == "__main__":
    main()
