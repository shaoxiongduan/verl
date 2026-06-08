"""Split v4 rollouts into CAPPED (n_tok >= 2040, likely loop) vs UNCAPPED
(n_tok < 2040, clean EOS) and report TPF distribution for each. Also dump
the top-N TPF in each category.

The question: among rollouts that genuinely finished (EOS, not max_new cap),
how high does TPF go? If high-TPF NON-loop rollouts exist, the model is
learning real noise-refinement. If high-TPF is exclusively from looped
capped rollouts, the model is just exploiting repetition.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import statistics
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--indir", required=True)
    p.add_argument("--cap_thresh", type=int, default=2040,
                   help="n_tokens >= this = treated as capped")
    p.add_argument("--top_n", type=int, default=8)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted([f for f in glob.glob(os.path.join(args.indir, f"{args.tag}__*.jsonl"))
                    if not f.endswith("merged.jsonl")])
    if not files:
        raise SystemExit(f"no files for {args.tag}")

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

    print(f"\n=== {args.tag}: TPF split by n_tokens cap (>= {args.cap_thresh}) ===\n")
    print(f"{'mode':18s} {'n_capped':>9s} {'TPF_capped':>11s} "
          f"{'n_uncapped':>11s} {'TPF_uncapped':>13s} {'n_tok_uncapped':>15s}")
    for mode in modes_order:
        rows = by_mode[mode]
        capped = [r for r in rows if r["n_tokens"] >= args.cap_thresh]
        uncapped = [r for r in rows if r["n_tokens"] < args.cap_thresh]
        tpf_c = [r["tpf"] for r in capped]
        tpf_u = [r["tpf"] for r in uncapped]
        ntok_u = [r["n_tokens"] for r in uncapped]
        mean_c = statistics.mean(tpf_c) if tpf_c else float("nan")
        mean_u = statistics.mean(tpf_u) if tpf_u else float("nan")
        mean_ntok_u = statistics.mean(ntok_u) if ntok_u else 0
        print(f"{mode:18s} {len(capped):9d} {mean_c:11.3f} "
              f"{len(uncapped):11d} {mean_u:13.3f} {mean_ntok_u:15.0f}")

    # Per-mode top-N TPF among uncapped rollouts (the cleanest signal).
    print(f"\n=== Top-{args.top_n} UNCAPPED TPF rollouts per mode (clean EOS) ===\n")
    for mode in modes_order:
        rows = sorted(
            [r for r in by_mode[mode] if r["n_tokens"] < args.cap_thresh],
            key=lambda r: -r["tpf"],
        )[: args.top_n]
        print(f"\n--- {mode} (top {len(rows)} uncapped) ---")
        for r in rows:
            print(f"  idx={r['batch_idx']:3d} n_tok={r['n_tokens']:4d} "
                  f"tpf={r['tpf']:6.3f}  "
                  f"preview={r.get('completion_preview', '')[:250]!r}")


if __name__ == "__main__":
    main()
