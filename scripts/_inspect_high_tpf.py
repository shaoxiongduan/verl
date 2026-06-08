"""Inspect the high-TPF tail of v4 oracle-reinit rollouts.

For each mode, take the top-N rollouts by TPF and report:
  - mode, batch_idx, n_tokens, tpf
  - is_text_rep   (substring detector from _check_v4_repetition)
  - is_hist_rep   (sustained high n_acc tail in n_acc_real_history)
  - completion preview (first ~250 chars after the chat prompt)

Headline: among top-N by TPF, how many are loops? Are the non-loop ones
producing meaningful math reasoning?
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from _check_v4_repetition import has_substring_repetition, history_looped


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--tag", required=True)
    p.add_argument("--indir", required=True)
    p.add_argument("--top_n", type=int, default=10,
                   help="per-mode top-N TPF rollouts to inspect")
    p.add_argument("--preview_chars", type=int, default=300)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted([f for f in glob.glob(os.path.join(args.indir, f"{args.tag}__*.jsonl"))
                    if not f.endswith("merged.jsonl")])
    if not files:
        raise SystemExit(f"no files for tag {args.tag}")

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

    print(f"\n=== Top-{args.top_n} TPF rollouts per mode ===\n")
    for mode in modes_order:
        rows = sorted(by_mode[mode], key=lambda r: -r["tpf"])[: args.top_n]
        print(f"\n--- {mode} (top {len(rows)}) ---")
        n_loop_text = 0
        n_loop_hist = 0
        n_loop_either = 0
        for r in rows:
            text = r.get("completion_preview", "")
            hist = r.get("n_acc_real_history")
            text_rep = has_substring_repetition(text)
            hist_rep = history_looped(hist)
            is_loop = text_rep or hist_rep
            n_loop_text += int(text_rep)
            n_loop_hist += int(hist_rep)
            n_loop_either += int(is_loop)
            flag = "LOOP" if is_loop else "OK  "
            ttag = "T" if text_rep else "-"
            htag = "H" if hist_rep else "-"
            print(f"  [{flag} {ttag}{htag}] idx={r['batch_idx']:3d} "
                  f"n_tok={r['n_tokens']:4d} tpf={r['tpf']:6.3f}  "
                  f"preview={text[: args.preview_chars]!r}")
        print(f"  > loops: text={n_loop_text}/{len(rows)} "
              f"hist={n_loop_hist}/{len(rows)} either={n_loop_either}/{len(rows)}")


if __name__ == "__main__":
    main()
