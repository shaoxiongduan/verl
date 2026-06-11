"""Repetition-artifact audit for vLLM TPF bench completions.

For each row in the output jsonl(s) (schema of vllm_tpf_trajectories.py:
{"completion": str, "num_tokens": int, "tpf": float, ...}), compute:

  uniq4      : unique-4gram ratio over whitespace tokens (1.0 = no repetition)
  tail_uniq4 : same, over the last 150 whitespace tokens (past-EOS loops live here)
  max_line_rep: max count of any identical non-empty line
  capped     : num_tokens >= cap (hit max_new_tokens)

Flags a row as REPETITIVE if tail_uniq4 < 0.35 or max_line_rep >= 8.
Prints per-file summary + worst offenders, and TPF with/without flagged rows.
"""
from __future__ import annotations
import argparse, json, sys
from collections import Counter


def ngram_uniq(words, n=4):
    if len(words) < n + 1:
        return 1.0
    grams = [tuple(words[i:i+n]) for i in range(len(words) - n + 1)]
    return len(set(grams)) / len(grams)


def audit_row(r, cap):
    text = r.get("completion") or ""
    words = text.split()
    lines = [l for l in text.splitlines() if l.strip()]
    lc = Counter(lines)
    max_line_rep = max(lc.values()) if lc else 0
    u4 = ngram_uniq(words)
    tail_u4 = ngram_uniq(words[-150:])
    capped = (r.get("num_tokens") or 0) >= cap
    rep = tail_u4 < 0.35 or max_line_rep >= 8
    return {"uniq4": u4, "tail_uniq4": tail_u4, "max_line_rep": max_line_rep,
            "capped": capped, "repetitive": rep}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--cap", type=int, default=2048)
    args = ap.parse_args()
    for path in args.files:
        rows = [json.loads(l) for l in open(path)]
        audits = [audit_row(r, args.cap) for r in rows]
        n = len(rows)
        nrep = sum(a["repetitive"] for a in audits)
        ncap = sum(a["capped"] for a in audits)
        tpfs = [r["tpf"] for r in rows if isinstance(r.get("tpf"), (int, float))]
        clean_tpfs = [r["tpf"] for r, a in zip(rows, audits)
                      if isinstance(r.get("tpf"), (int, float)) and not a["repetitive"]]
        mean = lambda x: sum(x) / len(x) if x else float("nan")
        print(f"=== {path}")
        print(f"  n={n}  repetitive={nrep} ({100*nrep/max(1,n):.0f}%)  capped={ncap}  "
              f"mean uniq4={mean([a['uniq4'] for a in audits]):.3f}  "
              f"mean tail_uniq4={mean([a['tail_uniq4'] for a in audits]):.3f}")
        print(f"  TPF all={mean(tpfs):.3f}  TPF excl-repetitive={mean(clean_tpfs):.3f}  "
              f"delta={mean(tpfs)-mean(clean_tpfs):+.3f}")
        worst = sorted(zip(rows, audits), key=lambda x: x[1]["tail_uniq4"])[:3]
        for r, a in worst:
            tail = " ".join((r.get("completion") or "").split()[-30:])
            print(f"  worst tail_uniq4={a['tail_uniq4']:.2f} line_rep={a['max_line_rep']} "
                  f"tpf={r.get('tpf')} ntok={r.get('num_tokens')}  …{tail[-140:]!r}")


if __name__ == "__main__":
    main()
