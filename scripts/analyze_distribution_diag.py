"""Analyze the JSONL outputs from scripts/diagnose_distribution.py.

Reports for each JSONL:
- Average per-token entropy across all completions.
- Average top-1 probability.
- Entropy at percentiles (P10, P50, P90) — to characterize the distribution shape.
- Position-binned average entropy (first 25% / second 25% / third 25% / last 25% of completion).
- If JS divergence present: mean JS, JS at percentiles, highest-JS positions.

Useful to compare base vs AR vs cons checkpoints side-by-side.

Usage:
  python scripts/analyze_distribution_diag.py \
    --jsonls base_he.jsonl ar300_he.jsonl ar_vs_cons.jsonl \
    --tags base ar_v2_300 ar_vs_cons300
"""
from __future__ import annotations
import argparse
import json
from statistics import mean


def percentile(xs: list[float], pct: float) -> float:
    if not xs:
        return float("nan")
    xs_sorted = sorted(xs)
    k = (len(xs_sorted) - 1) * pct
    f = int(k)
    c = min(f + 1, len(xs_sorted) - 1)
    if f == c:
        return xs_sorted[f]
    return xs_sorted[f] * (c - k) + xs_sorted[c] * (k - f)


def analyze(rows: list[dict]) -> dict:
    all_H = []
    all_top1 = []
    H_q1, H_q2, H_q3, H_q4 = [], [], [], []  # entropy by position quartile
    all_H2 = []
    all_js = []
    for r in rows:
        H = r.get("entropy_per_step", [])
        p1 = r.get("top1_p_per_step", [])
        if not H:
            continue
        all_H.extend(H)
        all_top1.extend(p1)
        n = len(H)
        q = max(1, n // 4)
        H_q1.extend(H[:q])
        H_q2.extend(H[q:2*q])
        H_q3.extend(H[2*q:3*q])
        H_q4.extend(H[3*q:])
        if "entropy2_per_step" in r:
            all_H2.extend(r["entropy2_per_step"])
        if "js_per_step" in r:
            all_js.extend(r["js_per_step"])
    out = {
        "n_prompts": len(rows),
        "n_tokens": len(all_H),
        "H_mean": mean(all_H) if all_H else None,
        "H_p10": percentile(all_H, 0.10),
        "H_p50": percentile(all_H, 0.50),
        "H_p90": percentile(all_H, 0.90),
        "top1_mean": mean(all_top1) if all_top1 else None,
        "H_quartile_means": (mean(H_q1) if H_q1 else None,
                             mean(H_q2) if H_q2 else None,
                             mean(H_q3) if H_q3 else None,
                             mean(H_q4) if H_q4 else None),
    }
    if all_H2:
        out["H2_mean"] = mean(all_H2)
        out["H2_p50"] = percentile(all_H2, 0.50)
    if all_js:
        out["JS_mean"] = mean(all_js)
        out["JS_p50"] = percentile(all_js, 0.50)
        out["JS_p90"] = percentile(all_js, 0.90)
        out["JS_p99"] = percentile(all_js, 0.99)
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--jsonls", nargs="+", required=True)
    p.add_argument("--tags", nargs="+", required=True)
    args = p.parse_args()
    assert len(args.jsonls) == len(args.tags)

    summaries = {}
    for tag, path in zip(args.tags, args.jsonls):
        rows = [json.loads(l) for l in open(path)]
        summaries[tag] = analyze(rows)

    print("=" * 100)
    print(f"{'metric':>22} | " + " | ".join(f"{t:>14}" for t in args.tags))
    print("-" * 100)
    for metric in ["n_prompts", "n_tokens", "H_mean", "H_p10", "H_p50", "H_p90",
                   "top1_mean", "H2_mean", "H2_p50", "JS_mean", "JS_p50",
                   "JS_p90", "JS_p99"]:
        row = f"{metric:>22} | "
        for t in args.tags:
            v = summaries[t].get(metric)
            if v is None:
                row += f"{'-':>14} | "
            elif isinstance(v, int):
                row += f"{v:>14d} | "
            else:
                row += f"{v:>14.4f} | "
        print(row)

    print()
    print("Entropy by position quartile (Q1=first 25% of tokens, Q4=last 25%):")
    print(f"{'tag':>22} | {'Q1':>8} {'Q2':>8} {'Q3':>8} {'Q4':>8}")
    print("-" * 70)
    for t in args.tags:
        qs = summaries[t].get("H_quartile_means", (None, None, None, None))
        row = f"{t:>22} | "
        row += " ".join(f"{q:>8.3f}" if q is not None else f"{'-':>8}" for q in qs)
        print(row)


if __name__ == "__main__":
    main()
