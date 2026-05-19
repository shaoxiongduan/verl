"""Repetition + TPF analysis on Jacobi trajectories.

Repetition definition (deliberately *not* local n-gram repeat, which is the
"from typing import List" pathology that frequency_penalty would catch):
  - "Big chunk repetition" = the completion contains a contiguous substring
    of length >= --min_chunk_chars that appears at least 2 times.
  - We compute the longest repeated substring (LRS) via suffix-array O(N log N)
    and treat LRS >= threshold as a repetition flag.
  - Plus a JF-specific check: count distinct `def NAME(` occurrences in the
    completion. If the model emits the SAME function name with >=2 defs,
    flag as 'restart' — the canonical "JF model restarts the solution"
    failure mode.

Outputs:
  - JSON summary: per-model mean TPF, fraction repeating, mean TPF stratified
  - PNG plots: TPF histograms (overall + repeating-vs-not), in eval_passk/plots/
  - Markdown report: eval_passk/plots/repetition_report.md
"""

import argparse
import json
import os
import re
from collections import Counter
from typing import Dict, List

import numpy as np


# ---------------------------------------------------------------------------
# Repetition primitives
# ---------------------------------------------------------------------------

def longest_repeated_substring(s: str) -> int:
    """Return the length of the longest substring that occurs at least 2x.
    O(N log N) via suffix array + LCP (Kasai). For N<10k this is fine."""
    n = len(s)
    if n < 2:
        return 0
    # Suffix array
    sa = sorted(range(n), key=lambda i: s[i:])
    # LCP via Kasai
    rank = [0] * n
    for i, p in enumerate(sa):
        rank[p] = i
    lcp_max = 0
    h = 0
    for i in range(n):
        if rank[i] > 0:
            j = sa[rank[i] - 1]
            while i + h < n and j + h < n and s[i + h] == s[j + h]:
                h += 1
            if h > lcp_max:
                lcp_max = h
            if h > 0:
                h -= 1
        else:
            h = 0
    return lcp_max


_DEF_RE = re.compile(r"^\s*def\s+([A-Za-z_][A-Za-z0-9_]*)\s*\(", re.MULTILINE)


def count_redefs(s: str) -> Dict[str, int]:
    """Count occurrences of each `def NAME(` in the completion."""
    return dict(Counter(_DEF_RE.findall(s)))


def has_restart(s: str) -> bool:
    return any(c >= 2 for c in count_redefs(s).values())


def repetition_flag(completion: str, min_chunk_chars: int) -> Dict:
    lrs = longest_repeated_substring(completion)
    restart = has_restart(completion)
    return {
        "lrs_chars": lrs,
        "lrs_repeats": lrs >= min_chunk_chars,
        "restart": restart,
        "repetition": (lrs >= min_chunk_chars) or restart,
    }


# ---------------------------------------------------------------------------
# Stats + plots
# ---------------------------------------------------------------------------

def load_rows(path: str, min_chunk_chars: int) -> List[Dict]:
    rows = []
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            r = json.loads(line)
            r["rep"] = repetition_flag(r["completion"], min_chunk_chars)
            rows.append(r)
    return rows


def summarize(rows: List[Dict], name: str) -> Dict:
    n = len(rows)
    tpf = np.array([r["tpf"] for r in rows], dtype=float)
    rep_mask = np.array([r["rep"]["repetition"] for r in rows], dtype=bool)
    restart_mask = np.array([r["rep"]["restart"] for r in rows], dtype=bool)
    lrs_mask = np.array([r["rep"]["lrs_repeats"] for r in rows], dtype=bool)
    return {
        "name": name,
        "n": n,
        "tpf_mean": float(tpf.mean()),
        "tpf_median": float(np.median(tpf)),
        "tpf_std": float(tpf.std()),
        "tpf_min": float(tpf.min()),
        "tpf_max": float(tpf.max()),
        "frac_repetition": float(rep_mask.mean()),
        "frac_restart": float(restart_mask.mean()),
        "frac_lrs": float(lrs_mask.mean()),
        "tpf_rep_mean": float(tpf[rep_mask].mean()) if rep_mask.any() else float("nan"),
        "tpf_nonrep_mean": float(tpf[~rep_mask].mean()) if (~rep_mask).any() else float("nan"),
        "n_rep": int(rep_mask.sum()),
        "n_nonrep": int((~rep_mask).sum()),
    }


def make_plots(baseline_rows: List[Dict], post_rows: List[Dict], out_dir: str):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    os.makedirs(out_dir, exist_ok=True)
    b_tpf = np.array([r["tpf"] for r in baseline_rows])
    p_tpf = np.array([r["tpf"] for r in post_rows])
    b_rep = np.array([r["rep"]["repetition"] for r in baseline_rows], dtype=bool)
    p_rep = np.array([r["rep"]["repetition"] for r in post_rows], dtype=bool)

    # Plot 1: overall TPF distribution overlap
    fig, ax = plt.subplots(figsize=(8, 5))
    bins = np.linspace(min(b_tpf.min(), p_tpf.min()), max(b_tpf.max(), p_tpf.max()), 40)
    ax.hist(b_tpf, bins=bins, alpha=0.55, label=f"baseline (μ={b_tpf.mean():.2f})", color="#4C72B0")
    ax.hist(p_tpf, bins=bins, alpha=0.55, label=f"step 260 (μ={p_tpf.mean():.2f})", color="#DD8452")
    ax.axvline(b_tpf.mean(), color="#4C72B0", linestyle="--")
    ax.axvline(p_tpf.mean(), color="#DD8452", linestyle="--")
    ax.set_xlabel("TPF (tokens per forward pass)")
    ax.set_ylabel("Count (prompts)")
    ax.set_title("Per-prompt TPF distribution — baseline vs RL step 260 (HumanEval+)")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tpf_distribution.png"), dpi=150)
    plt.close(fig)

    # Plot 2: TPF stratified by repetition flag
    fig, axes = plt.subplots(1, 2, figsize=(13, 5), sharey=True)
    for ax, tpf, mask, title in [
        (axes[0], b_tpf, b_rep, "baseline"),
        (axes[1], p_tpf, p_rep, "step 260"),
    ]:
        tr = tpf[mask]
        tnr = tpf[~mask]
        bins = np.linspace(tpf.min(), tpf.max(), 30)
        if tr.size:
            ax.hist(tr, bins=bins, alpha=0.65, label=f"repeating  n={tr.size} μ={tr.mean():.2f}", color="#C44E52")
        if tnr.size:
            ax.hist(tnr, bins=bins, alpha=0.65, label=f"non-rep    n={tnr.size} μ={tnr.mean():.2f}", color="#55A868")
        ax.set_xlabel("TPF")
        ax.set_title(title)
        ax.legend()
    axes[0].set_ylabel("Count")
    fig.suptitle("TPF stratified by repetition flag")
    fig.tight_layout()
    fig.savefig(os.path.join(out_dir, "tpf_by_repetition.png"), dpi=150)
    plt.close(fig)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--baseline", required=True, help="trajectory JSONL for baseline model")
    p.add_argument("--post", required=True, help="trajectory JSONL for post-RL model")
    p.add_argument("--out_dir", default="eval_passk/plots")
    p.add_argument("--min_chunk_chars", type=int, default=100,
                   help="LRS threshold for flagging 'big chunk' repetition")
    args = p.parse_args()

    baseline_rows = load_rows(args.baseline, args.min_chunk_chars)
    post_rows = load_rows(args.post, args.min_chunk_chars)

    b = summarize(baseline_rows, "baseline")
    p_ = summarize(post_rows, "step 260")
    print(json.dumps({"baseline": b, "step_260": p_}, indent=2))

    make_plots(baseline_rows, post_rows, args.out_dir)

    # Markdown report
    lines = [
        "# JF Coder TPF & repetition analysis (HumanEval+, T=0.6)",
        "",
        f"- LRS chunk threshold: {args.min_chunk_chars} chars",
        "",
        "## Summary",
        "",
        "| | Baseline | Step 260 |",
        "|---|---|---|",
        f"| n | {b['n']} | {p_['n']} |",
        f"| TPF mean | {b['tpf_mean']:.3f} | {p_['tpf_mean']:.3f} |",
        f"| TPF median | {b['tpf_median']:.3f} | {p_['tpf_median']:.3f} |",
        f"| % repeating | {b['frac_repetition']*100:.1f}% | {p_['frac_repetition']*100:.1f}% |",
        f"| % restart (def-redef) | {b['frac_restart']*100:.1f}% | {p_['frac_restart']*100:.1f}% |",
        f"| % LRS≥{args.min_chunk_chars}c | {b['frac_lrs']*100:.1f}% | {p_['frac_lrs']*100:.1f}% |",
        f"| TPF on repeating | {b['tpf_rep_mean']:.3f} (n={b['n_rep']}) | {p_['tpf_rep_mean']:.3f} (n={p_['n_rep']}) |",
        f"| TPF on non-rep | {b['tpf_nonrep_mean']:.3f} (n={b['n_nonrep']}) | {p_['tpf_nonrep_mean']:.3f} (n={p_['n_nonrep']}) |",
        "",
        "## Hypothesis test",
        "",
        "Pre-experiment claim: \"Repeating responses have higher TPF; RL may have reduced TPF *implicitly* by killing repetition.\"",
        "",
        f"- Baseline TPF gap rep − non-rep: **{b['tpf_rep_mean'] - b['tpf_nonrep_mean']:+.3f}**",
        f"- Step 260  TPF gap rep − non-rep: **{p_['tpf_rep_mean'] - p_['tpf_nonrep_mean']:+.3f}**",
        f"- Repetition rate Δ: **{b['frac_repetition']*100:.1f}% → {p_['frac_repetition']*100:.1f}%** (Δ={p_['frac_repetition']*100 - b['frac_repetition']*100:+.1f}pp)",
        f"- Non-repeating-only TPF Δ: **{p_['tpf_nonrep_mean'] - b['tpf_nonrep_mean']:+.3f}** (this isolates direct RL effect on TPF)",
        "",
        "## Plots",
        "- `tpf_distribution.png` — per-prompt TPF histograms, both models",
        "- `tpf_by_repetition.png` — TPF stratified by repetition flag",
    ]
    md_path = os.path.join(args.out_dir, "repetition_report.md")
    with open(md_path, "w") as f:
        f.write("\n".join(lines))
    print(f"\nWrote report: {md_path}")
    print(f"Wrote plots:  {args.out_dir}/tpf_distribution.png  {args.out_dir}/tpf_by_repetition.png")


if __name__ == "__main__":
    main()
