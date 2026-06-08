"""Analyze the oracle-reinit grid output: per-mode mean/SE TPF, per-mode mean
acceptance count, and delta vs the natural baseline.

Input: a directory containing JSONL files named {TAG}__<mode>.jsonl, each row
is one prompt's simulate() output (see _sim_jacobi_oracle_reinit.py).

Usage:
    python3 scripts/_analyze_oracle_reinit.py \\
        --tag oracle_reinit_v1 --indir eval_passk/tpf_results
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
    p.add_argument("--print_per_prompt", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    pattern = os.path.join(args.indir, f"{args.tag}__*.jsonl")
    files = sorted(glob.glob(pattern))
    files = [f for f in files if not f.endswith("__merged.jsonl")]
    if not files:
        raise SystemExit(f"no files matched {pattern}")

    # mode -> list of (batch_idx, tpf, n_tok, n_iters, n_acc_real, n_acc_verify)
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
                    "oracle_shift2", "oracle_shift3", "oracle_shift4"]
                   if m in by_mode]
    # Append any unknown modes after.
    for m in by_mode:
        if m not in modes_order:
            modes_order.append(m)

    baseline_tpf_by_idx = {r["batch_idx"]: r["tpf"]
                           for r in by_mode.get("natural", [])}

    print(f"\n=== {args.tag} ===")
    print(f"{'mode':16s} {'n':>4s} {'mean_TPF':>10s} {'SE':>7s} "
          f"{'min':>6s} {'max':>6s} {'mean_nacc_real':>15s} "
          f"{'mean_nacc_verify':>17s} {'Δ_vs_natural':>14s} "
          f"{'paired_Δ':>10s} {'paired_SE':>10s}")
    for mode in modes_order:
        rows = by_mode[mode]
        if not rows:
            continue
        tpfs = [r["tpf"] for r in rows]
        n = len(tpfs)
        mean_t = sum(tpfs) / n
        var = sum((x - mean_t) ** 2 for x in tpfs) / max(1, n - 1)
        se = math.sqrt(var / n)
        mean_acc_r = sum(r["mean_n_acc_real"] for r in rows) / n
        v_list = [r["mean_n_acc_verify"] for r in rows
                  if r.get("mean_n_acc_verify") is not None]
        mean_acc_v = (sum(v_list) / len(v_list)) if v_list else None
        v_str = f"{mean_acc_v:17.3f}" if mean_acc_v is not None else f"{'—':>17s}"

        # Δ vs baseline (paired by batch_idx).
        delta_str = "—"
        paired_d_str = "—"
        paired_se_str = "—"
        if mode != "natural" and baseline_tpf_by_idx:
            deltas = []
            for r in rows:
                bi = r["batch_idx"]
                if bi in baseline_tpf_by_idx:
                    deltas.append(r["tpf"] - baseline_tpf_by_idx[bi])
            if deltas:
                md = sum(deltas) / len(deltas)
                dvar = sum((x - md) ** 2 for x in deltas) / max(1, len(deltas) - 1)
                dse = math.sqrt(dvar / len(deltas))
                delta_str = f"{md:+.3f}"
                paired_d_str = f"{md:+.3f}"
                paired_se_str = f"{dse:.3f}"

        print(f"{mode:16s} {n:4d} {mean_t:10.3f} {se:7.3f} "
              f"{min(tpfs):6.2f} {max(tpfs):6.2f} {mean_acc_r:15.3f} "
              f"{v_str} {delta_str:>14s} {paired_d_str:>10s} {paired_se_str:>10s}")

    if args.print_per_prompt and baseline_tpf_by_idx:
        # Side-by-side per-prompt comparison.
        print("\nper-prompt TPF (sorted by batch_idx):")
        cols = modes_order
        print(f"  idx | " + " | ".join(f"{c:>13s}" for c in cols))
        all_idx = sorted({r["batch_idx"] for rows in by_mode.values() for r in rows})
        for idx in all_idx:
            row_vals = []
            for c in cols:
                v = next((r["tpf"] for r in by_mode[c] if r["batch_idx"] == idx), None)
                row_vals.append(f"{v:13.3f}" if v is not None else f"{'—':>13s}")
            print(f"  {idx:3d} | " + " | ".join(row_vals))


if __name__ == "__main__":
    main()
