"""Correlate noise-condition features with next-iter acceptance.

For each iter i (except last), build features that characterize the
"noisy condition" at iter i — i.e., the draft tokens fed to iter i and
the model's predictive uncertainty per position. Use n_acc[i+1] as the
quality score for iter i (since iter i's target_argmax becomes iter i+1's
draft, and n_acc[i+1] measures how many of those predictions agreed with
target_argmax of iter i+1).

Per-iter features (each is a K-dim vector or scalar):
  - entropy[j]      : H(p_θ(·|context_j)) at position j of iter i. (K-dim)
  - max_prob[j]     : max p_θ; (K-dim) — confidence.
  - draft[j] == target_argmax[j] : binary match per position. (K-dim, current iter)
  - n_acc[i]        : how many spec tokens iter i committed (PREVIOUS quality score).

Outputs:
  - HTML report with: scatter plots, per-bucket aggregations, heatmaps,
    and example iters for each n_acc[i+1] bucket.
"""
from __future__ import annotations
import argparse, glob, json, os, statistics, math
from collections import defaultdict
from html import escape


def load_runs(in_dir):
    runs = defaultdict(dict)
    for fp in sorted(glob.glob(os.path.join(in_dir, "p*__*.json"))):
        rec = json.load(open(fp))
        runs[rec["prompt_idx"]][rec["model_label"]] = rec
    return runs


def iter_features(rec):
    """Yield (i, features_dict) for each iter that has a 'next iter' to score it."""
    iters = rec["iters"]
    for i in range(len(iters) - 1):
        cur = iters[i]
        nxt = iters[i + 1]
        if not cur.get("num_draft") or not nxt.get("n_acc"):
            continue
        nd = cur["num_draft"][0]
        if nd <= 0:
            continue
        draft = cur["draft"][0][:nd]
        target = cur["target_argmax"][0][:nd]
        entropy = cur.get("target_entropy", [None])[0]
        max_prob = cur.get("target_max_prob", [None])[0]
        if entropy is not None:
            entropy = entropy[:nd]
            max_prob = max_prob[:nd]
        n_acc_cur = cur["n_acc"][0]
        n_acc_next = nxt["n_acc"][0]
        match_pattern = [int(d == t) for d, t in zip(draft, target)]
        yield i, {
            "iter": i,
            "n_acc_cur": n_acc_cur,
            "n_acc_next": n_acc_next,  # QUALITY SCORE for iter i
            "draft": draft,
            "target_argmax": target,
            "match_pattern": match_pattern,
            "entropy": entropy,
            "max_prob": max_prob,
            "K": nd,
        }


def bucket_iters_by_quality(features):
    """Group iters by n_acc_next bucket: 0, 1-3, 4-7, 8-15, 16+."""
    buckets = defaultdict(list)
    for f in features:
        n = f["n_acc_next"]
        if n == 0: bucket = "0"
        elif n <= 3: bucket = "1-3"
        elif n <= 7: bucket = "4-7"
        elif n <= 15: bucket = "8-15"
        else: bucket = "16+"
        buckets[bucket].append(f)
    return buckets


def aggregate_entropy_by_bucket(buckets):
    """For each bucket, compute mean entropy at each position (K-dim avg)."""
    out = {}
    for bname, feats in buckets.items():
        entropies = [f["entropy"] for f in feats if f["entropy"] is not None]
        if not entropies:
            out[bname] = None; continue
        K = max(len(e) for e in entropies)
        # Pad / truncate to K
        means = []
        for j in range(K):
            vals = [e[j] for e in entropies if j < len(e)]
            if vals: means.append(statistics.mean(vals))
            else: means.append(None)
        out[bname] = {"mean_entropy": means, "n": len(entropies)}
    return out


def aggregate_match_by_bucket(buckets):
    out = {}
    for bname, feats in buckets.items():
        K = max(f["K"] for f in feats) if feats else 0
        per_pos = []
        for j in range(K):
            vals = [f["match_pattern"][j] for f in feats if j < len(f["match_pattern"])]
            per_pos.append(statistics.mean(vals) if vals else None)
        out[bname] = {"match_freq": per_pos, "n": len(feats)}
    return out


def correlation(xs, ys):
    """Pearson r."""
    if len(xs) < 2: return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x - mx) * (y - my) for x, y in zip(xs, ys))
    dx = math.sqrt(sum((x - mx)**2 for x in xs))
    dy = math.sqrt(sum((y - my)**2 for y in ys))
    return num / (dx * dy) if dx > 0 and dy > 0 else float("nan")


def per_iter_scalars(features):
    """Reduce K-dim features to scalars for correlation analysis."""
    rows = []
    for f in features:
        if f["entropy"] is None:
            continue
        ent = f["entropy"]
        K = len(ent)
        rows.append({
            "n_acc_cur": f["n_acc_cur"],
            "n_acc_next": f["n_acc_next"],
            "mean_entropy": statistics.mean(ent) if ent else 0,
            "max_entropy": max(ent) if ent else 0,
            "min_entropy": min(ent) if ent else 0,
            "entropy_first_half": statistics.mean(ent[:K//2]) if K > 1 else 0,
            "entropy_second_half": statistics.mean(ent[K//2:]) if K > 1 else 0,
            "entropy_first_third": statistics.mean(ent[:K//3]) if K > 2 else 0,
            "entropy_last_third": statistics.mean(ent[-K//3:]) if K > 2 else 0,
            "match_count": sum(f["match_pattern"]),
            "match_ratio_first_half": (sum(f["match_pattern"][:K//2]) / (K//2)) if K > 1 else 0,
            "match_ratio_second_half": (sum(f["match_pattern"][K//2:]) / (K - K//2)) if K > 1 else 0,
            "K": K,
        })
    return rows


def html_report(runs, out_path):
    parts = ['<!doctype html><html><head><meta charset="utf-8">',
             '<title>Noise condition vs next-iter TPF</title><style>',
             'body{font-family:sans-serif;font-size:13px;background:#fafafa;max-width:1400px;margin:auto;padding:20px}',
             'h2{background:#333;color:#fff;padding:8px 12px;margin:24px 0 8px}',
             'h3{background:#666;color:#fff;padding:4px 8px;margin:12px 0 4px}',
             '.tbl{border-collapse:collapse;background:#fff;margin:8px 0;font-size:11px;font-family:monospace}',
             '.tbl th,.tbl td{border:1px solid #ddd;padding:3px 6px;text-align:right}',
             '.tbl th{background:#eee;text-align:center}',
             '.heatmap-row td{font-family:monospace;text-align:center;min-width:18px;font-size:9px}',
             '.posbucket-row{display:grid;grid-template-columns:90px repeat(32,1fr);gap:1px;margin-bottom:1px}',
             '.posbucket-row > div{padding:1px 0;text-align:center;font-size:9px;font-family:monospace}',
             '.posbucket-row > .lab{font-size:10px;text-align:left;padding-left:4px;background:#f3f3f3;font-weight:bold}',
             '.example-iter{background:#fff;padding:6px;margin:4px 0;border-left:3px solid #38f;font-family:monospace;font-size:11px}',
             '</style></head><body>',
             '<h1>Noise condition → next-iter n_acc correlation</h1>',
             '<p>For each iter <i>i</i> (except last), we measure the "noise condition" via the draft tokens fed to <i>i</i> and the per-position entropy of <i>i</i>\'s prediction. The quality score is <code>n_acc[i+1]</code> — how many of iter <i>i</i>\'s predictions iter <i>i+1</i> accepted as spec tokens.</p>']

    # ---- Per-model analysis ----
    for label in ("base", "consA_s300", "consB_s300"):
        all_feats = []
        for p_idx, by_label in runs.items():
            if label in by_label:
                all_feats.extend(list(f for _, f in iter_features(by_label[label])))
        if not all_feats:
            continue

        parts.append(f'<h2>{label}: {len(all_feats)} iter pairs</h2>')

        # n_acc_next distribution
        hist = defaultdict(int)
        for f in all_feats:
            hist[f["n_acc_next"]] += 1
        parts.append('<h3>n_acc[i+1] distribution (quality score for iter i)</h3>')
        parts.append('<table class="tbl"><tr><th>n_acc</th>')
        for n in sorted(hist.keys()):
            parts.append(f'<th>{n}</th>')
        parts.append('</tr><tr><th>count</th>')
        for n in sorted(hist.keys()):
            parts.append(f'<td>{hist[n]}</td>')
        parts.append('</tr><tr><th>%</th>')
        tot = sum(hist.values())
        for n in sorted(hist.keys()):
            parts.append(f'<td>{100*hist[n]/tot:.1f}</td>')
        parts.append('</tr></table>')

        # Buckets
        buckets = bucket_iters_by_quality(all_feats)
        order = ["0", "1-3", "4-7", "8-15", "16+"]

        # Bucket sizes
        parts.append('<h3>iter pairs per quality bucket</h3>')
        parts.append('<table class="tbl"><tr><th>bucket</th>')
        for b in order:
            parts.append(f'<th>{b}</th>')
        parts.append('</tr><tr><th>count</th>')
        for b in order:
            parts.append(f'<td>{len(buckets.get(b, []))}</td>')
        parts.append('</tr></table>')

        # Mean entropy per position per bucket (heatmap-ish)
        ent_agg = aggregate_entropy_by_bucket(buckets)
        if any(v is not None for v in ent_agg.values()):
            parts.append('<h3>Mean per-position prediction entropy (rows = quality bucket)</h3>')
            parts.append('<p>Each column is position j ∈ [0, K). Higher entropy = more uncertain prediction at that position. Color = entropy magnitude (red=high uncertainty).</p>')
            for b in order:
                d = ent_agg.get(b)
                if d is None: continue
                means = d["mean_entropy"]
                row = [f'<div class="lab">{b} (n={d["n"]})</div>']
                for v in means[:32]:
                    if v is None:
                        row.append('<div style="background:#eee"></div>')
                    else:
                        # color: low entropy=green, high=red
                        norm = min(1.0, v / 5.0)
                        r = int(220 + 35 * norm); g = int(255 - 80 * norm); bl = int(220 - 80 * norm)
                        row.append(f'<div style="background:rgb({r},{g},{bl})" title="ent={v:.2f}">{v:.1f}</div>')
                parts.append(f'<div class="posbucket-row">{"".join(row)}</div>')

        # Mean match frequency per position per bucket
        match_agg = aggregate_match_by_bucket(buckets)
        parts.append('<h3>Mean draft==target_argmax match frequency per position (rows = quality bucket)</h3>')
        parts.append('<p>Position-wise probability that <code>draft[j] == target_argmax[j]</code> at iter <i>i</i>. High = matched-prefix region. Color: red=high match, white=low match.</p>')
        for b in order:
            d = match_agg.get(b)
            if d is None or d["n"] == 0: continue
            mf = d["match_freq"]
            row = [f'<div class="lab">{b} (n={d["n"]})</div>']
            for v in mf[:32]:
                if v is None:
                    row.append('<div style="background:#eee"></div>')
                else:
                    g = int(255 * (1 - v)); r = 255; bl = int(255 * (1 - v))
                    row.append(f'<div style="background:rgb({r},{g},{bl})" title="match_p={v:.2f}">{v:.2f}</div>')
            parts.append(f'<div class="posbucket-row">{"".join(row)}</div>')

        # Scalar correlations
        scalars = per_iter_scalars(all_feats)
        if scalars:
            parts.append('<h3>Scalar feature correlations with n_acc[i+1]</h3>')
            features_to_correlate = ["n_acc_cur", "mean_entropy", "max_entropy",
                                      "entropy_first_half", "entropy_second_half",
                                      "entropy_first_third", "entropy_last_third",
                                      "match_count", "match_ratio_first_half", "match_ratio_second_half"]
            ys = [s["n_acc_next"] for s in scalars]
            parts.append('<table class="tbl"><tr><th>feature</th><th>Pearson r</th><th>n</th></tr>')
            for k in features_to_correlate:
                xs = [s[k] for s in scalars]
                r = correlation(xs, ys)
                parts.append(f'<tr><td style="text-align:left">{k}</td><td>{r:+.3f}</td><td>{len(xs)}</td></tr>')
            parts.append('</table>')

            # Mean of each feature per bucket
            parts.append('<h3>Feature means by quality bucket</h3>')
            parts.append('<table class="tbl"><tr><th>feature</th>')
            for b in order:
                parts.append(f'<th>{b}</th>')
            parts.append('</tr>')
            for k in features_to_correlate:
                parts.append(f'<tr><td style="text-align:left">{k}</td>')
                for b in order:
                    vals = [s[k] for s in scalars
                            if s["n_acc_next"] in {0:[0], "0":[0]}.get(b, []) and False] or \
                           [s[k] for s in scalars if (
                               (b=="0" and s["n_acc_next"]==0) or
                               (b=="1-3" and 1<=s["n_acc_next"]<=3) or
                               (b=="4-7" and 4<=s["n_acc_next"]<=7) or
                               (b=="8-15" and 8<=s["n_acc_next"]<=15) or
                               (b=="16+" and s["n_acc_next"]>=16))]
                    if vals:
                        m = statistics.mean(vals)
                        parts.append(f'<td>{m:.2f}</td>')
                    else:
                        parts.append('<td>—</td>')
                parts.append('</tr>')
            parts.append('</table>')

    parts.append('</body></html>')
    with open(out_path, "w") as fp:
        fp.write("".join(parts))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    p.add_argument("--html_out", required=True)
    args = p.parse_args()

    runs = load_runs(args.in_dir)
    print(f"Loaded {len(runs)} prompts × ~3 models")
    html_report(runs, args.html_out)
    print(f"Wrote {args.html_out}")


if __name__ == "__main__":
    main()
