"""Analyze captured trajectories + render HTML side-by-side.

Looks at per-iter records for {base, consA_s300, consB_s300} on the same prompts.
Output HTML showing:
  - per prompt, per model: each iter's draft tokens vs target_argmax tokens,
    colored by acceptance (green=accepted, orange=overwritten-by-target, gray=truncated)
  - per-iter n_acc bar
  - aggregate distribution: histogram of n_acc per model
  - "bottleneck" stat: when n_acc=0, what does the draft look like?
"""
from __future__ import annotations
import argparse, glob, json, os, statistics
from collections import Counter, defaultdict
from html import escape


def load_runs(in_dir):
    runs = defaultdict(dict)  # prompt_idx -> label -> record
    for fp in sorted(glob.glob(os.path.join(in_dir, "p*__*.json"))):
        rec = json.load(open(fp))
        runs[rec["prompt_idx"]][rec["model_label"]] = rec
    return runs


def tok_color(draft_tok, target_tok, accepted, K):
    if accepted:
        return "#bff6b3"  # green — accepted
    elif draft_tok == target_tok:
        return "#e8d5ff"  # purple — agreement but truncated by earlier reject
    else:
        return "#ffd5d5"  # red — mismatch


def decode_iter(rec, tokenizer):
    """Return (draft_strs, target_strs, n_acc) for the first request slot."""
    nd = rec["num_draft"][0] if rec["num_draft"] else 0
    if nd <= 0:
        return [], [], 0
    draft = rec["draft"][0] if rec["draft"] else []
    target = rec["target_argmax"][0] if rec["target_argmax"] else []
    n_acc = rec["n_acc"][0] if rec["n_acc"] else 0
    bonus = rec["bonus"][0] if rec.get("bonus") else None
    # Decode each token individually for display
    draft_strs = [tokenizer.decode([t]) if t >= 0 else "" for t in draft[:nd]]
    target_strs = [tokenizer.decode([t]) if t >= 0 else "" for t in target[:nd]]
    return draft_strs, target_strs, n_acc, draft, target, bonus


def render_iter_row(it_idx, draft_strs, target_strs, draft_ids, target_ids, n_acc, K):
    """Render one HTML row for a single iter."""
    cells = []
    nd = len(draft_strs)
    for i in range(nd):
        ds = escape(draft_strs[i].replace("\n", "\\n").replace(" ", "·")[:8]) or "·"
        ts = escape(target_strs[i].replace("\n", "\\n").replace(" ", "·")[:8]) or "·"
        accepted = i < n_acc
        # Match status
        match = draft_ids[i] == target_ids[i]
        if accepted:
            cls = "acc"  # green: both accept and match by definition
        elif i == n_acc:
            cls = "rej"  # red: this is where rejection happens
        else:
            cls = "trunc" if match else "trunc_mm"
        cells.append(
            f'<td class="{cls}" title="draft={draft_ids[i]} target={target_ids[i]}">'
            f'<div class="d">{ds}</div><div class="t">{ts}</div></td>'
        )
    # Pad to K for alignment
    for _ in range(K - nd):
        cells.append('<td class="pad"></td>')
    label = f"iter {it_idx:3d} | n_acc={n_acc}"
    return f'<tr><th class="iter-label">{label}</th>{"".join(cells)}</tr>'


def render_prompt_block(prompt_idx, runs_for_p, tokenizer, K):
    """Render one prompt section across all models."""
    parts = [f'<section><h2>Prompt {prompt_idx}</h2>']
    if "base" in runs_for_p:
        p = runs_for_p["base"]["prompt"][:200]
        parts.append(f'<div class="prompt-text">{escape(p)}{"..." if len(p)==200 else ""}</div>')
    for label in ("base", "consA_s300", "consB_s300"):
        if label not in runs_for_p:
            continue
        rec = runs_for_p[label]
        n_iters = rec["n_iters"]
        tpf = rec["tpf"]
        comp = rec["completion"][:300].replace("\n", "<br>")
        parts.append(f'<div class="model-block"><h3>{label} — TPF={tpf:.3f}, n_iters={n_iters}, n_tok={rec["n_tokens"]}</h3>')
        parts.append(f'<details><summary>completion (truncated 300c)</summary><div class="completion">{escape(comp)}</div></details>')
        parts.append('<table class="trace-table"><thead><tr><th>iter</th>' +
                     "".join(f'<th>p{i}</th>' for i in range(K)) +
                     '</tr></thead><tbody>')
        for it_idx, it in enumerate(rec["iters"]):
            draft_strs, target_strs, n_acc, draft_ids, target_ids, _ = decode_iter(it, tokenizer)
            if not draft_strs:
                continue
            parts.append(render_iter_row(it_idx, draft_strs, target_strs, draft_ids, target_ids, n_acc, K))
        parts.append('</tbody></table></div>')
    parts.append('</section>')
    return "".join(parts)


def aggregate_stats(runs):
    """Per-model: distribution of n_acc, p(n_acc=0|...), etc."""
    stats = defaultdict(lambda: {"n_acc_hist": Counter(), "total_iters": 0,
                                  "zero_acc_drafts": [], "high_acc_drafts": []})
    for p_idx, by_label in runs.items():
        for label, rec in by_label.items():
            for it in rec["iters"]:
                if not it.get("num_draft"):
                    continue
                nd = it["num_draft"][0]
                if nd <= 0:
                    continue
                n_acc = it["n_acc"][0]
                stats[label]["n_acc_hist"][n_acc] += 1
                stats[label]["total_iters"] += 1
                if n_acc == 0:
                    stats[label]["zero_acc_drafts"].append((it["draft"][0], it["target_argmax"][0]))
                elif n_acc >= nd - 2:  # nearly full acceptance
                    stats[label]["high_acc_drafts"].append((it["draft"][0], it["target_argmax"][0]))
    return stats


def render_stats(stats):
    parts = ['<section><h2>Aggregate statistics</h2>']
    # n_acc histogram per model
    parts.append('<h3>n_acc distribution (per iter)</h3>')
    parts.append('<table class="stats-table"><thead><tr><th>n_acc</th>'
                 + "".join(f'<th>{l}</th>' for l in stats.keys()) + '</tr></thead><tbody>')
    max_n = max(max(s["n_acc_hist"].keys() or [0]) for s in stats.values()) if stats else 0
    for n in range(max_n + 1):
        row = [f'<td>{n}</td>']
        for label, s in stats.items():
            count = s["n_acc_hist"].get(n, 0)
            pct = 100 * count / max(1, s["total_iters"])
            row.append(f'<td>{count} ({pct:.1f}%)</td>')
        parts.append('<tr>' + "".join(row) + '</tr>')
    parts.append('</tbody></table>')

    # Summary stats per model
    parts.append('<h3>Per-model summary</h3>')
    parts.append('<table class="stats-table"><thead><tr><th>metric</th>'
                 + "".join(f'<th>{l}</th>' for l in stats.keys()) + '</tr></thead><tbody>')
    for metric_name, fn in [
        ("total_iters", lambda s: s["total_iters"]),
        ("mean n_acc", lambda s: statistics.mean(
            [n * c for n, c in s["n_acc_hist"].items() for _ in range(c)]) if s["total_iters"] else 0),
        ("p(n_acc=0)", lambda s: s["n_acc_hist"].get(0, 0) / max(1, s["total_iters"])),
        ("p(n_acc>=K/2)", lambda s: sum(c for n, c in s["n_acc_hist"].items() if n >= 16) / max(1, s["total_iters"])),
    ]:
        row = [f'<td><b>{metric_name}</b></td>']
        for label, s in stats.items():
            val = fn(s)
            row.append(f'<td>{val:.3f}</td>' if isinstance(val, float) else f'<td>{val}</td>')
        parts.append('<tr>' + "".join(row) + '</tr>')
    parts.append('</tbody></table></section>')
    return "".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    p.add_argument("--model_path", required=True, help="HF model dir for tokenizer")
    p.add_argument("--html_out", required=True)
    p.add_argument("--stats_out", required=True)
    args = p.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    runs = load_runs(args.in_dir)
    print(f"Loaded {len(runs)} prompts")
    K = 32
    for p_idx, by_label in runs.items():
        for label, rec in by_label.items():
            K = rec.get("K", 32)
            break
        break

    stats = aggregate_stats(runs)
    json.dump({label: {"n_acc_hist": dict(s["n_acc_hist"]), "total_iters": s["total_iters"]}
               for label, s in stats.items()}, open(args.stats_out, "w"), indent=2)

    html_parts = ['<!doctype html><html><head><meta charset="utf-8">',
                  '<title>Jacobi trajectory viz</title><style>',
                  'body{font-family:monospace;font-size:11px;background:#fafafa}',
                  'h2{background:#333;color:#fff;padding:8px;margin:24px 0 8px}',
                  'h3{background:#666;color:#fff;padding:4px 8px;margin:12px 0 4px}',
                  '.prompt-text{padding:8px;background:#fff;border-left:4px solid #38f;margin-bottom:8px}',
                  '.model-block{margin-bottom:16px}',
                  '.completion{padding:6px;background:#fff;font-size:10px;max-height:120px;overflow:auto}',
                  '.trace-table{border-collapse:collapse;background:#fff}',
                  '.trace-table th,.trace-table td{border:1px solid #ddd;padding:1px 2px;text-align:center;min-width:24px}',
                  '.trace-table th.iter-label{text-align:left;padding:1px 6px;background:#eee;font-weight:normal}',
                  '.trace-table .d{font-size:9px;color:#666}',
                  '.trace-table .t{font-size:9px;font-weight:bold}',
                  '.trace-table .acc{background:#bff6b3}',
                  '.trace-table .rej{background:#ffb3b3}',
                  '.trace-table .trunc{background:#e8d5ff}',
                  '.trace-table .trunc_mm{background:#fff3cd}',
                  '.trace-table .pad{background:#f3f3f3}',
                  '.stats-table{border-collapse:collapse;margin:8px;background:#fff}',
                  '.stats-table th,.stats-table td{border:1px solid #ddd;padding:4px 8px}',
                  'details summary{cursor:pointer;background:#eee;padding:4px}',
                  '</style></head><body>',
                  '<h1>Jacobi trajectory viz — base vs consA-step300 vs consB-corrupt03-step300</h1>',
                  '<p><b>Legend (cell colors):</b> '
                  '<span style="background:#bff6b3">green = draft accepted (target_argmax == draft)</span> | '
                  '<span style="background:#ffb3b3">red = rejection point (first mismatch, becomes the bonus)</span> | '
                  '<span style="background:#e8d5ff">purple = match but truncated (rejected at earlier position)</span> | '
                  '<span style="background:#fff3cd">yellow = mismatch and truncated</span></p>',
                  '<p>Each cell shows draft token (small, top) and target_argmax token (bold, bottom). · = space.</p>',
                  render_stats(stats)]

    for p_idx in sorted(runs.keys()):
        html_parts.append(render_prompt_block(p_idx, runs[p_idx], tokenizer, K))

    html_parts.append('</body></html>')
    with open(args.html_out, "w") as fp:
        fp.write("".join(html_parts))
    print(f"Wrote {args.html_out}")
    print(f"Wrote {args.stats_out}")


if __name__ == "__main__":
    main()
