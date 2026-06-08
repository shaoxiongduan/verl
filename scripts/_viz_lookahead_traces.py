"""HTML visualization of Jacobi lookahead sim per-iter traces.

Renders one panel per prompt with:
  - vanilla TPF + n_acc_history sparkline (per-iter accepted count)
  - lookahead TPF + n_acc_history sparkline
  - vanilla completion preview vs lookahead completion preview vs ref

Usage:
    python3 scripts/_viz_lookahead_traces.py \\
        --in_jsonl eval_passk/diag_traces/base_sim/lookahead_K8_verbose.jsonl \\
        --prompts_jsonl eval_passk/deepscaler_tpf_prompts_16.jsonl \\
        --out_html eval_passk/diag_traces/base_sim/lookahead_viz.html
"""
from __future__ import annotations

import argparse
import html
import json


def sparkline(values: list[int], max_bar: int = 32, width: int = 600) -> str:
    if not values:
        return ""
    bar_w = max(2, width // max(1, len(values)))
    bars = []
    for v in values:
        h = min(max_bar, max(0, v))
        # color: green if v >= 28, blue if 10+, gray if <5, red if 0
        if v >= 28:
            color = "#2ecc71"
        elif v >= 10:
            color = "#3498db"
        elif v >= 3:
            color = "#95a5a6"
        elif v >= 1:
            color = "#e67e22"
        else:
            color = "#e74c3c"
        bars.append(
            f'<rect x="{len(bars)*bar_w}" y="{max_bar - h}" width="{bar_w-1}" '
            f'height="{h}" fill="{color}"/>'
        )
    return f'<svg width="{width}" height="{max_bar+4}" style="vertical-align: middle">{"".join(bars)}</svg>'


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--in_jsonl", required=True)
    p.add_argument("--prompts_jsonl", required=True)
    p.add_argument("--out_html", required=True)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    rows = [json.loads(l) for l in open(args.in_jsonl)]
    prompts = [json.loads(l) for l in open(args.prompts_jsonl)]

    html_lines = ['<!DOCTYPE html><html><head><meta charset="utf-8">',
                  '<title>Jacobi lookahead viz</title>',
                  '<style>',
                  'body{font-family:-apple-system,sans-serif;max-width:1200px;margin:20px auto;padding:0 20px;}',
                  '.panel{border:1px solid #ddd;border-radius:6px;padding:14px;margin:14px 0;background:#fafafa;}',
                  '.header{display:flex;justify-content:space-between;align-items:center;margin-bottom:8px;}',
                  '.tpf-box{display:inline-block;padding:4px 10px;border-radius:4px;font-weight:600;}',
                  '.tpf-v{background:#fae8e8;color:#8e2222;}',
                  '.tpf-l{background:#d4f1d4;color:#1e6e1e;}',
                  '.delta{background:#fcf4d4;color:#7a5b00;margin-left:6px;}',
                  '.section{margin:6px 0;}',
                  '.label{display:inline-block;width:90px;font-weight:600;color:#555;}',
                  '.prompt-text{font-style:italic;color:#444;background:#eef;padding:6px;border-radius:4px;font-size:0.95em;max-height:80px;overflow:auto;}',
                  'pre{background:#f4f4f4;padding:8px;border-radius:4px;font-size:0.85em;white-space:pre-wrap;max-height:120px;overflow:auto;margin:4px 0;}',
                  '.legend{font-size:0.85em;color:#666;margin-top:10px;}',
                  '.legend span{display:inline-block;padding:1px 6px;border-radius:3px;margin-right:8px;color:white;}',
                  '</style></head><body>',
                  '<h1>Jacobi lookahead vs vanilla — per-prompt visualization</h1>',
                  '<div class="legend">',
                  '<span style="background:#2ecc71">28+ accepted</span>',
                  '<span style="background:#3498db">10+ accepted</span>',
                  '<span style="background:#95a5a6">3-9 accepted</span>',
                  '<span style="background:#e67e22">1-2 accepted</span>',
                  '<span style="background:#e74c3c">0 accepted</span>',
                  '</div>']

    mean_v = sum(r["tpf_vanilla"] for r in rows)/len(rows)
    mean_l = sum(r["tpf_lookahead"] for r in rows)/len(rows)
    html_lines.append(f'<div class="panel" style="background:#eef9ff;">'
                      f'<b>OVERALL</b>: vanilla TPF mean = {mean_v:.3f} / lookahead TPF mean = {mean_l:.3f} / '
                      f'Δ = {mean_l-mean_v:+.3f} (n={len(rows)} prompts)</div>')

    for r in sorted(rows, key=lambda x: -x["tpf_lookahead"]):
        i = r["batch_idx"]
        prompt_text = prompts[i]["input"] if i < len(prompts) else ""
        nv = r["n_acc_v_history"]
        nl = r["n_acc_l_history"]

        html_lines.append('<div class="panel">')
        html_lines.append(f'<div class="header"><b>Prompt {i+1}</b> '
                          f'<span class="tpf-box tpf-v">vanilla TPF {r["tpf_vanilla"]:.2f} '
                          f'(n_acc mean {r["mean_n_acc_v"]:.2f}, {r["iters_vanilla"]} iters, '
                          f'{r["n_tok_vanilla"]} tok)</span> '
                          f'<span class="tpf-box tpf-l">lookahead TPF {r["tpf_lookahead"]:.2f} '
                          f'(n_acc mean {r["mean_n_acc_l"]:.2f}, {r["iters_lookahead"]} iters, '
                          f'{r["n_tok_lookahead"]} tok)</span> '
                          f'<span class="tpf-box delta">Δ {r["tpf_lookahead"]-r["tpf_vanilla"]:+.2f}</span>'
                          f'</div>')
        html_lines.append(f'<div class="section"><span class="label">Prompt:</span></div>'
                          f'<div class="prompt-text">{html.escape(prompt_text[:500])}</div>')

        html_lines.append(f'<div class="section"><span class="label">Vanilla n_acc per iter ({len(nv)} iters):</span>{sparkline(nv)}</div>')
        html_lines.append(f'<div class="section"><span class="label">Lookahead n_acc per iter ({len(nl)} iters):</span>{sparkline(nl)}</div>')

        html_lines.append('<details><summary>Completions preview (200 toks each)</summary>')
        html_lines.append(f'<div class="section"><span class="label">Vanilla:</span></div>'
                          f'<pre>{html.escape(r.get("completion_v_preview","")[:500])}</pre>')
        html_lines.append(f'<div class="section"><span class="label">Lookahead:</span></div>'
                          f'<pre>{html.escape(r.get("completion_l_preview","")[:500])}</pre>')
        html_lines.append(f'<div class="section"><span class="label">Greedy ref:</span></div>'
                          f'<pre>{html.escape(r.get("ref_preview","")[:500])}</pre>')
        html_lines.append('</details>')
        html_lines.append('</div>')

    html_lines.append('</body></html>')
    with open(args.out_html, "w") as f:
        f.write("\n".join(html_lines))
    print(f"[viz] wrote {args.out_html}", flush=True)


if __name__ == "__main__":
    main()
