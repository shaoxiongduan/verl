"""Visualize per-prompt per-block per-iter Jacobi block-decode trace.

Reads a JSON trace from _jf_block_trace.py (schema: list[prompt] with
nested blocks → inner iters with input_draft, greedy_pred, entropy,
num_accepted, L). Decodes token ids with the model's tokenizer and renders
an HTML page where each prompt is a panel and each block expands to show
the per-iter trajectory: input draft → model prediction with accept/reject
coloring per position + entropy bar.

Usage:
    python3 scripts/_viz_block_trace.py \\
        --in_json eval_passk/tpf_results/math_k3_ds_block_trace_DS_uniform_1024.json \\
        --tokenizer_path /mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_k3_ds_step_300 \\
        --out_html eval_passk/diag_traces/math_k3_block_viz.html
"""
from __future__ import annotations
import argparse, html, json
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--in_json", required=True)
    p.add_argument("--tokenizer_path", required=True)
    p.add_argument("--out_html", required=True)
    p.add_argument("--max_blocks_per_prompt", type=int, default=12,
                   help="Cap blocks shown per prompt (saves HTML size)")
    p.add_argument("--max_iters_per_block", type=int, default=8,
                   help="Cap inner iters shown per block")
    return p.parse_args()


def tok_to_html(token: str, accepted: bool, entropy: float, is_diff: bool) -> str:
    """Render a single token cell. Color: green if accepted, red if rejected,
    yellow if not yet evaluated. Background opacity reflects entropy."""
    # Entropy range typically [0, ~5]. Map to opacity 0.2..1.0.
    op = min(1.0, max(0.15, 0.15 + entropy / 5.0))
    if accepted:
        bg = f"rgba(46, 204, 113, {op:.2f})"
        bd = "#27ae60"
    elif is_diff:
        bg = f"rgba(231, 76, 60, {op:.2f})"
        bd = "#c0392b"
    else:
        bg = f"rgba(149, 165, 166, {op:.2f})"
        bd = "#7f8c8d"
    # Escape token text and replace whitespace explicitly for visibility
    visible = (token.replace("\n", "↵").replace("\t", "→").replace(" ", "·"))
    return (f'<span class="tok" style="background:{bg};border:1px solid {bd};" '
            f'title="ent={entropy:.2f}">{html.escape(visible)}</span>')


def render_block(blk: dict, tokenizer, max_iters: int) -> str:
    """Render one block as a collapsible section with per-iter rows."""
    K = blk.get("inner", [{}])[0].get("L", 32) if blk.get("inner") else 32
    iters = blk.get("inner", [])[:max_iters]
    # Build a 2D grid: rows = iters, cols = K positions
    rows = []
    for it in iters:
        draft = it["input_draft"]
        pred = it["greedy_pred"]
        ent = it["entropy"]
        n_acc = it.get("num_accepted", 0)
        cells = []
        for j in range(min(len(draft), len(pred))):
            d_tok = tokenizer.decode([draft[j]], skip_special_tokens=False)
            p_tok = tokenizer.decode([pred[j]], skip_special_tokens=False)
            is_acc = j < n_acc
            is_diff = draft[j] != pred[j]
            # Show pred token (what model wants); accept color if draft==pred & j<n_acc
            cells.append(tok_to_html(p_tok, accepted=is_acc, entropy=ent[j], is_diff=is_diff and not is_acc))
        rows.append(
            f'<div class="iter-row">'
            f'<div class="iter-label">iter {it["iter"]}<br><small>n_acc={n_acc}</small></div>'
            f'<div class="iter-cells">{"".join(cells)}</div>'
            f'</div>'
        )
    # Also decode the draft of the first iter (the noise input) for reference
    if iters:
        first_draft = iters[0]["input_draft"]
        first_draft_text = tokenizer.decode(first_draft, skip_special_tokens=False)
        first_draft_visible = first_draft_text.replace("\n", "↵").replace("\t", "→")
    else:
        first_draft_visible = "(no iters)"
    # Actual committed block content: concatenate first-n_acc of each iter's greedy_pred
    committed_ids = []
    for it in iters:
        committed_ids.extend(it["greedy_pred"][:it.get("num_accepted", 0)])
    last_visible = tokenizer.decode(committed_ids, skip_special_tokens=False).replace("\n", "↵").replace("\t", "→")
    return f'''
<details class="block-card" {"open" if blk.get("block_idx", 0) < 3 else ""}>
  <summary><b>Block {blk.get("block_idx", "?")}</b> — iter_count={blk.get("iter_count", "?")},
    first_iter_accepted={blk.get("first_iter_accepted", "?")},
    n_committed={blk.get("n_committed", "?")}</summary>
  <div class="block-body">
    <div class="ref-text"><b>Initial noisy draft (iter 1 input):</b><br><code>{html.escape(first_draft_visible[:400])}</code></div>
    <div class="ref-text"><b>Last-iter prediction (≈converged block):</b><br><code>{html.escape(last_visible[:400])}</code></div>
    <div class="grid-header"><b>Per-iter trajectory</b> (cells colored: <span style="background:rgba(46,204,113,0.6);padding:2px">accepted</span>,
      <span style="background:rgba(231,76,60,0.6);padding:2px">rejected/different</span>,
      <span style="background:rgba(149,165,166,0.6);padding:2px">post-rejection tail</span>; opacity ∝ entropy):</div>
    {"".join(rows)}
  </div>
</details>'''


def main():
    args = parse_args()
    tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_path)
    data = json.load(open(args.in_json))

    html_lines = [
        '<!DOCTYPE html><html><head><meta charset="utf-8"><title>Block-decode viz</title>',
        '<style>',
        'body{font-family:-apple-system,sans-serif;max-width:1500px;margin:20px auto;padding:0 20px;}',
        '.prompt-panel{border:1px solid #999;border-radius:6px;padding:14px;margin:18px 0;background:#fafafa;}',
        '.prompt-text{font-style:italic;background:#eef;padding:6px;border-radius:4px;font-size:0.95em;max-height:80px;overflow:auto;}',
        '.block-card{border:1px solid #ddd;border-radius:4px;margin:6px 0;padding:8px;background:#fff;}',
        '.block-card summary{cursor:pointer;font-size:0.95em;}',
        '.block-body{margin-top:8px;}',
        '.ref-text{margin:4px 0;font-size:0.85em;color:#555;}',
        '.ref-text code{background:#f4f4f4;padding:2px 4px;border-radius:3px;display:block;white-space:pre-wrap;}',
        '.grid-header{font-size:0.85em;color:#555;margin:6px 0;}',
        '.iter-row{display:grid;grid-template-columns:80px 1fr;gap:8px;align-items:center;margin:3px 0;}',
        '.iter-label{font-size:0.75em;color:#666;text-align:right;}',
        '.iter-cells{display:flex;flex-wrap:wrap;gap:2px;}',
        '.tok{display:inline-block;padding:2px 4px;border-radius:2px;font-family:Consolas,monospace;font-size:0.78em;min-width:8px;line-height:1.2;}',
        'h1,h2{margin-top:24px;}',
        '.headline{background:#fffbe6;border-left:4px solid #ffc107;padding:10px;border-radius:4px;margin-bottom:14px;}',
        '</style></head><body>',
        '<h1>Math k3 block-decode visualization (DS prompts, uniform noise drafts)</h1>',
        '<div class="headline">For each prompt: collapsible blocks. '
        'Each block shows the noisy initial draft, the per-iter model predictions, '
        'and the converged final block. Green cells = accepted at this iter (matched draft); '
        'red = model disagrees with draft. Opacity = entropy (darker = less confident). '
        'First 3 blocks per prompt are pre-expanded.</div>',
    ]

    for p in data[:8]:  # cap to first 8 prompts for HTML size
        html_lines.append(f'<div class="prompt-panel">')
        html_lines.append(f'<h2>Prompt {p["prompt_idx"]} — TPF={p.get("tpf_overall", 0):.2f}, '
                          f'{p["total_new_tokens"]} tok, {p["n_blocks"]} blocks, stop={p["stop_reason"]}</h2>')
        html_lines.append(f'<div class="prompt-text">{html.escape(p["prompt"][:400])}</div>')
        for blk in p["blocks"][:args.max_blocks_per_prompt]:
            html_lines.append(render_block(blk, tokenizer, args.max_iters_per_block))
        if len(p["blocks"]) > args.max_blocks_per_prompt:
            html_lines.append(f'<div style="color:#888;font-size:0.85em;">...{len(p["blocks"]) - args.max_blocks_per_prompt} more blocks not shown</div>')
        html_lines.append('</div>')

    html_lines.append('</body></html>')
    with open(args.out_html, "w") as f:
        f.write("\n".join(html_lines))
    print(f"[viz] wrote {args.out_html}", flush=True)


if __name__ == "__main__":
    main()
