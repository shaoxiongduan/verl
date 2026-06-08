"""Visualize the Jacobi cascade evolution per prompt.

For each prompt × model, build a 2D matrix:
  - rows = iter index (time)
  - cols = absolute response position (space)
  - each cell shows: target_argmax token at that (iter, position) cell
  - color: entropy of prediction (low=green, high=red)
  - status: accepted (green border), rejection point (red border),
            still-draft (no border / faded), committed-from-earlier-iter (gray)

This shows the cascade: at iter 0 a K-block of fuzzy predictions appears;
each subsequent iter refines/shifts; you can see noisy positions get
"locked in" as the window passes over them.
"""
from __future__ import annotations
import argparse, glob, json, os, sys, statistics
from collections import defaultdict
from html import escape


def load_runs(in_dir):
    runs = defaultdict(dict)
    for fp in sorted(glob.glob(os.path.join(in_dir, "p*__*.json"))):
        rec = json.load(open(fp))
        runs[rec["prompt_idx"]][rec["model_label"]] = rec
    return runs


def commit_positions(iters):
    """Return for each iter the (window_start, K) i.e. absolute pos of K-block.

    window_start_i = sum over j<i of (n_acc_j + 1). prefill not counted.
    """
    positions = []
    cursor = 0
    for it in iters:
        if not it.get("num_draft"): continue
        nd = it["num_draft"][0]
        if nd <= 0: continue
        positions.append(cursor)
        n_acc = it["n_acc"][0]
        cursor += n_acc + 1
    return positions


def render_cascade(rec, tokenizer, label):
    """Build HTML for one model's cascade on one prompt."""
    iters = [it for it in rec["iters"] if it.get("num_draft") and it["num_draft"][0] > 0]
    if not iters:
        return f'<div>(no iters for {label})</div>'
    K = iters[0]["num_draft"][0]
    positions = commit_positions(iters)
    n_iters = len(iters)
    # Total response length = positions[-1] + K
    total_pos = positions[-1] + K if positions else K

    # Build matrix[iter_idx][abs_pos] = (target_tok_str, entropy, status, draft_tok_str)
    # status: 'acc' (this iter's accepted), 'rej' (rejection point), 'draft' (drafted-only), 'committed' (already committed in earlier iter)
    cells = [[None] * total_pos for _ in range(n_iters)]
    final_committed = [False] * total_pos
    for i, it in enumerate(iters):
        n_acc = it["n_acc"][0]
        draft = it["draft"][0][:K]
        target = it["target_argmax"][0][:K]
        entropy = (it.get("target_entropy") or [[None]])[0]
        if entropy: entropy = entropy[:K]
        ws = positions[i]
        for j in range(K):
            abs_pos = ws + j
            if abs_pos >= total_pos: continue
            ent = entropy[j] if entropy and j < len(entropy) else None
            if j < n_acc:
                status = 'acc'
            elif j == n_acc:
                status = 'rej'  # bonus token from target_argmax
            else:
                status = 'draft'
            cells[i][abs_pos] = {
                "draft": draft[j] if j < len(draft) else -1,
                "target": target[j] if j < len(target) else -1,
                "entropy": ent,
                "status": status,
                "in_window_pos": j,
            }
    # Render
    parts = [f'<div class="model-section"><h3>{label} — n_iters={n_iters}, TPF={(sum(it["n_acc"][0] for it in iters) + n_iters)/n_iters:.2f}</h3>']
    parts.append('<div class="cascade-scroll"><table class="cascade"><thead><tr><th class="iter-col">iter</th>')
    # Position headers (every 10)
    for p in range(total_pos):
        if p % 10 == 0:
            parts.append(f'<th class="poshdr">{p}</th>')
        else:
            parts.append('<th class="poshdr">&nbsp;</th>')
    parts.append('</tr></thead><tbody>')

    for i, row in enumerate(cells):
        n_acc_i = iters[i]["n_acc"][0]
        parts.append(f'<tr><td class="iter-col" title="n_acc={n_acc_i}">{i:3d} (na={n_acc_i})</td>')
        for p, cell in enumerate(row):
            if cell is None:
                parts.append('<td class="empty"></td>')
                continue
            tok_str = tokenizer.decode([cell["target"]]) if cell["target"] >= 0 else ""
            tok_str = tok_str.replace("\n", "↵").replace(" ", "·")[:6]
            ent = cell["entropy"] if cell["entropy"] is not None else 0
            # color by entropy
            norm = min(1.0, ent / 6.0)
            r = int(180 + 75 * norm); g = int(245 - 80 * norm); bl = int(180 - 60 * norm)
            border_style = {
                'acc': 'border:2px solid #1a9c1a',
                'rej': 'border:2px solid #c40808',
                'draft': 'border:1px solid #ccc',
            }[cell["status"]]
            tip = f'iter={i} pos={p} winpos={cell["in_window_pos"]} draft={cell["draft"]} target={cell["target"]} ent={ent:.2f} status={cell["status"]}'
            parts.append(
                f'<td class="cellgrid" style="background:rgb({r},{g},{bl});{border_style}" title="{escape(tip)}">{escape(tok_str)}</td>'
            )
        parts.append('</tr>')
    parts.append('</tbody></table></div></div>')
    return "".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_dir", required=True)
    p.add_argument("--model_path", required=True, help="HF model dir for tokenizer")
    p.add_argument("--html_out", required=True)
    p.add_argument("--prompt_idxs", type=str, default="0,1,2",
                   help="Comma-separated prompt indices to include (cascade viz is large)")
    args = p.parse_args()

    from transformers import AutoTokenizer
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)

    runs = load_runs(args.in_dir)
    prompt_idxs = [int(x) for x in args.prompt_idxs.split(",")]

    parts = ['<!doctype html><html><head><meta charset="utf-8">',
             '<title>Jacobi cascade viz</title><style>',
             'body{font-family:sans-serif;font-size:13px;background:#fafafa;margin:10px}',
             'h1{margin:8px 0}',
             'h2{background:#333;color:#fff;padding:8px 12px;margin:24px 0 6px}',
             'h3{background:#666;color:#fff;padding:4px 8px;margin:8px 0 2px}',
             '.legend{padding:6px;background:#fff;border:1px solid #ccc;margin:6px 0}',
             '.legend-cell{display:inline-block;width:18px;height:18px;margin:0 4px;vertical-align:middle}',
             '.prompt-text{background:#fff;padding:6px;border-left:3px solid #38f;font-size:11px;margin-bottom:4px}',
             '.model-section{margin-bottom:16px;background:#fff;padding:6px;border:1px solid #ddd}',
             '.cascade-scroll{overflow-x:auto;max-width:100%}',
             'table.cascade{border-collapse:collapse;font-family:monospace;font-size:8px}',
             'table.cascade th,table.cascade td{padding:0;text-align:center}',
             'table.cascade th.iter-col,table.cascade td.iter-col{position:sticky;left:0;background:#eee;font-size:9px;text-align:right;padding:1px 4px;min-width:60px}',
             'table.cascade th.poshdr{font-size:8px;color:#666;background:#f6f6f6;width:14px}',
             'table.cascade td.empty{background:#fff;width:14px;height:14px}',
             'table.cascade td.cellgrid{width:14px;height:14px;font-size:7px;line-height:1}',
             '</style></head><body>',
             '<h1>Jacobi cascade viz — how the K-block evolves across iters</h1>',
             '<div class="legend">',
             '<b>Each cell</b> = (iter, absolute response position). Token shown = model\'s target_argmax for that position at that iter.<br>',
             '<b>Color (background)</b> = entropy of model\'s prediction. <span style="display:inline-block;width:18px;height:14px;background:rgb(180,245,180)"></span> low entropy = confident → ',
             '<span style="display:inline-block;width:18px;height:14px;background:rgb(255,165,120)"></span> high entropy = uncertain.<br>',
             '<b>Border</b>: <span style="border:2px solid #1a9c1a;padding:0 4px">green</span> accepted spec token | ',
             '<span style="border:2px solid #c40808;padding:0 4px">red</span> rejection point (bonus token, becomes committed) | ',
             '<span style="border:1px solid #ccc;padding:0 4px">gray</span> still in draft (not committed this iter).<br>',
             '<b>Window slide</b>: across iters, the K-block shifts right by (n_acc+1). Hover any cell for details.',
             '</div>']

    for p_idx in prompt_idxs:
        if p_idx not in runs:
            continue
        by_label = runs[p_idx]
        parts.append(f'<h2>Prompt {p_idx}</h2>')
        if "base" in by_label:
            ptext = by_label["base"]["prompt"][:300]
            parts.append(f'<div class="prompt-text">{escape(ptext)}{"..." if len(ptext)==300 else ""}</div>')
        for label in ("base", "consA_s300", "consB_s300"):
            if label not in by_label: continue
            parts.append(render_cascade(by_label[label], tokenizer, label))

    parts.append('</body></html>')
    with open(args.html_out, "w") as fp:
        fp.write("".join(parts))
    print(f"Wrote {args.html_out}")


if __name__ == "__main__":
    main()
