"""Render JF block-decode trace as cascade HTML (one matrix per block).

Each block: rows = inner iter, cols = position within K-block. Cells show the
greedy argmax token at that (iter, position) with entropy color and accept border.
"""
from __future__ import annotations
import argparse, json, os, statistics, sys
from html import escape

sys.path.insert(0, "/mnt/weka/home/hao.zhang/shao/JacobiForcing")
from transformers import AutoTokenizer


def render_block(block, b_idx, tokenizer, K):
    inner = block.get("inner") or []
    if not inner:
        return ''
    n_iters = len(inner)
    n_committed = block.get("n_committed", 0)
    first2 = inner[1]["num_accepted"] + 1 if len(inner) >= 2 else None
    parts = [f'<div class="block-section"><h4>Block {b_idx} — n_iters={n_iters}, committed={n_committed} ({n_committed/n_iters:.2f}/iter); iter-2 first-verif accept+1={first2}</h4>']
    parts.append('<table class="cascade"><thead><tr><th class="iter-col">iter</th>')
    for p in range(K):
        parts.append(f'<th class="poshdr">{p}</th>')
    parts.append('</tr></thead><tbody>')
    for j, it in enumerate(inner):
        in_draft = it["input_draft"]
        greedy = it["greedy_pred"]  # len = L-1 (predicts pos i+1 from pos i)
        ent = it.get("entropy", [])
        num_acc = it["num_accepted"]
        # Greedy predicts position p+1 from position p; convention: we visualize greedy as the model's vote at each position.
        # For simplicity, show input_draft as the cell content and greedy_pred as overlay.
        parts.append(f'<tr><td class="iter-col">{j+1} (acc={num_acc})</td>')
        for p in range(K):
            draft_tok = in_draft[p] if p < len(in_draft) else -1
            gp = greedy[p] if p < len(greedy) else -1
            e = ent[p] if p < len(ent) else 0
            # color by entropy
            norm = min(1.0, e / 6.0) if e is not None else 0
            r = int(180 + 75 * norm); g = int(245 - 80 * norm); bl = int(180 - 60 * norm)
            # border: accepted (matches & in accepted prefix) -> green; rejection point -> red; rest -> gray
            if p < num_acc:
                border = 'border:2px solid #1a9c1a'
            elif p == num_acc:
                border = 'border:2px solid #c40808'
            else:
                border = 'border:1px solid #ccc'
            draft_str = (tokenizer.decode([draft_tok]) if draft_tok >= 0 else '').replace('\n','↵').replace(' ','·')[:6]
            greedy_str = (tokenizer.decode([gp]) if gp >= 0 else '').replace('\n','↵').replace(' ','·')[:6]
            tip = f'iter={j+1} pos={p} draft={draft_tok} greedy={gp} ent={e:.2f} num_acc={num_acc}'
            parts.append(
                f'<td class="cellgrid" style="background:rgb({r},{g},{bl});{border}" title="{escape(tip)}">'
                f'<div class="d">{escape(draft_str)}</div><div class="g">{escape(greedy_str)}</div></td>'
            )
        parts.append('</tr>')
    parts.append('</tbody></table></div>')
    return "".join(parts)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--in_json", required=True)
    p.add_argument("--model_path", required=True)
    p.add_argument("--html_out", required=True)
    p.add_argument("--prompt_idxs", default="0,1,2")
    p.add_argument("--blocks_per_prompt", type=int, default=6)
    args = p.parse_args()

    data = json.load(open(args.in_json))
    tokenizer = AutoTokenizer.from_pretrained(args.model_path)
    prompt_idxs = {int(x) for x in args.prompt_idxs.split(",")}
    K = 32
    if data and data[0]["blocks"]:
        first = data[0]["blocks"][0]["inner"]
        if first:
            K = first[0]["L"]

    # Aggregate stats
    iter_acc = [[] for _ in range(8)]
    for r in data:
        for b in r["blocks"]:
            for j, it in enumerate(b.get("inner") or []):
                if j < 8:
                    iter_acc[j].append(it["num_accepted"])

    parts = ['<!doctype html><html><head><meta charset="utf-8">',
             '<title>k3 JF block-decode cascade</title><style>',
             'body{font-family:sans-serif;font-size:12px;background:#fafafa;margin:10px}',
             'h2{background:#333;color:#fff;padding:8px 12px;margin:20px 0 6px}',
             'h3{background:#666;color:#fff;padding:4px 8px;margin:8px 0 2px}',
             'h4{background:#a90;color:#fff;padding:3px 8px;margin:4px 0 2px;font-size:12px}',
             '.legend{padding:6px;background:#fff;border:1px solid #ccc;margin:6px 0}',
             '.block-section{margin-bottom:12px;background:#fff;padding:4px;border:1px solid #ddd}',
             'table.cascade{border-collapse:collapse;font-family:monospace;font-size:8px}',
             'table.cascade th,table.cascade td{padding:0;text-align:center}',
             'table.cascade .iter-col{background:#eee;font-size:9px;text-align:right;padding:1px 4px;min-width:80px}',
             'table.cascade th.poshdr{font-size:8px;color:#666;background:#f6f6f6;width:42px}',
             'table.cascade td.cellgrid{width:42px;height:24px;font-size:8px;line-height:1.1;vertical-align:middle}',
             'table.cascade .d{color:#666;font-size:7px}',
             'table.cascade .g{font-weight:bold;font-size:8px}',
             '.stats{background:#fff;padding:8px;border:1px solid #ccc;margin:8px 0}',
             '</style></head><body>',
             '<h1>k3 JF block-decode cascade — verifies "TPF=6 at first verification iter" claim</h1>',
             '<div class="legend">',
             '<b>Each table</b> = one K-block. Rows = inner Jacobi iter (1=initial noise→pred, 2=first verification of model\'s own argmax, ...).<br>',
             '<b>Each cell</b> = position within the K-block. Top row = input draft token; bold = model\'s greedy prediction.<br>',
             '<b>Color</b>: entropy heat — green=confident → orange=uncertain.<br>',
             '<b>Border</b>: <span style="border:2px solid #1a9c1a;padding:0 4px">green</span>=accepted | ',
             '<span style="border:2px solid #c40808;padding:0 4px">red</span>=rejection point (becomes bonus) | ',
             '<span style="border:1px solid #ccc;padding:0 4px">gray</span>=draft (refreshed for next iter).',
             '</div>']

    parts.append('<div class="stats"><h2>Aggregate per-inner-iter accept counts (k3 block-decode, math)</h2>')
    parts.append('<table class="cascade"><thead><tr><th>inner iter</th><th>mean num_accepted</th><th>median</th><th>+1 bonus = TPF</th><th>n blocks</th></tr></thead><tbody>')
    for i, lst in enumerate(iter_acc):
        if not lst: continue
        m = statistics.mean(lst); md = statistics.median(lst)
        parts.append(f'<tr><td>{i+1}</td><td>{m:.3f}</td><td>{md:.1f}</td><td>{m+1:.3f}</td><td>{len(lst)}</td></tr>')
    parts.append('</tbody></table></div>')

    for r in data:
        if r["prompt_idx"] not in prompt_idxs:
            continue
        parts.append(f'<h2>Prompt {r["prompt_idx"]} — TPF_overall={r["tpf_overall"]:.3f}, n_blocks={r["n_blocks"]}</h2>')
        parts.append(f'<div style="background:#fff;padding:4px;font-size:10px">{escape(r["prompt"][:300])}...</div>')
        for b_idx, b in enumerate(r["blocks"][:args.blocks_per_prompt]):
            parts.append(render_block(b, b_idx, tokenizer, K))
    parts.append('</body></html>')
    with open(args.html_out, "w") as fp:
        fp.write("".join(parts))
    print(f"Wrote {args.html_out}")


if __name__ == "__main__":
    main()
