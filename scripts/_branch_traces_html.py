"""Render branch5 trace jsonl(s) into a self-contained HTML for visual analysis.

Per branch point (sorted by winner advantage): context, greedy future, and the
full per-iteration cascade of VANILLA / WINNER / WORST candidates. Token cells:
  green bg          = committed this iteration (accepted prefix)
  blue underline    = token equals the greedy future at its offset
  orange background = token repeats the previous window token (degeneracy)
Usage:
  python scripts/_branch_traces_html.py out.html base=branch5_base.jsonl fwdv8=branch5_fwdv8.jsonl
"""
from __future__ import annotations
import html as html_mod
import json, sys
from transformers import AutoTokenizer

CSS = """
body{font-family:ui-monospace,Menlo,monospace;font-size:12px;background:#fafafa;margin:18px}
h1{font-size:18px} h2{font-size:15px;margin:24px 0 6px;border-bottom:2px solid #888}
.bp{background:#fff;border:1px solid #ccc;border-radius:6px;padding:10px;margin:14px 0}
.hdr{font-weight:bold;margin-bottom:4px}
.ctx{color:#555;background:#f0f0f0;padding:4px;border-radius:4px;white-space:pre-wrap}
.fut{color:#0a4;background:#eafaef;padding:4px;border-radius:4px;white-space:pre-wrap;margin:4px 0}
.cand{margin:8px 0;padding:6px;border-left:4px solid #bbb}
.cand.win{border-left-color:#2a7} .cand.van{border-left-color:#36c} .cand.lose{border-left-color:#c44}
.cname{font-weight:bold}
.curve{color:#777}
.iter{margin:2px 0;white-space:nowrap;overflow-x:auto}
.it{display:inline-block;width:54px;color:#999}
.tk{display:inline-block;padding:0 2px;margin:0 1px;border-radius:3px;background:#f4f4f4;border:1px solid #e8e8e8}
.tk.acc{background:#bdf0c9;border-color:#7fcf95;font-weight:bold}
.tk.fut{text-decoration:underline;text-decoration-color:#36c;text-decoration-thickness:2px}
.tk.rep{background:#ffd9a8;border-color:#f0a44c}
.tk.acc.rep{background:#d8eaa0}
.legend span{margin-right:14px}
details summary{cursor:pointer;color:#36c}
"""


def render_window(tok, draft, n_acc, future):
    cells = []
    prev = None
    for j, t in enumerate(draft):
        s = html_mod.escape(tok.decode([t]).replace("\n", "\\n")) or "·"
        cls = ["tk"]
        if j < n_acc:
            cls.append("acc")
        if j < len(future) and t == future[j]:
            cls.append("fut")
        if prev is not None and t == prev:
            cls.append("rep")
        prev = t
        cells.append(f'<span class="{" ".join(cls)}">{s}</span>')
    return "".join(cells)


def render_cand(tok, name, cls, cand, future, h):
    cum = cand["cum"]
    curve = "→".join(str(c) for c in cum)
    rows = []
    fut = list(future)
    consumed = 0
    for i, it in enumerate(cand["iters"]):
        # future alignment shifts as tokens commit
        rows.append(f'<div class="iter"><span class="it">it{i+1} +{it["n_acc"]}</span>'
                    + render_window(tok, it["draft"], it["n_acc"], fut[consumed:]) + "</div>")
        consumed += it["n_acc"]
    return (f'<div class="cand {cls}"><span class="cname">{name}</span> '
            f'<span class="curve">cum: {curve}  (h-TPF {cum[h-1]/h:.2f})</span>'
            + "".join(rows) + "</div>")


def main():
    out_path = sys.argv[1]
    sources = [a.split("=", 1) for a in sys.argv[2:]]
    tok = AutoTokenizer.from_pretrained("ckpts_hf/math_k3_ds_step_300")
    parts = [f"<html><head><meta charset='utf-8'><style>{CSS}</style></head><body>",
             "<h1>Jacobi branching traces — cascade evolution per candidate</h1>",
             "<div class='legend'><span class='tk acc'>committed this iter</span>"
             "<span class='tk fut'>= greedy future</span>"
             "<span class='tk rep'>repeats prev token</span></div>"]
    for label, path in sources:
        rows = [json.loads(l) for l in open(path)]
        B = [b for r in rows for b in r["branches"]]
        h = len(B[0]["vanilla"]["cum"])
        scored = []
        for b in B:
            outs = [c["cum"][h-1] for c in b["cands"]]
            wi = max(range(len(outs)), key=lambda i: outs[i])
            li = min(range(len(outs)), key=lambda i: outs[i])
            adv = outs[wi] - b["vanilla"]["cum"][h-1]
            scored.append((adv, b, wi, li))
        scored.sort(key=lambda x: -x[0])
        parts.append(f"<h2>{html_mod.escape(label)} — {len(B)} branch points "
                     f"(showing top 10 by winner advantage + 2 negative)</h2>")
        sel = scored[:10] + scored[-2:]
        for adv, b, wi, li in sel:
            ctx = html_mod.escape(tok.decode(b["ctx_tail"]))
            fut_txt = html_mod.escape(tok.decode(b["future"]))
            parts.append(f"<div class='bp'><div class='hdr'>pos={b['pos']}  "
                         f"winner advantage = {adv:+d} tokens over {h} forwards</div>")
            parts.append(f"<div class='ctx'>CTX …{ctx}</div>")
            parts.append(f"<div class='fut'>GREEDY FUTURE: {fut_txt}</div>")
            parts.append(render_cand(tok, "VANILLA (argmax window)", "van",
                                     b["vanilla"], b["future"], h))
            parts.append(render_cand(tok, f"WINNER (sample #{wi})", "win",
                                     b["cands"][wi], b["future"], h))
            parts.append("<details><summary>worst candidate</summary>"
                         + render_cand(tok, f"WORST (sample #{li})", "lose",
                                       b["cands"][li], b["future"], h)
                         + "</details></div>")
    parts.append("</body></html>")
    with open(out_path, "w") as f:
        f.write("\n".join(parts))
    print(f"wrote {out_path}")


if __name__ == "__main__":
    main()
