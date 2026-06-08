"""Repetition viz: show decoded completions across models with repetition highlighting.

Reads N completion JSONLs (each has 64 rows from the DeepScaler-sampled prompts,
schema: {task_id, prompt, completion, num_tokens, tpf}). Aligns by prompt-idx
order, detects repetition at multiple scales, and renders a per-prompt HTML
panel with side-by-side completions and repetition highlights.

Repetition detection:
  - "trail-loop": last 30-char tail repeats >=3 times in the completion
  - "ngram-rep": any 6+ token n-gram appears 4+ times consecutively (typical
    degenerate loop)
  - per-row counts of repetition severity
"""
from __future__ import annotations
import argparse, html, json, os, re
from collections import Counter


def detect_repetitions(text: str) -> dict:
    """Return repetition stats + the spans to highlight."""
    out = {"trail_loop": False, "max_ngram_run": 0, "spans": []}
    if len(text) < 60:
        return out

    # 1) Trail-loop detector: 30-char tail repeated 3+ times
    tail = text[-50:]
    if text.count(tail) >= 3 and len(tail) > 5:
        out["trail_loop"] = True

    # 2) Token-level: find consecutive repetition of n-grams of words (n=3..10).
    words = text.split()
    if len(words) < 12:
        return out
    best_run = 0
    best_span = None
    for n in range(3, 12):
        i = 0
        while i + n * 3 <= len(words):
            ng = tuple(words[i:i+n])
            run = 1
            j = i + n
            while j + n <= len(words) and tuple(words[j:j+n]) == ng:
                run += 1
                j += n
            if run >= 3 and run > best_run:
                best_run = run
                best_span = (i, j, n)
            i += 1
    if best_span:
        out["max_ngram_run"] = best_run
        # Compute approximate char span of the run
        start_char = len(" ".join(words[:best_span[0]])) + (1 if best_span[0] > 0 else 0)
        end_char = len(" ".join(words[:best_span[1]]))
        out["spans"].append((start_char, end_char, best_span[2], best_run))

    # 3) Simple character substring repetition (e.g., "111111" or "ToToToTo")
    # Find any substring of length >=3 that repeats >=8 times consecutively
    m = re.search(r"(.{3,40}?)\1{4,}", text)
    if m:
        out["spans"].append((m.start(), m.end(), len(m.group(1)), m.end()-m.start()))
        out["max_ngram_run"] = max(out["max_ngram_run"], (m.end()-m.start()) // max(1, len(m.group(1))))

    return out


def highlight_spans(text: str, spans: list[tuple[int,int,int,int]]) -> str:
    """Render with <mark> around repetition spans."""
    if not spans:
        return html.escape(text)
    spans = sorted(spans, key=lambda x: x[0])
    out = []
    cursor = 0
    for s, e, n, run in spans:
        if s < cursor:
            continue
        out.append(html.escape(text[cursor:s]))
        out.append(f'<mark title="ngram={n} run={run}" style="background:#ffd6d6;border:1px solid #d22;">')
        out.append(html.escape(text[s:e]))
        out.append('</mark>')
        cursor = e
    out.append(html.escape(text[cursor:]))
    return "".join(out)


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--inputs", nargs="+", required=True,
                   help="<label>=<path> entries, e.g. base=eval_passk/tpf_results/base_jf_math_7b__vllm_deepscaler_T1.jsonl")
    p.add_argument("--out_html", required=True)
    p.add_argument("--max_chars_per_completion", type=int, default=3000)
    return p.parse_args()


def main():
    args = parse_args()
    inputs = []
    for s in args.inputs:
        label, path = s.split("=", 1)
        rows = [json.loads(l) for l in open(path)]
        inputs.append((label, rows))
        print(f"[viz] loaded {label}: {len(rows)} rows from {path}")

    n_prompts = min(len(r) for _, r in inputs)
    print(f"[viz] aligning on {n_prompts} prompts")

    # Compute repetition stats per (model, prompt)
    stats = {}
    for label, rows in inputs:
        loops = sum(1 for r in rows[:n_prompts] if detect_repetitions(r.get("completion", ""))["trail_loop"])
        bad_ngrams = sum(1 for r in rows[:n_prompts] if detect_repetitions(r.get("completion", ""))["max_ngram_run"] >= 4)
        stats[label] = {"trail_loops": loops, "bad_ngram_runs": bad_ngrams, "n": n_prompts}

    html_lines = [
        '<!DOCTYPE html><html><head><meta charset="utf-8"><title>Repetition viz</title>',
        '<style>',
        'body{font-family:-apple-system,sans-serif;max-width:1400px;margin:20px auto;padding:0 20px;}',
        '.panel{border:1px solid #ddd;border-radius:6px;padding:14px;margin:14px 0;background:#fafafa;}',
        '.prompt-text{font-style:italic;color:#444;background:#eef;padding:6px;border-radius:4px;font-size:0.95em;}',
        '.completion{background:#fff;border:1px solid #ddd;padding:8px;margin:4px 0;border-radius:4px;white-space:pre-wrap;font-family:Consolas,monospace;font-size:0.85em;max-height:280px;overflow:auto;}',
        '.model-row{display:grid;grid-template-columns:140px 100px 1fr;gap:8px;margin:6px 0;align-items:start;}',
        '.label{font-weight:600;color:#333;}',
        '.tpf{display:inline-block;padding:2px 6px;border-radius:3px;font-weight:600;background:#eaf2fa;color:#1e4f8d;}',
        '.tpf-bad{background:#fae8e8;color:#8e2222;}',
        '.flag{display:inline-block;padding:2px 6px;border-radius:3px;background:#ffd6d6;color:#7a1a1a;font-size:0.85em;margin-left:4px;}',
        'h2{margin-top:30px;}',
        'mark{padding:1px;}',
        '.summary{background:#fffbe6;border-left:4px solid #ffc107;padding:10px;border-radius:4px;}',
        '</style></head><body>',
        '<h1>Repetition viz: model completions side-by-side</h1>',
        '<div class="summary"><b>Repetition prevalence:</b><br>',
    ]
    for label, s in stats.items():
        html_lines.append(f'{label}: trail-loops={s["trail_loops"]}/{s["n"]}, '
                          f'bad-ngram-runs(>=4 repeats)={s["bad_ngram_runs"]}/{s["n"]}<br>')
    html_lines.append('</div>')

    # Per-prompt panel — sort by worst-repetition model's badness
    # so the user sees the worst cases first.
    prompts_ordered = list(range(n_prompts))
    # Sort: prompts where ANY model has max_ngram_run >= 4 first, by max run desc
    def prompt_severity(i):
        worst = 0
        for label, rows in inputs:
            r = detect_repetitions(rows[i].get("completion", ""))
            worst = max(worst, r["max_ngram_run"])
        return -worst
    prompts_ordered.sort(key=prompt_severity)

    for pi in prompts_ordered:
        prompt_text = inputs[0][1][pi].get("prompt", "")
        html_lines.append(f'<div class="panel"><b>Prompt {pi}</b>')
        html_lines.append(f'<div class="prompt-text">{html.escape(prompt_text[:400])}</div>')
        for label, rows in inputs:
            r = rows[pi]
            comp = r.get("completion", "")[:args.max_chars_per_completion]
            rep = detect_repetitions(r.get("completion", ""))
            tpf = r.get("tpf", 0)
            ntok = r.get("num_tokens", 0)
            tpf_cls = "tpf-bad" if tpf < 2.5 else "tpf"
            flags = []
            if rep["trail_loop"]:
                flags.append("trail-loop")
            if rep["max_ngram_run"] >= 4:
                flags.append(f"ngram×{rep['max_ngram_run']}")
            flag_html = "".join(f'<span class="flag">{f}</span>' for f in flags)
            html_lines.append('<div class="model-row">')
            html_lines.append(f'<div class="label">{html.escape(label)}{flag_html}</div>')
            html_lines.append(f'<div><span class="{tpf_cls}">TPF {tpf:.2f}</span><br><small>{ntok} tok</small></div>')
            html_lines.append(f'<div class="completion">{highlight_spans(comp, rep["spans"])}</div>')
            html_lines.append('</div>')
        html_lines.append('</div>')

    html_lines.append('</body></html>')
    with open(args.out_html, "w") as f:
        f.write("\n".join(html_lines))
    print(f"[viz] wrote {args.out_html}")


if __name__ == "__main__":
    main()
