"""Truncate completions at first <|im_end|>; recompute TPF + repetition stats.

The vllm Jacobi bench doesn't pass `stop_token_ids` to SamplingParams, so when
EOS lands inside a multi-token spec-decode commit batch, the LAST token of the
batch (which isn't EOS) is checked against the stop list — generation continues
past EOS into degenerate-loop garbage until max_new fires.

This script:
  1. For each completion JSONL: find first <|im_end|> in completion text;
     report fraction-content-vs-loop per row + repetition prevalence in the
     content-only portion.
  2. For math_k3 (where trajectory files exist): replay iters per request
     to find the exact iter where the first EOS landed in the committed
     output; compute "pre-EOS TPF" exactly.

Usage:
    python3 scripts/_analyze_post_eos.py
"""
from __future__ import annotations
import json, re, glob
from collections import Counter

EOS_STR = "<|im_end|>"
EOS_ID = 151645


def detect_repetition(text: str) -> dict:
    if len(text) < 60:
        return {"trail_loop": False, "max_ngram_run": 0}
    tail = text[-50:]
    trail = (text.count(tail) >= 3 and len(tail) > 5)
    words = text.split()
    best_run = 0
    for n in range(3, 12):
        i = 0
        while i + n * 3 <= len(words):
            ng = tuple(words[i:i+n])
            run = 1
            j = i + n
            while j + n <= len(words) and tuple(words[j:j+n]) == ng:
                run += 1
                j += n
            if run > best_run:
                best_run = run
            i += 1
    m = re.search(r"(.{3,40}?)\1{4,}", text)
    if m:
        run = (m.end() - m.start()) // max(1, len(m.group(1)))
        best_run = max(best_run, run)
    return {"trail_loop": trail, "max_ngram_run": best_run}


def split_at_first_eos(text: str) -> tuple[str, str]:
    idx = text.find(EOS_STR)
    if idx == -1:
        return text, ""
    return text[:idx], text[idx + len(EOS_STR):]


def analyze_completion_file(label: str, path: str) -> dict:
    rows = [json.loads(l) for l in open(path)]
    total = len(rows)
    pre_lens = []
    post_lens = []
    pre_rep = 0
    post_rep = 0
    pre_trail = 0
    eos_found = 0
    tpfs = []
    for r in rows:
        comp = r.get("completion", "")
        pre, post = split_at_first_eos(comp)
        if post:
            eos_found += 1
        pre_lens.append(len(pre))
        post_lens.append(len(post))
        rep_pre = detect_repetition(pre)
        rep_post = detect_repetition(post)
        if rep_pre["max_ngram_run"] >= 4:
            pre_rep += 1
        if rep_post["max_ngram_run"] >= 4:
            post_rep += 1
        if rep_pre["trail_loop"]:
            pre_trail += 1
        tpfs.append(r.get("tpf", 0))
    return {
        "label": label,
        "n": total,
        "n_eos_found": eos_found,
        "mean_pre_chars": sum(pre_lens) / total if total else 0,
        "mean_post_chars": sum(post_lens) / total if total else 0,
        "frac_post_content": sum(post_lens) / max(1, sum(pre_lens) + sum(post_lens)),
        "pre_eos_repeaters": pre_rep,
        "pre_eos_trail_loops": pre_trail,
        "post_eos_repeaters": post_rep,
        "mean_tpf_full_rollout": sum(tpfs) / total if total else 0,
    }


def main():
    # All completion JSONLs we want to analyze.
    files = [
        ("base",         "/mnt/weka/home/hao.zhang/shao/verl/eval_passk/tpf_results/base_jf_math_7b__vllm_deepscaler_T1.jsonl"),
        ("AR_step300",   "/mnt/weka/home/hao.zhang/shao/verl/eval_passk/tpf_results/math_ar_ds_step_300__vllm_deepscaler_T1.jsonl"),
        ("v4_step20",    "/mnt/weka/home/hao.zhang/shao/verl/eval_passk/tpf_results/dflashce_corrupt03_v4_step_20__vllm_deepscaler_T1.jsonl"),
        ("math_k3_ds",   "/mnt/weka/home/hao.zhang/shao/verl/eval_passk/tpf_results/math_k3_ds_step_300__vllm_deepscaler_T1.jsonl"),
        ("ce_noisy_k3",  "/mnt/weka/home/hao.zhang/shao/verl/eval_passk/tpf_results/dflashce_v2_step_20__vllm_deepscaler_T1.jsonl"),  # for comparison
        ("k3_ds64_run",  "/mnt/weka/home/hao.zhang/shao/verl/eval_passk/diag_traces/k3_ds64/completions.jsonl"),
    ]
    print(f"{'label':14s} | {'n':>3s} | {'EOS_found':>9s} | {'pre_chars':>9s} | {'post_chars':>10s} | {'frac_post':>9s} | {'pre_rep>=4':>10s} | {'pre_trail':>9s} | {'post_rep>=4':>11s} | {'TPF_full':>8s}")
    print('-' * 130)
    for label, path in files:
        try:
            s = analyze_completion_file(label, path)
        except FileNotFoundError:
            continue
        print(f"{label:14s} | {s['n']:>3d} | {s['n_eos_found']:>9d} | {s['mean_pre_chars']:>9.0f} | {s['mean_post_chars']:>10.0f} | "
              f"{s['frac_post_content']:>9.2%} | {s['pre_eos_repeaters']:>10d} | {s['pre_eos_trail_loops']:>9d} | {s['post_eos_repeaters']:>11d} | {s['mean_tpf_full_rollout']:>8.3f}")

    # Now: for math_k3_ds, use the trajectory file to compute EXACT pre-EOS TPF.
    print()
    print("=" * 80)
    print("EXACT PRE-EOS TPF (math_k3_ds DS_64 run, from trajectory file):")
    print("=" * 80)
    traj_files = glob.glob("/mnt/weka/home/hao.zhang/shao/verl/eval_passk/diag_traces/k3_ds64/traj.jsonl.*")
    if not traj_files:
        print("No trajectory files found")
        return
    # vllm_jacobi_patch records batched-per-iter format. Each rec has num_draft, n_acc per request.
    # Walk per request: accumulate (committed_tokens, iter_count); stop when first EOS is committed.
    # We don't have draft tokens decoded to know exactly when EOS lands in the commit, so use proxy:
    # n_iters_to_eos = round(n_chars_pre_eos / mean_tpf_full * 1) — too approximate.
    # Better proxy: for each request, walk iters, accumulate n_acc+1 tokens, find iter where
    # accumulated tokens hits the pre-EOS content length. Then TPF_pre_eos = pre_eos_n / iters_to_eos.
    # First need to know pre_eos_n in TOKENS not chars; use a quick re-tokenize.
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained("/mnt/weka/home/hao.zhang/shao/verl/ckpts_hf/math_k3_ds_step_300")
    completions_path = "/mnt/weka/home/hao.zhang/shao/verl/eval_passk/diag_traces/k3_ds64/completions.jsonl"
    comp_rows = [json.loads(l) for l in open(completions_path)]
    # Find pre-EOS token count per prompt
    pre_eos_tok = {}
    for i, r in enumerate(comp_rows):
        comp = r.get("completion", "")
        pre, _ = split_at_first_eos(comp)
        toks = tok(pre, add_special_tokens=False, return_tensors=None).input_ids
        pre_eos_tok[i] = len(toks)

    # Walk per-request through trajectory; the request index in batched-per-iter is the batch slot.
    # Assuming order matches prompts (which is how vllm_jacobi_patch records them).
    per_req_tok = {}
    per_req_iters = {}
    per_req_eos_iter = {}
    for fp in traj_files:
        with open(fp) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "num_draft" not in rec or not isinstance(rec["num_draft"], list):
                    continue
                if len(rec["num_draft"]) > 256 and set(rec["num_draft"]) == {1}:
                    continue
                for i, k in enumerate(rec["num_draft"]):
                    if k <= 0:
                        continue
                    na = rec["n_acc"][i] if i < len(rec["n_acc"]) else 0
                    per_req_tok[i] = per_req_tok.get(i, 0) + (na + 1)
                    per_req_iters[i] = per_req_iters.get(i, 0) + 1
                    # If this request's pre-EOS token count has been reached, record iter.
                    if i not in per_req_eos_iter and i in pre_eos_tok and per_req_tok[i] >= pre_eos_tok[i]:
                        per_req_eos_iter[i] = per_req_iters[i]

    # Aggregate pre-EOS TPF: pre_eos_tok / iters_to_first_pass_pre_eos
    pre_tpfs = []
    full_tpfs = []
    for i, n_pre in pre_eos_tok.items():
        if i in per_req_eos_iter and per_req_eos_iter[i] > 0:
            pre_tpfs.append(n_pre / per_req_eos_iter[i])
        if i in per_req_iters and per_req_iters[i] > 0 and i in per_req_tok:
            full_tpfs.append(per_req_tok[i] / per_req_iters[i])
    print(f"math_k3_ds (DS 64 prompts, k3_ds64 run):")
    print(f"  N requests with EOS reached: {len(pre_tpfs)} / {len(pre_eos_tok)}")
    if pre_tpfs:
        mean_pre = sum(pre_tpfs) / len(pre_tpfs)
        mean_full = sum(full_tpfs) / len(full_tpfs)
        print(f"  mean PRE-EOS TPF (real content only): {mean_pre:.3f}")
        print(f"  mean FULL-rollout TPF (incl post-EOS loops): {mean_full:.3f}")
        print(f"  delta (loop inflation): {mean_full - mean_pre:+.3f}")
        # Per-request preview
        print(f"\n  Per-prompt sample:")
        for i in sorted(pre_eos_tok.keys())[:10]:
            n_pre = pre_eos_tok[i]
            iters_pre = per_req_eos_iter.get(i, None)
            tpf_pre = (n_pre / iters_pre) if iters_pre else None
            n_full = per_req_tok.get(i, 0)
            iters_full = per_req_iters.get(i, 0)
            tpf_full = (n_full / iters_full) if iters_full else 0
            print(f"    [{i:2d}] pre_eos_tok={n_pre:4d} iters_pre={iters_pre} tpf_pre={tpf_pre:.2f}  "
                  f"| full_tok={n_full:4d} iters_full={iters_full:4d} tpf_full={tpf_full:.2f}")


if __name__ == "__main__":
    main()
