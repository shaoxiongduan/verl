"""TPF analysis: distribution width, correct/incorrect split, abnormal-run detection.

Reads result jsonls produced by vllm_tpf_trajectories.py and tpf_trajectories.py,
and JF reference logs. Scores completions with verl's math_dapo verifier (minerva
mode). Flags outputs as 'repetitive' if the last 200 chars contain a sub-string
of length >= 30 that repeats > 3 times.
"""
import json
import os
import re
import sys
import statistics
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, "/mnt/weka/home/hao.zhang/shao/verl")
from verl.utils.reward_score.math_dapo import last_boxed_only_string, remove_boxed, normalize_final_answer  # noqa: E402


def _norm(s: str) -> str:
    if s is None:
        return ""
    s = s.strip()
    # strip trailing periods, $, whitespace, surrounding {}
    s = s.replace("$", "").strip()
    while s.startswith("{") and s.endswith("}"):
        s = s[1:-1].strip()
    s = normalize_final_answer(s)
    # convert "3.5" / "3.50" / "3.500" to canonical numeric where possible
    try:
        f = float(s.replace(",", ""))
        if f == int(f):
            return str(int(f))
        return f"{f:.6g}"
    except ValueError:
        return s


def verify(completion: str, expected: str):
    boxed = last_boxed_only_string(completion)
    if boxed is None:
        return False, "[NO_BOX]"
    try:
        pred = remove_boxed(boxed)
    except Exception:
        return False, "[NO_BOX]"
    return _norm(pred) == _norm(expected), pred

EVAL_PROMPTS_JSONL = "/mnt/weka/home/hao.zhang/shao/verl/eval_passk/eval_prompts_tpf.jsonl"
OUT_DIR = Path("/mnt/weka/home/hao.zhang/shao/verl/eval_passk/tpf_results")

# Load ground-truth answer by prompt text
gt = {}
with open(EVAL_PROMPTS_JSONL) as f:
    for line in f:
        r = json.loads(line)
        gt[r["input"]] = r["expected_answer"]


def is_repetitive(text: str) -> bool:
    if not text or len(text) < 200:
        return False
    # check for any 30+ char substring repeating > 3x in the last 600 chars
    tail = text[-600:]
    for L in (30, 50, 80):
        for i in range(len(tail) - L * 4):
            seg = tail[i:i + L]
            if seg.count(seg[:10]) > 3 and tail.count(seg) > 3:
                return True
    return False


def fmt_q(xs):
    xs = sorted(xs)
    n = len(xs)

    def q(p):
        if n == 0:
            return float("nan")
        k = max(0, min(n - 1, int(p * (n - 1))))
        return xs[k]

    return q(0.10), q(0.25), q(0.5), q(0.75), q(0.90)


def analyze_jsonl(path: str, label: str):
    rows = [json.loads(l) for l in open(path)]
    out = {
        "label": label,
        "n": 0, "n_corr": 0, "n_inc": 0, "n_rep": 0,
        "tpf_all": [], "tpf_corr": [], "tpf_inc": [],
        "top_outliers": [],
    }
    for r in rows:
        if r.get("tpf") in (None, 0, 0.0) and r.get("num_tokens", 0) == 0:
            continue
        tpf = float(r["tpf"])
        prompt = r["prompt"]
        completion = r.get("completion", "") or ""
        ans = gt.get(prompt)
        if ans is None:
            corr = None
        else:
            try:
                corr, _ = verify(completion, ans)
            except Exception:
                corr = False
        rep = is_repetitive(completion)
        out["n"] += 1
        out["tpf_all"].append(tpf)
        if corr:
            out["n_corr"] += 1
            out["tpf_corr"].append(tpf)
        else:
            out["n_inc"] += 1
            out["tpf_inc"].append(tpf)
        if rep:
            out["n_rep"] += 1
        out["top_outliers"].append((tpf, r.get("num_tokens", 0), rep, corr, prompt[:60]))
    out["top_outliers"].sort(reverse=True)
    out["top_outliers"] = out["top_outliers"][:5]
    return out


def analyze_jfref_log(path: str, prompts_in_order: list[str], label: str):
    # Format per row: "  [k/64] new_toks= XX iters= XX tpf= XX reason=eos"
    # JF ref doesn't dump completion, only stats. Use prompt order for GT match.
    # But we can't check correctness without completion text. So skip correctness here.
    out = {
        "label": label, "n": 0, "n_corr": 0, "n_inc": 0, "n_rep": 0,
        "tpf_all": [], "tpf_corr": [], "tpf_inc": [], "top_outliers": [],
    }
    pat = re.compile(r"\[\s*(\d+)/\s*\d+\]\s+new_toks=\s*(\d+)\s+iters=\s*(\d+)\s+tpf=\s*([\d.]+)\s+reason=(\w+)")
    with open(path) as f:
        for line in f:
            m = pat.search(line)
            if not m:
                continue
            idx, ntok, iters, tpf, reason = m.groups()
            out["n"] += 1
            out["tpf_all"].append(float(tpf))
    return out


def report(a):
    if a["n"] == 0:
        print(f"  {a['label']:36}  EMPTY")
        return
    m = statistics.mean(a["tpf_all"])
    s = statistics.stdev(a["tpf_all"]) if len(a["tpf_all"]) > 1 else 0.0
    se = s / (len(a["tpf_all"]) ** .5) if a["tpf_all"] else 0.0
    q10, q25, q50, q75, q90 = fmt_q(a["tpf_all"])
    cv = s / m if m else 0
    line = (
        f"  {a['label']:36}  n={a['n']:2d}  mean={m:.3f}±{se:.3f}  "
        f"std={s:.3f}  CV={cv:.2f}  "
        f"q10={q10:.2f} q25={q25:.2f} med={q50:.2f} q75={q75:.2f} q90={q90:.2f}  "
        f"max={max(a['tpf_all']):.2f}"
    )
    print(line)
    if a["n_corr"] + a["n_inc"] > 0:
        mc = statistics.mean(a["tpf_corr"]) if a["tpf_corr"] else float("nan")
        mi = statistics.mean(a["tpf_inc"]) if a["tpf_inc"] else float("nan")
        print(f"      correct: n={a['n_corr']:2d} mean={mc:.3f}    incorrect: n={a['n_inc']:2d} mean={mi:.3f}    repetitive: n={a['n_rep']}")
    if a["top_outliers"]:
        print(f"      top-5 by TPF:")
        for tpf, ntok, rep, corr, prompt in a["top_outliers"]:
            ct = "✓" if corr else ("✗" if corr is False else "?")
            rt = "REP" if rep else "   "
            print(f"        tpf={tpf:6.2f}  ntok={ntok:4d}  {ct} {rt}  {prompt!r}")


def main():
    runs = [
        # vLLM Jacobi T=0 max_new=1024 (original)
        ("base T=0 vllm m1024", OUT_DIR / "base_jf_math_7b__vllm_T0.jsonl"),
        ("AR T=0 vllm m1024",   OUT_DIR / "math_ar_ds_step_300__vllm_T0.jsonl"),
        ("consA T=0 vllm m1024", OUT_DIR / "dflashce_v2_step_20__vllm_T0.jsonl"),
        ("consB T=0 vllm m1024", OUT_DIR / "dflashce_fixed_step_20__vllm_T0.jsonl"),
        # vLLM Jacobi T=0.6 max_new=1024
        ("base T=.6 vllm m1024", OUT_DIR / "base_jf_math_7b__vllm_T06.jsonl"),
        ("AR T=.6 vllm m1024",   OUT_DIR / "math_ar_ds_step_300__vllm_T06.jsonl"),
        ("consA T=.6 vllm m1024", OUT_DIR / "dflashce_v2_step_20__vllm_T06.jsonl"),
        ("consB T=.6 vllm m1024", OUT_DIR / "dflashce_fixed_step_20__vllm_T06.jsonl"),
        # vLLM Jacobi T=0 max_new=2048
        ("base T=0 vllm m2048", OUT_DIR / "base_jf_math_7b__vllm_greedy2048.jsonl"),
        ("AR T=0 vllm m2048",   OUT_DIR / "math_ar_ds_step_300__vllm_greedy2048.jsonl"),
        ("consA T=0 vllm m2048", OUT_DIR / "dflashce_v2_step_20__vllm_greedy2048.jsonl"),
        ("consB T=0 vllm m2048", OUT_DIR / "dflashce_fixed_step_20__vllm_greedy2048.jsonl"),
        # vLLM Jacobi T=1.0 max_new=2048 (train params)
        ("base T=1 vllm m2048", OUT_DIR / "base_jf_math_7b__vllm_trainparams.jsonl"),
        ("AR T=1 vllm m2048",   OUT_DIR / "math_ar_ds_step_300__vllm_trainparams.jsonl"),
        ("consA T=1 vllm m2048", OUT_DIR / "dflashce_v2_step_20__vllm_trainparams.jsonl"),
        ("consB T=1 vllm m2048", OUT_DIR / "dflashce_fixed_step_20__vllm_trainparams.jsonl"),
    ]
    print("=" * 100)
    print("vLLM Jacobi engine — per-run TPF distributions + correctness")
    print("=" * 100)
    for label, p in runs:
        if not p.exists():
            print(f"  {label:36}  MISSING ({p.name})")
            continue
        report(analyze_jsonl(str(p), label))

    print()
    print("=" * 100)
    print("JF reference engine (greedy block-decode) — TPF distributions (no completion -> no correctness)")
    print("=" * 100)
    jfref_runs = [
        ("base jf-ref m2048",     OUT_DIR / "base_jf_math_7b__jfref_trainparams.log"),
        ("AR jf-ref m2048",       OUT_DIR / "math_ar_ds_step_300__jfref_trainparams.log"),
        ("consA jf-ref m2048",    OUT_DIR / "dflashce_v2_step_20__jfref_trainparams.log"),
        ("consB jf-ref m2048",    OUT_DIR / "dflashce_fixed_step_20__jfref_trainparams.log"),
    ]
    for label, p in jfref_runs:
        if not p.exists():
            print(f"  {label:36}  MISSING ({p.name})")
            continue
        report(analyze_jfref_log(str(p), [], label))


if __name__ == "__main__":
    main()
