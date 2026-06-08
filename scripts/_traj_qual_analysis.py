"""Qualitative + quantitative trace analysis.

For each captured trajectory iter, we have:
  draft[0..K-1]            — input tokens fed to the model (the noise condition)
  target_argmax[0..K-1]    — model's argmax prediction at each position
  target_entropy[0..K-1]   — entropy of model's prediction at each position
  target_max_prob[0..K-1]  — max prob (confidence) at each position
  n_acc                    — # accepted spec tokens (= leading match prefix)

For correlation we use TPF = n_acc + 1 (matches user's convention).

Sections:
  A) Per-iter feature → TPF correlations (Pearson)
  B) Top/bottom TPF iters: full text decode + entropy histogram
  C) Per-prompt position-aligned consB vs base GAIN in TPF — sort by gain, read samples
"""
from __future__ import annotations
import argparse, glob, json, os, statistics, math
from collections import defaultdict

import sys
sys.path.insert(0, "/mnt/weka/home/hao.zhang/shao/verl")
from transformers import AutoTokenizer

TOKENIZER = AutoTokenizer.from_pretrained(
    "/mnt/weka/home/hao.zhang/.cache/huggingface/hub/models--JacobiForcing--JacobiForcing_Math_7B_v1/snapshots/e65283c1b3d205b23c2bdf9946158035c409d3a1"
)


def load_runs(in_dir):
    runs = defaultdict(dict)
    for fp in sorted(glob.glob(f"{in_dir}/p*__*.json")):
        rec = json.load(open(fp))
        runs[rec["prompt_idx"]][rec["model_label"]] = rec
    return runs


def decode_compact(ids, max_chars=200):
    text = TOKENIZER.decode(ids, skip_special_tokens=False)
    text = text.replace("\n", "↵").replace("\t", "→")
    if len(text) > max_chars:
        return text[:max_chars] + "..."
    return text


def per_iter_features(rec):
    """For each iter i, features describe its NOISE CONDITION (its draft + its prediction
    quality) and target_tpf = TPF of iter i+1 (which validates iter i's target_argmax).

    iter i's target_argmax → becomes iter i+1's draft (shift-by-1 windowed)
    iter i+1's n_acc       → measures how many of iter i's predictions matched iter i+1's argmax
    So 'how good was iter i as conditioning' = (n_acc[i+1] + 1).
    """
    rows = []
    cursor = 0
    iters_data = []
    # First pass — collect all iters with positions
    for i, it in enumerate(rec["iters"]):
        if not it.get("num_draft"): continue
        nd = it["num_draft"][0]
        if nd <= 0: continue
        draft = it["draft"][0][:nd]
        target = it["target_argmax"][0][:nd]
        ent = it.get("target_entropy", [None])[0] or []
        ent = ent[:nd] if ent else []
        maxp = it.get("target_max_prob", [None])[0] or []
        maxp = maxp[:nd] if maxp else []
        n_acc = it["n_acc"][0]
        if not ent:
            cursor += n_acc + 1
            iters_data.append(None)
            continue
        iters_data.append({
            "iter_idx": i,
            "pos": cursor,
            "self_tpf": n_acc + 1,  # how many tokens THIS iter committed
            "n_acc": n_acc,
            "draft": draft,
            "target": target,
            "entropy": ent,
            "max_prob": maxp,
            "K": nd,
            "ent_mean": statistics.mean(ent),
            "ent_max": max(ent),
            "ent_min": min(ent),
            "ent_first_third": statistics.mean(ent[:nd//3]) if nd >= 3 else statistics.mean(ent),
            "ent_last_third": statistics.mean(ent[-nd//3:]) if nd >= 3 else statistics.mean(ent),
            "max_prob_mean": statistics.mean(maxp) if maxp else 0,
            "match_count": sum(1 for d,t in zip(draft, target) if d==t),
            "draft_unique_ratio": len(set(draft)) / len(draft),
        })
        cursor += n_acc + 1
    # Second pass — attach NEXT iter's n_acc as the "tpf" target (= conditioning quality)
    for k, cur in enumerate(iters_data):
        if cur is None: continue
        nxt = iters_data[k + 1] if k + 1 < len(iters_data) else None
        if nxt is None: continue  # last iter has no next; drop
        cur["tpf"] = nxt["self_tpf"]    # quality of cur as conditioning for next
        rows.append(cur)
    return rows


def corr(xs, ys):
    if len(xs) < 2: return float("nan")
    mx, my = statistics.mean(xs), statistics.mean(ys)
    num = sum((x-mx)*(y-my) for x,y in zip(xs,ys))
    dx = math.sqrt(sum((x-mx)**2 for x in xs))
    dy = math.sqrt(sum((y-my)**2 for y in ys))
    return num/(dx*dy) if dx>0 and dy>0 else float("nan")


def section_a_correlations(runs):
    print("=" * 80)
    print("(A) Per-iter feature → TPF correlations")
    print("=" * 80)
    feature_keys = ["n_acc", "ent_mean", "ent_max", "ent_min",
                    "ent_first_third", "ent_last_third",
                    "max_prob_mean", "match_count", "draft_unique_ratio", "pos"]
    print(f"{'model':14}", end="")
    for k in feature_keys:
        print(f"  {k:>16}", end="")
    print()
    for label in ("base", "consA_s300", "consB_s300"):
        all_rows = []
        for p_idx, by_label in runs.items():
            if label in by_label:
                all_rows.extend(per_iter_features(by_label[label]))
        if not all_rows: continue
        tpfs = [r["tpf"] for r in all_rows]
        print(f"{label:14}", end="")
        for k in feature_keys:
            xs = [r[k] for r in all_rows]
            r = corr(xs, tpfs)
            print(f"  {r:>+16.3f}", end="")
        print(f"   (n={len(all_rows)})")


def section_b_top_bottom(runs, label="consB_s300", n=5):
    print()
    print("=" * 80)
    print(f"(B) Top & bottom TPF iters for {label}")
    print("=" * 80)
    all_rows = []
    for p_idx, by_label in runs.items():
        if label in by_label:
            for r in per_iter_features(by_label[label]):
                r["prompt_idx"] = p_idx
                all_rows.append(r)
    all_rows.sort(key=lambda r: r["tpf"], reverse=True)
    print(f"\n--- TOP {n} TPF iters ({label}) ---")
    for r in all_rows[:n]:
        print(f"\n  prompt={r['prompt_idx']} iter={r['iter_idx']} pos={r['pos']} TPF={r['tpf']} n_acc={r['n_acc']} | "
              f"ent[mean/min/max]={r['ent_mean']:.2f}/{r['ent_min']:.2f}/{r['ent_max']:.2f} "
              f"first3rd_ent={r['ent_first_third']:.2f} last3rd_ent={r['ent_last_third']:.2f}")
        print(f"    DRAFT  : {decode_compact(r['draft'])}")
        print(f"    TARGET : {decode_compact(r['target'])}")
        print(f"    entropy bin profile (per pos): " + " ".join(f"{e:4.1f}" for e in r['entropy']))
    print(f"\n--- BOTTOM {n} TPF iters ({label}) ---")
    for r in all_rows[-n:]:
        print(f"\n  prompt={r['prompt_idx']} iter={r['iter_idx']} pos={r['pos']} TPF={r['tpf']} n_acc={r['n_acc']} | "
              f"ent[mean/min/max]={r['ent_mean']:.2f}/{r['ent_min']:.2f}/{r['ent_max']:.2f} "
              f"first3rd_ent={r['ent_first_third']:.2f} last3rd_ent={r['ent_last_third']:.2f}")
        print(f"    DRAFT  : {decode_compact(r['draft'])}")
        print(f"    TARGET : {decode_compact(r['target'])}")
        print(f"    entropy bin profile (per pos): " + " ".join(f"{e:4.1f}" for e in r['entropy']))


def section_c_position_aligned_gain(runs, ref="base", trained="consB_s300", n=5, pos_bucket=64):
    """Pair-align iters by (prompt, position_bucket) and compute trained_tpf - ref_tpf."""
    print()
    print("=" * 80)
    print(f"(C) {trained} − {ref} TPF gain at position-aligned iters (bucket={pos_bucket})")
    print("=" * 80)
    # Collect per (prompt, pos_bucket) the average TPF for each model
    by_key = defaultdict(lambda: {"base": [], "trained": []})
    for p_idx, by_label in runs.items():
        for who, label in (("base", ref), ("trained", trained)):
            if label not in by_label: continue
            for r in per_iter_features(by_label[label]):
                key = (p_idx, r["pos"] // pos_bucket * pos_bucket)
                by_key[key][who].append(r)
    pairs = []
    for key, d in by_key.items():
        if not d["base"] or not d["trained"]: continue
        b_tpf = statistics.mean(r["tpf"] for r in d["base"])
        t_tpf = statistics.mean(r["tpf"] for r in d["trained"])
        # Take the trained side's first iter as representative trace
        rep = d["trained"][0]
        pairs.append({"prompt_idx": key[0], "pos_bucket": key[1],
                      "base_tpf": b_tpf, "trained_tpf": t_tpf,
                      "gain": t_tpf - b_tpf, "trace": rep})
    pairs.sort(key=lambda x: x["gain"], reverse=True)
    print(f"\n--- TOP {n} GAIN (cons helped most) ---")
    for p in pairs[:n]:
        r = p["trace"]
        print(f"\n  prompt={p['prompt_idx']} pos~{p['pos_bucket']}: "
              f"base_TPF={p['base_tpf']:.2f}  {trained}_TPF={p['trained_tpf']:.2f}  GAIN={p['gain']:+.2f}")
        print(f"    ({trained} iter) DRAFT  : {decode_compact(r['draft'])}")
        print(f"    ({trained} iter) TARGET : {decode_compact(r['target'])}")
        print(f"    ent_mean={r['ent_mean']:.2f} ent_first={r['ent_first_third']:.2f} ent_last={r['ent_last_third']:.2f}")
    print(f"\n--- BOTTOM {n} GAIN (cons hurt most) ---")
    for p in pairs[-n:]:
        r = p["trace"]
        print(f"\n  prompt={p['prompt_idx']} pos~{p['pos_bucket']}: "
              f"base_TPF={p['base_tpf']:.2f}  {trained}_TPF={p['trained_tpf']:.2f}  GAIN={p['gain']:+.2f}")
        print(f"    ({trained} iter) DRAFT  : {decode_compact(r['draft'])}")
        print(f"    ({trained} iter) TARGET : {decode_compact(r['target'])}")
        print(f"    ent_mean={r['ent_mean']:.2f} ent_first={r['ent_first_third']:.2f} ent_last={r['ent_last_third']:.2f}")


def main():
    runs = load_runs("eval_passk/tpf_results/trajectory_viz")
    section_a_correlations(runs)
    section_b_top_bottom(runs, label="consB_s300", n=5)
    section_b_top_bottom(runs, label="base", n=5)
    section_c_position_aligned_gain(runs)


if __name__ == "__main__":
    main()
