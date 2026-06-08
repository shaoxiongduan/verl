"""Analyze saved Jacobi trajectories to find the clean / confusing / noise
boundary in per-iter drafts.

Per Jacobi iteration of a request, we have:
  - draft[K]:           the K draft tokens proposed (slot of K=32 per iter)
  - n_acc:              number accepted as committed prefix
  - target_argmax[K]:   model's top-1 prediction at each draft position
  - target_entropy[K]:  entropy of model's softmax at each draft position
  - target_max_prob[K]: max probability of model's softmax at each draft position

Categorization per position j:
  - j < n_acc:                          CLEAN (accepted, draft[j] matches model)
  - j >= n_acc, draft[j] == target[j]:  IMPLICIT_CLEAN (would be accepted but
                                        prefix already broke earlier — rare)
  - j >= n_acc, draft[j] in top-k:      CONFUSING (model is "uncertain": its
                                        argmax differs but draft is plausible)
  - j >= n_acc, draft[j] not in top-k:  NOISE (model strongly disagrees)

We use `target_max_prob` as the confidence proxy:
  - low max_prob (e.g. < 0.3): model is uncertain → wider distribution →
    "confusing zone" where draft tokens have non-trivial probability
  - high max_prob (e.g. > 0.7): model is confident → if draft != argmax, draft
    is in the tail → "noise zone"

Note vllm_jacobi_patch records BATCHED per-iter format (num_draft is a list
per request, draft is a list of lists, etc.). We iterate the request dimension.
"""
from __future__ import annotations

import argparse
import glob
import json
import math
from collections import defaultdict


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser()
    p.add_argument("--traj_glob", required=True,
                   help="glob pattern matching per-PID JSONL files")
    p.add_argument("--out_summary", required=True)
    p.add_argument("--K", type=int, default=32)
    return p.parse_args()


def main() -> None:
    args = parse_args()
    files = sorted(glob.glob(args.traj_glob))
    if not files:
        raise SystemExit(f"No traj files matching {args.traj_glob}")
    print(f"[analyze] {len(files)} traj files", flush=True)

    # Aggregate by position-within-block (0..K-1) and by relative offset
    # from n_acc (-n_acc, ..., 0, +1, ..., K-1-n_acc).
    n_iters = 0
    n_accs: list[int] = []
    # Per-position absolute (j=0..K-1)
    by_pos = defaultdict(lambda: {"entropy": [], "max_prob": [], "log_p_draft": [],
                                   "draft_eq_argmax": 0, "n": 0})
    # Per-relative offset (offset = j - n_acc; 0 = first rejected, 1 = second, ...)
    by_rel = defaultdict(lambda: {"entropy": [], "max_prob": [], "draft_eq_argmax": 0, "n": 0})

    # Sample histograms of (entropy, max_prob) for j >= n_acc
    post_acc_samples: list[tuple[int, int, float, float, int]] = []  # (rel_off, j, ent, mp, draft_eq_argmax)

    for fp in files:
        with open(fp) as f:
            for line in f:
                try:
                    rec = json.loads(line)
                except json.JSONDecodeError:
                    continue
                if "num_draft" not in rec or not isinstance(rec["num_draft"], list):
                    continue
                # Skip warmup batched records (all 1s, very long list)
                if len(rec["num_draft"]) > 256 and set(rec["num_draft"]) == {1}:
                    continue
                # `target_entropy`/`target_max_prob` keys are optional.
                ent_per_req = rec.get("target_entropy")
                mp_per_req = rec.get("target_max_prob")
                if ent_per_req is None or mp_per_req is None:
                    continue
                drafts = rec["draft"]
                targs  = rec["target_argmax"]
                n_accs_list = rec["n_acc"]
                num_draft = rec["num_draft"]
                for i, k in enumerate(num_draft):
                    if k <= 0 or k != args.K:
                        continue
                    if i >= len(drafts) or i >= len(targs):
                        continue
                    d = drafts[i]
                    t = targs[i]
                    e = ent_per_req[i] if i < len(ent_per_req) else None
                    m = mp_per_req[i] if i < len(mp_per_req) else None
                    na = n_accs_list[i] if i < len(n_accs_list) else 0
                    if len(d) != k or len(t) != k or e is None or len(e) != k:
                        continue
                    n_iters += 1
                    n_accs.append(na)
                    for j in range(k):
                        d_eq_t = int(d[j] == t[j])
                        # log p that model assigns to the draft token. We don't
                        # have full probs, only argmax + max_prob + entropy. So
                        # we cannot compute log_p_draft exactly. Approximate:
                        # if d_eq_t, log_p_draft = log(max_prob); else unknown
                        # (skip in aggregation).
                        by_pos[j]["entropy"].append(e[j])
                        by_pos[j]["max_prob"].append(m[j])
                        by_pos[j]["draft_eq_argmax"] += d_eq_t
                        by_pos[j]["n"] += 1
                        if d_eq_t:
                            by_pos[j]["log_p_draft"].append(math.log(max(m[j], 1e-9)))

                        rel = j - na
                        by_rel[rel]["entropy"].append(e[j])
                        by_rel[rel]["max_prob"].append(m[j])
                        by_rel[rel]["draft_eq_argmax"] += d_eq_t
                        by_rel[rel]["n"] += 1

                        if j >= na:
                            post_acc_samples.append((rel, j, e[j], m[j], d_eq_t))

    print(f"[analyze] n_iters={n_iters} n_acc mean={sum(n_accs)/max(1,len(n_accs)):.2f}", flush=True)

    out = {
        "n_iters": n_iters,
        "n_acc_distribution": {
            "mean": sum(n_accs)/max(1,len(n_accs)),
            "median": sorted(n_accs)[len(n_accs)//2] if n_accs else 0,
            "p25":  sorted(n_accs)[len(n_accs)//4]   if n_accs else 0,
            "p75":  sorted(n_accs)[3*len(n_accs)//4] if n_accs else 0,
            "max":  max(n_accs) if n_accs else 0,
            "min":  min(n_accs) if n_accs else 0,
        },
        "per_position_in_block": {},
        "per_relative_offset_from_n_acc": {},
    }
    for j in sorted(by_pos.keys()):
        d = by_pos[j]
        out["per_position_in_block"][str(j)] = {
            "n": d["n"],
            "ent_mean": sum(d["entropy"])/max(1,d["n"]),
            "mp_mean":  sum(d["max_prob"])/max(1,d["n"]),
            "draft_eq_argmax_frac": d["draft_eq_argmax"] / max(1, d["n"]),
        }
    for rel in sorted(by_rel.keys()):
        d = by_rel[rel]
        out["per_relative_offset_from_n_acc"][str(rel)] = {
            "n": d["n"],
            "ent_mean": sum(d["entropy"])/max(1,d["n"]),
            "mp_mean":  sum(d["max_prob"])/max(1,d["n"]),
            "draft_eq_argmax_frac": d["draft_eq_argmax"] / max(1, d["n"]),
        }

    # Save histogram of (entropy, max_prob) at relative offsets 0, 1, 2, 3, 4
    # to see how "confusion" decays past the accepted prefix.
    hist_thresholds = [0.2, 0.4, 0.6, 0.8]
    out["post_acc_max_prob_distribution"] = {}
    for rel in [0, 1, 2, 3, 4, 5, 10, 20]:
        rel_samples = [s for s in post_acc_samples if s[0] == rel]
        if not rel_samples:
            continue
        ent = [s[2] for s in rel_samples]
        mp  = [s[3] for s in rel_samples]
        bucket = {f"mp<{th}": sum(1 for x in mp if x < th)/len(mp) for th in hist_thresholds}
        bucket["mp>=0.8"] = sum(1 for x in mp if x >= 0.8)/len(mp)
        bucket["draft_eq_argmax_frac"] = sum(s[4] for s in rel_samples)/len(rel_samples)
        bucket["mp_mean"] = sum(mp)/len(mp)
        bucket["ent_mean"] = sum(ent)/len(ent)
        bucket["n"] = len(rel_samples)
        out["post_acc_max_prob_distribution"][f"rel_{rel}"] = bucket

    with open(args.out_summary, "w") as f:
        json.dump(out, f, indent=2)
    print(f"[analyze] wrote summary to {args.out_summary}", flush=True)


if __name__ == "__main__":
    main()
