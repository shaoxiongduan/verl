"""Compare the decode-time canvas state distribution (from
_sim_assembly_decode.py --trace_jsonl) against the v11 training-pair
corruption construction.

Decode side, aggregated per window position j (0..W-1, j<W_ar = AR zone):
  - agree[j]   : P(window token before update == finally-committed token)
                 — the empirical "cleanliness" profile the model actually sees
  - keep[j]    : P(entropy <= tau) at canvas positions (survives re-noise)
  - commit dist: tokens committed per forward (frontier speed nu)

Training side (same machinery as pack.py): marginal corruption probability
per position under f ~ U{levels}, |R| = round(f*N) drawn without replacement
with weights w_j ∝ 0.5 + j/N, plus rho plausible-subs on kept positions.

Run:  python scripts/_analyze_assembly_trace.py --trace x.jsonl --W 32 --W_ar 8
"""
import argparse
import json
from collections import defaultdict

import torch


def training_marginal(N=32, levels=(1.0, 0.75, 0.5, 0.25, 0.125), rho=0.1,
                      n_mc=20000, seed=0):
    """Monte-Carlo marginal P(position j is corrupted) in a canvas pair."""
    g = torch.Generator().manual_seed(seed)
    w = 0.5 + torch.arange(N, dtype=torch.float32) / N
    hits = torch.zeros(N)
    for _ in range(n_mc):
        f = levels[int(torch.randint(0, len(levels), (1,), generator=g))]
        n_r = min(N, int(round(f * N)))
        R = torch.multinomial(w, n_r, replacement=False, generator=g)
        m = torch.zeros(N)
        m[R] = 1.0
        # plausible subs corrupt kept positions w.p. rho (alt-pool token)
        m = torch.where(m.bool(), m, torch.full_like(m, rho))
        hits += m
    return (hits / n_mc).tolist()


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--trace", required=True)
    ap.add_argument("--W", type=int, default=32)
    ap.add_argument("--W_ar", type=int, default=8)
    ap.add_argument("--label", default="")
    args = ap.parse_args()
    W, W_ar = args.W, args.W_ar

    finals = {}
    steps = defaultdict(list)
    for line in open(args.trace):
        r = json.loads(line)
        if "final" in r:
            finals[r["p"]] = r["final"]
        else:
            steps[r["p"]].append(r)

    agree_n = [0] * W
    agree_d = [0] * W
    keep_n = [0] * (W - W_ar)
    keep_d = [0] * (W - W_ar)
    commits = []
    cand_agree_n = [0] * W
    for p, recs in steps.items():
        fin = finals.get(p)
        if fin is None:
            continue
        for r in recs:
            L0 = r["L0"]
            commits.append(r["n_commit"])
            for j in range(W):
                ap_ = L0 + j
                if ap_ >= len(fin):
                    continue
                agree_d[j] += 1
                if r["win"][j] == fin[ap_]:
                    agree_n[j] += 1
                if r["cand"][j] == fin[ap_]:
                    cand_agree_n[j] += 1
            if r.get("keep") is not None:
                for j, k in enumerate(r["keep"]):
                    keep_d[j] += 1
                    keep_n[j] += int(k)

    n_fwd = len(commits)
    nu = sum(commits) / max(1, n_fwd)
    print(f"=== {args.label or args.trace} ===")
    print(f"forwards={n_fwd}  mean commit/fwd (nu)={nu:.3f}")
    print(f"{'j':>3} {'zone':>6} {'P(win==final)':>14} {'P(cand==final)':>15} {'P(keep)':>8}")
    tm = training_marginal(N=32)
    for j in range(W):
        zone = "AR" if j < W_ar else "canvas"
        a = agree_n[j] / max(1, agree_d[j])
        c = cand_agree_n[j] / max(1, agree_d[j])
        k = ""
        if j >= W_ar and keep_d[j - W_ar] > 0:
            k = f"{keep_n[j - W_ar] / keep_d[j - W_ar]:.3f}"
        print(f"{j:>3} {zone:>6} {a:>14.3f} {c:>15.3f} {k:>8}")
    print("\nTraining-pair corruption marginal (N=32, levels mix, rho=0.1):")
    print("  P(corrupt at j): " + " ".join(f"{x:.2f}" for x in tm))
    print("  -> implied P(clean at j): " + " ".join(f"{1 - x:.2f}" for x in tm))


if __name__ == "__main__":
    main()
