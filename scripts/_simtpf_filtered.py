"""Same as _simtpf_clean_v2 but filters cycles to exclude the repetition tail.

Filter rule: drop cycles where the running 5-gram repetition rate of the
committed-so-far tokens exceeds 0.20, OR where n_acc >= 16 (since real n_acc
rarely exceeds 16 in math text).
"""
from __future__ import annotations
import argparse
import numpy as np
from collections import defaultdict, Counter


SLOPES = {"math_k3": 0.10, "base": -0.16}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--probe", required=True)
    p.add_argument("--model", required=True, choices=["math_k3", "base"])
    p.add_argument("--holdout_prompts_lt", type=int, default=16)
    p.add_argument("--K", type=int, default=32)
    p.add_argument("--frac_keep", type=float, default=0.7,
                   help="keep only first frac_keep of cycles per prompt (drops repetition tail)")
    return p.parse_args()


def compute_P(z, probe):
    H = z["hidden"].astype(np.float32)
    N, K, d = H.shape
    aux = np.stack([z["top1_prob"], z["entropy"], z["margin_log"], z["top5_prob_sum"]], axis=-1).astype(np.float32)
    pos = np.tile(np.arange(K, dtype=np.float32)/K, (N, 1))[..., None]
    feats3 = np.concatenate([H, aux, pos], axis=-1)
    d_in = feats3.shape[-1]
    pz = np.load(probe, allow_pickle=True)
    mu = pz["mu"].astype(np.float32); sd = pz["sd"].astype(np.float32)
    with_aux = bool(pz["with_aux"])
    if not with_aux:
        feats3 = H; d_in = d
    feats2 = feats3.reshape(N*K, d_in)
    Xs = (feats2 - mu) / sd
    if "W1" in pz.files:
        W1 = pz["W1"].astype(np.float32); b1 = pz["b1"].astype(np.float32)
        W2 = pz["W2"].astype(np.float32); b2 = pz["b2"].astype(np.float32)
        h = np.maximum(0, Xs @ W1 + b1)
        z2 = (h @ W2 + b2).flatten()
    else:
        W = pz["W"].astype(np.float32); b = float(pz["b"])
        z2 = Xs @ W + b
    return (1/(1+np.exp(-z2))).reshape(N, K)


def decode_first(P, thr):
    K = P.shape[1]; out = []
    for r in P:
        below = np.where(r < thr)[0]
        out.append(int(below[0]) if len(below) else K)
    return np.array(out)


def decode_last(P, thr):
    K = P.shape[1]; out = []
    for r in P:
        above = np.where(r >= thr)[0]
        out.append(int(above[-1] + 1) if len(above) else 0)
    return np.array(out)


def main():
    args = parse_args()
    z = np.load(args.data)
    P = compute_P(z, args.probe)
    pi = z["prompt_idx"]; ci = z["cycle_idx"]; nacc = z["n_acc"]
    va = np.where(pi < args.holdout_prompts_lt)[0]

    # Build per-prompt ordered cycle indices, drop the tail
    by_p = defaultdict(list)
    for i in va:
        by_p[int(pi[i])].append((int(ci[i]), i))
    kept = []
    for p, lst in by_p.items():
        lst.sort()
        n_keep = int(len(lst) * args.frac_keep)
        kept.extend([idx for _, idx in lst[:n_keep]])
    kept = np.array(kept)

    P_va = P[kept]; nacc_va = nacc[kept].astype(int)
    print(f"val cycles BEFORE filter: {len(va)}  AFTER (keep first {args.frac_keep:.0%} per prompt): {len(kept)}")
    print(f"true mean n_acc AFTER filter = {nacc_va.mean():.2f}")
    print(f"ORACLE TPF (perfect probe) = mean(true+1) = {(nacc_va+1).mean():.3f}")
    slope = SLOPES[args.model]
    print(f"model={args.model}  leak slope = {slope:+.3f}\n")

    def sim(pb, tb):
        pb = np.clip(pb, 0, args.K)
        commits = np.where(pb < tb, pb, tb + 1).astype(np.float64)
        overshoot = np.maximum(0, pb - tb - 1)
        leak = slope * overshoot
        tpf = (commits.sum() - leak.sum()) / len(pb)
        return tpf, (pb < tb).mean(), (pb > tb).mean(), overshoot.mean()

    print(f"{'rule':<6} {'thr':>5} {'off':>4} {'TPF':>6} {'P(under)':>9} {'P(over)':>8} {'over_amt':>9}")
    for rule in ["first", "last"]:
        for thr in [0.2, 0.3, 0.5]:
            base_b = decode_first(P_va, thr) if rule == "first" else decode_last(P_va, thr)
            for off in [-1, 0, 1, 2, 3, 5]:
                pb = base_b + off
                tpf, pu, po, ov = sim(pb, nacc_va)
                print(f"{rule:<6} {thr:>5.2f} {off:>+4d} {tpf:>6.3f} {pu*100:>8.1f}% {po*100:>7.1f}% {ov:>9.2f}")


if __name__ == "__main__":
    main()
