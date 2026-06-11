"""Cleaner TPF model — no double counting.

For each cycle (in time-order within each prompt):
  commits_i:
    if pred_i < true_i (under-pred):  commits = pred_i
    if pred_i >= true_i:               commits = true_i + 1   (full oracle this iter)
  this_overshoot = max(0, pred_i - true_i - 1)

Leak penalty (next-iter degradation):
  For each cycle i with overshoot s, the NEXT cycle loses (slope * s) TPF.
  Empirical slope (per leaked token):
    math_k3: 0.10   (shift=0 → 5.94, shift=5 → 5.46, slope ≈ 0.10)
    base:   -0.16   (shift=0 → 4.94, shift=5 → 5.74, slope NEG; over-pred HELPS base)

TPF_total = (Σ commits - Σ leak_penalty) / N_cycles

Oracle TPF on this val set = mean(true+1) — this is the UPPER BOUND. Any predictor
strategy's TPF must be ≤ oracle.
"""
from __future__ import annotations
import argparse
import numpy as np


SLOPES = {"math_k3": 0.10, "base": -0.16}


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--probe", required=True)
    p.add_argument("--model", required=True, choices=["math_k3", "base"])
    p.add_argument("--holdout_prompts_lt", type=int, default=16)
    p.add_argument("--K", type=int, default=32)
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


def simulate(pred_b, true_b, slope, K):
    pred_b = np.clip(pred_b, 0, K)
    n = len(pred_b)
    commits = np.where(pred_b < true_b, pred_b, true_b + 1).astype(np.float64)
    overshoot = np.maximum(0, pred_b - true_b - 1)
    leak_penalty = slope * overshoot
    total_commits = commits.sum()
    total_penalty = leak_penalty.sum()
    tpf = (total_commits - total_penalty) / n
    return {
        "tpf": tpf,
        "oracle_tpf": (true_b + 1).mean(),  # if probe were perfect
        "n": n,
        "p_under": (pred_b < true_b).mean(),
        "p_match": (pred_b == true_b).mean(),
        "p_over":  (pred_b > true_b).mean(),
        "mean_commits": commits.mean(),
        "mean_overshoot": overshoot.mean(),
        "total_penalty_per_cycle": total_penalty / n,
        "shortfall_mean": ((true_b - pred_b)[pred_b < true_b]).mean() if (pred_b < true_b).sum() > 0 else 0,
    }


def main():
    args = parse_args()
    z = np.load(args.data)
    P = compute_P(z, args.probe)
    pi = z["prompt_idx"]; nacc = z["n_acc"]
    va = np.where(pi < args.holdout_prompts_lt)[0]
    P_va = P[va]; nacc_va = nacc[va].astype(int)
    print(f"val cycles = {len(va)}, true mean n_acc = {nacc_va.mean():.2f}")
    print(f"ORACLE TPF (perfect probe) = mean(true+1) = {(nacc_va+1).mean():.3f}")
    slope = SLOPES[args.model]
    print(f"model={args.model}  leak slope = {slope:+.3f} TPF / overshoot-token\n")

    print(f"{'rule':<6} {'thr':>5} {'off':>4} {'TPF':>6} {'mean_cmt':>9} {'penalty/cyc':>12} {'P(under)':>9} {'P(over)':>8} {'over_amt':>9} {'shortfall':>10}")
    strategies = []
    for rule in ["first", "last"]:
        for thr in [0.2, 0.3, 0.5]:
            for off in [-1, 0, 1, 2, 3, 5]:
                strategies.append((rule, thr, off))

    for rule, thr, off in strategies:
        base_b = decode_first(P_va, thr) if rule == "first" else decode_last(P_va, thr)
        pred_b = base_b + off
        r = simulate(pred_b, nacc_va, slope, args.K)
        print(f"{rule:<6} {thr:>5.2f} {off:>+4d} {r['tpf']:>6.3f} {r['mean_commits']:>9.3f} {r['total_penalty_per_cycle']:>12.3f} {r['p_under']*100:>8.1f}% {r['p_over']*100:>7.1f}% {r['mean_overshoot']:>9.2f} {r['shortfall_mean']:>10.2f}")


if __name__ == "__main__":
    main()
