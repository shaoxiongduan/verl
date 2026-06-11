"""Compute TPF model across many predictor strategies.

For each strategy in the cross-product {first, last} × {thr} × {offset}:
  - apply on val cycles (prompt-holdout)
  - generate pred_b
  - pair with true_b
  - run the shift-leak-based TPF model
  - report TPF + under/over rates
"""
from __future__ import annotations
import argparse
import numpy as np
from collections import defaultdict


SHIFT_TABLE = {
    "math_k3": [5.94, 5.79, 5.76, 5.67, 5.53, 5.46, 5.40, 5.34, 5.28, 5.22, 5.16],
    "base":    [4.94, 5.23, 5.51, 5.63, 5.67, 5.74, 5.74, 5.74, 5.74, 5.74, 5.74],
}


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


def decode_first(P_row, thr):
    below = np.where(P_row < thr)[0]
    return int(below[0]) if len(below) else len(P_row)


def decode_last(P_row, thr):
    above = np.where(P_row >= thr)[0]
    return int(above[-1] + 1) if len(above) else 0


def simulate_tpf(pred_b, true_b, table, K, max_shift=10):
    """Process cycles in given order; assume they came from one stream so shifts carry over."""
    total_tok = 0.0
    n_under = n_match = n_over = 0
    last_shift = 0
    for pb, tb in zip(pred_b, true_b):
        if pb < tb:
            commits = float(pb)
            n_under += 1
            this_shift = 0
        elif pb == tb:
            commits = float(tb + 1)
            n_match += 1
            this_shift = 0
        else:
            s = min(last_shift, max_shift)
            commits = (tb + 1) * (table[s] / table[0])
            n_over += 1
            this_shift = min(pb - tb - 1, max_shift)
        total_tok += commits
        last_shift = this_shift
    n = len(pred_b)
    return {
        "tpf": total_tok / max(1, n),
        "n": n,
        "p_under": n_under / n,
        "p_match": n_match / n,
        "p_over": n_over / n,
    }


def main():
    args = parse_args()
    z = np.load(args.data)
    P = compute_P(z, args.probe)
    pi = z["prompt_idx"]; ci = z["cycle_idx"]
    nacc = z["n_acc"]
    va = np.where(pi < args.holdout_prompts_lt)[0]
    # Sort by (prompt, cycle) to ensure chronological order within prompt
    order = sorted(va, key=lambda i: (int(pi[i]), int(ci[i])))
    order = np.array(order)
    P_va = P[order]; nacc_va = nacc[order].astype(int)
    print(f"val cycles = {len(order)}, true mean n_acc = {nacc_va.mean():.2f}")
    table = SHIFT_TABLE[args.model]

    # Sweep
    strategies = []
    for rule in ["first", "last"]:
        for thr in [0.2, 0.3, 0.5, 0.7]:
            for off in [-1, 0, 1, 2, 3, 5]:
                strategies.append((rule, thr, off))

    print(f"\n{'rule':<6} {'thr':>5} {'off':>4} {'TPF':>6} {'P(under)':>9} {'P(match)':>9} {'P(over)':>8} {'shortfall_mean':>15}")
    for rule, thr, off in strategies:
        pred_b = np.array([
            (decode_first(P_va[i], thr) if rule == "first" else decode_last(P_va[i], thr)) + off
            for i in range(len(P_va))
        ])
        pred_b = np.clip(pred_b, 0, args.K)
        result = simulate_tpf(pred_b, nacc_va, table, args.K)
        # Shortfall stats for under-pred
        under_mask = pred_b < nacc_va
        shortfall_mean = (nacc_va[under_mask] - pred_b[under_mask]).mean() if under_mask.sum() > 0 else 0
        print(f"{rule:<6} {thr:>5.2f} {off:>+4d} {result['tpf']:>6.3f} {result['p_under']*100:>8.1f}% {result['p_match']*100:>8.1f}% {result['p_over']*100:>7.1f}% {shortfall_mean:>15.2f}")


if __name__ == "__main__":
    main()
