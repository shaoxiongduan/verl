"""Evaluate a per-position probe's BOUNDARY-LEVEL accuracy.

For each cycle in the (held-out) val split:
  - apply probe to per-position features → P[j] for j=0..K-1
  - predicted_b = first j where P[j] < threshold (else K)
  - true_b = n_acc
  - err = predicted_b - true_b
  - report MAE, mean signed err, and bias direction

Threshold sweep: 0.3, 0.5, 0.7.
"""
from __future__ import annotations
import argparse, json
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--probe", required=True)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--target", choices=["accept", "before_boundary"], default="before_boundary")
    p.add_argument("--thresholds", type=str, default="0.2,0.3,0.4,0.5,0.6,0.7,0.8")
    return p.parse_args()


def main():
    args = parse_args()
    z = np.load(args.data)
    pz = np.load(args.probe, allow_pickle=True)
    H = z["hidden"]; nacc = z["n_acc"]
    N, K, d = H.shape
    W = pz["W"].astype(np.float32); b = float(pz["b"])
    mu = pz["mu"].astype(np.float32); sd = pz["sd"].astype(np.float32)
    with_aux = bool(pz["with_aux"])
    print(f"N={N}  K={K}  d_hidden={d}  with_aux={with_aux}  threshold sweep on val split")

    aux_keys = ["top1_prob", "entropy", "margin_log", "top5_prob_sum"]
    # Build the same flat per-position feature matrix as the trainer
    X_h = H.reshape(N * K, d).astype(np.float32)
    if with_aux:
        aux_feats = []
        for k in aux_keys:
            aux_feats.append(z[k].reshape(N * K).astype(np.float32))
        pos = np.tile(np.arange(K, dtype=np.float32) / K, N)
        aux_feats.append(pos)
        X = np.concatenate([X_h, np.stack(aux_feats, axis=1)], axis=1)
    else:
        X = X_h
    Xs = (X - mu) / sd
    P = 1 / (1 + np.exp(-(Xs @ W + b)))  # (N*K,)
    P = P.reshape(N, K)

    # Use the SAME split as the trainer
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    n_tr = int(N * (1 - args.val_frac))
    va = perm[n_tr:]
    P_va = P[va]
    n_va = nacc[va]
    print(f"val n_cycles = {len(va)}")
    print(f"true n_acc distribution head: {np.bincount(n_va, minlength=K+1)[:10]}")
    print(f"true mean n_acc = {n_va.mean():.3f}")

    print(f"\n{'thr':>5}  {'mean(pred_b)':>13}  {'mean(true_b)':>13}  {'mean signed':>11}  {'MAE':>6}  {'pred<true':>10}  {'pred>true':>10}  {'within±1':>9}  {'within±2':>9}  {'within±3':>9}")
    for thr in [float(x) for x in args.thresholds.split(",")]:
        pred_b = []
        for i in range(len(P_va)):
            row = P_va[i]
            below = np.where(row < thr)[0]
            b_pred = int(below[0]) if len(below) else K
            pred_b.append(b_pred)
        pred_b = np.array(pred_b)
        err = pred_b - n_va
        mae = np.abs(err).mean()
        signed = err.mean()
        pl = (pred_b < n_va).mean()
        pg = (pred_b > n_va).mean()
        w1 = (np.abs(err) <= 1).mean()
        w2 = (np.abs(err) <= 2).mean()
        w3 = (np.abs(err) <= 3).mean()
        print(f"{thr:>5.2f}  {pred_b.mean():>13.3f}  {n_va.mean():>13.3f}  {signed:>+11.3f}  {mae:>6.3f}  {pl:>10.3f}  {pg:>10.3f}  {w1:>9.3f}  {w2:>9.3f}  {w3:>9.3f}")


if __name__ == "__main__":
    main()
