"""Quickly compute boundary-MAE separately on train and val splits to expose
overfit gap."""
from __future__ import annotations
import argparse
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--probe", required=True)
    p.add_argument("--threshold", type=float, default=0.3)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    z = np.load(args.data)
    pz = np.load(args.probe, allow_pickle=True)
    H = z["hidden"]; nacc = z["n_acc"]
    top1 = z["top1_prob"]; ent = z["entropy"]; marg = z["margin_log"]; t5s = z["top5_prob_sum"]
    N, K, d = H.shape
    W = pz["W"].astype(np.float32); b = float(pz["b"])
    mu = pz["mu"].astype(np.float32); sd = pz["sd"].astype(np.float32)
    with_aux = bool(pz["with_aux"])

    X_h = H.reshape(N * K, d).astype(np.float32)
    if with_aux:
        feats = [top1.reshape(N*K).astype(np.float32),
                 ent.reshape(N*K).astype(np.float32),
                 marg.reshape(N*K).astype(np.float32),
                 t5s.reshape(N*K).astype(np.float32),
                 np.tile(np.arange(K, dtype=np.float32) / K, N)]
        X = np.concatenate([X_h, np.stack(feats, axis=1)], axis=1)
    else:
        X = X_h
    Xs = (X - mu) / sd
    P = (1 / (1 + np.exp(-(Xs @ W + b)))).reshape(N, K)

    pred_b = []
    for i in range(N):
        below = np.where(P[i] < args.threshold)[0]
        pred_b.append(int(below[0]) if len(below) else K)
    pred_b = np.array(pred_b); true_b = nacc.astype(int)
    err = pred_b - true_b

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    n_tr = int(N * (1 - args.val_frac))
    tr = perm[:n_tr]; va = perm[n_tr:]

    def stats(idx, name):
        e = err[idx]
        print(f"  {name}: N={len(idx)}  MAE={np.abs(e).mean():.3f}  signed={e.mean():+.3f}  ±1={(np.abs(e)<=1).mean():.3f}  ±2={(np.abs(e)<=2).mean():.3f}  ±3={(np.abs(e)<=3).mean():.3f}")
        # Tail
        worst = np.sort(np.abs(e))[::-1][:10]
        print(f"        worst |err| (top 10): {worst.tolist()}")

    print(f"probe: {args.probe}  threshold={args.threshold}")
    stats(np.arange(N), "ALL    ")
    stats(tr, "TRAIN  ")
    stats(va, "VAL    ")


if __name__ == "__main__":
    main()
