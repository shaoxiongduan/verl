"""Stacked boundary classifier.

Stage 1 (already done): per-position linear probe → P_correct[j] for j=0..K-1.
Stage 2 (this script): tiny K → 1 model that maps the K per-position probabilities
into a single boundary estimate.

Architecture: input = P_correct[0..K-1] (32-dim) + optionally K extra logit-derived
features (top1, entropy, margin, top5_sum at each position) flattened.
Two heads tested:
  A) Regression: linear K → 1, target = n_acc (continuous) + L2 + Huber loss.
  B) Classification: linear K → (K+1), softmax over boundary location.

Use regression (head A) by default — simpler and matches "predict a single
value of where it is."
"""
from __future__ import annotations
import argparse, time
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help="boundary_h_*.npz")
    p.add_argument("--probe", required=True, help="trained stage-1 per-pos probe.npz")
    p.add_argument("--out", required=True)
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--iters", type=int, default=2000)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--l2", type=float, default=1e-2)
    p.add_argument("--head", choices=["regression", "classification"], default="regression")
    p.add_argument("--add_aux", action="store_true", help="also include per-position top1_p / entropy / margin / top5_sum as features")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    print(f"loading {args.data}")
    z = np.load(args.data)
    pz = np.load(args.probe, allow_pickle=True)
    H = z["hidden"]; nacc = z["n_acc"].astype(np.int64)
    top1 = z["top1_prob"].astype(np.float32); ent = z["entropy"].astype(np.float32)
    marg = z["margin_log"].astype(np.float32); t5s = z["top5_prob_sum"].astype(np.float32)
    N, K, d = H.shape
    print(f"N={N}  K={K}  d_hidden={d}")

    # Compute stage-1 P_correct[j] for every cycle (vectorized)
    W1 = pz["W"].astype(np.float32); b1 = float(pz["b"])
    mu1 = pz["mu"].astype(np.float32); sd1 = pz["sd"].astype(np.float32)
    with_aux1 = bool(pz["with_aux"])
    X_h = H.reshape(N * K, d).astype(np.float32)
    if with_aux1:
        aux_feats = [top1.reshape(N*K), ent.reshape(N*K), marg.reshape(N*K), t5s.reshape(N*K),
                     np.tile(np.arange(K, dtype=np.float32) / K, N)]
        X1 = np.concatenate([X_h, np.stack(aux_feats, axis=1)], axis=1)
    else:
        X1 = X_h
    X1s = (X1 - mu1) / sd1
    P1 = (1 / (1 + np.exp(-(X1s @ W1 + b1)))).reshape(N, K).astype(np.float32)

    # Stage-2 feature: K P values per cycle, optionally augmented with raw features
    if args.add_aux:
        feats = [P1, top1, ent / 8.0, np.clip(marg / 8.0, 0, 1), t5s]
        X = np.concatenate(feats, axis=1).astype(np.float32)
    else:
        X = P1
    d_feat = X.shape[1]
    print(f"feature dim = {d_feat}  (P[0..K-1]{'+aux' if args.add_aux else ''})")

    # Standardize
    mu = X.mean(axis=0); sd = X.std(axis=0) + 1e-6
    Xs = (X - mu) / sd
    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    n_tr = int(N * (1 - args.val_frac))
    tr = perm[:n_tr]; va = perm[n_tr:]
    Xtr = Xs[tr]; Xva = Xs[va]
    ytr = nacc[tr].astype(np.float32); yva = nacc[va].astype(np.float32)
    print(f"split: tr={len(tr)} va={len(va)}")

    if args.head == "regression":
        # Linear regression with Huber loss
        W = np.zeros(d_feat, dtype=np.float32); b = float(ytr.mean())
        m_W = np.zeros_like(W); v_W = np.zeros_like(W); m_b = 0.0; v_b = 0.0
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        delta = 2.0  # Huber threshold
        n = len(ytr)
        print(f"regression train: {args.iters} iters lr={args.lr} l2={args.l2} Huber_δ={delta}")
        t0 = time.time()
        for it in range(args.iters):
            pred = Xtr @ W + b
            r = pred - ytr  # residual
            # Huber gradient: |r| <= δ → r; else δ * sign(r)
            gd_r = np.where(np.abs(r) <= delta, r, delta * np.sign(r)).astype(np.float32)
            gW = (Xtr.T @ gd_r) / n + args.l2 * W
            gb = gd_r.mean()
            t = it + 1
            m_W = beta1 * m_W + (1 - beta1) * gW; v_W = beta2 * v_W + (1 - beta2) * gW * gW
            mW_hat = m_W / (1 - beta1 ** t); vW_hat = v_W / (1 - beta2 ** t)
            W -= args.lr * mW_hat / (np.sqrt(vW_hat) + eps)
            m_b = beta1 * m_b + (1 - beta1) * gb; v_b = beta2 * v_b + (1 - beta2) * gb * gb
            mb_hat = m_b / (1 - beta1 ** t); vb_hat = v_b / (1 - beta2 ** t)
            b -= args.lr * mb_hat / (np.sqrt(vb_hat) + eps)
            if it % 200 == 0 or it == args.iters - 1:
                # val MAE on the rounded prediction
                pv = Xva @ W + b
                pv_int = np.clip(np.round(pv), 0, K).astype(int)
                yva_int = yva.astype(int)
                mae = np.abs(pv_int - yva_int).mean()
                mae_cont = np.abs(pv - yva).mean()
                signed = (pv_int - yva_int).mean()
                w1 = (np.abs(pv_int - yva_int) <= 1).mean()
                w2 = (np.abs(pv_int - yva_int) <= 2).mean()
                elapsed = time.time() - t0
                print(f"  iter {it:>4d}  MAE_int={mae:.3f}  MAE={mae_cont:.3f}  signed={signed:+.3f}  ±1={w1:.3f}  ±2={w2:.3f} ({elapsed:.1f}s)")
        out_dict = {"W": W, "b": np.array([b], dtype=np.float32), "mu": mu, "sd": sd,
                    "with_aux": args.add_aux, "K": K, "head": "regression"}
    else:
        # Classification softmax(K+1)
        n_cls = K + 1
        W = np.zeros((d_feat, n_cls), dtype=np.float32); b = np.zeros(n_cls, dtype=np.float32)
        m_W = np.zeros_like(W); v_W = np.zeros_like(W); m_b = np.zeros_like(b); v_b = np.zeros_like(b)
        beta1, beta2, eps = 0.9, 0.999, 1e-8
        n = len(ytr)
        Y_tr = np.zeros((n, n_cls), dtype=np.float32); Y_tr[np.arange(n), ytr.astype(int)] = 1.0
        print(f"classification train: {args.iters} iters")
        for it in range(args.iters):
            z_tr = Xtr @ W + b
            e = np.exp(z_tr - z_tr.max(axis=1, keepdims=True)); p = e / e.sum(axis=1, keepdims=True)
            diff = (p - Y_tr)
            gW = (Xtr.T @ diff) / n + args.l2 * W
            gb = diff.mean(axis=0)
            t = it + 1
            m_W = beta1 * m_W + (1 - beta1) * gW; v_W = beta2 * v_W + (1 - beta2) * gW * gW
            mW_hat = m_W / (1 - beta1 ** t); vW_hat = v_W / (1 - beta2 ** t)
            W -= args.lr * mW_hat / (np.sqrt(vW_hat) + eps)
            m_b = beta1 * m_b + (1 - beta1) * gb; v_b = beta2 * v_b + (1 - beta2) * gb * gb
            mb_hat = m_b / (1 - beta1 ** t); vb_hat = v_b / (1 - beta2 ** t)
            b -= args.lr * mb_hat / (np.sqrt(vb_hat) + eps)
            if it % 200 == 0 or it == args.iters - 1:
                z_v = Xva @ W + b
                e_v = np.exp(z_v - z_v.max(axis=1, keepdims=True)); p_v = e_v / e_v.sum(axis=1, keepdims=True)
                pred = p_v.argmax(axis=1)
                yva_int = yva.astype(int)
                mae = np.abs(pred - yva_int).mean()
                signed = (pred - yva_int).mean()
                w1 = (np.abs(pred - yva_int) <= 1).mean()
                w2 = (np.abs(pred - yva_int) <= 2).mean()
                print(f"  iter {it:>4d}  MAE={mae:.3f}  signed={signed:+.3f}  ±1={w1:.3f}  ±2={w2:.3f}")
        out_dict = {"W": W, "b": b, "mu": mu, "sd": sd,
                    "with_aux": args.add_aux, "K": K, "head": "classification"}

    np.savez(args.out, **out_dict)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
