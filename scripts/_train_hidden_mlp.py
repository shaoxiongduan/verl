"""Per-position 2-layer MLP probe on last-layer hidden state + aux features.

Predicts before_boundary[j] = int(j < n_acc[cycle]).

Architecture: Linear(d_in → h) → ReLU → Dropout(0.1) → Linear(h → 1) → sigmoid.
Trained with weighted BCE (pos_weight) + L2 via Adam.

Saves: W1, b1, W2, b2, mu, sd, with_aux, hidden_dim.
"""
from __future__ import annotations
import argparse, time
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--out", required=True)
    p.add_argument("--with_aux", action="store_true")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--iters", type=int, default=800)
    p.add_argument("--lr", type=float, default=0.01)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--hidden", type=int, default=128)
    p.add_argument("--dropout", type=float, default=0.1)
    p.add_argument("--pos_weight", type=float, default=1.0)
    p.add_argument("--target", choices=["accept", "before_boundary"], default="before_boundary")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def auc(scores, labels):
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0: return float("nan")
    combined = np.concatenate([pos, neg]); order = combined.argsort()
    ranks = np.empty_like(order, dtype=np.float64); ranks[order] = np.arange(1, len(combined)+1)
    return (ranks[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))


def relu(x): return np.maximum(0, x)


def main():
    args = parse_args()
    print(f"loading {args.data}")
    z = np.load(args.data)
    H = z["hidden"]; A = z["accept"]; nacc = z["n_acc"]
    N, K, d = H.shape
    print(f"N={N}  K={K}  d_hidden={d}")

    X_h = H.reshape(N * K, d).astype(np.float32)
    if args.target == "accept":
        y = A.reshape(N * K).astype(np.float32)
    else:
        Y = (np.arange(K)[None, :] < nacc[:, None]).astype(np.float32)
        y = Y.reshape(N * K)
    print(f"target={args.target}  positive rate={y.mean():.4f}")

    if args.with_aux:
        aux_feats = [z["top1_prob"].reshape(N*K).astype(np.float32),
                     z["entropy"].reshape(N*K).astype(np.float32),
                     z["margin_log"].reshape(N*K).astype(np.float32),
                     z["top5_prob_sum"].reshape(N*K).astype(np.float32),
                     np.tile(np.arange(K, dtype=np.float32) / K, N)]
        X = np.concatenate([X_h, np.stack(aux_feats, axis=1)], axis=1)
    else:
        X = X_h
    d_in = X.shape[1]
    print(f"feature dim = {d_in}")

    mu = X.mean(axis=0); sd = X.std(axis=0) + 1e-6
    Xs = ((X - mu) / sd).astype(np.float32)

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    n_tr = int(N * (1 - args.val_frac))
    tr_cyc = perm[:n_tr]; va_cyc = perm[n_tr:]
    tr_idx = np.concatenate([np.arange(c*K, (c+1)*K) for c in tr_cyc])
    va_idx = np.concatenate([np.arange(c*K, (c+1)*K) for c in va_cyc])
    print(f"split: tr={len(tr_idx)} va={len(va_idx)}")

    h_dim = args.hidden
    rng2 = np.random.default_rng(args.seed)
    W1 = rng2.normal(0, 1.0/np.sqrt(d_in), (d_in, h_dim)).astype(np.float32)
    b1 = np.zeros(h_dim, dtype=np.float32)
    W2 = rng2.normal(0, 1.0/np.sqrt(h_dim), (h_dim, 1)).astype(np.float32)
    b2 = np.zeros(1, dtype=np.float32)

    m_W1 = np.zeros_like(W1); v_W1 = np.zeros_like(W1)
    m_b1 = np.zeros_like(b1); v_b1 = np.zeros_like(b1)
    m_W2 = np.zeros_like(W2); v_W2 = np.zeros_like(W2)
    m_b2 = np.zeros_like(b2); v_b2 = np.zeros_like(b2)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    lr = args.lr; l2 = args.l2
    pw = args.pos_weight

    print(f"MLP[{d_in}-{h_dim}-1] iters={args.iters} lr={lr} l2={l2} dropout={args.dropout} pos_weight={pw}")
    t0 = time.time()
    Xtr_full = Xs[tr_idx]; ytr = y[tr_idx]
    Xva_full = Xs[va_idx]; yva = y[va_idx]
    w_tr = np.where(ytr > 0.5, pw, 1.0).astype(np.float32)
    n = len(ytr)
    # Minibatch SGD (mb size = 4096)
    mb = 4096
    for it in range(args.iters):
        idx = rng2.choice(n, size=mb, replace=False)
        Xb = Xtr_full[idx]; yb = ytr[idx]; wb = w_tr[idx]
        # Forward with dropout
        h_pre = Xb @ W1 + b1
        h_act = relu(h_pre)
        if args.dropout > 0:
            mask = (rng2.random(h_act.shape) > args.dropout).astype(np.float32) / (1 - args.dropout)
            h_act_drop = h_act * mask
        else:
            mask = None; h_act_drop = h_act
        z_out = (h_act_drop @ W2 + b2).flatten()
        p = 1 / (1 + np.exp(-z_out))
        # Weighted BCE gradient
        diff = wb * (p - yb)
        # Backprop
        dz = diff / len(yb)
        gW2 = h_act_drop.T @ dz[:, None] + l2 * W2
        gb2 = dz.sum(keepdims=True)
        dh_drop = dz[:, None] @ W2.T
        if mask is not None: dh = dh_drop * mask
        else: dh = dh_drop
        dh[h_pre <= 0] = 0
        gW1 = Xb.T @ dh + l2 * W1
        gb1 = dh.sum(axis=0)
        # Adam updates
        t = it + 1
        for (W, gW, m, v) in [(W1, gW1, m_W1, v_W1), (W2, gW2, m_W2, v_W2)]:
            m[:] = beta1 * m + (1 - beta1) * gW
            v[:] = beta2 * v + (1 - beta2) * gW * gW
            mhat = m / (1 - beta1 ** t); vhat = v / (1 - beta2 ** t)
            W -= lr * mhat / (np.sqrt(vhat) + eps)
        for (B, gB, m, v) in [(b1, gb1, m_b1, v_b1), (b2, gb2, m_b2, v_b2)]:
            m[:] = beta1 * m + (1 - beta1) * gB
            v[:] = beta2 * v + (1 - beta2) * gB * gB
            mhat = m / (1 - beta1 ** t); vhat = v / (1 - beta2 ** t)
            B -= lr * mhat / (np.sqrt(vhat) + eps)
        if it % 50 == 0 or it == args.iters - 1:
            # Full val pass
            h_va = relu(Xva_full @ W1 + b1)
            z_va = (h_va @ W2 + b2).flatten()
            p_va = 1 / (1 + np.exp(-z_va))
            auc_va = auc(p_va, yva.astype(int))
            pred = (p_va > 0.5).astype(int)
            acc_va = (pred == yva.astype(int)).mean()
            tp = ((pred == 1) & (yva == 1)).sum(); fp = ((pred == 1) & (yva == 0)).sum()
            fn = ((pred == 0) & (yva == 1)).sum()
            print(f"  iter {it:>4d}  val_AUC={auc_va:.4f}  val_acc={acc_va:.4f}  FN/FP={fn}/{fp}  ({time.time()-t0:.1f}s)")
    # Final
    h_va = relu(Xva_full @ W1 + b1)
    p_va = 1 / (1 + np.exp(-(h_va @ W2 + b2).flatten()))
    print(f"\nfinal val_AUC={auc(p_va, yva.astype(int)):.4f}")
    np.savez(args.out, W1=W1, b1=b1, W2=W2, b2=b2, mu=mu, sd=sd,
             with_aux=args.with_aux, hidden_dim=h_dim)
    print(f"saved {args.out}")


if __name__ == "__main__":
    main()
