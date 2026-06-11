"""Global boundary classifier: takes ALL K hidden states + aux features per
cycle as input, outputs a (K+1)-way softmax over boundary location, predicts
n_acc via argmax.

Architecture: flatten(K * (d_hidden + 5)) → Linear → softmax(K+1).
Loss: cross-entropy with true n_acc as the class label.

Saves: W (n_features, K+1), b (K+1,), mu, sd, with_aux, K.
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
    p.add_argument("--iters", type=int, default=600)
    p.add_argument("--lr", type=float, default=0.05)
    p.add_argument("--l2", type=float, default=1e-1,  # strong default given low data + huge feature dim
                   help="L2 reg coefficient. Default 0.1 — needed given ~1100 train cycles vs 100k features.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def main():
    args = parse_args()
    z = np.load(args.data)
    H = z["hidden"]; A = z["accept"]; nacc = z["n_acc"]
    N, K, d = H.shape
    print(f"[gbc] N={N}  K={K}  d_hidden={d}", flush=True)

    # Per-cycle feature: flatten K hidden states + (optionally) K aux per position
    X = H.reshape(N, K * d).astype(np.float32)
    aux_keys = ["top1_prob", "entropy", "margin_log", "top5_prob_sum"]
    if args.with_aux:
        aux = []
        for k in aux_keys:
            aux.append(z[k].astype(np.float32))  # (N, K)
        pos_idx = np.tile(np.arange(K, dtype=np.float32) / K, (N, 1))
        aux.append(pos_idx)
        aux_stack = np.stack(aux, axis=-1)  # (N, K, 5)
        X = np.concatenate([X, aux_stack.reshape(N, -1)], axis=1)
    print(f"[gbc] feature dim per cycle = {X.shape[1]}  ({'+aux' if args.with_aux else 'hidden only'})", flush=True)
    n_classes = K + 1
    y = nacc.astype(np.int64)  # (N,)
    print(f"[gbc] n_classes = {n_classes} (boundary in [0..K])  class distribution head: {np.bincount(y, minlength=n_classes)[:15]}", flush=True)

    # Standardize features
    mu = X.mean(axis=0); sd = X.std(axis=0) + 1e-6
    Xs = (X - mu) / sd

    rng = np.random.default_rng(args.seed)
    perm = rng.permutation(N)
    n_tr = int(N * (1 - args.val_frac))
    tr = perm[:n_tr]; va = perm[n_tr:]
    Xtr = Xs[tr]; ytr = y[tr]
    Xva = Xs[va]; yva = y[va]
    print(f"[gbc] split: tr={len(tr)} va={len(va)}", flush=True)

    # Logits = Xs @ W + b ; W: (d_feat, K+1), b: (K+1,)
    d_feat = X.shape[1]
    W = np.zeros((d_feat, n_classes), dtype=np.float32)
    b = np.zeros(n_classes, dtype=np.float32)

    # Adam
    m_W = np.zeros_like(W); v_W = np.zeros_like(W)
    m_b = np.zeros_like(b); v_b = np.zeros_like(b)
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    lr = args.lr; l2 = args.l2
    print(f"[gbc] training {args.iters} iters  lr={lr}  l2={l2}", flush=True)
    t0 = time.time()
    n = len(ytr)
    # one-hot encoding
    Y_tr = np.zeros((n, n_classes), dtype=np.float32)
    Y_tr[np.arange(n), ytr] = 1.0

    for it in range(args.iters):
        z_tr = Xtr @ W + b  # (n, K+1)
        # log-softmax for numerical stability
        z_max = z_tr.max(axis=1, keepdims=True)
        e = np.exp(z_tr - z_max)
        p = e / e.sum(axis=1, keepdims=True)
        # gradient: (p - Y) for cross-entropy with softmax
        diff = (p - Y_tr)
        gW = (Xtr.T @ diff) / n + l2 * W
        gb = diff.mean(axis=0)
        # Adam
        t = it + 1
        m_W = beta1 * m_W + (1 - beta1) * gW
        v_W = beta2 * v_W + (1 - beta2) * gW * gW
        mW_hat = m_W / (1 - beta1 ** t)
        vW_hat = v_W / (1 - beta2 ** t)
        W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        m_b = beta1 * m_b + (1 - beta1) * gb
        v_b = beta2 * v_b + (1 - beta2) * gb * gb
        mb_hat = m_b / (1 - beta1 ** t)
        vb_hat = v_b / (1 - beta2 ** t)
        b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

        if it % 50 == 0 or it == args.iters - 1:
            loss_tr = -np.mean(np.log(p[np.arange(n), ytr] + 1e-9))
            z_va = Xva @ W + b
            z_va_max = z_va.max(axis=1, keepdims=True)
            e_va = np.exp(z_va - z_va_max); p_va = e_va / e_va.sum(axis=1, keepdims=True)
            pred_va = p_va.argmax(axis=1)
            acc_va = (pred_va == yva).mean()
            # Boundary error metrics
            err = pred_va - yva
            mae = np.abs(err).mean(); mse = (err ** 2).mean()
            within_1 = (np.abs(err) <= 1).mean(); within_2 = (np.abs(err) <= 2).mean()
            mean_signed = err.mean()
            elapsed = time.time() - t0
            print(f"[gbc] iter {it:>4d} tr_loss={loss_tr:.4f} val_acc={acc_va:.4f}  MAE={mae:.3f}  ±1={within_1:.3f}  ±2={within_2:.3f}  mean_err={mean_signed:+.2f} ({elapsed:.1f}s)", flush=True)

    # Final eval + per-class breakdown
    z_va = Xva @ W + b
    e_va = np.exp(z_va - z_va.max(axis=1, keepdims=True)); p_va = e_va / e_va.sum(axis=1, keepdims=True)
    pred_va = p_va.argmax(axis=1)
    err = pred_va - yva
    print(f"\n[gbc] FINAL val_acc={(pred_va == yva).mean():.4f}  MAE={np.abs(err).mean():.3f}  ±1={(np.abs(err) <= 1).mean():.3f}  ±2={(np.abs(err) <= 2).mean():.3f}", flush=True)
    print(f"[gbc] signed-err hist (pred-true): {[int(x) for x in np.histogram(err, bins=[-32,-10,-5,-2,0,2,5,10,33])[0]]}")
    print(f"[gbc]   bins:                     [-32,-10) [-10,-5) [-5,-2) [-2,0) [0,2) [2,5) [5,10) [10,33)")
    np.savez(args.out, W=W, b=b, mu=mu, sd=sd, with_aux=args.with_aux, K=K)
    print(f"[gbc] saved {args.out}", flush=True)


if __name__ == "__main__":
    main()
