"""Train a linear probe on last-layer hidden states -> P(warm_argmax == verify_argmax).

Optionally adds the 5 logit-derived features (top1_prob, entropy, margin, top5_sum, pos/K)
as auxiliary inputs.

Trains via numpy-only L2-regularized logistic regression (Newton's method on the
inverse Hessian is too expensive at d=4096 — use Adam/GD).

Saves the trained weights as .npz with keys: W (d+aux,), b (scalar),
                                              mu/sd for hidden + aux features.
"""
from __future__ import annotations
import argparse, json, math, sys, time
import numpy as np


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True, help=".npz from _collect_boundary_dataset_hidden.py")
    p.add_argument("--out", required=True, help="output .npz with W, b, mu, sd")
    p.add_argument("--with_aux", action="store_true",
                   help="include the 5 logit-derived features in addition to hidden state")
    p.add_argument("--val_frac", type=float, default=0.2)
    p.add_argument("--iters", type=int, default=400)
    p.add_argument("--lr", type=float, default=0.1)
    p.add_argument("--l2", type=float, default=1e-3)
    p.add_argument("--pos_weight", type=float, default=1.0,
                   help="Multiply y=1 BCE term by this. >1 penalizes false negatives more (underreporting).")
    p.add_argument("--target", choices=["accept", "before_boundary"], default="accept",
                   help="Label: 'accept' = warm[j]==verify[j] (noisy). 'before_boundary' = j < n_acc (monotonic).")
    p.add_argument("--holdout_prompts_lt", type=int, default=None,
                   help="If set: train on cycles with prompt_idx >= N; eval val on cycles with prompt_idx < N. Overrides val_frac.")
    p.add_argument("--seed", type=int, default=0)
    return p.parse_args()


def auc(scores, labels):
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    combined = np.concatenate([pos, neg]); order = combined.argsort()
    ranks = np.empty_like(order, dtype=np.float64); ranks[order] = np.arange(1, len(combined)+1)
    return (ranks[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))


def main():
    args = parse_args()
    print(f"[probe] loading {args.data}", flush=True)
    z = np.load(args.data)
    H = z["hidden"]  # (N, K, d) fp16
    A = z["accept"]  # (N, K) int8
    N, K, d = H.shape
    print(f"[probe] N_cycles={N}  K={K}  d_hidden={d}  total_pos={N*K}", flush=True)

    # Flatten to per-position
    X_h = H.reshape(N * K, d).astype(np.float32)
    if args.target == "accept":
        y = A.reshape(N * K).astype(np.float32)
    else:
        # before_boundary: per-position label = 1 iff j < n_acc[cycle]
        nacc = z["n_acc"]  # (N,)
        Y = (np.arange(K)[None, :] < nacc[:, None]).astype(np.float32)
        y = Y.reshape(N * K)
    print(f"[probe] target={args.target}  positive rate = {y.mean():.4f}", flush=True)

    aux_keys = ["top1_prob", "entropy", "margin_log", "top5_prob_sum"]
    if args.with_aux:
        aux_feats = []
        for k in aux_keys:
            aux_feats.append(z[k].reshape(N * K).astype(np.float32))
        pos = np.tile(np.arange(K, dtype=np.float32) / K, N)
        aux_feats.append(pos)
        X_aux = np.stack(aux_feats, axis=1)  # (N*K, 5)
        X = np.concatenate([X_h, X_aux], axis=1)
        print(f"[probe] feature dim = {X.shape[1]} (hidden {d} + aux 5)")
    else:
        X = X_h
        print(f"[probe] feature dim = {X.shape[1]} (hidden only)")

    # Standardize features
    mu = X.mean(axis=0); sd = X.std(axis=0) + 1e-6
    Xs = (X - mu) / sd
    print(f"[probe] standardized; ||mu||={np.linalg.norm(mu):.4f}  median sd={np.median(sd):.4f}")

    # Train/val split
    rng = np.random.default_rng(args.seed)
    if args.holdout_prompts_lt is not None:
        pi = z["prompt_idx"]
        tr_cyc = np.where(pi >= args.holdout_prompts_lt)[0]
        va_cyc = np.where(pi < args.holdout_prompts_lt)[0]
        print(f"[probe] PROMPT-LEVEL split: train on prompt_idx >= {args.holdout_prompts_lt} ({len(tr_cyc)} cycles); val on prompt_idx < {args.holdout_prompts_lt} ({len(va_cyc)} cycles)")
    else:
        perm = rng.permutation(N)
        n_tr = int(N * (1 - args.val_frac))
        tr_cyc = perm[:n_tr]; va_cyc = perm[n_tr:]
    tr_idx = np.concatenate([np.arange(c*K, (c+1)*K) for c in tr_cyc])
    va_idx = np.concatenate([np.arange(c*K, (c+1)*K) for c in va_cyc])
    print(f"[probe] split: tr={len(tr_idx)}  va={len(va_idx)}")

    # Initialize linear model
    W = np.zeros(Xs.shape[1], dtype=np.float32)
    b = 0.0
    Xtr = Xs[tr_idx]; ytr = y[tr_idx]
    Xva = Xs[va_idx]; yva = y[va_idx]

    # Adam optimizer
    m_W = np.zeros_like(W); v_W = np.zeros_like(W)
    m_b = 0.0; v_b = 0.0
    beta1, beta2, eps = 0.9, 0.999, 1e-8
    lr = args.lr
    l2 = args.l2

    pw = args.pos_weight
    print(f"[probe] training {args.iters} iters, lr={lr}, l2={l2}, pos_weight={pw}", flush=True)
    # per-sample weight: alpha for y=1, 1 for y=0
    w_tr = np.where(ytr > 0.5, pw, 1.0).astype(np.float32)
    t0 = time.time()
    for it in range(args.iters):
        z_tr = Xtr @ W + b
        p_tr = 1 / (1 + np.exp(-z_tr))
        # weighted gradient
        diff = (p_tr - ytr) * w_tr
        gW = (Xtr.T @ diff) / len(ytr) + l2 * W
        gb = diff.mean()
        # Adam
        m_W = beta1 * m_W + (1 - beta1) * gW
        v_W = beta2 * v_W + (1 - beta2) * gW * gW
        mW_hat = m_W / (1 - beta1 ** (it + 1))
        vW_hat = v_W / (1 - beta2 ** (it + 1))
        W -= lr * mW_hat / (np.sqrt(vW_hat) + eps)
        m_b = beta1 * m_b + (1 - beta1) * gb
        v_b = beta2 * v_b + (1 - beta2) * gb * gb
        mb_hat = m_b / (1 - beta1 ** (it + 1))
        vb_hat = v_b / (1 - beta2 ** (it + 1))
        b -= lr * mb_hat / (np.sqrt(vb_hat) + eps)

        if it % 50 == 0 or it == args.iters - 1:
            loss_tr = -np.mean(w_tr * (ytr * np.log(p_tr + 1e-9) + (1 - ytr) * np.log(1 - p_tr + 1e-9)))
            z_va = Xva @ W + b
            p_va = 1 / (1 + np.exp(-z_va))
            auc_va = auc(p_va, yva.astype(int))
            pred = (p_va > 0.5).astype(int)
            acc_va = (pred == yva.astype(int)).mean()
            tp = ((pred == 1) & (yva == 1)).sum(); fp = ((pred == 1) & (yva == 0)).sum()
            fn = ((pred == 0) & (yva == 1)).sum(); tn = ((pred == 0) & (yva == 0)).sum()
            elapsed = time.time() - t0
            print(f"[probe] iter {it:>4d} tr_loss={loss_tr:.4f} val_AUC={auc_va:.4f} val_acc={acc_va:.4f}  FN/FP={fn}/{fp} ({elapsed:.1f}s)", flush=True)

    # Final eval
    z_va = Xva @ W + b
    p_va = 1 / (1 + np.exp(-z_va))
    print(f"\n[probe] final val_AUC={auc(p_va, yva.astype(int)):.4f}", flush=True)

    np.savez(args.out, W=W, b=b, mu=mu, sd=sd, with_aux=args.with_aux,
             aux_keys=np.array(aux_keys, dtype="<U16"))
    print(f"[probe] saved {args.out}")


if __name__ == "__main__":
    main()
