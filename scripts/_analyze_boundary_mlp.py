"""Push further: tiny MLP and richer feature combinations.

Compares:
  1. 5-feature logistic regression: (top1_prob, entropy, margin, top5_sum, pos)
  2. 5-feature → 16h MLP (1 hidden layer, ReLU)
  3. Adds top-5 probabilities (10 extra features) → 15-feature MLP

Reports AUC, accuracy, and per-position accept-rate calibration.
"""
from __future__ import annotations
import argparse, json
import numpy as np


def load_features(fn: str, with_top5: bool = False):
    pos, accept, top1, ent, marg, t5sum = [], [], [], [], [], []
    top5_probs = []
    for line in open(fn):
        r = json.loads(line)
        K = len(r["accept"])
        for j in range(K):
            pos.append(j)
            accept.append(r["accept"][j])
            top1.append(r["top1_prob"][j])
            ent.append(r["entropy"][j])
            marg.append(r["margin_log"][j])
            t5sum.append(r["top5_prob_sum"][j])
            if with_top5:
                top5_probs.append(r["top5_probs"][j])
    X = np.stack([top1, ent, marg, t5sum, [p/32.0 for p in pos]], axis=1).astype(np.float64)
    if with_top5:
        T5 = np.array(top5_probs, dtype=np.float64)
        X = np.concatenate([X, T5], axis=1)
    y = np.array(accept, dtype=np.float64)
    return X, y


def auc(scores, labels):
    pos = scores[labels == 1]; neg = scores[labels == 0]
    if len(pos) == 0 or len(neg) == 0:
        return float("nan")
    combined = np.concatenate([pos, neg]); order = combined.argsort()
    ranks = np.empty_like(order, dtype=np.float64); ranks[order] = np.arange(1, len(combined)+1)
    return (ranks[:len(pos)].sum() - len(pos)*(len(pos)+1)/2) / (len(pos)*len(neg))


def standardize(X):
    mu = X.mean(axis=0); sd = X.std(axis=0) + 1e-6
    return (X - mu) / sd, mu, sd


def logreg(X, y, iters=600, lr=0.05):
    Xb = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    w = np.zeros(Xb.shape[1])
    n = len(Xb)
    for it in range(iters):
        z = Xb @ w; p = 1 / (1 + np.exp(-z))
        grad = Xb.T @ (p - y) / n
        w -= lr * grad
    z = Xb @ w; p = 1 / (1 + np.exp(-z))
    return p, w


def relu(x): return np.maximum(0, x)


def mlp_train(X, y, hidden=16, iters=600, lr=0.05, seed=0):
    rng = np.random.default_rng(seed)
    d = X.shape[1]
    W1 = rng.normal(0, 0.5, (d, hidden)) / np.sqrt(d)
    b1 = np.zeros(hidden)
    W2 = rng.normal(0, 0.5, (hidden, 1)) / np.sqrt(hidden)
    b2 = np.zeros(1)
    n = len(X)
    for it in range(iters):
        h = relu(X @ W1 + b1)
        z = (h @ W2 + b2).flatten()
        p = 1 / (1 + np.exp(-z))
        dp = (p - y) / n
        dW2 = h.T @ dp[:, None]
        db2 = dp.sum(keepdims=True)
        dh = dp[:, None] @ W2.T
        dh[h <= 0] = 0
        dW1 = X.T @ dh
        db1 = dh.sum(axis=0)
        W1 -= lr * dW1; b1 -= lr * db1; W2 -= lr * dW2; b2 -= lr * db2
        if it % 200 == 0:
            loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
            print(f"    iter {it} loss={loss:.4f}")
    h = relu(X @ W1 + b1); z = (h @ W2 + b2).flatten()
    p = 1 / (1 + np.exp(-z))
    return p


def eval_model(name, scores, y):
    pred = (scores > 0.5).astype(int)
    a = (pred == y.astype(int)).mean()
    auc_v = auc(scores, y.astype(int))
    print(f"  [{name}] acc={a:.4f}  AUC={auc_v:.4f}")
    return auc_v


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--label", default="")
    args = p.parse_args()
    print(f"[mlp] {args.label}  loading {args.data}")
    X5, y = load_features(args.data, with_top5=False)
    X15, _ = load_features(args.data, with_top5=True)
    print(f"  N positions = {len(y)}, accept rate = {y.mean():.4f}")
    # 80/20 train/val split — simple deterministic split
    n = len(y)
    rng = np.random.default_rng(0)
    perm = rng.permutation(n)
    tr = perm[: int(0.8 * n)]; va = perm[int(0.8 * n) :]

    print("\n5-feature logistic regression (top1_prob, entropy, margin, top5_sum, pos):")
    Xs, mu, sd = standardize(X5[tr])
    Xv = (X5[va] - mu) / sd
    p, w = logreg(Xs, y[tr])
    pv = 1 / (1 + np.exp(-(np.concatenate([Xv, np.ones((len(Xv),1))], 1) @ w)))
    print(f"    train AUC={auc(p, y[tr].astype(int)):.4f}")
    eval_model("logreg-5  val", pv, y[va])
    print(f"    weights std-scale (top1,ent,marg,t5sum,pos,bias) = {w}")

    print("\n5-feature MLP-16:")
    p_mlp_tr = mlp_train(Xs, y[tr], hidden=16, iters=600, lr=0.05)
    # apply
    # (use the trained W1/b1/W2/b2 from a re-run — simpler: re-train and capture)
    rng = np.random.default_rng(0); d=Xs.shape[1]; h=16
    W1 = rng.normal(0, 0.5, (d, h))/np.sqrt(d); b1=np.zeros(h)
    W2 = rng.normal(0, 0.5, (h, 1))/np.sqrt(h); b2=np.zeros(1)
    lr=0.05
    for it in range(600):
        H = relu(Xs @ W1 + b1); z = (H @ W2 + b2).flatten(); pp = 1/(1+np.exp(-z))
        dp = (pp - y[tr])/len(tr)
        dW2 = H.T @ dp[:,None]; db2 = dp.sum(keepdims=True)
        dH = dp[:,None] @ W2.T; dH[H<=0]=0
        dW1 = Xs.T @ dH; db1 = dH.sum(axis=0)
        W1 -= lr*dW1; b1 -= lr*db1; W2 -= lr*dW2; b2 -= lr*db2
    H = relu(Xv @ W1 + b1); z = (H @ W2 + b2).flatten(); pv = 1/(1+np.exp(-z))
    eval_model("mlp-5     val", pv, y[va])

    print("\n15-feature MLP-32 (5 base + top-5 probs):")
    Xs15, mu15, sd15 = standardize(X15[tr])
    Xv15 = (X15[va] - mu15) / sd15
    rng = np.random.default_rng(0); d=Xs15.shape[1]; h=32
    W1 = rng.normal(0, 0.5, (d, h))/np.sqrt(d); b1=np.zeros(h)
    W2 = rng.normal(0, 0.5, (h, 1))/np.sqrt(h); b2=np.zeros(1)
    lr=0.05
    for it in range(800):
        H = relu(Xs15 @ W1 + b1); z = (H @ W2 + b2).flatten(); pp = 1/(1+np.exp(-z))
        dp = (pp - y[tr])/len(tr)
        dW2 = H.T @ dp[:,None]; db2 = dp.sum(keepdims=True)
        dH = dp[:,None] @ W2.T; dH[H<=0]=0
        dW1 = Xs15.T @ dH; db1 = dH.sum(axis=0)
        W1 -= lr*dW1; b1 -= lr*db1; W2 -= lr*dW2; b2 -= lr*db2
        if it % 200 == 0:
            loss = -np.mean(y[tr]*np.log(pp+1e-9)+(1-y[tr])*np.log(1-pp+1e-9))
            print(f"    iter {it} loss={loss:.4f}")
    H = relu(Xv15 @ W1 + b1); z = (H @ W2 + b2).flatten(); pv = 1/(1+np.exp(-z))
    eval_model("mlp-15    val", pv, y[va])


if __name__ == "__main__":
    main()
