"""Feasibility analysis for boundary-predictor: how well do simple features
(top1_prob, entropy, margin, position) separate accept from reject positions?

Loads the JSONL dump from _collect_boundary_dataset.py and computes:
  1. Per-feature accept-vs-reject distribution stats (mean, std).
  2. Threshold-based per-position classifier ROC/AUC (top1_prob, entropy).
  3. Boundary detection accuracy: predict n_acc = first j where feature crosses threshold; measure |pred - true|.
  4. Position-conditional analysis: do features behave differently at j=0 vs j=20?
  5. Simple logistic regression with feature combinations (no sklearn dep —
     numpy gradient descent on small data).
"""
from __future__ import annotations
import argparse, json, math
import numpy as np


def load_jsonl(fn: str):
    """Flatten cycles into per-position rows. Returns dict of np arrays:
      pos (N,), accept (N,), top1_prob (N,), entropy (N,), margin (N,),
      top5_sum (N,), L_committed (N,), cycle_idx (N,), n_acc (N,)
    """
    rows = []
    cycles = []
    for line in open(fn):
        rec = json.loads(line)
        cycles.append({"n_acc": rec["n_acc"], "K": len(rec["accept"]),
                       "top1_prob": rec["top1_prob"], "entropy": rec["entropy"],
                       "margin_log": rec["margin_log"], "accept": rec["accept"]})
        K = len(rec["accept"])
        for j in range(K):
            rows.append({
                "pos": j,
                "accept": rec["accept"][j],
                "top1_prob": rec["top1_prob"][j],
                "entropy": rec["entropy"][j],
                "margin": rec["margin_log"][j],
                "top5_sum": rec["top5_prob_sum"][j],
                "L_committed": rec["L_committed"],
                "cycle_idx": rec["cycle_idx"],
                "n_acc": rec["n_acc"],
            })
    keys = ["pos", "accept", "top1_prob", "entropy", "margin", "top5_sum",
            "L_committed", "cycle_idx", "n_acc"]
    arr = {k: np.array([r[k] for r in rows], dtype=np.float64) for k in keys}
    return arr, cycles


def per_feature_stats(arr):
    a = arr["accept"].astype(bool)
    print("\n=== Per-position feature stats (accept vs reject) ===")
    print(f"{'feature':<14} {'accept μ':>10} {'accept σ':>10} {'reject μ':>10} {'reject σ':>10} {'sep (Δ/σ)':>12}")
    for k in ["top1_prob", "entropy", "margin", "top5_sum"]:
        acc_v = arr[k][a]; rej_v = arr[k][~a]
        mu_a, sd_a = acc_v.mean(), acc_v.std()
        mu_r, sd_r = rej_v.mean(), rej_v.std()
        pooled_sd = math.sqrt(0.5 * (sd_a**2 + sd_r**2)) + 1e-9
        sep = abs(mu_a - mu_r) / pooled_sd
        print(f"{k:<14} {mu_a:>10.4f} {sd_a:>10.4f} {mu_r:>10.4f} {sd_r:>10.4f} {sep:>12.3f}")
    n_pos = len(arr["accept"]); n_acc = a.sum()
    print(f"\nN positions = {n_pos}, accept rate = {n_acc/n_pos:.4f}")


def auc_score(scores, labels):
    """Compute ROC AUC by Mann-Whitney U formula. Higher score should mean 'accept'."""
    pos = scores[labels == 1]
    neg = scores[labels == 0]
    n_pos, n_neg = len(pos), len(neg)
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    # rank-based AUC
    combined = np.concatenate([pos, neg])
    order = combined.argsort()
    ranks = np.empty_like(order, dtype=np.float64)
    ranks[order] = np.arange(1, len(combined) + 1)
    sum_ranks_pos = ranks[:n_pos].sum()
    return (sum_ranks_pos - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg)


def threshold_classifier(arr):
    print("\n=== Per-position binary classifier — feature thresholds ===")
    labels = arr["accept"].astype(int)
    for k, sign in [("top1_prob", +1), ("entropy", -1), ("margin", +1), ("top5_sum", +1)]:
        # sign=+1: high score → accept ; sign=-1: low score → accept (entropy)
        s = sign * arr[k]
        auc = auc_score(s, labels)
        # Find best threshold
        sorted_s = np.sort(s)
        # Subsample to ~200 thresholds for speed
        step = max(1, len(sorted_s) // 200)
        best = (0.0, 0.0, 0.0, 0.0, 0.0)  # (acc, thr, tp, fp, fn)
        for thr in sorted_s[::step]:
            pred = (s >= thr).astype(int)
            tp = ((pred == 1) & (labels == 1)).sum()
            tn = ((pred == 0) & (labels == 0)).sum()
            fp = ((pred == 1) & (labels == 0)).sum()
            fn = ((pred == 0) & (labels == 1)).sum()
            acc = (tp + tn) / len(labels)
            if acc > best[0]:
                best = (acc, thr, tp, fp, fn)
        acc, thr, tp, fp, fn = best
        tpr = tp / max(1, tp + fn); fpr = fp / max(1, fp + (len(labels) - labels.sum()))
        print(f"  {k:<12} AUC={auc:.4f}  best_acc={acc:.4f}  (sign={sign})  thr={thr:.3f}  TPR={tpr:.3f}  FPR={fpr:.3f}")


def position_conditional_accept(arr):
    print("\n=== Accept rate vs position j ===")
    print(f"{'j':>4} {'n':>8} {'accept_rate':>12} {'mean(top1_prob)':>16} {'mean(entropy)':>14}")
    for j in range(0, int(arr["pos"].max()) + 1):
        mask = arr["pos"] == j
        n = mask.sum()
        if n == 0: continue
        ar = arr["accept"][mask].mean()
        tp = arr["top1_prob"][mask].mean()
        en = arr["entropy"][mask].mean()
        print(f"{j:>4} {int(n):>8} {ar:>12.4f} {tp:>16.4f} {en:>14.4f}")


def boundary_prediction_error(arr, cycles, feature="top1_prob", thr=0.5, sign=+1):
    """Predict n_acc as: first j where feature falls below threshold (if sign=+1)
    or rises above threshold (if sign=-1).  Compare to true n_acc per cycle."""
    errs = []
    over = under = exact = 0
    for c in cycles:
        K = c["K"]
        # Recompute boundary from feature
        f = np.array(c[feature]) if feature in c else None
        if f is None:
            continue
        if sign == +1:
            below = np.where(f < thr)[0]
            pred = int(below[0]) if len(below) else K
        else:
            above = np.where(f > thr)[0]
            pred = int(above[0]) if len(above) else K
        true = c["n_acc"]
        errs.append(pred - true)
        if pred > true:
            over += 1
        elif pred < true:
            under += 1
        else:
            exact += 1
    errs = np.array(errs)
    print(f"\n=== Boundary prediction error (feature={feature}, thr={thr}, sign={sign}) ===")
    print(f"  n_cycles={len(errs)}  exact={exact}  over={over}  under={under}")
    print(f"  mean(pred - true) = {errs.mean():+.3f}")
    print(f"  mean |pred - true| = {np.abs(errs).mean():.3f}")
    print(f"  median |pred - true| = {np.median(np.abs(errs)):.3f}")
    print(f"  abs err distribution: <=0: {(np.abs(errs) <= 0).mean():.3f}  <=1: {(np.abs(errs) <= 1).mean():.3f}  <=2: {(np.abs(errs) <= 2).mean():.3f}  <=3: {(np.abs(errs) <= 3).mean():.3f}  <=5: {(np.abs(errs) <= 5).mean():.3f}")
    # In our protocol over-estimating boundary is BAD (we keep wrong tokens),
    # under-estimating is OK (we reinit more positions than needed = wasted noise but no harm).
    # Show signed-error histogram
    bins = [-32, -10, -5, -3, -1, 0, 1, 3, 5, 10, 33]
    hist, _ = np.histogram(errs, bins=bins)
    print(f"  signed-err histogram (pred-true):")
    for i in range(len(hist)):
        print(f"    [{bins[i]:>+3d}, {bins[i+1]:>+3d}): {hist[i]:>5d}  ({hist[i]/len(errs)*100:.1f}%)")


def logreg_2feat(arr):
    """Tiny logistic regression on (top1_prob, entropy) → accept. Numpy GD.
    Useful sanity-check: does combining features help over either alone?"""
    X = np.stack([arr["top1_prob"], arr["entropy"], arr["pos"] / 32.0], axis=1)
    # add bias column
    X = np.concatenate([X, np.ones((len(X), 1))], axis=1)
    y = arr["accept"].astype(np.float64)
    # Normalize features
    mu = X[:, :3].mean(axis=0); sd = X[:, :3].std(axis=0) + 1e-6
    X[:, :3] = (X[:, :3] - mu) / sd
    w = np.zeros(X.shape[1])
    lr = 0.05
    n = len(X)
    for it in range(400):
        z = X @ w
        p = 1 / (1 + np.exp(-z))
        grad = X.T @ (p - y) / n
        w -= lr * grad
        if it % 100 == 0:
            loss = -np.mean(y * np.log(p + 1e-9) + (1 - y) * np.log(1 - p + 1e-9))
            print(f"  [logreg] iter {it} loss={loss:.4f}")
    z = X @ w
    p = 1 / (1 + np.exp(-z))
    pred = (p > 0.5).astype(int)
    acc = (pred == arr["accept"].astype(int)).mean()
    auc = auc_score(p, arr["accept"].astype(int))
    print(f"  [logreg] final accuracy={acc:.4f}  AUC={auc:.4f}")
    print(f"  [logreg] weights (top1_prob, entropy, pos/32, bias): {w}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--feature_thr_top1", type=float, default=0.5)
    p.add_argument("--feature_thr_entropy", type=float, default=2.0)
    args = p.parse_args()

    print(f"[ana] loading {args.data}")
    arr, cycles = load_jsonl(args.data)
    print(f"[ana] cycles={len(cycles)}  positions={len(arr['accept'])}")

    per_feature_stats(arr)
    threshold_classifier(arr)
    position_conditional_accept(arr)

    # Boundary prediction with top1_prob threshold sweep
    for thr in [0.3, 0.5, 0.7, 0.8, 0.9, 0.95]:
        boundary_prediction_error(arr, cycles, feature="top1_prob", thr=thr, sign=+1)

    print("\n=== Tiny logistic regression: (top1_prob, entropy, pos/K) → accept ===")
    logreg_2feat(arr)


if __name__ == "__main__":
    main()
