"""Extended boundary visualizations.

Three sets of panels (each saves a separate PNG):
  1. 24 RANDOM cycles (unbiased sample)
  2. 16 WORST cases (largest |pred_b_firstcrossing - true_b|)
  3. 16 BEST cases (predicted exactly right)
Plus one aggregate plot: median + 25/75 quantile bands of probe(P) as a
function of position OFFSET from true_b, averaged over all val cycles.
"""
from __future__ import annotations
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from transformers import AutoTokenizer


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--data", required=True)
    p.add_argument("--probe", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--out_prefix", required=True)
    p.add_argument("--threshold", type=float, default=0.3, help="for first-crossing pred_b")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--val_only", action="store_true", help="restrict viz to val split only (matches probe's training split)")
    p.add_argument("--val_frac", type=float, default=0.2)
    return p.parse_args()


def first_crossing(P_row, thr):
    K = len(P_row)
    below = np.where(P_row < thr)[0]
    return int(below[0]) if len(below) else K


def render_panels(picks, tok, P, top1, ent, marg, warm, nacc, args, out_path, title):
    n = len(picks)
    cols = 4
    rows = (n + cols - 1) // cols
    fig, axes = plt.subplots(rows, cols, figsize=(cols * 5.5, rows * 3.6), squeeze=False)
    plt.subplots_adjust(hspace=0.9, wspace=0.25)
    K = P.shape[1]
    js = np.arange(K)
    for k, idx in enumerate(picks):
        ax = axes[k // cols][k % cols]
        true_b = int(nacc[idx])
        pred_b = first_crossing(P[idx], args.threshold)
        ax.plot(js, top1[idx].astype(np.float32), label="top1_p", color="C0", linewidth=1.5)
        ax.plot(js, np.clip(ent[idx].astype(np.float32) / 8.0, 0, 1), label="entropy/8", color="C1", linewidth=1.0, linestyle="--")
        ax.plot(js, np.clip(marg[idx].astype(np.float32) / 8.0, 0, 1), label="margin/8", color="C2", linewidth=1.0, linestyle=":")
        ax.plot(js, P[idx], label="probe P", color="C3", linewidth=2.0)
        ax.axhline(args.threshold, color="gray", linewidth=0.5, alpha=0.5)
        ax.axvline(true_b - 0.5, color="red", linewidth=2.0, alpha=0.8, label=f"true_b={true_b}")
        ax.axvline(pred_b - 0.5, color="purple", linewidth=1.5, alpha=0.7, linestyle="--", label=f"pred_b={pred_b}")
        err = pred_b - true_b
        sign = "+" if err >= 0 else ""
        ax.set_xlim(-0.5, K - 0.5); ax.set_ylim(-0.05, 1.1)
        ax.set_title(f"n_acc={true_b} pred={pred_b} err={sign}{err}  (cyc#{idx})", fontsize=9)
        ax.set_xlabel("position j", fontsize=8)
        if k % cols == 0:
            ax.set_ylabel("feature value", fontsize=8)
        toks = warm[idx].tolist()
        decoded = []
        for j in range(K):
            s = tok.decode([int(toks[j])])
            s = s.replace("\n", "\\n").replace(" ", "·")
            if len(s) > 8: s = s[:7] + "…"
            decoded.append(s)
        for j in range(K):
            color = "darkgreen" if j < true_b else "darkred"
            ax.text(j, -0.18 - (j % 2) * 0.08, decoded[j], rotation=90, fontsize=5, ha="center", va="top", color=color)
        ax.set_xticks(range(0, K, 4))
        ax.tick_params(axis="both", labelsize=7)
        if k == 0:
            ax.legend(fontsize=6, loc="upper right")
    for kk in range(n, rows * cols):
        axes[kk // cols][kk % cols].axis("off")
    plt.suptitle(title, fontsize=11)
    plt.savefig(out_path, dpi=140, bbox_inches="tight")
    print(f"saved {out_path}")
    plt.close()


def main():
    args = parse_args()
    z = np.load(args.data)
    pz = np.load(args.probe, allow_pickle=True)
    H = z["hidden"]; nacc = z["n_acc"]
    top1 = z["top1_prob"]; ent = z["entropy"]; marg = z["margin_log"]; t5s = z["top5_prob_sum"]
    warm = z["warm_argmax"]
    N, K, d = H.shape

    W = pz["W"].astype(np.float32); b = float(pz["b"])
    mu = pz["mu"].astype(np.float32); sd = pz["sd"].astype(np.float32)
    with_aux = bool(pz["with_aux"])

    X_h = H.reshape(N * K, d).astype(np.float32)
    if with_aux:
        aux_feats = [top1.reshape(N*K).astype(np.float32),
                     ent.reshape(N*K).astype(np.float32),
                     marg.reshape(N*K).astype(np.float32),
                     t5s.reshape(N*K).astype(np.float32),
                     np.tile(np.arange(K, dtype=np.float32) / K, N)]
        X = np.concatenate([X_h, np.stack(aux_feats, axis=1)], axis=1)
    else:
        X = X_h
    Xs = (X - mu) / sd
    P = (1 / (1 + np.exp(-(Xs @ W + b)))).reshape(N, K).astype(np.float32)

    # Compute pred_b for ALL cycles
    pred_b_all = np.array([first_crossing(P[i], args.threshold) for i in range(N)])
    err_all = pred_b_all - nacc.astype(int)
    print(f"global: MAE={np.abs(err_all).mean():.3f}  signed={err_all.mean():+.3f}  ±1={(np.abs(err_all)<=1).mean():.3f}  ±2={(np.abs(err_all)<=2).mean():.3f}")

    rng = np.random.default_rng(args.seed)
    if args.val_only:
        perm = rng.permutation(N)
        n_tr = int(N * (1 - args.val_frac))
        eligible = perm[n_tr:]
        print(f"VAL-ONLY mode: restricting to {len(eligible)} cycles")
    else:
        eligible = np.arange(N)
    rng2 = np.random.default_rng(args.seed + 1)

    tok = AutoTokenizer.from_pretrained(args.tokenizer)

    # 1. RANDOM 24 (from eligible)
    rand_idx = rng2.choice(eligible, size=min(24, len(eligible)), replace=False)
    rand_idx = sorted(rand_idx, key=lambda i: int(nacc[i]))
    render_panels(rand_idx, tok, P, top1, ent, marg, warm, nacc, args,
                  args.out_prefix + "_random.png",
                  f"RANDOM 24 cycles  (probe MAE={np.abs(err_all).mean():.2f}, ±1={(np.abs(err_all)<=1).mean():.1%})")

    # 2. WORST 16 by |err| (within eligible set)
    worst_within = sorted(eligible, key=lambda i: -abs(int(err_all[i])))
    worst_idx = worst_within[:16]
    render_panels(worst_idx, tok, P, top1, ent, marg, warm, nacc, args,
                  args.out_prefix + "_worst.png",
                  f"WORST 16 by |pred-true| (left = larger absolute error)")

    # 3. BEST 16 by err==0 (within eligible set)
    zero_idx = np.array([i for i in eligible if err_all[i] == 0])
    if len(zero_idx) >= 16:
        sel = rng2.choice(zero_idx, size=16, replace=False)
    else:
        sel = zero_idx
    sel = sorted(sel, key=lambda i: int(nacc[i]))
    render_panels(sel, tok, P, top1, ent, marg, warm, nacc, args,
                  args.out_prefix + "_best.png",
                  f"BEST: 16 cycles where pred_b == true_b (across all n_acc)")

    # 4. Aggregate: probe value at position true_b + offset (over eligible cycles)
    offsets = np.arange(-8, 8)
    rows = []
    for i in eligible:
        tb = int(nacc[i])
        for off in offsets:
            j = tb + int(off)
            if 0 <= j < K:
                rows.append((int(off), float(P[i][j])))
    rows = np.array(rows)
    medians = []; lo = []; hi = []
    for off in offsets:
        v = rows[rows[:, 0] == off, 1]
        if len(v) > 0:
            medians.append(np.median(v)); lo.append(np.quantile(v, 0.25)); hi.append(np.quantile(v, 0.75))
        else:
            medians.append(np.nan); lo.append(np.nan); hi.append(np.nan)
    fig, ax = plt.subplots(figsize=(7, 4.5))
    ax.plot(offsets, medians, color="C3", linewidth=2.5, label="median P")
    ax.fill_between(offsets, lo, hi, color="C3", alpha=0.2, label="25/75 quantile")
    ax.axhline(args.threshold, color="gray", linewidth=0.8, linestyle="--", label=f"thr={args.threshold}")
    ax.axvline(0, color="red", linewidth=1.2, alpha=0.7, label="true boundary")
    ax.set_xlabel("position offset from true boundary (true_b + offset)")
    ax.set_ylabel("probe P_correct")
    ax.set_title(f"Aggregate probe transition across all {N} cycles (median + IQR)")
    ax.legend()
    plt.savefig(args.out_prefix + "_aggregate.png", dpi=140, bbox_inches="tight")
    print(f"saved {args.out_prefix}_aggregate.png")


if __name__ == "__main__":
    main()
